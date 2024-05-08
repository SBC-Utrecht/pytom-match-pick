import cupy as cp
import cupy.typing as cpt
import numpy.typing as npt
import voltools as vt
import gc
from typing import Optional
from cupyx.scipy.fft import rfftn, irfftn
from tqdm import tqdm
from pytom_tm.correlation import mean_under_mask, std_under_mask
from pytom_tm.template import phase_randomize_template


class TemplateMatchingPlan:
    def __init__(
            self,
            volume: npt.NDArray[float],
            template: npt.NDArray[float],
            mask: npt.NDArray[float],
            device_id: int,
            wedge: Optional[npt.NDArray[float]] = None,
            phase_randomized_template: Optional[npt.NDArray[float]] = None,
    ):
        """Initialize a template matching plan. All the necessary cupy arrays will be allocated on the GPU.

        Parameters
        ----------
        volume: npt.NDArray[float]
            3D numpy array representing the search tomogram
        template: npt.NDArray[float]
            3D numpy array representing the template for the search, a square box of size sx
        mask: npt.NDArray[float]
            3D numpy array representing the mask for the search, same dimensions as template
        device_id: int
            GPU device id to load arrays on
        wedge: Optional[npt.NDArray[float]], default None
            3D numpy array that contains the Fourier space weighting for the template, it should be in Fourier
            reduced form, with dimensions (sx, sx, sx // 2 + 1)
        phase_randomized_template: Optional[npt.NDArray[float]], default None
            initialize the plan with a phase randomized version of the template for noise correction
        """
        # Search volume + and fft transform plan for the volume
        volume_shape = volume.shape
        cp_vol = cp.asarray(volume, dtype=cp.float32, order='C')
        self.volume_rft_conj = rfftn(cp_vol).conj()
        self.volume_sq_rft_conj = rfftn(cp_vol ** 2).conj()
        # Explicit fft plan is no longer necessary as cupy generates a plan behind the scene which leads to
        # comparable timings

        # Array for storing local standard deviations
        self.std_volume = cp.zeros(volume_shape, dtype=cp.float32)

        # Data for the mask
        self.mask = cp.asarray(mask, dtype=cp.float32, order='C')
        self.mask_texture = vt.StaticVolume(self.mask, interpolation='filt_bspline', device=f'gpu:{device_id}')
        self.mask_padded = cp.zeros(volume_shape, dtype=cp.float32)
        self.mask_weight = self.mask.sum()  # weight of the mask

        # Init template data
        self.template = cp.asarray(template, dtype=cp.float32, order='C')
        self.template_texture = vt.StaticVolume(self.template, interpolation='filt_bspline', device=f'gpu:{device_id}')
        self.template_padded = cp.zeros(volume_shape, dtype=cp.float32)

        # fourier binary wedge weight for the template
        self.wedge = cp.asarray(wedge, order='C', dtype=cp.float32) if wedge is not None else None

        # Initialize result volumes
        self.ccc_map = cp.zeros(volume_shape, dtype=cp.float32)
        self.scores = cp.ones(volume_shape, dtype=cp.float32)*-1000
        self.angles = cp.ones(volume_shape, dtype=cp.float32)*-1000

        self.random_phase_template_texture = None
        self.noise_scores = None
        if phase_randomized_template is not None:
            self.random_phase_template_texture = vt.StaticVolume(
                cp.asarray(phase_randomized_template, dtype=cp.float32, order='C'),
                interpolation='filt_bspline',
                device=f'gpu:{device_id}',
            )
            self.noise_scores = cp.ones(volume_shape, dtype=cp.float32)*-1000

        # wait for stream to complete the work
        cp.cuda.stream.get_current_stream().synchronize()

    def clean(self) -> None:
        """Remove all stored cupy arrays from the GPU's memory pool."""
        gpu_memory_pool = cp.get_default_memory_pool()
        del (
            self.volume_rft_conj, self.volume_sq_rft_conj, self.mask, self.mask_texture, self.mask_padded,
            self.template, self.template_texture, self.template_padded, self.wedge, self.ccc_map, self.scores,
            self.angles, self.std_volume
        )
        gc.collect()
        gpu_memory_pool.free_all_blocks()


class TemplateMatchingGPU:
    def __init__(
            self,
            job_id: str,
            device_id: int,
            volume: npt.NDArray[float],
            template: npt.NDArray[float],
            mask: npt.NDArray[float],
            angle_list: list[tuple[float, float, float]],
            angle_ids: list[int],
            mask_is_spherical: bool = True,
            wedge: Optional[npt.NDArray[float]] = None,
            stats_roi: Optional[tuple[slice, slice, slice]] = None,
            noise_correction: bool = False,
            rng_seed: int = 321,
    ):
        """Initialize a template matching run.

        For other great implementations see:
        - STOPGAP: https://github.com/wan-lab-vanderbilt/STOPGAP
        - pyTME: https://github.com/KosinskiLab/pyTME

        The precalculation of conjugated FTs of the tomo was (AFAIK) introduced
        by STOPGAP!

        Parameters
        ----------
        job_id: str
            string for job identification
        device_id: int
            GPU device id to run the job on
        volume: npt.NDArray[float]
            3D numpy array of tomogram
        template: npt.NDArray[float]
            3D numpy array of template, square box of size sx
        mask: npt.NDArray[float]
            3D numpy array with mask, same box size as template
        angle_list: list[tuple[float, float, float]]
            list of tuples with 3 floats representing Euler angle rotations
        angle_ids: list[int]
            list of indices for angle_list to actually search, this can be a subset of the full list
        mask_is_spherical: bool, default True
            True (default) if mask is spherical, set to False for non-spherical mask which increases computation time
        wedge: Optional[npt.NDArray[float]], default None
            3D numpy array that contains the Fourier space weighting for the template, it should be in Fourier
            reduced form, with dimensions (sx, sx, sx // 2 + 1)
        stats_roi: Optional[tuple[slice, slice, slice]], default None
            region of interest to calculate statistics on the search volume, default will just take the full search
            volume
        noise_correction: bool, default False
            initialize template matching with a phase randomized version of the template that is used to subtract
            background noise from the score map; expense is more gpu memory and computation time
        rng_seed: int, default 321
            seed for rng in phase randomization
        """
        cp.cuda.Device(device_id).use()

        self.job_id = job_id
        self.device_id = device_id
        self.active = True
        self.completed = False
        self.mask_is_spherical = mask_is_spherical  # whether mask is spherical
        self.angle_list = angle_list
        self.angle_ids = angle_ids
        self.stats = {'search_space': 0, 'variance': 0., 'std': 0.}
        if stats_roi is None:
            self.stats_roi = (
                slice(None),
                slice(None),
                slice(None)
            )
        else:
            self.stats_roi = stats_roi
        self.noise_correction = noise_correction
        shuffled_template = (
            phase_randomize_template(template, rng_seed)
            if noise_correction else None
        )

        self.plan = TemplateMatchingPlan(
            volume,
            template,
            mask,
            device_id,
            wedge=wedge,
            phase_randomized_template=shuffled_template,
        )

    def run(self) -> tuple[npt.NDArray[float], npt.NDArray[float], dict]:
        """Run the template matching job.

        Returns
        -------
        results: tuple[npt.NDArray[float], npt.NDArray[float], dict]
            results is a tuple with tree elements:
                - score_map with the locally normalised maximum score over all the angles searched; a 3D numpy array
                with same size as search volume
                - angle_map with an index into the angle list corresponding to the maximum of the correlation scores;
                a 3D numpy array with same size as search volume
                - a dictionary with three floats of search statistics - 'search_space', 'variance', and 'std'
        """
        print("Progress job_{} on device {:d}:".format(self.job_id, self.device_id))

        # Size x template (sxz) and center x template (cxt)
        sxt, syt, szt = self.plan.template.shape
        cxt, cyt, czt = sxt // 2, syt // 2, szt // 2
        mx, my, mz = sxt % 2, syt % 2, szt % 2  # odd or even

        # Size x volume (sxv) and center x volume (xcv)
        sxv, syv, szv = self.plan.template_padded.shape
        cxv, cyv, czv = sxv // 2, syv // 2, szv // 2

        # create slice for padding
        pad_index = (
            slice(cxv - cxt, cxv + cxt + mx),
            slice(cyv - cyt, cyv + cyt + my),
            slice(czv - czt, czv + czt + mz),
        )

        # calculate roi mask
        shift = cp.floor(cp.array(self.plan.scores.shape) / 2).astype(int) + 1
        roi_mask = cp.zeros(self.plan.scores.shape, dtype=bool)
        roi_mask[self.stats_roi] = True
        roi_mask = cp.flip(cp.roll(roi_mask, -shift, (0, 1, 2)))
        roi_size = self.plan.scores[roi_mask].size

        if self.mask_is_spherical:  # Then we only need to calculate std volume once
            self.plan.mask_padded[pad_index] = self.plan.mask
            self.plan.std_volume = std_under_mask_convolution(
                self.plan.volume_rft_conj,
                self.plan.volume_sq_rft_conj,
                self.plan.mask_padded,
                self.plan.mask_weight,
            ) * self.plan.mask_weight

        # Track iterations with a tqdm progress bar
        for i in tqdm(range(len(self.angle_ids))):

            # tqdm cannot loop over zipped lists, so need to do it like this
            angle_id, rotation = self.angle_ids[i], self.angle_list[i]

            if not self.mask_is_spherical:
                self.plan.mask_texture.transform(
                    rotation=(rotation[0], rotation[1], rotation[2]),
                    rotation_order='rzxz',
                    output=self.plan.mask,
                    rotation_units='rad'
                )
                self.plan.mask_padded[pad_index] = self.plan.mask
                # Std volume needs to be recalculated for every rotation of the mask, expensive step
                self.plan.std_volume = std_under_mask_convolution(
                    self.plan.volume_rft_conj,
                    self.plan.volume_sq_rft_conj,
                    self.plan.mask_padded,
                    self.plan.mask_weight,
                ) * self.plan.mask_weight

            # Rotate template
            self.plan.template_texture.transform(
                rotation=(rotation[0], rotation[1], rotation[2]),
                rotation_order='rzxz',
                output=self.plan.template,
                rotation_units='rad'
            )

            if self.plan.wedge is not None:
                # Add wedge to the template after rotating
                self.plan.template = irfftn(
                    rfftn(self.plan.template) * self.plan.wedge,
                    s=self.plan.template.shape
                )

            # Normalize and mask template
            mean = mean_under_mask(self.plan.template, self.plan.mask, mask_weight=self.plan.mask_weight)
            std = std_under_mask(self.plan.template, self.plan.mask, mean, mask_weight=self.plan.mask_weight)
            self.plan.template = ((self.plan.template - mean) / std) * self.plan.mask

            # Paste in center
            self.plan.template_padded[pad_index] = self.plan.template

            # Fast local correlation function between volume and template, norm is the standard deviation at each
            # point in the volume in the masked area
            self.plan.ccc_map = (
                irfftn(self.plan.volume_rft_conj * rfftn(self.plan.template_padded),
                       s=self.plan.template_padded.shape)
                / self.plan.std_volume
            )

            # Update the scores and angle_lists
            update_results_kernel(
                self.plan.scores,
                self.plan.ccc_map,
                angle_id,
                self.plan.scores,
                self.plan.angles
            )

            self.stats['variance'] += (
                square_sum_kernel(self.plan.ccc_map * roi_mask) / roi_size
            )

        # Get correct orientation back!
        # Use same method as William Wan's STOPGAP
        # (https://doi.org/10.1107/S205979832400295X): the search volume is Fourier
        # transformed and conjugated before the iterations this means the eventual
        # score map needs to be flipped back. The map is also rolled due to the ftshift
        # effect of a Fourier space correlation function.
        self.plan.scores = cp.roll(cp.flip(self.plan.scores), shift, axis=(0, 1, 2))
        self.plan.angles = cp.roll(cp.flip(self.plan.angles), shift, axis=(0, 1, 2))

        self.stats['search_space'] = int(roi_size * len(self.angle_ids))
        self.stats['variance'] = float(self.stats['variance'] / len(self.angle_ids))
        self.stats['std'] = float(cp.sqrt(self.stats['variance']))

        # package results back to the CPU
        results = (self.plan.scores.get(), self.plan.angles.get(), self.stats)

        # clear all the used gpu memory
        self.plan.clean()

        return results


def std_under_mask_convolution(
        volume_rft_conj: cpt.NDArray[float],
        volume_sq_rft_conj: cpt.NDArray[float],
        padded_mask: cpt.NDArray[float],
        mask_weight: float,
) -> cpt.NDArray[float]:
    """Calculate the local standard deviation under the mask for each position in the volume. Calculation is done in
    Fourier space as this is a convolution between volume and mask.

    Parameters
    ----------
    volume_rft_conj: cpt.NDArray[float]
        complex conjugate of the rft of the search volume
    volume_sq_rft_conj: cpt.NDArray[float]
        complex conjugate of the rft of the squared search volume
    padded_mask: cpt.NDArray[float]
        template mask that has been padded to dimensions of volume
    mask_weight: float
        weight of the mask, usually calculated as mask.sum()

    Returns
    -------
    std_v: cpt.NDArray[float]
        array with local standard deviations in volume
    """
    padded_mask_rft = rfftn(padded_mask)
    std_v = (
            irfftn(volume_sq_rft_conj * padded_mask_rft, s=padded_mask.shape) / mask_weight -
            (irfftn(volume_rft_conj * padded_mask_rft, s=padded_mask.shape) / mask_weight) ** 2
    )
    std_v[std_v <= cp.float32(1e-18)] = 1  # prevent potential sqrt of negative value and division by zero
    std_v = cp.sqrt(std_v)
    return std_v


"""Update scores and angles if a new maximum is found."""
update_results_kernel = cp.ElementwiseKernel(
    'float32 scores, float32 ccc_map, float32 angle_id',
    'float32 out1, float32 out2',
    'if (scores < ccc_map) {out1 = ccc_map; out2 = angle_id;}',
    'update_results'
)


"""Calculate the sum of squares in a volume. Mean is assumed to be 0 which makes this operation a lot faster."""
square_sum_kernel = cp.ReductionKernel(
    'T x',  # input params
    'T y',  # output params
    'x * x',  # pre-processing expression
    'a + b',  # reduction operation
    'y = a',  # post-reduction output processing
    '0',  # identity value
    'variance'  # kernel name
)
