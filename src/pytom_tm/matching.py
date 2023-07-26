import cupy as cp
import cupy.typing as cpt
import numpy.typing as npt
import voltools as vt
from typing import Optional
from cupyx.scipy.fft import rfftn, irfftn, fftshift
from tqdm import tqdm
from pytom_tm.correlation import mean_under_mask, std_under_mask


class TemplateMatchingPlan:
    def __init__(
            self,
            volume: npt.NDArray[float],
            template: npt.NDArray[float],
            mask: npt.NDArray[float],
            device_id: int,
            wedge: Optional[npt.NDArray[float]] = None
    ):
        # Search volume + and fft transform plan for the volume
        self.volume = cp.asarray(volume, dtype=cp.float32, order='C')
        self.volume_rft = rfftn(self.volume)
        # Explicit fft plan is no longer necessary as cupy generates a plan behind the scene which leads to
        # comparable timings

        # Data for the mask
        self.mask = cp.asarray(mask, dtype=cp.float32, order='C')
        self.mask_texture = vt.StaticVolume(self.mask, interpolation='filt_bspline', device=f'gpu:{device_id}')
        self.mask_padded = cp.zeros_like(self.volume).astype(cp.float32)
        self.mask_weight = self.mask.sum()  # weight of the mask

        # Init template data
        self.template = cp.asarray(template, dtype=cp.float32, order='C')
        self.template_texture = vt.StaticVolume(self.template, interpolation='filt_bspline', device=f'gpu:{device_id}')
        self.template_padded = cp.zeros_like(self.volume)

        # fourier binary wedge weight for the template
        self.wedge = cp.asarray(wedge, order='C', dtype=cp.float32) if wedge is not None else None

        # Initialize result volumes
        self.ccc_map = cp.zeros_like(self.volume)
        self.scores = cp.ones_like(self.volume)*-1000
        self.angles = cp.ones_like(self.volume)*-1000

        # wait for stream to complete the work
        cp.cuda.stream.get_current_stream().synchronize()


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
            wedge: Optional[npt.NDArray[float]] = None
    ):
        cp.cuda.Device(device_id).use()

        self.job_id = job_id
        self.device_id = device_id
        self.active = True
        self.completed = False
        self.mask_is_spherical = mask_is_spherical  # whether mask is spherical
        self.angle_list = angle_list
        self.angle_ids = angle_ids
        self.stats = {'search_space': 0, 'variance': 0., 'std': 0.}

        self.plan = TemplateMatchingPlan(volume, template, mask, device_id, wedge=wedge)

    def run(self):
        print("Progress job_{} on device {:d}:".format(self.job_id, self.device_id))

        # Size x template (sxz) and center x template (cxt)
        sxt, syt, szt = self.plan.template.shape
        cxt, cyt, czt = sxt // 2, syt // 2, szt // 2
        mx, my, mz = sxt % 2, syt % 2, szt % 2  # odd or even

        # Size x volume (sxv) and center x volume (xcv)
        sxv, syv, szv = self.plan.template_padded.shape
        cxv, cyv, czv = sxv // 2, syv // 2, szv // 2

        if self.mask_is_spherical:  # Then we only need to calculate std volume once
            self.plan.mask_padded[cxv - cxt:cxv + cxt + mx,
                                  cyv - cyt:cyv + cyt + my,
                                  czv - czt:czv + czt + mz] = self.plan.mask
            std_volume = std_under_mask_convolution(
                self.plan.volume,
                self.plan.mask_padded,
                self.plan.mask_weight,
                volume_rft=self.plan.volume_rft
            )
        else:
            std_volume = None  # but will be calculated during iterations

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
                self.plan.mask_padded[cxv - cxt:cxv + cxt + mx,
                                      cyv - cyt:cyv + cyt + my,
                                      czv - czt:czv + czt + mz] = self.plan.mask
                # Std volume needs to be recalculated for every rotation of the mask, expensive step
                std_volume = std_under_mask_convolution(
                    self.plan.volume,
                    self.plan.mask_padded,
                    self.plan.mask_weight,
                    volume_rft=self.plan.volume_rft,
                )

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
                ).real

            # Normalize and mask template
            mean = mean_under_mask(self.plan.template, self.plan.mask, mask_weight=self.plan.mask_weight)
            std = std_under_mask(self.plan.template, self.plan.mask, mean, mask_weight=self.plan.mask_weight)
            self.plan.template = ((self.plan.template - mean) / std) * self.plan.mask

            # Paste in center
            self.plan.template_padded[cxv - cxt:cxv + cxt + mx,
                                      cyv - cyt:cyv + cyt + my,
                                      czv - czt:czv + czt + mz] = self.plan.template

            # Fast local correlation function between volume and template, norm is the standard deviation at each
            # point in the volume in the masked area
            self.plan.ccc_map = fftshift(
                irfftn(self.plan.volume_rft * rfftn(self.plan.template_padded).conj(),
                       s=self.plan.template_padded.shape).real
                / (self.plan.mask_weight * std_volume)
            )

            # Update the scores and angle_lists
            update_results_kernel(
                self.plan.scores,
                self.plan.ccc_map,
                angle_id,
                self.plan.scores,
                self.plan.angles
            )

            self.stats['variance'] += (square_sum_kernel(self.plan.ccc_map) / self.plan.volume.size)

        self.stats['search_space'] = int(self.plan.volume.size * len(self.angle_ids))
        self.stats['variance'] = float(self.stats['variance'] / len(self.angle_ids))
        self.stats['std'] = float(cp.sqrt(self.stats['variance']))


def std_under_mask_convolution(volume, padded_mask, mask_weight, volume_rft=None):
    """
    std convolution of volume and mask
    """
    volume_rft = rfftn(volume) if volume_rft is None else volume_rft
    std_v = (
            mean_under_mask_convolution(rfftn(volume ** 2), padded_mask, mask_weight) -
            mean_under_mask_convolution(volume_rft, padded_mask, mask_weight) ** 2
    )
    std_v[std_v <= cp.float32(1e-09)] = 1
    return cp.sqrt(std_v)


def mean_under_mask_convolution(volume_rft, mask, mask_weight):
    """
    mean convolution of volume and mask
    """
    return irfftn(
        volume_rft * rfftn(mask).conj(), s=mask.shape
    ).real / mask_weight


update_results_kernel = cp.ElementwiseKernel(
    'float32 scores, float32 ccc_map, float32 angle_id',
    'float32 out1, float32 out2',
    'if (scores < ccc_map) {out1 = ccc_map; out2 = angle_id;}',
    'update_results'
)


# mean is assumed to be 0 which makes this operation a lot faster
square_sum_kernel = cp.ReductionKernel(
    'T x',  # input params
    'T y',  # output params
    'x * x',  # map
    'a + b',  # reduce
    'y = a',  # post-reduction map
    '0',  # identity value
    'variance'  # kernel name
)
