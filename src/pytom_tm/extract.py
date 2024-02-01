from packaging import version
import pandas as pd
import numpy as np
import numpy.typing as npt
import logging
import scipy.ndimage as ndimage
import pathlib
from typing import Optional
from pytom_tm.tmjob import TMJob
from pytom_tm.mask import spherical_mask
from pytom_tm.angles import load_angle_list, convert_euler
from pytom_tm.io import read_mrc
from scipy.special import erfcinv
from scipy.optimize import curve_fit
from tqdm import tqdm


plotting_available = False
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(context='talk', style='ticks')
    plotting_available = True
except ModuleNotFoundError:
    pass


def predict_tophat_mask(
        score_volume: npt.NDArray[float],
        output_path: Optional[pathlib.Path] = None,
        n_false_positives: int = 1,
) -> npt.NDArray[int]:
    tophat = ndimage.white_tophat(
        score_volume,
        structure=ndimage.generate_binary_structure(
            rank=3,
            connectivity=1
        )
    )
    y, bins = np.histogram(tophat.flatten(), bins=50)
    x_raw, y_raw = (bins[2:-1] + bins[3:]) / 2, y[2:]  # discard first two bin because of over-representation of zeros

    # take second derivative and discard inaccurate boundary value (hence the [2:])
    with np.errstate(divide='ignore'):
        y_log = np.where(np.log(y_raw) == np.inf, 0, np.log(y_raw))
    second_derivative = np.gradient(np.gradient(y_log))[2:]
    m1 = second_derivative[:-1] < 0  # where the derivative is negative
    m2 = np.sign(second_derivative[1:] * second_derivative[:-1]) == -1  # switches from neg. to pos. and vice versa
    idx = (
            int(np.argwhere(np.all(np.vstack([m1, m2]), axis=0))[0])  # first switch from negative to positive
            + 2  # +2 for 2nd derivative discarded part
            + 1  # +1 to include last value
    )
    x_fit, y_fit = x_raw[:idx], y_raw[:idx]

    def gauss(x, amp, mu, sigma):  # gaussian for fitting
        return amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    def log_gauss(x, amp, mu, sigma):  # log of gaussian for fitting
        return np.log(amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)))

    guess = np.array([y.max(), 0, score_volume.std()])  # rough initial guess for amp, mu and sigma
    coeff = curve_fit(gauss, x_fit, y_fit, p0=guess)[0]  # first fit regular gauss to get better guesses
    coeff_log = curve_fit(log_gauss, x_fit, np.log(y_fit), p0=coeff)[0]  # now go for accurate fit to log of gauss
    search_space = coeff_log[0] / (coeff_log[2] * np.sqrt(2 * np.pi))
    cut_off = erfcinv((2 * n_false_positives) / search_space) * np.sqrt(2) * coeff_log[2] + coeff_log[1]

    if plotting_available and output_path is not None:
        fig, ax = plt.subplots()
        ax.scatter(x_raw, y_raw, label='scores', marker='o')
        ax.plot(x_raw, gauss(x_raw, *coeff_log), label='pred', color='tab:orange')
        ax.axvline(cut_off, color='gray', linestyle='dashed', label='cut-off')
        ax.axvspan(x_fit[0], x_fit[-1], alpha=0.25, color='gray', label='fitted data')
        ax.set_yscale('log')
        ax.set_ylim(bottom=.1)
        ax.set_ylabel('Occurence')
        ax.set_xlabel('Tophat scores')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, transparent=False, bbox_inches='tight')

    peak_mask = (tophat > cut_off)

    return peak_mask


def extract_particles(
        job: TMJob,
        particle_radius_px: int,
        n_particles: int,
        n_false_positives: int = 1,
        cut_off: Optional[float] = None,
        tophat_filter: bool = False,
        tomogram_mask_path: Optional[pathlib.Path] = None,
) -> tuple[pd.DataFrame, list[float, ...]]:
    """
    Parameters
    ----------
    job: pytom_tm.tmjob.TMJob
        template matching job to annotate particles from
    particle_radius_px: int
        particle radius to remove peaks with after a score has been annotated
    n_particles: int
        maximum number of particles to extract
    n_false_positives: int
        tune the number of false positives to be included for automated error function cut-off estimation: default is 1;
        lowering the value up to zero reduces sensitivity; increasing the value above 1 increases sensitivity,
        it should roughly relate to the number of false positives (i.e. 500 expects 500 FPs)
    cut_off: Optional[float]
        manually override the automated score cut-off estimation, value between 0 and 1
    tophat_filter: bool
        attempt to only select sharp peaks with the tophat filter
    tomogram_mask_path: Optional[pathlib.Path]
        path to a tomographic binary mask for extraction

    Returns
    -------
    dataframe, scores: tuple[pd.DataFrame, list[float, ...]]
        dataframe with annotations that can be written out as a STAR file and a list of the selected scores
    """

    score_volume = read_mrc(job.output_dir.joinpath(f'{job.tomo_id}_scores.mrc'))
    angle_volume = read_mrc(job.output_dir.joinpath(f'{job.tomo_id}_angles.mrc'))
    angle_list = load_angle_list(
        job.rotation_file,
        sort_angles=version.parse(job.pytom_tm_version_number) > version.parse('0.3.0')
    )

    if tophat_filter:  # constrain the extraction with a tophat filter
        predicted_peaks = predict_tophat_mask(
            score_volume,
            output_path=job.output_dir.joinpath(f'{job.tomo_id}_tophat_filter.svg'),
            n_false_positives=n_false_positives,
        )
        score_volume *= predicted_peaks  # multiply with predicted peaks to keep only those

    # apply tomogram mask if provided
    if tomogram_mask_path is not None:
        tomogram_mask = read_mrc(tomogram_mask_path)[
            job.search_origin[0]: job.search_origin[0] + job.search_size[0],
            job.search_origin[1]: job.search_origin[1] + job.search_size[1],
            job.search_origin[2]: job.search_origin[2] + job.search_size[2]
        ]  # mask should be larger than zero in regions of interest!
        score_volume[tomogram_mask <= 0] = 0

    # mask edges of score volume
    score_volume[0: particle_radius_px, :, :] = 0
    score_volume[:, 0: particle_radius_px, :] = 0
    score_volume[:, :, 0: particle_radius_px] = 0
    score_volume[-particle_radius_px:, :, :] = 0
    score_volume[:, -particle_radius_px:, :] = 0
    score_volume[:, :, -particle_radius_px:] = 0

    sigma = job.job_stats['std']
    if cut_off is None:
        # formula rickgauer (2017) should be: 10**-13 = erfc( theta / ( sigma * sqrt(2) ) ) / 2
        search_space = (
            # wherever the score volume has not been explicitly set to -1 is the size of the search region
            (score_volume > -1).sum() *
            int(np.ceil(job.n_rotations / job.rotational_symmetry))
        )
        cut_off = erfcinv((2 * n_false_positives) / search_space) * np.sqrt(2) * sigma
        logging.info(f'cut off for particle extraction: {cut_off}')
    elif cut_off < 0:
        logging.warning('Provided extraction score cut-off is smaller than 0. Changing to 0 as that is smallest '
                        'allowed value.')
        cut_off = 0

    # mask for iteratively selecting peaks
    cut_box = int(particle_radius_px) * 2 + 1
    cut_mask = (spherical_mask(cut_box, particle_radius_px, cut_box // 2) == 0) * 1

    # data for star file
    pixel_size = job.voxel_size
    tomogram_id = job.tomo_id

    data = []
    scores = []

    for _ in tqdm(range(n_particles)):

        ind = np.unravel_index(score_volume.argmax(), score_volume.shape)

        lcc_max = score_volume[ind]

        if lcc_max <= cut_off:
            break

        scores.append(lcc_max)

        # assumed that here also need to multiply with -1
        rotation = convert_euler(
            [-1 * a for a in angle_list[int(angle_volume[ind])]],
            order_in='ZXZ',
            order_out='ZYZ',
            degrees_in=False,
            degrees_out=True
        )

        location = [i + o for i, o in zip(job.search_origin, ind)]

        data.append((
            location[0],  # CoordinateX
            location[1],  # CoordinateY
            location[2],  # CoordinateZ
            rotation[0],  # AngleRot
            rotation[1],  # AngleTilt
            rotation[2],  # AnglePsi
            lcc_max,  # LCCmax
            cut_off,  # Extraction cut off
            sigma,  # Add sigma of template matching search, LCCmax can be divided by sigma to obtain SNR
            pixel_size,  # DetectorPixelSize
            tomogram_id,  # MicrographName
        ))

        # box out the particle
        start = [i - particle_radius_px for i in ind]
        score_volume[
            start[0]: start[0] + cut_box,
            start[1]: start[1] + cut_box,
            start[2]: start[2] + cut_box
        ] *= cut_mask

    return pd.DataFrame(data, columns=[
        'ptmCoordinateX',
        'ptmCoordinateY',
        'ptmCoordinateZ',
        'ptmAngleRot',
        'ptmAngleTilt',
        'ptmAnglePsi',
        'ptmLCCmax',
        'ptmCutOff',
        'ptmSearchStd',
        'ptmDetectorPixelSize',
        'ptmMicrographName',
    ]), scores
