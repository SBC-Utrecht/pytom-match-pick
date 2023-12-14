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
from tqdm import tqdm


def predict_gold_marker_mask(
        volume: npt.NDArray[float],
        voxel_size: float,
        blob_size: float,
) -> npt.NDArray[int]:
    # gold marker radius in number of voxels
    blob_radius = (blob_size / voxel_size) / 2

    # Laplacian of Gaussian (LoG) filter where sigma is gaussian width corresponding to object size
    sigma = blob_radius / np.sqrt(3)
    log = ndimage.gaussian_laplace(volume, sigma)

    # fudge factor for the number of iterations for binary opening and dilation
    iters = int(blob_radius / (5 / 3))  # 5/3 is a fudge ratio

    mask = ndimage.binary_opening(  # binary opening keeps objects that consist of plenty connected components
        (log > log.std() * 3) * 1,  # LoG is threshold with 3x the standard deviation
        iterations=iters,  # iterations should be more for smaller voxel spacings
        structure=ndimage.generate_binary_structure(
            rank=3, connectivity=1
        )
    )  # dilate strongly to ensure gold is fully covered
    return ndimage.binary_dilation(
        mask,
        iterations=iters + 2,
        structure=ndimage.generate_binary_structure(rank=3, connectivity=2),
    )


def extract_particles(
        job: TMJob,
        particle_radius_px: int,
        n_particles: int,
        n_false_positives: int = 1,
        cut_off: Optional[float] = None,
        tophat_filter_cut_off: Optional[float] = None,
        gold_marker_diameter: Optional[float] = None,
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
    tophat_filter_cut_off: Optional[float]
        float between 0 and 1 to tune the specificity: default is 0.1; value of 1 removes everything, value of 0
        includes everything
    gold_marker_diameter: Optional[float]
        average diameter of gold markers in angstrom
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

    if gold_marker_diameter is not None:
        mask = predict_gold_marker_mask(
            read_mrc(job.tomograms),  # the gold mark mask is created on the tomogram
            job.voxel_size,
            gold_marker_diameter,
        )  # zero all elements where gold beads are predicted
        score_volume[mask == 1] = 0

    if tophat_filter_cut_off is not None:  # constrain the extraction with a tophat filter
        layer1 = ndimage.white_tophat(
            score_volume,
            structure=ndimage.generate_binary_structure(
                rank=3,
                connectivity=1
            )
        )
        layer2 = ndimage.white_tophat(
            score_volume,
            structure=ndimage.generate_binary_structure(
                rank=3,
                connectivity=2
            )
        )
        tophat_filter = layer1 * layer2
        tophat_filter = (tophat_filter - tophat_filter.min()) / (tophat_filter.max() - tophat_filter.min())
        # zero places where the tophat filter has too small values
        score_volume[tophat_filter < tophat_filter_cut_off] = 0  # threshold tophat

    # apply tomogram mask if provided
    if tomogram_mask_path is not None:
        tomogram_mask = read_mrc(tomogram_mask_path)[
            job.search_origin[0]: job.search_origin[0] + job.search_size[0],
            job.search_origin[1]: job.search_origin[1] + job.search_size[1],
            job.search_origin[2]: job.search_origin[2] + job.search_size[2]
        ]
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
