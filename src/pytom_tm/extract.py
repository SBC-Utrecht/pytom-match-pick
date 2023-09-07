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


def detect_blobs(
        volume: npt.NDArray[float],
        voxel_size: float,
        blob_size: float,
        threshold: float = 1.
) -> npt.NDArray[int]:
    blob_radius = (blob_size / voxel_size) / 2
    sigma = blob_radius / np.sqrt(3)
    filtered = ndimage.gaussian_laplace((volume - volume.mean()) / volume.std(), sigma)
    return ((filtered > threshold) * 1).astype(int)


def extract_particles(
        job: TMJob,
        particle_radius_px: int,
        n_particles: int,
        cut_off: Optional[float] = None,
        n_false_positives: int = 1,
        tomogram_mask_path: Optional[pathlib.Path] = None
) -> tuple[pd.DataFrame, list[float, ...]]:

    score_volume = read_mrc(job.output_dir.joinpath(f'{job.tomo_id}_scores.mrc'))
    angle_volume = read_mrc(job.output_dir.joinpath(f'{job.tomo_id}_angles.mrc'))
    angle_list = load_angle_list(job.rotation_file)

    # mask edges of score volume
    score_volume[0: particle_radius_px, :, :] = -1
    score_volume[:, 0: particle_radius_px, :] = -1
    score_volume[:, :, 0: particle_radius_px] = -1
    score_volume[-particle_radius_px:, :, :] = -1
    score_volume[:, -particle_radius_px:, :] = -1
    score_volume[:, :, -particle_radius_px:] = -1

    # apply tomogram mask if provided
    if tomogram_mask_path is not None:
        tomogram_mask = read_mrc(tomogram_mask_path)[
            job.search_origin[0]: job.search_origin[0] + job.search_size[0],
            job.search_origin[1]: job.search_origin[1] + job.search_size[1],
            job.search_origin[2]: job.search_origin[2] + job.search_size[2]
        ]
        score_volume[tomogram_mask <= 0] = -1

    if cut_off is None:
        # formula rickgauer (2017) should be: 10**-13 = erfc( theta / ( sigma * sqrt(2) ) ) / 2
        sigma = job.job_stats['std']
        search_space = (
            # wherever the score volume has not been explicitly set to -1 is the size of the search region
            (score_volume > -1).sum() *
            job.n_rotations
        )
        cut_off = erfcinv((2 * n_false_positives) / search_space) * np.sqrt(2) * sigma
        logging.info(f'cut off for particle extraction: {cut_off}')

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

        if lcc_max < cut_off:
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
            # lcc_max,  # LCCmax
            pixel_size,  # DetectorPixelSize
            tomogram_id,  # MicrographName
            10000.0,  # Magnification
            0  # GroupNumber
        ))

        # box out the particle
        start = [i - particle_radius_px for i in ind]
        score_volume[
            start[0]: start[0] + cut_box,
            start[1]: start[1] + cut_box,
            start[2]: start[2] + cut_box
        ] *= cut_mask

    return pd.DataFrame(data, columns=[
        'rlnCoordinateX',
        'rlnCoordinateY',
        'rlnCoordinateZ',
        'rlnAngleRot',
        'rlnAngleTilt',
        'rlnAnglePsi',
        # 'ptmLCCmax',
        'rlnDetectorPixelSize',
        'rlnMicrographName',
        'rlnMagnification',
        'rlnGroupNumber'
    ]), scores
