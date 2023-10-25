import numpy as np
import numpy.typing as npt
import logging
import scipy.ndimage as ndimage
from typing import Optional, Union
from pytom_tm.io import UnequalSpacingError


constants = {
    # Dictionary of physical constants required for calculation.
    "c": 299792458,  # m/s
    "el": 1.60217646e-19,  # C
    "h": 6.62606896e-34,  # J*S
    "h_ev": 4.13566733e-15,  # eV*s
    "h_bar": 1.054571628e-34,  # J*s
    "h_bar_ev": 6.58211899e-16,  # eV*s

    "na": 6.02214179e23,  # mol-1
    "re": 2.817940289458e-15,  # m
    "rw": 2.976e-10,  # m

    "me": 9.10938215e-31,  # kg
    "me_ev": 0.510998910e6,  # ev/c^2
    "kb": 1.3806503e-23,  # m^2 kgs^-2 K^-1

    "eps0": 8.854187817620e-12  # F/m
}


def hwhm_to_sigma(hwhm: float) -> float:
    return hwhm / (np.sqrt(2 * np.log(2)))


def sigma_to_hwhm(sigma: float) -> float:
    return sigma * (np.sqrt(2 * np.log(2)))


def wavelength_ev2m(voltage: float) -> float:
    """
    Calculate wavelength of electrons from voltage.

    @param voltage: voltage of wave in eV
    @return: wavelength of electrons in m

    @author: Marten Chaillet
    """
    h = constants["h"]
    e = constants["el"]
    m = constants["me"]
    c = constants["c"]

    _lambda = h / np.sqrt(e * voltage * m * (e / m * voltage / c ** 2 + 2))

    return _lambda


def radial_reduced_grid(shape: Union[tuple[int, int, int], tuple[int, int]]) -> npt.NDArray[float]:
    if len(shape) == 3:
        x = (np.abs(np.arange(
            -shape[0] // 2 + shape[0] % 2,
            shape[0] // 2 + shape[0] % 2, 1.
        )) / (shape[0] // 2))[:, np.newaxis, np.newaxis]
        y = (np.abs(np.arange(
            -shape[1] // 2 + shape[1] % 2,
            shape[1] // 2 + shape[1] % 2, 1.
        )) / (shape[1] // 2))[:, np.newaxis]
        z = np.arange(0, shape[2] // 2 + 1, 1.) / (shape[2] // 2)
        return np.sqrt(x ** 2 + y ** 2 + z ** 2)
    elif len(shape) == 2:
        x = (np.abs(np.arange(
            -shape[0] // 2 + shape[0] % 2,
            shape[0] // 2 + shape[0] % 2, 1.
        )) / (shape[0] // 2))[:, np.newaxis]
        y = np.arange(0, shape[1] // 2 + 1, 1.) / (shape[1] // 2)
        return np.sqrt(x ** 2 + y ** 2)


def create_gaussian_low_pass(shape: tuple[int, int, int], spacing: float, resolution: float) -> npt.NDArray[float]:
    """
    Create a 3D Gaussian low-pass filter with cutoff (or HWHM) that is reduced in fourier space.
    @param shape: shape tuple with x,y or x,y,z dimension
    @param spacing: voxel size in real space
    @param resolution: resolution in real space to filter towards
    @return: sphere in square volume
    """
    q = radial_reduced_grid(shape)

    # 2 * spacing / resolution is cutoff in fourier space
    # then convert cutoff (hwhm) to sigma for gaussian function
    sigma_fourier = hwhm_to_sigma(2 * spacing / resolution)

    return np.fft.ifftshift(np.exp(-q ** 2 / (2 * sigma_fourier ** 2)), axes=(0, 1))


def create_gaussian_high_pass(shape: tuple[int, int, int], spacing: float, resolution: float) -> npt.NDArray[float]:
    """
    Create a 3D Gaussian high-pass filter with cutoff (or HWHM) that is reduced in fourier space.
    @param shape: shape tuple with x,y or x,y,z dimension
    @param spacing: voxel size in real space
    @param resolution: resolution in real space to filter towards
    @return: sphere in square volume
    """
    q = radial_reduced_grid(shape)

    # 2 * spacing / resolution is cutoff in fourier space
    # then convert cutoff (hwhm) to sigma for gaussian function
    sigma_fourier = hwhm_to_sigma(2 * spacing / resolution)

    return np.fft.ifftshift(1 - np.exp(-q ** 2 / (2 * sigma_fourier ** 2)), axes=(0, 1))


def create_gaussian_band_pass(
        shape: tuple[int, int, int],
        spacing: float,
        low_pass: Optional[float] = None,
        high_pass: Optional[float] = None
) -> npt.NDArray[float]:
    """
    Resolution bands presents the resolution shells where information needs to be maintained. For example the bands
    might be (150A, 40A). For a spacing of 15A (nyquist resolution is 30A) this is a mild low pass filter. However,
    quite some low spatial frequencies will be cut by it.
    @param shape: shape of output, will return fourier reduced shape
    @param spacing: voxel size of input shape in real space
    @param low_pass:
    @param high_pass:
    @return: a volume with a gaussian bandapss
    """
    if high_pass is None and low_pass is None:
        raise ValueError('Either low-pass or high-pass needs to be set for band-pass')

    if high_pass is None:
        return create_gaussian_low_pass(shape, spacing, low_pass)
    elif low_pass is None:
        return create_gaussian_high_pass(shape, spacing, high_pass)
    elif low_pass >= high_pass:
        raise ValueError('Second value of band-pass needs to be a high resolution shell.')
    else:
        q = radial_reduced_grid(shape)

        # 2 * spacing / resolution is cutoff in fourier space
        # then convert cutoff (hwhm) to sigma for gaussian function
        sigma_high_pass = hwhm_to_sigma(2 * spacing / high_pass)
        sigma_low_pass = hwhm_to_sigma(2 * spacing / low_pass)

        return np.fft.ifftshift(
            (1 - np.exp(-q ** 2 / (2 * sigma_high_pass ** 2))) * np.exp(-q ** 2 / (2 * sigma_low_pass ** 2)),
            axes=(0, 1)
        )


def create_wedge(
        shape: tuple[int, int, int],
        tilt_angles: list[float, ...],
        voxel_size: float,
        cut_off_radius: float = 1.,
        angles_in_degrees: bool = True,
        low_pass: Optional[float] = None,
        high_pass: Optional[float] = None,
        tilt_weighting: bool = False,
        accumulated_dose_per_tilt: Optional[list[float, ...]] = None,
        ctf_params_per_tilt: Optional[list[dict]] = None
) -> npt.NDArray[float]:
    """This function returns a wedge volume that is either symmetric or asymmetric depending on wedge angle input.

    Parameters
    ----------
    shape
        real space shape of volume to which it needs to be applied
    tilt_angles
        tilt angles used for reconstructing the tomogram
    voxel_size
        voxel size is needed for the calculation of various filters
    cut_off_radius
        cutoff as a fraction of nyquist, i.e. 1.0 means all the way to nyquist
    angles_in_degrees
        whether angles are in degrees or radians units
    low_pass
        low pass filter resolution in A
    high_pass
        high pass filter resolution in A
    tilt_weighting
        apply tilt weighting
    accumulated_dose_per_tilt
        accumulated dose for each tilt for dose weighting
    ctf_params_per_tilt
        ctf parameters for each tilt

    Returns
    -------
    wedge: npt.NDArray[float]
        wedge volume that is a reduced fourier space object in z, i.e. shape[2] // 2 + 1
    """
    if not isinstance(tilt_angles, list) or len(tilt_angles) < 2:
        raise ValueError('Wedge generation needs at least a list of two tilt angles.')

    if voxel_size <= 0.:
        raise ValueError('Voxel size in create wedge is smaller or equal to 0, which is an invalid voxel spacing.')

    if cut_off_radius > 1:
        print('Warning: wedge cutoff needs to be defined as a fraction of nyquist 0 < c <= 1. Setting value to 1.0.')
        cut_off_radius = 1.0
    elif cut_off_radius <= 0:
        raise ValueError('Invalid wedge cutoff: needs to be larger than 0')

    if angles_in_degrees:
        tilt_angles_rad = [np.deg2rad(w) for w in tilt_angles]
    else:
        tilt_angles_rad = tilt_angles

    if tilt_weighting:
        wedge = _create_tilt_weighted_wedge(
            shape,
            tilt_angles_rad,
            cut_off_radius,
            voxel_size,
            accumulated_dose_per_tilt=accumulated_dose_per_tilt,
            ctf_params_per_tilt=ctf_params_per_tilt
        ).astype(np.float32)
    else:
        wedge_angles = (np.pi / 2 - np.abs(min(tilt_angles_rad)), np.pi / 2 - np.abs(max(tilt_angles_rad)))
        if np.round(wedge_angles[0], 2) == np.round(wedge_angles[1], 2):
            wedge = _create_symmetric_wedge(
                shape,
                wedge_angles[0],
                cut_off_radius
            ).astype(np.float32)
        else:
            wedge = _create_asymmetric_wedge(
                shape,
                (wedge_angles[0], wedge_angles[1]),
                cut_off_radius
            ).astype(np.float32)

    if not (low_pass is None and high_pass is None):
        return wedge * create_gaussian_band_pass(shape, voxel_size, low_pass, high_pass).astype(np.float32)
    else:
        return wedge


def _create_symmetric_wedge(
        shape: tuple[int, int, int],
        wedge_angle: float,
        cut_off_radius: float
) -> npt.NDArray[float]:
    """
    This function returns a symmetric wedge object. Function should not be imported, user should call create_wedge().
    @param shape: real space shape of volume to which it needs to be applied
    @param wedge_angle: angle describing symmetric wedge in radians
    @param cut_off_radius: cutoff as a fraction of nyquist, i.e. 1.0 means all the way to nyquist
    @return: wedge volume that is a reduced fourier space object in z, i.e. shape[2] // 2 + 1
    """
    x = (np.abs(np.arange(
        -shape[0] // 2 + shape[0] % 2,
        shape[0] // 2 + shape[0] % 2, 1.
    )) / (shape[0] // 2))[:, np.newaxis]
    z = np.arange(0, shape[2] // 2 + 1, 1.) / (shape[2] // 2)

    # calculate the wedge mask with smooth edges
    wedge_2d = x - np.tan(wedge_angle) * z
    limit = (wedge_2d.max() - wedge_2d.min()) / (2 * min(shape[0], shape[2]) // 2)
    wedge_2d[wedge_2d > limit] = limit
    wedge_2d[wedge_2d < -limit] = -limit
    wedge_2d = (wedge_2d - wedge_2d.min()) / (wedge_2d.max() - wedge_2d.min())
    wedge_2d[shape[0] // 2 + 1, 0] = 1  # ensure that the zero frequency point equals 1

    # duplicate in x
    wedge = np.tile(wedge_2d[:, np.newaxis, :], (1, shape[1], 1))

    wedge[radial_reduced_grid(shape) > cut_off_radius] = 0

    # fourier shift to origin
    return np.fft.ifftshift(wedge, axes=(0, 1))


def _create_asymmetric_wedge(
        shape: tuple[int, int, int],
        wedge_angles: tuple[float, float],
        cut_off_radius: float
) -> npt.NDArray[float]:
    """
    This function returns an asymmetric wedge object. Function should not be imported, user should call create_wedge().
    @param shape: real space shape of volume to which it needs to be applied
    @param wedge_angles: two angles describing asymmetric missing wedge in radians
    @param cut_off_radius: cutoff as a fraction of nyquist, i.e. 1.0 means all the way to nyquist
    @return: wedge volume that is a reduced fourier space object in z, i.e. shape[2] // 2 + 1
    """
    x = (np.abs(np.arange(
        -shape[0] // 2 + shape[0] % 2,
        shape[0] // 2 + shape[0] % 2, 1.
    )) / (shape[0] // 2))[:, np.newaxis]
    z = np.arange(0, shape[2] // 2 + 1, 1.) / (shape[2] // 2)

    # calculate wedge for first angle
    wedge_section = x - np.tan(wedge_angles[0]) * z
    limit = (wedge_section.max() - wedge_section.min()) / (2 * min(shape[0], shape[2]) // 2)
    wedge_section[wedge_section > limit] = limit
    wedge_section[wedge_section < -limit] = -limit
    wedge_section = (wedge_section - wedge_section.min()) / (wedge_section.max() - wedge_section.min())

    # set top of the wedge
    wedge_2d = wedge_section.copy()

    # calculate wedge for second angle
    wedge_section = x - np.tan(wedge_angles[1]) * z
    limit = (wedge_section.max() - wedge_section.min()) / (2 * min(shape[0], shape[2]) // 2)
    wedge_section[wedge_section > limit] = limit
    wedge_section[wedge_section < -limit] = -limit
    wedge_section = (wedge_section - wedge_section.min()) / (wedge_section.max() - wedge_section.min())

    # set bottom of wedge and put 0 frequency to 1
    wedge_2d[shape[0] // 2 + 1:] = wedge_section[shape[0] // 2 + 1:]
    wedge_2d[shape[0] // 2 + 1, 0] = 1

    # duplicate in x
    wedge = np.tile(wedge_2d[:, np.newaxis, :], (1, shape[1], 1))

    wedge[radial_reduced_grid(shape) > cut_off_radius] = 0

    # fourier shift to origin
    return np.fft.ifftshift(wedge, axes=(0, 1))


def _create_tilt_weighted_wedge(
        shape: tuple[int, int, int],
        tilt_angles: list[float, ...],
        cut_off_radius: float,
        pixel_size_angstrom: float,
        accumulated_dose_per_tilt: Optional[list[float, ...]] = None,
        ctf_params_per_tilt: Optional[list[dict]] = None
) -> npt.NDArray[float]:
    """
    The following B-factor heuristic is used (as mentioned in the M paper, and introduced in RELION 1.4):
        "The B factor is increased by 4Å2 per 1e− Å−2 of exposure, and each tilt is weighted as cos θ."

    Relation between B-factor and the sigma of a gaussian:

        B = 8 * pi ** 2 * sigma_motion ** 2

    i.e. sigma_motion = sqrt( B / (8 * pi ** 2)). Belonging to a Gaussian blur:

        exp( -2 * pi ** 2 * sigma_motion ** 2 * q ** 2 )

    Parameters
    ----------
    shape: tuple[int, int, int]
        shape of volume to model the wedge for
    tilt_angles: list[float, ...]
        tilt angles is a list of angle in radian units
    cut_off_radius: float
        cut off for the mask as a fraction of nyquist, value between 0 and 1
    pixel_size_angstrom: float
        the pixel size as a value in Å
    accumulated_dose_per_tilt: list[float, ...]
        the accumulated dose in e− Å−2
    ctf_params_per_tilt: list[dict, ...]
        the ctf parameters per tilt angle, list of dicts where each dict has the following keys:
        - 'defocus'; in um
        - 'amplitude'; fraction of amplitude contrast between 0 and 1
        - 'voltage'; in keV
        - 'cs'; spherical abberation in mm
        ctf_phase_flip: bool
            whether the CTFs need to be phase-flipped (made fully positive), needed for template matching in tomograms that
            were CTF-corrected through phase-flipping

    Returns
    -------
    wedge: npt.NDArray[float]
        structured wedge mask in fourier reduced form, i.e. output shape is (shape[0], shape[1], shape[2] // 2 + 1)
    """
    if accumulated_dose_per_tilt is not None and len(accumulated_dose_per_tilt) != len(tilt_angles):
        raise ValueError('in _create_tilt_weighted_wedge the list of accumulated dose per tilt does not have the same '
                         'length as the tilt angle list!')
    if ctf_params_per_tilt is not None and len(ctf_params_per_tilt) != len(tilt_angles):
        raise ValueError('in _create_tilt_weighted_wedge the list of CTF parameters per tilt does not have the same '
                         'length as the tilt angle list!')
    if not all([shape[0] == s for s in shape[1:]]):
        raise UnequalSpacingError('Input shape for structured wedge needs to be a square box. '
                                  'Otherwise the frequencies in fourier space are not equal across dimensions.')

    tilt = np.zeros(shape)
    tilt_sum = np.zeros((shape[0], shape[1], shape[2] // 2 + 1))

    for i, alpha in enumerate(tilt_angles):
        if ctf_params_per_tilt is not None:
            ctf = np.fft.fftshift(
                create_ctf(
                    (shape[0], shape[1]),
                    pixel_size_angstrom * 1e-10,
                    ctf_params_per_tilt[i]['defocus'] * 1e-6,
                    ctf_params_per_tilt[i]['amplitude'],
                    ctf_params_per_tilt[i]['voltage'] * 1e3,
                    ctf_params_per_tilt[i]['cs'] * 1e-3,
                    flip_phase=True  # creating a per tilt ctf is hard if the phase is not flipped
                ), axes=0,
            )
            # make ctf non-reduced for easy of writing code
            tilt[:, :, shape[2] // 2] = np.concatenate((np.flip(ctf[:, 2 - shape[0] % 2:], axis=1), ctf), axis=1)
        else:
            tilt[:, :, shape[2] // 2] = 1

        # rotate the image weights to the tilt angle
        rotated = np.flip(
            ndimage.rotate(
                tilt,
                np.rad2deg(alpha),
                axes=(0, 2),
                reshape=False,
                order=3
            )[:, :, : shape[2] // 2 + 1]
        )

        # weight with exposure and tilt dampening
        if accumulated_dose_per_tilt is not None:
            q_squared = (radial_reduced_grid(shape) / (2 * pixel_size_angstrom)) ** 2
            sigma_motion = np.sqrt(accumulated_dose_per_tilt[i] * 4 / (8 * np.pi ** 2))
            weighted_tilt = (
                    rotated *
                    np.cos(alpha) *  # apply tilt-dependent weighting
                    np.exp(-2 * np.pi ** 2 * sigma_motion ** 2 * q_squared)  # apply dose-weighting
            )
        else:
            weighted_tilt = (
                    rotated *
                    np.cos(alpha)  # apply tilt-dependent weighting
            )
        tilt_sum[weighted_tilt > tilt_sum] = weighted_tilt[weighted_tilt > tilt_sum]

    tilt_sum[radial_reduced_grid(shape) > cut_off_radius] = 0

    return np.fft.fftshift(tilt_sum, axes=(0, 1))


def create_ctf(
        shape: Union[tuple[int, int, int], tuple[int, int]],
        pixel_size: float,
        defocus: float,
        amplitude_contrast: float,
        voltage: float,
        spherical_aberration: float,
        cut_after_first_zero: bool = False,
        flip_phase: bool = False
) -> npt.NDArray[float]:
    """Create a CTF in a 3D volume in reduced format.

    Parameters
    ----------
    shape: tuple[int, int, int]
        dimensions of volume to create ctf in
    pixel_size: float
        pixel size for ctf in m
    defocus: float
        defocus for ctf in m
    amplitude_contrast: float
        the fraction of amplitude contrast in the ctf
    voltage: float
        acceleration voltage of the microscope in eV
    spherical_aberration: float
        spherical aberration in m
    cut_after_first_zero: bool
        whether to cut ctf after first zero crossing
    flip_phase: bool
        make ctf fully positive/negative to imitate ctf correction by phase flipping

    Returns
    -------
    ctf: npt.NDArray[float]
        CTF in 3D
    """
    k = radial_reduced_grid(shape) / (2 * pixel_size)  # frequencies in fourier space

    _lambda = wavelength_ev2m(voltage)

    # phase contrast transfer
    chi = np.pi * _lambda * defocus * k ** 2 - 0.5 * np.pi * spherical_aberration * _lambda ** 3 * k ** 4
    # amplitude contrast term
    tan_term = np.arctan(amplitude_contrast / np.sqrt(1 - amplitude_contrast ** 2))

    # determine the ctf
    ctf = - np.sin(chi + tan_term)

    if cut_after_first_zero:  # find frequency to cut first zero
        def chi_1d(q):
            return np.pi * _lambda * defocus * q ** 2 - 0.5 * np.pi * spherical_aberration * _lambda ** 3 * q ** 4

        def ctf_1d(q):
            return - np.sin(chi_1d(q) + tan_term)

        # sample 1d ctf and get indices of zero crossing
        k_range = np.arange(max(k.shape)) / max(k.shape) / (2 * pixel_size)
        values = ctf_1d(k_range)
        zero_crossings = np.where(np.diff(np.sign(values)))[0]

        # for overfocus the first crossing needs to be skipped, for example see: Yonekura et al. 2006 JSB
        k_cutoff = k_range[zero_crossings[0]] if defocus > 0 else k_range[zero_crossings[1]]

        # filter the ctf with the cutoff frequency
        ctf[k > k_cutoff] = 0

    if flip_phase:  # phase flip but consider whether contrast should be black/white
        # ctf = np.abs(ctf) * -1 if defocus > 0 else np.abs(ctf)
        ctf = np.abs(ctf)

    return np.fft.ifftshift(ctf, axes=(0, 1) if len(shape) == 3 else 0)


def radial_average(image: npt.NDArray[float]) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """ This calculates the radial average of an image.

    Parameters
    ----------
    image: npt.NDArray[float]
        3D array to be radially averaged: in fourier reduced form and assumed to have origin in corner.

    Returns
    -------
    (q, mean): tuple[npt.NDArray[float], npt.NDArray[float]]
        A tuple of two 1d numpy arrays. Their length equals half of largest input image dimension.
    """
    if len(image.shape) not in [2, 3]:
        raise ValueError('Radial average calculation only works for 2d/3d arrays')
    if len(set(image.shape)) != 2:
        raise ValueError('Radial average calculation only works for images with equal dimensions')

    size = image.shape[0]
    center = size // 2  # fourier space center
    if len(image.shape) == 3:
        xx, yy, zz = (
            np.arange(size) - center,
            np.arange(size) - center,
            np.arange(size // 2 + 1))
        r = np.sqrt(xx[:, np.newaxis, np.newaxis] ** 2 + yy[:, np.newaxis] ** 2 + zz ** 2)
    else:
        xx, yy = (
            np.arange(size) - center,
            np.arange(size // 2 + 1)
        )
        r = np.sqrt(xx[:, np.newaxis] ** 2 + yy ** 2)

    logging.debug(f'shape of image for radial average {image.shape} and determined grid {r.shape}')

    q = np.arange(size // 2)
    mean = np.vectorize(lambda x: np.fft.fftshift(image, axes=(0, 1))[(r >= x - .5) & (r < x + .5)].mean())(q)

    return q, mean
