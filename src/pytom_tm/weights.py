import numpy as np
import numpy.typing as npt
import scipy.ndimage as ndimage
import voltools as vt
from typing import Generator
from pytom_tm.io import UnequalSpacingError
from pytom_tm.dataclass import CtfData, TiltSeriesMetaData
from itertools import pairwise

# typing imports

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
    "eps0": 8.854187817620e-12,  # F/m
}


def hwhm_to_sigma(hwhm: float) -> float:
    """Convert half width of half maximum of a Gaussian to sigma by dividing by
    sqrt(2 * ln(2)).

    Parameters
    ----------
    hwhm: float
        half width of half maximum of Gaussian

    Returns
    -------
    sigma: float
        sigma of Gaussian
    """
    return hwhm / (np.sqrt(2 * np.log(2)))


def sigma_to_hwhm(sigma: float) -> float:
    """Convert sigma to half width of half maximum of a Gaussian by multiplying with
    sqrt(2 * ln(2)).

    Parameters
    ----------
    sigma: float
        sigma of Gaussian

    Returns
    -------
    hwhm: float
        half width of half maximum of Gaussian
    """
    return sigma * (np.sqrt(2 * np.log(2)))


def wavelength_ev2m(voltage: float) -> float:
    """Calculate wavelength of electrons from voltage.

    Parameters
    ----------
    voltage: float
        voltage of wave in eV

    Returns
    -------
    lambda: float
        wavelength of electrons in m
    """
    h = constants["h"]
    e = constants["el"]
    m = constants["me"]
    c = constants["c"]

    _lambda = h / np.sqrt(e * voltage * m * (e / m * voltage / c**2 + 2))

    return _lambda


def radial_reduced_grid(
    shape: tuple[int, int, int] | tuple[int, int] | tuple[int],
    shape_is_reduced: bool = False,
) -> npt.NDArray[float]:
    """Calculates a Fourier space radial reduced grid for the given input shape, with
    the 0 frequency in the center of the output image. Values range from 0 in the
    center to 1 at Nyquist frequency.

    By default, it is assumed shape belongs to a real space array, which causes the
    function to return a grid with the last dimension reduced, i.e. shape[-1] // 2 + 1
    (ideal for creating frequency dependent filters). However, setting
    radial_reduced_grid(..., shape_is_reduced=True) the shape is assumed to already be
    in a reduced form.

    Parameters
    ----------
    shape: Union[tuple[int, int, int], tuple[int, int], tuple[int]]
        1D/2D/3D input shape, usually the .shape attribute of a numpy array
    shape_is_reduced: bool, default False
        whether the shape is already in a reduced fourier format, False by default

    Returns
    ----------
    radial_reduced_grid: npt.NDArray[float]
        fourier space frequency grid, 0 in center, 1 at nyquist
    """
    if len(shape) not in [1, 2, 3]:
        raise ValueError("radial_reduced_grid() only works for 1D, 2D or 3D shapes")
    reduced_dim = shape[-1] if shape_is_reduced else shape[-1] // 2 + 1
    if len(shape) == 1:
        return np.arange(0, reduced_dim, 1.0) / (reduced_dim - 1)
    elif len(shape) == 3:
        x = (
            np.abs(
                np.arange(
                    -shape[0] // 2 + shape[0] % 2, shape[0] // 2 + shape[0] % 2, 1.0
                )
            )
            / (shape[0] // 2)
        )[:, np.newaxis, np.newaxis]
        y = (
            np.abs(
                np.arange(
                    -shape[1] // 2 + shape[1] % 2, shape[1] // 2 + shape[1] % 2, 1.0
                )
            )
            / (shape[1] // 2)
        )[:, np.newaxis]
        z = np.arange(0, reduced_dim, 1.0) / (reduced_dim - 1)
        return np.sqrt(x**2 + y**2 + z**2)
    elif len(shape) == 2:
        x = (
            np.abs(
                np.arange(
                    -shape[0] // 2 + shape[0] % 2, shape[0] // 2 + shape[0] % 2, 1.0
                )
            )
            / (shape[0] // 2)
        )[:, np.newaxis]
        y = np.arange(0, reduced_dim, 1.0) / (reduced_dim - 1)
        return np.sqrt(x**2 + y**2)


def create_gaussian_low_pass(
    shape: tuple[int, int, int] | tuple[int, int] | tuple[int],
    spacing: float,
    resolution: float,
) -> npt.NDArray[float]:
    """Create a 3D Gaussian low-pass filter with cutoff (or HWHM) that is reduced in
    fourier space.

    Parameters
    ----------
    shape: Union[tuple[int, int, int], tuple[int, int], tuple[int]]
        shape tuple with x,y,z or x,y or x dimension
    spacing: float
        voxel size in real space
    resolution: float
        resolution in real space to filter towards

    Returns
    ----------
    output: npt.NDArray[float]
        array containing the filter
    """
    q = radial_reduced_grid(shape)

    # 2 * spacing / resolution is cutoff in fourier space
    # then convert cutoff (hwhm) to sigma for gaussian function
    sigma_fourier = hwhm_to_sigma(2 * spacing / resolution)

    low_pass = np.exp(-(q**2) / (2 * sigma_fourier**2))
    return low_pass if len(shape) == 1 else np.fft.ifftshift(low_pass, axes=(0, 1))


def create_gaussian_high_pass(
    shape: tuple[int, int, int] | tuple[int, int] | tuple[int],
    spacing: float,
    resolution: float,
) -> npt.NDArray[float]:
    """Create a 3D Gaussian high-pass filter with cutoff (or HWHM) that is reduced in
    fourier space.

    Parameters
    ----------
    shape: Union[tuple[int, int, int], tuple[int, int], tuple[int]]
        shape tuple with x,y,z or x,y or x dimension
    spacing: float
        voxel size in real space
    resolution: float
        resolution in real space to filter towards

    Returns
    ----------
    output: npt.NDArray[float]
        array containing the filter
    """
    q = radial_reduced_grid(shape)

    # 2 * spacing / resolution is cutoff in fourier space
    # then convert cutoff (hwhm) to sigma for gaussian function
    sigma_fourier = hwhm_to_sigma(2 * spacing / resolution)

    high_pass = 1 - np.exp(-(q**2) / (2 * sigma_fourier**2))
    return high_pass if len(shape) == 1 else np.fft.ifftshift(high_pass, axes=(0, 1))


def create_gaussian_band_pass(
    shape: tuple[int, int, int] | tuple[int, int] | tuple[int],
    spacing: float,
    low_pass: float | None = None,
    high_pass: float | None = None,
) -> npt.NDArray[float]:
    """Resolution bands presents the resolution shells where information needs to be
    maintained. For example the bands might be (150A, 40A). For a spacing of 15A
    (nyquist resolution is 30A) this is a mild low pass filter. However, quite some low
    spatial frequencies will be cut by it.

    Parameters
    ----------
    shape: Union[tuple[int, int, int], tuple[int, int], tuple[int]]
        shape tuple with x,y,z or x,y or x dimension
    spacing: float
        voxel size in real space
    low_pass: Optional[float], default None
        resolution of low-pass filter
    high_pass: Optional[float], default None
        resolution of high-pass filter

    Returns
    ----------
    output: npt.NDArray[float]
        array containing the band-pass filter
    """
    if high_pass is None and low_pass is None:
        raise ValueError("Either low-pass or high-pass needs to be set for band-pass")

    if high_pass is None:
        return create_gaussian_low_pass(shape, spacing, low_pass)
    elif low_pass is None:
        return create_gaussian_high_pass(shape, spacing, high_pass)
    elif low_pass >= high_pass:
        raise ValueError(
            "Second value of band-pass needs to be a high resolution shell."
        )
    else:
        q = radial_reduced_grid(shape)

        # 2 * spacing / resolution is cutoff in fourier space
        # then convert cutoff (hwhm) to sigma for gaussian function
        sigma_high_pass = hwhm_to_sigma(2 * spacing / high_pass)
        sigma_low_pass = hwhm_to_sigma(2 * spacing / low_pass)

        band_pass = (1 - np.exp(-(q**2) / (2 * sigma_high_pass**2))) * np.exp(
            -(q**2) / (2 * sigma_low_pass**2)
        )
        return (
            band_pass if len(shape) == 1 else np.fft.ifftshift(band_pass, axes=(0, 1))
        )


def create_wedge(
    shape: tuple[int, int, int],
    ts_metadata: TiltSeriesMetaData,
    voxel_size: float,
    cut_off_radius: float = 1.0,
    low_pass: float | None = None,
    high_pass: float | None = None,
    per_tilt_weighting: bool | None = None,
) -> npt.NDArray[float]:
    """This function returns a wedge volume that is either symmetric or asymmetric
    depending on wedge angle input.

    Parameters
    ----------
    shape: tuple[int, int, int]
        real space shape of volume to which it needs to be applied
    ts_metadata: TiltSeriesMetadata
        tiltseries metadata for reconstructing the tomogram
    voxel_size: float
        voxel size is needed for the calculation of various filters
    cut_off_radius: float, default 1.
        cutoff as a fraction of nyquist, i.e. 1.0 means all the way to nyquist
    low_pass: Optional[float], default None
        low pass filter resolution in A
    high_pass: Optional[float], default None
        high pass filter resolution in A
    per_tilt_weighting: bool | None, default None
        if given, use this instead of ts_metadata.per_tilt_weighting (default)

    Returns
    -------
    wedge: npt.NDArray[float]
        wedge volume that is a reduced fourier space object in z, i.e. shape[2] // 2 + 1
    """
    if voxel_size <= 0.0:
        raise ValueError(
            "Voxel size in create wedge is smaller or equal to 0, which is an invalid "
            "voxel spacing."
        )

    if cut_off_radius > 1:
        print(
            "Warning: wedge cutoff needs to be defined as a fraction of nyquist "
            "0 < c <= 1. Setting value to 1.0."
        )
        cut_off_radius = 1.0
    elif cut_off_radius <= 0:
        raise ValueError("Invalid wedge cutoff: needs to be larger than 0")

    if ts_metadata.angles_in_degrees:
        tilt_angles_rad = np.deg2rad(ts_metadata.tilt_angles)
    else:
        tilt_angles_rad = ts_metadata.tilt_angles

    if per_tilt_weighting is None:
        per_tilt_weighting = ts_metadata.per_tilt_weighting
    if per_tilt_weighting:
        wedge = _create_tilt_weighted_wedge(
            shape,
            tilt_angles_rad,
            cut_off_radius,
            voxel_size,
            accumulated_dose_per_tilt=ts_metadata.dose_accumulation,
            ctf_params_per_tilt=ts_metadata.ctf_data,
        ).astype(np.float32)
    else:
        wedge_angles = (
            np.pi / 2 - np.abs(min(tilt_angles_rad)),
            np.pi / 2 - np.abs(max(tilt_angles_rad)),
        )
        if np.round(wedge_angles[0], 2) == np.round(wedge_angles[1], 2):
            wedge = _create_symmetric_wedge(
                shape, wedge_angles[0], cut_off_radius
            ).astype(np.float32)
        else:
            wedge = _create_asymmetric_wedge(
                shape, (wedge_angles[0], wedge_angles[1]), cut_off_radius
            ).astype(np.float32)
        if ts_metadata.ctf_data is not None:
            # - take ctf params from approx. middle tilt as those are most accurate
            ctf_data = ts_metadata.ctf_data[len(ts_metadata) // 2]
            wedge *= create_ctf(
                shape,
                voxel_size * 1e-10,
                ctf_data,
            )

    if not (low_pass is None and high_pass is None):
        return wedge * create_gaussian_band_pass(
            shape, voxel_size, low_pass, high_pass
        ).astype(np.float32)
    else:
        return wedge


def _create_symmetric_wedge(
    shape: tuple[int, int, int], wedge_angle: float, cut_off_radius: float
) -> npt.NDArray[float]:
    """This function returns a symmetric wedge object.
    Function should not be imported, user should call create_wedge().

    Parameters
    ----------
    shape: tuple[int, int, int]
        real space shape of volume to which it needs to be applied
    wedge_angle: float
        angle describing symmetric wedge in radians
    cut_off_radius: float
        cutoff as a fraction of nyquist, i.e. 1.0 means all the way to nyquist

    Returns
    ----------
    wedge: npt.NDArray[float]
        wedge volume that is a reduced fourier space object in z, i.e. shape[2] // 2 + 1
    """
    if wedge_angle < 0:
        raise ValueError("Negative wedge angles are not defined")
    elif wedge_angle > np.pi:
        raise ValueError("Wedge angles bigger than 90 degrees are not defined")

    # special treatment for the 0.0 angles
    if wedge_angle == 0.0:
        new_shape = (shape[0], shape[2] // 2 + 1)
        wedge_2d = np.ones(shape=new_shape)
    else:
        x = (
            np.abs(
                np.arange(
                    -shape[0] // 2 + shape[0] % 2, shape[0] // 2 + shape[0] % 2, 1.0
                )
            )
            / (shape[0] // 2)
        )[:, np.newaxis]
        z = np.arange(0, shape[2] // 2 + 1, 1.0) / (shape[2] // 2)

        # calculate the wedge mask with smooth edges
        wedge_2d = x - np.tan(wedge_angle) * z
        limit = (wedge_2d.max() - wedge_2d.min()) / (2 * min(shape[0], shape[2]) // 2)
        wedge_2d[wedge_2d > limit] = limit
        wedge_2d[wedge_2d < -limit] = -limit
        wedge_2d = (wedge_2d - wedge_2d.min()) / (wedge_2d.max() - wedge_2d.min())
        wedge_2d[shape[0] // 2 + 1, 0] = (
            1  # ensure that the zero frequency point equals 1
        )

    # duplicate in x
    wedge = np.tile(wedge_2d[:, np.newaxis, :], (1, shape[1], 1))

    wedge[radial_reduced_grid(shape) > cut_off_radius] = 0

    # fourier shift to origin
    return np.fft.ifftshift(wedge, axes=(0, 1))


def _create_asymmetric_wedge(
    shape: tuple[int, int, int],
    wedge_angles: tuple[float, float],
    cut_off_radius: float,
) -> npt.NDArray[float]:
    """This function returns an asymmetric wedge object.
    Function should not be imported, user should call create_wedge().

    Parameters
    ----------
    shape: tuple[int, int, int]
        real space shape of volume to which it needs to be applied
    wedge_angles: tuple[float, float]
        two angles describing asymmetric missing wedge in radians
    cut_off_radius: float
        cutoff as a fraction of nyquist, i.e. 1.0 means all the way to nyquist

    Returns
    ----------
    wedge: npt.NDArray[float]
        wedge volume that is a reduced fourier space object in z, i.e. shape[2] // 2 + 1
    """
    x = (
        np.abs(
            np.arange(-shape[0] // 2 + shape[0] % 2, shape[0] // 2 + shape[0] % 2, 1.0)
        )
        / (shape[0] // 2)
    )[:, np.newaxis]
    z = np.arange(0, shape[2] // 2 + 1, 1.0) / (shape[2] // 2)

    # calculate wedge for first angle
    wedge_section = x - np.tan(wedge_angles[0]) * z
    limit = (wedge_section.max() - wedge_section.min()) / (
        2 * min(shape[0], shape[2]) // 2
    )
    wedge_section[wedge_section > limit] = limit
    wedge_section[wedge_section < -limit] = -limit
    wedge_section = (wedge_section - wedge_section.min()) / (
        wedge_section.max() - wedge_section.min()
    )

    # set top of the wedge
    wedge_2d = wedge_section.copy()

    # calculate wedge for second angle
    wedge_section = x - np.tan(wedge_angles[1]) * z
    limit = (wedge_section.max() - wedge_section.min()) / (
        2 * min(shape[0], shape[2]) // 2
    )
    wedge_section[wedge_section > limit] = limit
    wedge_section[wedge_section < -limit] = -limit
    wedge_section = (wedge_section - wedge_section.min()) / (
        wedge_section.max() - wedge_section.min()
    )

    # set bottom of wedge and put 0 frequency to 1
    wedge_2d[shape[0] // 2 + 1 :] = wedge_section[shape[0] // 2 + 1 :]
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
    accumulated_dose_per_tilt: list[float, ...] | None = None,
    ctf_params_per_tilt: list[CtfData] | None = None,
) -> npt.NDArray[float]:
    """
    The following B-factor heuristic is used (as mentioned in the M paper, and
    introduced in RELION 1.4):
        "The B factor is increased by 4Å2 per 1e− Å−2 of exposure, and each tilt
        is weighted as cos θ."

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
    accumulated_dose_per_tilt: list[float, ...], default None
        the accumulated dose in e− Å−2
    ctf_params_per_tilt: list[CtfData, ...], default None
        the ctf parameters per tilt angle, list of CtfData data classes

    Returns
    -------
    wedge: npt.NDArray[float]
        structured wedge mask in fourier reduced form, i.e. output shape is
        (shape[0], shape[1], shape[2] // 2 + 1)
    """
    if accumulated_dose_per_tilt is not None and len(accumulated_dose_per_tilt) != len(
        tilt_angles
    ):
        raise ValueError(
            "in _create_tilt_weighted_wedge the list of accumulated dose per tilt does "
            "not have the same length as the tilt angle list!"
        )
    if ctf_params_per_tilt is not None and len(ctf_params_per_tilt) != len(tilt_angles):
        raise ValueError(
            "in _create_tilt_weighted_wedge the list of CTF parameters per tilt does "
            "not have the same length as the tilt angle list!"
        )
    if not all([shape[0] == s for s in shape[1:]]):
        raise UnequalSpacingError(
            "Input shape for structured wedge needs to be a square box. "
            "Otherwise the frequencies in fourier space are not equal across "
            "dimensions."
        )

    image_size = shape[0]  # assign to size variable as all dimensions are equal size
    tilt = np.zeros(shape)
    q_grid = radial_reduced_grid(shape)
    tilt_weighted_wedge = np.zeros((image_size, image_size, image_size // 2 + 1))

    # create ramp weights to correct tilt summation for overlap
    tilt_increment = min([abs(x - y) for x, y in pairwise(tilt_angles)])
    # Crowther freq. determines till what point adjacent tilts overlap in Fourier space
    overlap_frequency = 1 / (tilt_increment * image_size)
    freq_1d = (
        np.abs(
            np.arange(
                -image_size // 2 + image_size % 2, image_size // 2 + image_size % 2, 1.0
            )
        )
        / (image_size // 2)
        * 0.5
    )  # multiply with .5 for nyquist frequency
    ramp_filter = freq_1d / overlap_frequency
    ramp_filter[ramp_filter > 1] = 1  # linear increase up to overlap frequency

    # generate 2d weights along the tilt axis
    ramp_weighting = np.tile(ramp_filter[:, np.newaxis], (1, image_size))

    for i, alpha in enumerate(tilt_angles):
        if ctf_params_per_tilt is not None:
            ctf = np.fft.fftshift(
                create_ctf(
                    (image_size,) * 2,
                    pixel_size_angstrom * 1e-10,
                    ctf_params_per_tilt[i],
                ),
                axes=0,
            )
            tilt[:, :, image_size // 2] = (
                np.concatenate(
                    (  # duplicate and flip the CTF around the 0 frequency;
                        # then concatenate to make it non-reduced
                        np.flip(ctf[:, 1 : 1 + image_size - ctf.shape[1]], axis=1),
                        ctf,
                    ),
                    axis=1,
                )
                * ramp_weighting
            )
        else:
            tilt[:, :, image_size // 2] = ramp_weighting

        # rotate the image weights to the tilt angle
        rotated = np.flip(
            vt.transform(
                tilt,
                rotation=(0, alpha, 0),
                rotation_units="rad",
                rotation_order="rxyz",
                center=(image_size // 2,) * 3,
                interpolation="filt_bspline",
                device="cpu",
            )[:, :, : image_size // 2 + 1],  # crop back z-axis to reduced Fourier form
            axis=2,
        )

        # weight with exposure and tilt dampening
        if accumulated_dose_per_tilt is not None:
            q_squared = (q_grid / (2 * pixel_size_angstrom)) ** 2
            sigma_motion = np.sqrt(accumulated_dose_per_tilt[i] * 4 / (8 * np.pi**2))
            weighted_tilt = (
                rotated
                * np.cos(alpha)  # apply tilt-dependent weighting
                * np.exp(
                    -2 * np.pi**2 * sigma_motion**2 * q_squared
                )  # apply dose-weighting
            )
        else:
            weighted_tilt = (
                rotated * np.cos(alpha)  # apply tilt-dependent weighting
            )

        tilt_weighted_wedge += weighted_tilt

    tilt_weighted_wedge[q_grid > cut_off_radius] = 0

    return np.fft.ifftshift(tilt_weighted_wedge, axes=(0, 1))


def create_ctf(
    shape: tuple[int, int, int] | tuple[int, int],
    pixel_size: float,
    ctf_data: CtfData,
    cut_after_first_zero: bool = False,
) -> npt.NDArray[float]:
    """Create a CTF in a 3D volume in reduced format.

    Parameters
    ----------
    shape: Union[tuple[int, int, int], tuple[int, int]]
        dimensions of volume to create ctf in
    pixel_size: float
        pixel size for ctf in m
    ctf_data: CtfData
        The ctf data for a tilt, see pytom_tm.dataclass.CtfData for definitions
    cut_after_first_zero: bool, default False
        whether to cut ctf after first zero crossing

    Returns
    -------
    ctf: npt.NDArray[float]
        CTF in 3D
    """
    k = radial_reduced_grid(shape) / (2 * pixel_size)  # frequencies in fourier space

    _lambda = wavelength_ev2m(ctf_data.voltage)

    # phase contrast transfer
    chi = (
        np.pi * _lambda * ctf_data.defocus * k**2
        - 0.5 * np.pi * ctf_data.spherical_aberration * _lambda**3 * k**4
    )
    # amplitude contrast term
    tan_term = np.arctan(
        ctf_data.amplitude_contrast / np.sqrt(1 - ctf_data.amplitude_contrast**2)
    )

    # determine the ctf
    ctf = -np.sin(chi + tan_term + np.deg2rad(ctf_data.phase_shift_deg))

    if cut_after_first_zero:  # find frequency to cut first zero

        def chi_1d(q):
            return (
                np.pi * _lambda * ctf_data.defocus * q**2
                - 0.5 * np.pi * ctf_data.spherical_aberration * _lambda**3 * q**4
            )

        def ctf_1d(q):
            return -np.sin(chi_1d(q) + tan_term)

        # sample 1d ctf and get indices of zero crossing
        k_range = np.arange(max(k.shape)) / max(k.shape) / (2 * pixel_size)
        values = ctf_1d(k_range)
        zero_crossings = np.where(np.diff(np.sign(values)))[0]

        # for overfocus the first crossing needs to be skipped,
        # for example see: Yonekura et al. 2006 JSB
        k_cutoff = (
            k_range[zero_crossings[0]]
            if ctf_data.defocus > 0
            else k_range[zero_crossings[1]]
        )

        # filter the ctf with the cutoff frequency
        ctf[k > k_cutoff] = 0

    if ctf_data.flip_phase:  # take absolute, ensures matching contrast
        ctf = np.abs(ctf)
    else:  # multiply the ctf with -1 if we have overfocus, this allows the user to
        # always match the contrast of the input template with the contrast of the
        # tomogram: if the tomogram is black the reference should be black.
        ctf *= -1 if ctf_data.defocus > 0 else 1

    return np.fft.ifftshift(ctf, axes=(0, 1) if len(shape) == 3 else 0)


def estimate_whitening_filter(
    tomogram: npt.NDArray[float],
    ts_metadata: TiltSeriesMetaData,
    patch_size: int,
    overlap: float = 0.5,
    reject_frac: float = 0.10,
    exclude: npt.NDArray[bool] | None = None,
    statistic: str = "median",
    voxel_size: float = 1.0,
) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """Estimate a whitening filter from the radially averaged noise power spectrum
    of a tomogram, sampled on overlapping cubic patches.

    The power spectrum is averaged only over Fourier voxels the tilt series
    actually sampled (via a tilt coverage mask), so the estimate is not biased by
    the empty missing-wedge / inter-tilt-gap voxels. Estimation is done on many
    overlapping, windowed, mean-subtracted patches, and the highest-variance
    patches (e.g. gold, carbon, ice, volume edges) are rejected before averaging.

    Parameters
    ----------
    tomogram: npt.NDArray[float]
        3D real-space array to estimate the whitening filter from
    ts_metadata: TiltSeriesMetaData
        tilt series metadata of the tomogram, used to build the coverage mask that
        restricts the radial average to Fourier voxels the tilt series actually
        sampled. CTF and dose weighting are not applied to this mask, only the
        tilt geometry matters here, regardless of what ts_metadata specifies
    patch_size: int
        edge length of the cubic estimation box, also the box the tilt coverage
        mask is built for. A larger patch reaches lower frequencies but leaves
        fewer patches to average over in a thin tomogram. The estimated profile
        can be interpolated to a different box size afterwards, so estimation
        size and matching size are decoupled
    overlap: float, default 0.5
        fractional overlap between patches
    reject_frac: float, default 0.10
        fraction of the highest-variance patches to reject, 0 disables rejection
    exclude: Optional[npt.NDArray[bool]], default None
        boolean array, same shape as tomogram, True where voxels must not be used
        (e.g. gold, carbon, contamination). Should not be used to mask out
        biological structure, as that would bias the estimate low
    statistic: str, default "median"
        radial averaging statistic, either "median" (robust, bias-corrected to
        the mean) or "mean"
    voxel_size: float, default 1.0
        voxel size in Angstrom, sets the physical units of the returned frequency
        axis

    Returns
    -------
    (q, w): tuple[npt.NDArray[float], npt.NDArray[float]]
        q is the frequency of each shell in cycles/Angstrom (0 to Nyquist), with
        shape (patch_size // 2 + 1,). w is the whitening filter derived from the
        power spectrum profile (DC set to 0, high frequencies tapered, and
        normalized to a maximum of 1), with the same shape as q
    """
    tomogram = np.asarray(tomogram, dtype=np.float32)
    length = int(patch_size)

    # tilt coverage mask, built once for the cubic patch (geometry identical per
    # patch). CTF and dose data are stripped from the metadata so only the tilt
    # geometry determines which Fourier voxels count as sampled.
    mask_metadata = ts_metadata.replace(ctf_data=None, dose_accumulation=None)
    wedge = create_wedge((length, length, length), mask_metadata, voxel_size)
    mask = wedge > 0.05
    mask[0, 0, 0] = 0

    # window (mandatory: controls leakage across the 3+ decade dynamic range)
    win = _hann3d(length)
    win_power = float((win**2).sum())

    # collect windowed periodograms over overlapping patches
    step = max(1, int(round(length * (1.0 - overlap))))
    psds, variances = [], []
    for sl in _patch_slices(tomogram.shape, length, step):
        if exclude is not None and exclude[sl].mean() > 0.01:
            continue
        v = tomogram[sl]
        if not np.all(np.isfinite(v)):
            continue
        v = v - v.mean()
        f = np.fft.rfftn(v * win)
        psds.append((f.real**2 + f.imag**2) / win_power)
        variances.append(float(v.var()))

    if len(psds) < 4:
        raise RuntimeError(f"only {len(psds)} usable patches; reduce patch_size")

    # reject high-variance patches (gold / carbon / ice / edges)
    variances = np.asarray(variances)
    keep = np.ones(len(psds), bool)
    if reject_frac > 0 and len(psds) >= 10:
        keep = variances <= np.quantile(variances, 1.0 - reject_frac)
        if keep.sum() < 4:
            keep = np.ones(len(psds), bool)

    psd = np.mean([p for p, k in zip(psds, keep) if k], axis=0)
    dof = int(keep.sum())

    # masked radial average
    q, prof = _masked_radial(psd, mask, length, voxel_size, statistic, dof)

    def cosine_cutoff():
        r_frac = radial_reduced_grid((length,))
        lo, hi = 0.9, 1.0
        ramp = np.clip((r_frac - lo) / (hi - lo), 0, 1)
        return 0.5 * (1 + np.cos(np.pi * ramp))  # 1 below lo, 0 at hi

    # transform into a whitening filter
    w = np.where(prof > 0, 1 / np.sqrt(prof), np.zeros_like(prof))
    w[0] = 0.0  # zero DC because its not needed
    w = w * cosine_cutoff()  # tamper high frequency estimate
    w /= w.max()

    return q, w


def _hann3d(length: int) -> npt.NDArray[float]:
    w = np.hanning(length + 2)[1:-1].astype(np.float32)  # drop the exact zeros
    return (w[:, None, None] * w[None, :, None] * w[None, None, :]).astype(np.float32)


def _patch_slices(
    shape: tuple[int, int, int], length: int, step: int
) -> Generator[tuple[slice, slice, slice], None, None]:
    def starts(n):
        s = list(range(0, n - length + 1, step))
        if not s:
            raise ValueError(f"patch_size {length} larger than tomogram extent {n}")
        if s[-1] != n - length:
            s.append(n - length)
        return s

    zs, ys, xs = (starts(n) for n in shape)
    for z in zs:
        for y in ys:
            for x in xs:
                yield (
                    slice(z, z + length),
                    slice(y, y + length),
                    slice(x, x + length),
                )


def _masked_radial(
    psd: npt.NDArray[float],
    mask: npt.NDArray[bool],
    length: int,
    voxel_size: float,
    statistic: str,
    dof: int,
) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    r_frac = np.fft.ifftshift(
        radial_reduced_grid((length, length, length)), axes=(0, 1)
    )

    k_nyq = 0.5 / voxel_size
    nb = length // 2 + 1
    shell = np.floor(r_frac * (nb - 1) + 0.5).astype(int)
    q = np.arange(nb) / (nb - 1) * k_nyq

    labels = shell.copy()
    labels[(shell >= nb) | (shell < 0)] = -1
    if mask is not None:
        labels[~mask] = -1

    idx = np.arange(nb)
    counts = ndimage.sum(np.ones_like(psd), labels=labels, index=idx)
    fn = ndimage.median if statistic == "median" else ndimage.mean
    prof = np.asarray(fn(psd, labels=labels, index=idx), dtype=float)
    prof[counts == 0] = np.nan  # empty shells -> NaN, not 0

    if statistic == "median":
        prof /= (1.0 - 1.0 / (9.0 * max(dof, 1))) ** 3  # median -> mean, Gamma(dof)

    good = np.isfinite(prof) & (prof > 0)
    if good.sum() >= 2:
        prof = np.interp(q, q[good], prof[good])
    return q, prof


def profile_to_weighting(
    profile: npt.NDArray[float], shape: tuple[int, int] | tuple[int, int, int]
) -> npt.NDArray[float]:
    """Calculate a radial weighing (filter) from a spectrum profile.

    Parameters
    ----------
    profile: npt.NDArray[float]
        power spectrum profile (or other 1d profile) to transform in a fourier space
        filter
    shape: Union[tuple[int, int], tuple[int, int, int]]
        2D/3D array shape in real space for which the fourier reduced weights are
        calculated

    Returns
    -------
    weighting: npt.NDArray[float]
        Reduced Fourier space weighting for shape, with the DC component set to 0
    """
    if len(profile.shape) != 1:
        raise ValueError("Profile passed to profile_to_weighting is not 1-dimensional.")
    if len(shape) not in [2, 3]:
        raise ValueError("Shape passed to profile_to_weighting needs to be 2D/3D.")

    q_grid = radial_reduced_grid(shape)

    weights = ndimage.map_coordinates(
        profile,
        q_grid.flatten()[np.newaxis, :] * (profile.shape[0] - 1),
        order=1,
        mode="nearest",
    ).reshape(q_grid.shape)

    weights[q_grid > 1] = 0

    weights = np.fft.ifftshift(weights, axes=(0, 1) if len(shape) == 3 else 0)
    weights[(0,) * len(shape)] = 0

    return weights
