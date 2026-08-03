from collections.abc import Generator
from itertools import pairwise

import numpy as np
import numpy.typing as npt
import voltools as vt
from scipy import ndimage

from pytom_tm.dataclass import CtfData, TiltSeriesMetaData
from pytom_tm.io import UnequalSpacingError

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


def radial_grid(
    shape: tuple[int, int, int] | tuple[int, int] | tuple[int],
    reduced: bool = True,
    fftshifted: bool = False,
    shape_is_reduced: bool = False,
) -> npt.NDArray[float]:
    """Calculates a Fourier space radial grid for the given input shape. Values
    range from 0 at the 0 frequency to 1 at Nyquist frequency.

    By default the last dimension is reduced to shape[-1] // 2 + 1 and the 0
    frequency sits at index 0 of each axis, matching the unshifted output of
    numpy.fft.rfftn. Set reduced=False for a full (non-reduced) last dimension, and
    fftshifted=True to place the 0 frequency in the center of each axis instead
    (matching numpy.fft.fftshift).

    Parameters
    ----------
    shape: Union[tuple[int, int, int], tuple[int, int], tuple[int]]
        1D/2D/3D input shape, usually the .shape attribute of a numpy array
    reduced: bool, default True
        whether the last dimension should be reduced to shape[-1] // 2 + 1, as in
        the output of numpy.fft.rfftn
    fftshifted: bool, default False
        whether the 0 frequency should be centered (True) or at index 0 of each
        axis (False)
    shape_is_reduced: bool, default False
        whether the shape is already in a reduced fourier format, only relevant
        when reduced=True, False by default

    Returns
    ----------
    radial_grid: npt.NDArray[float]
        fourier space frequency grid
    """
    if len(shape) not in {1, 2, 3}:
        raise ValueError("radial_grid() only works for 1D, 2D or 3D shapes")

    def full_axis(n: int) -> npt.NDArray[float]:
        # magnitude of frequency for a full (non-reduced) axis: 0 in the center,
        # 1 at nyquist
        values = np.abs(np.arange(-n // 2 + n % 2, n // 2 + n % 2, 1.0)) / (n // 2)
        return values if fftshifted else np.fft.ifftshift(values)

    def last_axis(n: int) -> npt.NDArray[float]:
        if not reduced:
            return full_axis(n)
        reduced_dim = n if shape_is_reduced else n // 2 + 1
        return np.arange(0, reduced_dim, 1) / (reduced_dim - 1)

    if len(shape) == 3:
        x = full_axis(shape[0])[:, np.newaxis, np.newaxis]
        y = full_axis(shape[1])[:, np.newaxis]
        z = last_axis(shape[2])
        return np.sqrt(x**2 + y**2 + z**2)
    elif len(shape) == 2:
        x = full_axis(shape[0])[:, np.newaxis]
        y = last_axis(shape[1])
        return np.sqrt(x**2 + y**2)
    else:
        return last_axis(shape[0])


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
    q = radial_grid(shape)

    # 2 * spacing / resolution is cutoff in fourier space
    # then convert cutoff (hwhm) to sigma for gaussian function
    sigma_fourier = hwhm_to_sigma(2 * spacing / resolution)

    return np.exp(-(q**2) / (2 * sigma_fourier**2))


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
    q = radial_grid(shape)

    # 2 * spacing / resolution is cutoff in fourier space
    # then convert cutoff (hwhm) to sigma for gaussian function
    sigma_fourier = hwhm_to_sigma(2 * spacing / resolution)

    return 1 - np.exp(-(q**2) / (2 * sigma_fourier**2))


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
        q = radial_grid(shape)

        # 2 * spacing / resolution is cutoff in fourier space
        # then convert cutoff (hwhm) to sigma for gaussian function
        sigma_high_pass = hwhm_to_sigma(2 * spacing / high_pass)
        sigma_low_pass = hwhm_to_sigma(2 * spacing / low_pass)

        return (1 - np.exp(-(q**2) / (2 * sigma_high_pass**2))) * np.exp(
            -(q**2) / (2 * sigma_low_pass**2)
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

    # WarpTools sample-leveling angles (see WarpTiltSeriesMetaData), always in degrees
    level_angle_x_rad = np.deg2rad(ts_metadata.level_angle_x)
    level_angle_y_rad = np.deg2rad(ts_metadata.level_angle_y)

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
            level_angle_x=level_angle_x_rad,
            level_angle_y=level_angle_y_rad,
        ).astype(np.float32)
    else:
        alpha_min = min(tilt_angles_rad) + level_angle_y_rad
        alpha_max = max(tilt_angles_rad) + level_angle_y_rad
        wedge = _create_binary_wedge(
            shape,
            alpha_min,
            alpha_max,
            cut_off_radius,
            level_angle_x=level_angle_x_rad,
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


def _create_binary_wedge(
    shape: tuple[int, int, int],
    alpha_min: float,
    alpha_max: float,
    cut_off_radius: float,
    level_angle_x: float = 0.0,
) -> npt.NDArray[float]:
    """This function returns a wedge object, built directly from the extreme
    tilt angles.

    By the central-slice theorem, each tilt's 2D Fourier transform is a plane
    through the origin of 3D Fourier space. Ignoring the tilt axis y, this is
    just a line through the origin in the x-z' plane (z' = z * cos(level_angle_x)
    - y * sin(level_angle_x) is just z after a fixed rigid tilt by the
    sample-leveling angle, applied once up front). As alpha sweeps from
    alpha_min to alpha_max, that line rotates, sweeping out covered vs.
    missing directions. arctan2(x, z') gives the angle phi of a point relative
    to this plane. So, relative to the min and max tilt angle alpha, we can
    determine the wedge. The rest of the code in this function is needed
    to give the wedge as smooth edge.

    Parameters
    ----------
    shape: tuple[int, int, int]
        real space shape of volume to which it needs to be applied
    alpha_min: float
        lowest tilt angle (including level_angle_y) in radians, in [-pi/2, pi/2]
    alpha_max: float
        highest tilt angle (including level_angle_y) in radians, in [-pi/2, pi/2]
    cut_off_radius: float
        cutoff as a fraction of nyquist, i.e. 1.0 means all the way to nyquist
    level_angle_x: float, default 0.0
        WarpTools sample-leveling angle in radians that rotates the wedge around the
        x-axis

    Returns
    ----------
    wedge: npt.NDArray[float]
        wedge volume that is a reduced fourier space object in z, i.e. shape[2] // 2 + 1
    """
    if abs(alpha_min) > np.pi / 2 or abs(alpha_max) > np.pi / 2:
        raise ValueError(
            "alpha_min and alpha_max (tilt angle plus level_angle_y) must lie "
            "within [-90, 90] degrees"
        )

    # x and y are negated to preserve the missing-wedge orientation convention
    # of this codebase (which corresponds with AreTomo). This has been verified
    # against both WarpTools and AreTomo reconstructions (their conventions
    # differ from each other, see PR #334)
    x = -(np.fft.fftfreq(shape[0]) * shape[0] / (shape[0] // 2))[
        :, np.newaxis, np.newaxis
    ]
    y = -(np.fft.fftfreq(shape[1]) * shape[1] / (shape[1] // 2))[
        np.newaxis, :, np.newaxis
    ]
    z = radial_grid((shape[2],))[np.newaxis, np.newaxis, :]

    # z rotated by level_angle_x within the (y, z) plane - see docstring
    z_eff = z * np.cos(level_angle_x) - y * np.sin(level_angle_x)

    r = np.sqrt(x**2 + z_eff**2)
    phi = np.arctan2(x, z_eff)

    f_min = x * np.sin(alpha_min) + z_eff * np.cos(alpha_min)
    f_max = x * np.sin(alpha_max) + z_eff * np.cos(alpha_max)
    phi_in_range = (phi >= alpha_min) & (phi <= alpha_max)

    lo = np.minimum(f_min, f_max)
    hi = np.where(phi_in_range, r, np.maximum(f_min, f_max))

    # positive inside the sampled interval, negative outside, continuous and zero
    # at the boundary
    wedge = np.minimum(-lo, hi)

    # normalize against the fixed clip bound
    limit = (wedge.max() - wedge.min()) / (2 * min(shape[0], shape[2]) // 2)
    wedge = np.clip(wedge, -limit, limit)
    wedge = (wedge + limit) / (2 * limit)

    wedge[radial_grid(shape) > cut_off_radius] = 0

    return wedge


def _create_tilt_weighted_wedge(
    shape: tuple[int, int, int],
    tilt_angles: list[float, ...],
    cut_off_radius: float,
    pixel_size_angstrom: float,
    accumulated_dose_per_tilt: list[float, ...] | None = None,
    ctf_params_per_tilt: list[CtfData] | None = None,
    level_angle_x: float = 0.0,
    level_angle_y: float = 0.0,
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
    level_angle_x: float, default 0.0
        WarpTools sample-leveling angle in radians that rotates the wedge around the
        x-axis, combined into the same per-tilt rotation as the tilt angle below
    level_angle_y: float, default 0.0
        WarpTools sample-leveling angle in radians that is added to each tilt angle

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
    if not all(shape[0] == s for s in shape[1:]):
        raise UnequalSpacingError(
            "Input shape for structured wedge needs to be a square box. "
            "Otherwise the frequencies in fourier space are not equal across "
            "dimensions."
        )

    image_size = shape[0]  # assign to size variable as all dimensions are equal size
    tilt = np.zeros(shape)
    q_squared_2d = (
        radial_grid((image_size, image_size), reduced=False, fftshifted=True)
        / (2 * pixel_size_angstrom)
    ) ** 2
    q_grid_3d = radial_grid(shape)
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
            ctf = create_ctf(
                (image_size,) * 2,
                pixel_size_angstrom * 1e-10,
                ctf_params_per_tilt[i],
                reduced=False,
                fftshifted=True,
            )
            plane = ctf * ramp_weighting
        else:
            plane = ramp_weighting

        plane = plane * np.cos(alpha)  # tilt-dependent dampening heuristic
        if accumulated_dose_per_tilt is not None:
            sigma_motion = np.sqrt(accumulated_dose_per_tilt[i] * 4 / (8 * np.pi**2))
            plane = plane * np.exp(-2 * np.pi**2 * sigma_motion**2 * q_squared_2d)

        if image_size % 2 == 0:
            # rotation with image_size//2 as the center is not symmetric
            # -> this makes it symmetric
            plane[0, :] = 0.0
            plane[:, 0] = 0.0

        tilt[:, :, image_size // 2] = plane

        # WarpTools composes this as TiltMatrix = Euler(alpha) * RotateX(level_angle_x)
        # matrix product, so RotateX is applied first (innermost), Euler second
        # (outermost). voltools' rotation_order names which slot maps to which
        # axis, and the first slot ends up applied last (outermost)
        rotated_full = vt.transform(
            tilt,
            rotation=(alpha + level_angle_y, level_angle_x, 0),
            rotation_units="rad",
            rotation_order="ryxz",
            center=(image_size // 2,) * 3,
            interpolation="filt_bspline",
            device="cpu",
        )
        # this is the correct way to go from full FT with DC
        # in the center to reduced FT form with DC at the origin
        tilt_weighted_wedge += np.fft.ifftshift(rotated_full, axes=(0, 1, 2))[
            :, :, : image_size // 2 + 1
        ]

    # because of the now correct full->reduced transform above, we need
    # to flip the xy-plane to stay consistent with previous behaviour.
    # the flip inverts the plane, the roll gets the DC to the origin.
    tilt_weighted_wedge = np.roll(
        np.flip(tilt_weighted_wedge, axis=(0, 1)), shift=(1, 1), axis=(0, 1)
    )

    tilt_weighted_wedge[q_grid_3d > cut_off_radius] = 0

    return tilt_weighted_wedge


def create_ctf(
    shape: tuple[int, int, int] | tuple[int, int],
    pixel_size: float,
    ctf_data: CtfData,
    reduced: bool = True,
    fftshifted: bool = False,
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
    reduced: bool, default True
        whether the last dimension should be reduced to shape[-1] // 2 + 1, as in
        the output of numpy.fft.rfftn
    fftshifted: bool, default False
        whether the 0 frequency should be centered (True) or at index 0 of each
        axis (False)

    Returns
    -------
    ctf: npt.NDArray[float]
        CTF in 3D
    """
    k = radial_grid(shape, reduced=reduced, fftshifted=fftshifted) / (
        2 * pixel_size
    )  # frequencies in fourier space

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

    if ctf_data.flip_phase:  # take absolute, ensures matching contrast
        ctf = np.abs(ctf)
    else:  # multiply the ctf with -1 if we have overfocus, this allows the user to
        # always match the contrast of the input template with the contrast of the
        # tomogram: if the tomogram is black the reference should be black.
        ctf *= -1 if ctf_data.defocus > 0 else 1

    return ctf


def estimate_whitening_filter(
    tomogram: npt.NDArray[float],
    ts_metadata: TiltSeriesMetaData,
    patch_size: int,
    overlap: float = 0.5,
    reject_frac: float = 0.10,
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
    mask = wedge > 0.05  # binarize the mask and set DC to zero
    mask[0, 0, 0] = 0

    # window (mandatory: controls leakage across the 3+ decade dynamic range)
    win = _hann3d(length)
    win_power = float((win**2).sum())

    # collect windowed periodograms over overlapping patches
    step = max(1, int(length * (1.0 - overlap) + 0.5))  # round half up
    psds, variances = [], []
    for sl in _patch_slices(tomogram.shape, length, step):
        v = tomogram[sl]
        if not np.all(np.isfinite(v)):
            continue
        v = v - v.mean()
        f = np.fft.rfftn(v * win)
        psds.append((f.real**2 + f.imag**2) / win_power)
        variances.append(float(v.var()))

    # reject high-variance patches (gold / carbon / ice / edges). Only attempted
    # with enough patches for the quantile estimate to be meaningful.
    psds = np.asarray(psds)
    variances = np.asarray(variances)
    keep = np.ones(len(psds), bool)
    if reject_frac > 0 and len(psds) >= 10:
        keep = variances <= np.quantile(variances, 1.0 - reject_frac)

    if keep.sum() == 0:
        raise RuntimeError("no usable patches for whitening filter estimation")

    psd = psds[keep].mean(axis=0)
    dof = int(keep.sum())

    # masked radial average
    q, prof = _masked_radial(psd, mask, length, voxel_size, statistic, dof)

    def cosine_cutoff():
        r_frac = radial_grid((length,))
        lo, hi = 0.9, 1.0
        ramp = np.clip((r_frac - lo) / (hi - lo), 0, 1)
        return 0.5 * (1 + np.cos(np.pi * ramp))  # 1 below lo, 0 at hi

    # transform into a whitening filter
    w = np.where(prof > 0, 1 / np.sqrt(prof), 0.0)
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

    xs, ys, zs = (starts(n) for n in shape)
    for x in xs:
        for y in ys:
            for z in zs:
                yield (
                    slice(x, x + length),
                    slice(y, y + length),
                    slice(z, z + length),
                )


def _masked_radial(
    psd: npt.NDArray[float],
    mask: npt.NDArray[bool],
    length: int,
    voxel_size: float,
    statistic: str,
    dof: int,
) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    r_frac = radial_grid((length, length, length))

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
    if good.sum() < 2:
        raise RuntimeError(
            "too few valid radial shells to interpolate a whitening filter profile"
        )
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

    q_grid = radial_grid(shape)

    weights = ndimage.map_coordinates(
        profile,
        q_grid.flatten()[np.newaxis, :] * (profile.shape[0] - 1),
        order=1,
        mode="nearest",
    ).reshape(q_grid.shape)

    weights[q_grid > 1] = 0
    weights[(0,) * len(shape)] = 0

    return weights
