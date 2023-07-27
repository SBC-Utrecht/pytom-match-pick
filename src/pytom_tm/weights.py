import numpy as np
import numpy.typing as npt
from typing import Optional
from scipy.ndimage import zoom
from pytom_tm._utils import hwhm_to_sigma


def radial_reduced_grid(shape: tuple[int, int, int]) -> npt.NDArray[float]:
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


def create_gaussian_low_pass(shape: tuple[int, int, int], spacing: float, resolution: float) -> npt.NDArray:
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


def create_gaussian_high_pass(shape: tuple[int, int, int], spacing: float, resolution: float) -> npt.NDArray:
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


def create_gaussian_bandpass(
        shape: tuple[int, int, int],
        spacing: float,
        lowpass: Optional[float] = None,
        highpass: Optional[float] = None
) -> npt.NDArray:
    """
    Resolution bands presents the resolution shells where information needs to be maintained. For example the bands
    might be (150A, 40A). For a spacing of 15A (nyquist resolution is 30A) this is a mild low pass filter. However,
    quite some low spatial frequencies will be cut by it.
    @param shape: shape of output, will return fourier reduced shape
    @param spacing: voxel size of input shape in real space
    @param lowpass:
    @param highpass:
    @return: a volume with a gaussian bandapss
    """
    if (highpass is None and lowpass is None) or (lowpass >= highpass):
        raise ValueError('Second value of bandpass needs to be a high resolution shell.')
    elif highpass is None:
        return create_gaussian_low_pass(shape, spacing, lowpass)
    elif lowpass is None:
        return create_gaussian_high_pass(shape, spacing, highpass)
    else:
        q = radial_reduced_grid(shape)

        # 2 * spacing / resolution is cutoff in fourier space
        # then convert cutoff (hwhm) to sigma for gaussian function
        sigma_high_pass = hwhm_to_sigma(2 * spacing / highpass)
        sigma_low_pass = hwhm_to_sigma(2 * spacing / lowpass)

        return np.fft.ifftshift(
            (1 - np.exp(-q ** 2 / (2 * sigma_high_pass ** 2))) * np.exp(-q ** 2 / (2 * sigma_low_pass ** 2)),
            axes=(0, 1)
        )


def create_wedge(
        shape: tuple[int, int, int],
        wedge_angles: tuple[float, float],
        cut_off_radius: float,
        angles_in_degrees: bool = True,
        voxel_size: float = 1.,
        lowpass: Optional[float] = None,
        highpass: Optional[float] = None
) -> npt.NDArray[float]:
    """
    This function returns a wedge volume that is either symmetric or asymmetric depending on wedge angle input.
    @param shape: real space shape of volume to which it needs to be applied
    @param wedge_angles: two angles describing asymmetric missing wedge in degrees
    @param cut_off_radius: cutoff as a fraction of nyquist, i.e. 1.0 means all the way to nyquist
    @param angles_in_degrees: whether angles are in degrees or radians units
    @param voxel_size:
    @param lowpass:
    @param highpass:
    @return: wedge volume that is a reduced fourier space object in z, i.e. shape[2] // 2 + 1
    """
    if cut_off_radius > 1:
        print('Warning: wedge cutoff needs to be defined as a fraction of nyquist 0 < c <= 1. Setting value to 1.0.')
        cut_off_radius = 1.0
    elif cut_off_radius <= 0:
        raise ValueError('Invalid wedge cutoff: needs to be larger than 0')

    if angles_in_degrees:
        wedge_angles_rad = [np.deg2rad(w) for w in wedge_angles]
    else:
        wedge_angles_rad = wedge_angles

    if wedge_angles_rad[0] == wedge_angles_rad[1]:
        wedge = _create_symmetric_wedge(shape, wedge_angles_rad[0], cut_off_radius,).astype(np.float32)
    else:
        wedge = _create_asymmetric_wedge(shape, wedge_angles_rad, cut_off_radius).astype(np.float32)

    if not (lowpass is None and highpass is None):
        return wedge * create_gaussian_bandpass(shape, voxel_size, lowpass, highpass).astype(np.float32)
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
    @param smooth: smoothing around wedge object in number of pixels
    @return: wedge volume that is a reduced fourier space object in z, i.e. shape[2] // 2 + 1
    """
    size = max(shape)
    
    # numpy meshgrid by default returns indexing with cartesian coordinates (xy)
    # shape N, M, P returns meshgrid with M, N, P (see numpy meshgrid documentation)
    # the naming here is therefore weird
    y = np.abs(np.arange(-size // 2 + size % 2, size // 2 + size % 2, 1.))[:, np.newaxis]
    z = np.arange(0, size // 2 + 1, 1.)

    # calculate the wedge mask with smooth edges
    wedge_2d = y - np.tan(wedge_angle) * z
    wedge_2d[wedge_2d > 0.5] = 0.5
    wedge_2d[wedge_2d < -0.5] = -0.5
    wedge_2d = (wedge_2d - wedge_2d.min()) / (wedge_2d.max() - wedge_2d.min())

    # scale to right dimensions
    scaled_wedge_2d = np.zeros((shape[1], shape[2] // 2 + 1))
    zoom(wedge_2d, [shape[1] / size, (shape[2] // 2 + 1) / (size // 2 + 1)], output=scaled_wedge_2d)

    # duplicate in x
    wedge = np.tile(scaled_wedge_2d, (shape[0], 1, 1))
    wedge[shape[0] // 2, :, 0] = 1  # put 1 in the zero frequency

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
    @param smooth: smoothing around wedge object in number of pixels
    @return: wedge volume that is a reduced fourier space object in z, i.e. shape[2] // 2 + 1
    """
    raise NotImplementedError('Asymmetric wedge needs to be fixed')

    size_x, size_y, size_z = shape
    angle1, angle2 = wedge_angles
    wedge = np.zeros((size_x, size_y, size_z // 2 + 1))
    
    # see comment above with symmetric wedge function about meshgrid
    z, y, x = np.meshgrid(np.arange(-size_y // 2 + size_y % 2, size_y // 2 + size_y % 2),
                          np.arange(-size_x // 2 + size_x % 2, size_x // 2 + size_x % 2),
                          np.arange(0, size_z // 2 + 1))

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    
    with np.errstate(all='ignore'):
        wedge[np.tan(angle1) < y / z] = 1
        wedge[np.tan(-angle2) > y / z] = 1
    wedge[size_x // 2, :, 0] = 1  # put 1 in the zero frequency

    smooth = None
    if smooth is not None:
        range_angle1_smooth = smooth / np.sin(angle1)
        range_angle2_smooth = smooth / np.sin(angle2)

        area = np.abs(x - (y / np.tan(angle1))) <= range_angle1_smooth
        strip = 1 - (np.abs(x - (y / np.tan(angle1))) * np.sin(angle1) / smooth)
        wedge += (strip * area * (1 - wedge) * (y > 0))

        area2 = np.abs(x + (y / np.tan(angle2))) <= range_angle2_smooth
        strip2 = 1 - (np.abs(x + (y / np.tan(angle2))) * np.sin(angle2) / smooth)
        wedge += (strip2 * area2 * (1 - wedge) * (y <= 0))

    wedge[r > cut_off_radius] = 0

    return np.fft.ifftshift(wedge, axes=(0, 1))


def ctf():
    raise NotImplementedError()
