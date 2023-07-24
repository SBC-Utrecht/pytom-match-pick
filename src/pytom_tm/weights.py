import numpy as np
import numpy.typing as npt
from typing import Optional


def create_wedge(
        shape: tuple[int, int, int],
        wedge_angles: tuple[float, float],
        cut_off_radius: float,
        smooth: Optional[float] = None,
        angles_in_degrees: bool = True
) -> npt.NDArray[float]:
    """
    This function returns a wedge volume that is either symmetric or asymmetric depending on wedge angle input.
    @param shape: real space shape of volume to which it needs to be applied
    @param wedge_angles: two angles describing asymmetric missing wedge in degrees
    @param cut_off_radius: cutoff as a fraction of nyquist, i.e. 1.0 means all the way to nyquist
    @param smooth: smoothing around wedge object in number of pixels
    @param angles_in_degrees: whether angles are in degrees or radians units
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
        return _create_symmetric_wedge(shape, wedge_angles_rad[0], cut_off_radius, smooth).astype(np.float32)
    else:
        return _create_asymmetric_wedge(shape, wedge_angles_rad, cut_off_radius, smooth).astype(np.float32)


def _create_symmetric_wedge(
        shape: tuple[int, int, int],
        wedge_angle: float,
        cut_off_radius: float,
        smooth: Optional[float] = None
) -> npt.NDArray[float]:
    """
    This function returns a symmetric wedge object. Function should not be imported, user should call create_wedge().
    @param shape: real space shape of volume to which it needs to be applied
    @param wedge_angle: angle describing symmetric wedge in radians
    @param cut_off_radius: cutoff as a fraction of nyquist, i.e. 1.0 means all the way to nyquist
    @param smooth: smoothing around wedge object in number of pixels
    @return: wedge volume that is a reduced fourier space object in z, i.e. shape[2] // 2 + 1
    """
    size_x, size_y, size_z = shape
    
    # numpy meshgrid by default returns indexing with cartesian coordinates (xy)
    # shape N, M, P returns meshgrid with M, N, P (see numpy meshgrid documentation)
    # the naming here is therefore weird

    x = np.abs(np.arange(-size_x // 2 + size_x % 2, size_x // 2 + size_x % 2, 1.)) / (size_x // 2)
    y = np.abs(np.arange(-size_y // 2 + size_y % 2, size_y // 2 + size_y % 2, 1.)) / (size_y // 2)
    z = np.arange(0, size_z // 2 + 1, 1.) / (size_z // 2)
    x, y, z = np.meshgrid(x, y, z, indexing='ij')

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    # calculate the wedge mask with smooth edges
    edge_pixels = 1 / ((size_y + size_z) / 4)
    wedge = y - np.tan(wedge_angle) * z
    wedge[wedge > edge_pixels] = edge_pixels
    wedge[wedge < -edge_pixels] = -edge_pixels
    wedge = (wedge - wedge.min()) / (wedge.max() - wedge.min())

    wedge[size_x // 2, :, 0] = 1  # ensure central slice equals 1

    wedge[r > cut_off_radius] = 0  # cut wedge after nyquist frequency

    return np.fft.ifftshift(wedge, axes=(0, 1))


def _create_asymmetric_wedge(
        shape: tuple[int, int, int],
        wedge_angles: tuple[float, float],
        cut_off_radius: float,
        smooth: Optional[float] = None
) -> npt.NDArray[float]:
    """
    This function returns an asymmetric wedge object. Function should not be imported, user should call create_wedge().
    @param shape: real space shape of volume to which it needs to be applied
    @param wedge_angles: two angles describing asymmetric missing wedge in radians
    @param cut_off_radius: cutoff as a fraction of nyquist, i.e. 1.0 means all the way to nyquist
    @param smooth: smoothing around wedge object in number of pixels
    @return: wedge volume that is a reduced fourier space object in z, i.e. shape[2] // 2 + 1
    """
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
    wedge[size_x // 2, :, 0] = 1

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
