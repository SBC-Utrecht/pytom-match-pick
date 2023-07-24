import numpy as np
from typing import Optional


def create_wedge(wedge_angles: tuple[float, float], cut_off_radius, size_x, size_y, size_z,
                 smooth: Optional[float] = None):
    """This function returns a wedge object. For speed reasons it decides whether to generate a symmetric or assymetric wedge.
    @param wedge_angle1: angle of wedge1 in degrees
    @type wedge_angle1: int
    @param wedge_angle2: angle of wedge2 in degrees
    @type wedge_angle2: int
    @param cut_off_radius: radius from center beyond which the wedge is set to zero.
    @type cut_off_radius: int
    @param size_x: the size of the box in x-direction.
    @type size_x: int
    @param size_y: the size of the box in y-direction.
    @type size_y: int
    @param size_z: the size of the box in z-direction.
    @type size_z: int
    @param smooth: smoothing parameter that defines the amount of smoothing  at the edge of the wedge.
    @type smooth: float
    @return: 3D array determining the wedge object.
    @rtype: ndarray of np.float64"""
    # TODO update so that uneven values for size_z still match with pytom.lib.pytom.lib.pytom_volume

    import numpy as np

    if cut_off_radius < 1:
        cut_off_radius = size_x // 2

    if wedge_angles[0] == wedge_angles[1]:
        return create_symmetric_wedge(wedge_angles[0], cut_off_radius, size_x, size_y, size_z, smooth).astype(
            np.float32)
    else:
        return create_asymmetric_wedge(wedge_angles, cut_off_radius, size_x, size_y, size_z, smooth).astype(np.float32)


def create_symmetric_wedge(wedge_angle: float, cut_off_radius, size_x, size_y, size_z,
                           smooth: Optional[float] = None):
    """This function returns a symmetric wedge object.
    @param angle1: angle of wedge1 in degrees
    @type angle1: int
    @param angle2: angle of wedge2 in degrees
    @type angle2: int
    @param cut_off_radius: radius from center beyond which the wedge is set to zero.
    @type cut_off_radius: int
    @param size_x: the size of the box in x-direction.
    @type size_x: int
    @param size_y: the size of the box in y-direction.
    @type size_y: int
    @param size_z: the size of the box in z-direction.
    @type size_z: int
    @param smooth: smoothing parameter that defines the amount of smoothing  at the edge of the wedge.
    @type smooth: float
    @return: 3D array determining the wedge object.
    @rtype: ndarray of np.float64"""
    wedge = np.zeros((size_x, size_y, size_z // 2 + 1), dtype=np.float64)
    
    # numpy meshgrid by default returns indexing with cartesian coordinates (xy)
    # shape N, M, P returns meshgrid with M, N, P (see numpy meshgrid documentation)
    # the naming here is therefore weird
    z, y, x = np.meshgrid(np.abs(np.arange(-size_x // 2 + size_x % 2, size_x // 2 + size_x % 2, 1.)),
                          np.abs(np.arange(-size_y // 2 + size_y % 2, size_y // 2 + size_y % 2, 1.)),
                          np.arange(0, size_z // 2 + 1, 1.))

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if wedge_angle > 1E-3:

        with np.errstate(all='ignore'):
            wedge[np.tan(np.float32(wedge_angle) * np.pi / np.float32(180.)) <= y / x] = 1

        wedge[size_x // 2, :, 0] = 1

        if smooth is not None:
            range_angle_smooth = smooth / np.sin(wedge_angle * np.pi / 180.)
            area = np.abs(x - (y / np.tan(wedge_angle * np.pi / 180))) < range_angle_smooth
            strip = 1 - ((np.abs(x - (y / np.tan(wedge_angle * np.pi / 180.)))) * np.sin(wedge_angle * np.pi / 180.) / smooth)
            wedge += (strip * area * (1 - wedge))

    else:
        wedge += 1
    wedge[r > cut_off_radius] = 0
    return np.fft.ifftshift(wedge, axes=(0, 1))  # TODO should be ifftshift, because centered is shifted to corner


def create_asymmetric_wedge(wedge_angles: tuple[float, float], cut_off_radius, size_x, size_y, size_z,
                            smooth: Optional[float] = None):
    """This function returns an asymmetric wedge object.
    @param angle1: angle of wedge1 in degrees
    @type angle1: int
    @param angle2: angle of wedge2 in degrees
    @type angle2: int
    @param cut_off_radius: radius from center beyond which the wedge is set to zero.
    @type cut_off_radius: int
    @param size_x: the size of the box in x-direction.
    @type size_x: int
    @param size_y: the size of the box in y-direction.
    @type size_y: int
    @param size_z: the size of the box in z-direction.
    @type size_z: int
    @param smooth: smoothing parameter that defines the amount of smoothing  at the edge of the wedge.
    @type smooth: float
    @return: 3D array determining the wedge object.
    @rtype: ndarray of np.float64"""
    angle1, angle2 = wedge_angles
    wedge = np.zeros((size_x, size_y, size_z // 2 + 1))
    
    # see comment above with symmetric wedge function about meshgrid
    z, y, x = np.meshgrid(np.arange(-size_y // 2 + size_y % 2, size_y // 2 + size_y % 2),
                          np.arange(-size_x // 2 + size_x % 2, size_x // 2 + size_x % 2),
                          np.arange(0, size_z // 2 + 1))

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    
    with np.errstate(all='ignore'):
        wedge[np.tan(angle1 * np.pi / 180) < y / x] = 1
        wedge[np.tan(-angle2 * np.pi / 180) > y / x] = 1
    wedge[size_x // 2, :, 0] = 1

    if smooth is not None:
        range_angle1_smooth = smooth / np.sin(angle1 * np.pi / 180.)
        range_angle2_smooth = smooth / np.sin(angle2 * np.pi / 180.)

        area = np.abs(x - (y / np.tan(angle1 * np.pi / 180))) <= range_angle1_smooth
        strip = 1 - (np.abs(x - (y / np.tan(angle1 * np.pi / 180.))) * np.sin(angle1 * np.pi / 180.) / smooth)
        wedge += (strip * area * (1 - wedge) * (y > 0))

        area2 = np.abs(x + (y / np.tan(angle2 * np.pi / 180))) <= range_angle2_smooth
        strip2 = 1 - (np.abs(x + (y / np.tan(angle2 * np.pi / 180.))) * np.sin(angle2 * np.pi / 180.) / smooth)
        wedge += (strip2 * area2 * (1 - wedge) * (y <= 0))

    wedge[r > cut_off_radius] = 0

    return np.fft.ifftshift(wedge, axes=(0, 1))  # TODO should be ifftshift, because centered is shifted to corner
