import numpy as np


def create_wedge(wedge_angle1, wedge_angle2, cutOffRadius, size_x, size_y, size_z, smooth=0, rotation=None):
    '''This function returns a wedge object. For speed reasons it decides whether to generate a symmetric or assymetric wedge.
    @param wedge_angle1: angle of wedge1 in degrees
    @type wedge_angle1: int
    @param wedge_angle2: angle of wedge2 in degrees
    @type wedge_angle2: int
    @param cutOffRadius: radius from center beyond which the wedge is set to zero.
    @type cutOffRadius: int
    @param size_x: the size of the box in x-direction.
    @type size_x: int
    @param size_y: the size of the box in y-direction.
    @type size_y: int
    @param size_z: the size of the box in z-direction.
    @type size_z: int
    @param smooth: smoothing parameter that defines the amount of smoothing  at the edge of the wedge.
    @type smooth: float
    @return: 3D array determining the wedge object.
    @rtype: ndarray of np.float64'''
    # TODO update so that uneven values for size_z still match with pytom.lib.pytom.lib.pytom_volume

    import numpy as np

    if cutOffRadius < 1:
        cutOffRadius = size_x // 2

    if wedge_angle1 == wedge_angle2:
        return create_symmetric_wedge(wedge_angle1, wedge_angle2, cutOffRadius, size_x, size_y, size_z, smooth, rotation).astype(np.float32)
    else:
        return create_asymmetric_wedge(wedge_angle1, wedge_angle2, cutOffRadius, size_x, size_y, size_z, smooth, rotation).astype(np.float32)

def create_symmetric_wedge(angle1, angle2, cutoffRadius, size_x, size_y, size_z, smooth, rotation=None):
    '''This function returns a symmetric wedge object.
    @param angle1: angle of wedge1 in degrees
    @type angle1: int
    @param angle2: angle of wedge2 in degrees
    @type angle2: int
    @param cutOffRadius: radius from center beyond which the wedge is set to zero.
    @type cutOffRadius: int
    @param size_x: the size of the box in x-direction.
    @type size_x: int
    @param size_y: the size of the box in y-direction.
    @type size_y: int
    @param size_z: the size of the box in z-direction.
    @type size_z: int
    @param smooth: smoothing parameter that defines the amount of smoothing  at the edge of the wedge.
    @type smooth: float
    @return: 3D array determining the wedge object.
    @rtype: ndarray of np.float64'''
    wedge = xp.zeros((size_x, size_y, size_z // 2 + 1), dtype=xp.float64)
    if rotation is None:
        # numpy meshgrid by default returns indexing with cartesian coordinates (xy)
        # shape N, M, P returns meshgrid with M, N, P (see numpy meshgrid documentation)
        # the naming here is therefore weird
        z, y, x = xp.meshgrid(xp.abs(xp.arange(-size_y // 2 + size_y % 2, size_y // 2 + size_y % 2, 1.)),
                              xp.abs(xp.arange(-size_x // 2 + size_x % 2, size_x // 2 + size_x % 2, 1.)),
                              xp.arange(0, size_z // 2 + 1, 1.))

    else:
        # here its different again, but result is correct.
        cx,cy,cz = [s//2 for s in (size_x,size_y,size_z)]
        grid = xp.mgrid[-cx:size_x - cx, -cy:size_y - cy, :size_z // 2 + 1]

        phi, the, psi = rotation

        phi = -float(phi) * xp.pi / 180.0
        the = -float(the) * xp.pi / 180.0
        psi = -float(psi) * xp.pi / 180.0
        sin_alpha = xp.sin(phi)
        cos_alpha = xp.cos(phi)
        sin_beta = xp.sin(the)
        cos_beta = xp.cos(the)
        sin_gamma = xp.sin(psi)
        cos_gamma = xp.cos(psi)

        # Calculate inverse rotation matrix
        Inv_R = xp.zeros((3, 3), dtype='float32')

        Inv_R[0, 0] = cos_alpha * cos_gamma - cos_beta * sin_alpha \
                      * sin_gamma
        Inv_R[0, 1] = -cos_alpha * sin_gamma - cos_beta * sin_alpha \
                      * cos_gamma
        Inv_R[0, 2] = sin_beta * sin_alpha

        Inv_R[1, 0] = sin_alpha * cos_gamma + cos_beta * cos_alpha \
                      * sin_gamma
        Inv_R[1, 1] = -sin_alpha * sin_gamma + cos_beta * cos_alpha \
                      * cos_gamma
        Inv_R[1, 2] = -sin_beta * cos_alpha

        Inv_R[2, 0] = sin_beta * sin_gamma
        Inv_R[2, 1] = sin_beta * cos_gamma
        Inv_R[2, 2] = cos_beta

        temp = grid.reshape((3, grid.size // 3))
        temp = xp.dot(Inv_R, temp)
        grid = xp.reshape(temp, grid.shape)

        y = abs(grid[0, :, :, :])
        z = abs(grid[1, :, :, :])
        x = abs(grid[2, :, :, :])

    r = xp.sqrt(x ** 2 + y ** 2 + z ** 2)
    if angle1 > 1E-3:
        range_angle1Smooth = smooth / xp.sin(angle1 * xp.pi / 180.)

        with np.errstate(all='ignore'):
            wedge[xp.tan(xp.float32(angle1) * xp.pi / xp.float32(180.)) <= y / x] = 1

        if rotation is None:
            wedge[size_x // 2, :, 0] = 1
        else:
            phi,the,psi = rotation
            if phi < 1E-6 and psi < 1E-6 and the<1E-6:
                wedge[size_x // 2, :, 0] = 1

        if smooth:
            area = xp.abs(x - (y / xp.tan(angle1 * xp.pi / 180))) < range_angle1Smooth
            strip = 1 - ((xp.abs((x) - ((y) / xp.tan(angle1 * xp.pi / 180.)))) * xp.sin(angle1 * xp.pi / 180.) / smooth)
            wedge += (strip * area * (1 - wedge))

    else:
        wedge += 1
    wedge[r > cutoffRadius] = 0
    return xp.fft.ifftshift(wedge, axes=(0, 1))  # TODO should be ifftshift, because centered is shifted to corner

def create_asymmetric_wedge(angle1, angle2, cutoffRadius, size_x, size_y, size_z, smooth, rotation=None):
    '''This function returns an asymmetric wedge object.
    @param angle1: angle of wedge1 in degrees
    @type angle1: int
    @param angle2: angle of wedge2 in degrees
    @type angle2: int
    @param cutOffRadius: radius from center beyond which the wedge is set to zero.
    @type cutOffRadius: int
    @param size_x: the size of the box in x-direction.
    @type size_x: int
    @param size_y: the size of the box in y-direction.
    @type size_y: int
    @param size_z: the size of the box in z-direction.
    @type size_z: int
    @param smooth: smoothing parameter that defines the amount of smoothing  at the edge of the wedge.
    @type smooth: float
    @return: 3D array determining the wedge object.
    @rtype: ndarray of xp.float64'''
    range_angle1Smooth = smooth / xp.sin(angle1 * xp.pi / 180.)
    range_angle2Smooth = smooth / xp.sin(angle2 * xp.pi / 180.)
    wedge = xp.zeros((size_x, size_y, size_z // 2 + 1))

    if rotation is None:
        # see comment above with symmetric wedge function about meshgrid
        z, y, x = xp.meshgrid(xp.arange(-size_y // 2 + size_y % 2, size_y // 2 + size_y % 2),
                              xp.arange(-size_x // 2 + size_x % 2, size_x // 2 + size_x % 2),
                              xp.arange(0, size_z // 2 + 1))

    else:
        cx, cy, cz = [s // 2 for s in (size_x, size_y, size_z)]
        grid = xp.mgrid[-cx:size_x - cx, -cy:size_y - cy, :size_z // 2 + 1]

        phi, the, psi = rotation

        phi = -float(phi) * xp.pi / 180.0
        the = -float(the) * xp.pi / 180.0
        psi = -float(psi) * xp.pi / 180.0
        sin_alpha = xp.sin(phi)
        cos_alpha = xp.cos(phi)
        sin_beta = xp.sin(the)
        cos_beta = xp.cos(the)
        sin_gamma = xp.sin(psi)
        cos_gamma = xp.cos(psi)

        # Calculate inverse rotation matrix
        Inv_R = xp.zeros((3, 3), dtype='float32')

        Inv_R[0, 0] = cos_alpha * cos_gamma - cos_beta * sin_alpha \
                      * sin_gamma
        Inv_R[0, 1] = -cos_alpha * sin_gamma - cos_beta * sin_alpha \
                      * cos_gamma
        Inv_R[0, 2] = sin_beta * sin_alpha

        Inv_R[1, 0] = sin_alpha * cos_gamma + cos_beta * cos_alpha \
                      * sin_gamma
        Inv_R[1, 1] = -sin_alpha * sin_gamma + cos_beta * cos_alpha \
                      * cos_gamma
        Inv_R[1, 2] = -sin_beta * cos_alpha

        Inv_R[2, 0] = sin_beta * sin_gamma
        Inv_R[2, 1] = sin_beta * cos_gamma
        Inv_R[2, 2] = cos_beta

        temp = grid.reshape((3, grid.size // 3))
        temp = xp.dot(Inv_R, temp)
        grid = xp.reshape(temp, grid.shape)

        y = grid[0, :, :, :]
        z = grid[1, :, :, :]
        x = grid[2, :, :, :]

    r = xp.sqrt(x ** 2 + y ** 2 + z ** 2)


    with np.errstate(all='ignore'):
        wedge[xp.tan(angle1 * xp.pi / 180) < y / x] = 1
        wedge[xp.tan(-angle2 * xp.pi / 180) > y / x] = 1
    wedge[size_x // 2, :, 0] = 1

    if smooth:
        area = xp.abs(x - (y / xp.tan(angle1 * xp.pi / 180))) <= range_angle1Smooth
        strip = 1 - (xp.abs(x - (y / xp.tan(angle1 * xp.pi / 180.))) * xp.sin(angle1 * xp.pi / 180.) / smooth)
        wedge += (strip * area * (1 - wedge) * (y > 0))

        area2 = xp.abs(x + (y / xp.tan(angle2 * xp.pi / 180))) <= range_angle2Smooth
        strip2 = 1 - (xp.abs(x + (y / xp.tan(angle2 * xp.pi / 180.))) * xp.sin(angle2 * xp.pi / 180.) / smooth)
        wedge += (strip2 * area2 * (1 - wedge) * (y <= 0))

    wedge[r > cutoffRadius] = 0

    return xp.fft.ifftshift(wedge, axes=(0, 1))  # TODO should be ifftshift, because centered is shifted to corner