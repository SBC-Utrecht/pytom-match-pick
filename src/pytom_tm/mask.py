import cupy as cp
import cupy.typing as cpt
from typing import Optional
# TODO create mask on CPU?


def spherical_mask(
        box_size: int,
        radius: float,
        smooth: Optional[float] = None,
        cutoff_sd: float = 3.
) -> cpt.NDArray[float]:
    return ellipsoidal_mask(box_size, radius, radius, radius, smooth, cutoff_sd=cutoff_sd)


def ellipsoidal_mask(
        box_size: int,
        major: float,
        minor1: float,
        minor2: float,
        smooth: Optional[float] = None,
        cutoff_sd: float = 3.
) -> cpt.NDArray[float]:
    """
    Center of the ellipsoid or sphere is (box_size - 1) / 2 => important for rotation center in template matching.
    @param box_size: box size of the mask, equal in each dimension
    @param major: radius of ellipsoid in x
    @param minor1: radius of ellipsoid in y
    @param minor2: radius of ellipsoid in z
    @param smooth: sigma (float relative to number of pixels) of gaussian falloff of mask
    @param cutoff_sd: how many standard deviations of the falloff to include, default of 3 is a good choice
    @return: volume with the mask
    """
    if not all([box_size > 0, major > 0, minor1 > 0, minor2 > 0]):
        raise ValueError('Invalid input for mask creation: box_size or radii are <= 0')

    x, y, z = (cp.arange(box_size) - (box_size - 1) / 2,
               cp.arange(box_size) - (box_size - 1) / 2,
               cp.arange(box_size) - (box_size - 1) / 2)

    # use broadcasting
    r = cp.sqrt(((x / major)**2)[:, cp.newaxis, cp.newaxis] +
                ((y / minor1)**2)[:, cp.newaxis] +
                (z / minor2)**2).astype(cp.float32)

    if smooth is not None:

        if not all([smooth >= 0, cutoff_sd >= 0]):
            raise ValueError('Invalid input for mask smoothing: smooth or sd cutoff are <= 0')

        r[r <= 1] = 1
        sigma = (smooth / ((major + minor1 + minor2) / 3))
        mask = cp.exp(-1 * ((r - 1) / sigma) ** 2)
        mask[mask <= cp.exp(-cutoff_sd**2/2.)] = 0

    else:

        mask = cp.zeros_like(r)
        mask[r <= 1] = 1.

    return mask
