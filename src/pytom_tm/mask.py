import numpy as np
import numpy.typing as npt
from typing import Optional


def spherical_mask(
        box_size: int,
        radius: float,
        smooth: Optional[float] = None,
        cutoff_sd: int = 3,
        center: Optional[float] = None
) -> npt.NDArray[float]:
    """Wrapper around ellipsoidal_mask() to create a spherical mask with just a single radius.

    Parameters
    ----------
    box_size: int
        box size of the mask, equal in each dimension
    radius: float
        radius of sphere
    smooth: Optional[float], default None
        sigma (float relative to number of pixels) of gaussian falloff around mask
    cutoff_sd: int, default 3
        how many standard deviations of the Gaussian falloff to include, default of 3 is a good choice
    center: Optional[float], default None
        alternative center for the mask, default is (size - 1) / 2

    Returns
    -------
    mask: npt.NDArray[float]
        3D numpy array with mask at center
    """
    return ellipsoidal_mask(box_size, radius, radius, radius, smooth, cutoff_sd=cutoff_sd, center=center)


def ellipsoidal_mask(
        box_size: int,
        major: float,
        minor1: float,
        minor2: float,
        smooth: Optional[float] = None,
        cutoff_sd: int = 3,
        center: Optional[float] = None
) -> npt.NDArray[float]:
    """Create an ellipsoidal mask in the specified square box. Ellipsoid is defined by 3 radius on x,y, and z axis.

    Parameters
    ----------
    box_size: int
        box size of the mask, equal in each dimension
    major: float
        radius of ellipsoid in x
    minor1: float
        radius of ellipsoid in y
    minor2: float
        radius of ellipsoid in z
    smooth: Optional[float], default None
        sigma (float relative to number of pixels) of gaussian falloff around mask
    cutoff_sd: int, default 3
        how many standard deviations of the Gaussian falloff to include, default of 3 is a good choice
    center: Optional[float], default None
        alternative center for the mask, default is (size - 1) / 2

    Returns
    -------
    mask: npt.NDArray[float]
        3D numpy array with mask at center
    """
    if not all([box_size > 0, major > 0, minor1 > 0, minor2 > 0]):
        raise ValueError('Invalid input for mask creation: box_size or radii are <= 0')

    center = (box_size - 1) / 2 if center is None else center
    x, y, z = (np.arange(box_size) - center,
               np.arange(box_size) - center,
               np.arange(box_size) - center)

    # use broadcasting
    r = np.sqrt(((x / major)**2)[:, np.newaxis, np.newaxis] +
                ((y / minor1)**2)[:, np.newaxis] +
                (z / minor2)**2).astype(np.float32)

    if smooth is not None:

        if not all([smooth >= 0, cutoff_sd >= 0]):
            raise ValueError('Invalid input for mask smoothing: smooth or sd cutoff are <= 0')

        r[r <= 1] = 1
        sigma = (smooth / ((major + minor1 + minor2) / 3))
        mask = np.exp(-1 * ((r - 1) / sigma) ** 2)
        mask[mask <= np.exp(-cutoff_sd**2/2.)] = 0

    else:

        mask = np.zeros_like(r)
        mask[r <= 1] = 1.

    return mask
