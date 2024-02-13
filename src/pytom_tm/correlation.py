"""
Functions in this file are cpu/gpu agnostic.
"""
import numpy.typing as npt
import cupy.typing as cpt
from typing import Optional, Union


def mean_under_mask(
        data: Union[npt.NDArray[float], cpt.NDArray[float]],
        mask: Union[npt.NDArray[float], cpt.NDArray[float]],
        mask_weight: Optional[float] = None
) -> Union[float, cpt.NDArray[float]]:
    """Calculate mean of array in the mask region.

    data and mask can be cupy or numpy arrays.

    Parameters
    ----------
    data: Union[npt.NDArray[float], cpt.NDArray[float]]
        input array
    mask: Union[npt.NDArray[float], cpt.NDArray[float]]
        input mask, same dimensions as data
    mask_weight: Optional[float], default None
        optional weight of mask, if not provided mask.sum() is used to determine weight

    Returns
    -------
    output: Union[float, cpt.NDArray[float]]
        mean of data in the region of the mask
    """
    output = (data * mask).sum() / (mask_weight if mask_weight is not None else mask.sum())
    return output


def std_under_mask(
        data: Union[npt.NDArray[float], cpt.NDArray[float]],
        mask: Union[npt.NDArray[float], cpt.NDArray[float]],
        mean: float,
        mask_weight: Optional[float] = None
) -> Union[float, cpt.NDArray[float]]:
    """Calculate standard deviation of array in the mask region. Uses mean_under_mask() to calculate the mean of
    data**2 within the mask.

    data and mask can be cupy or numpy arrays.

    Parameters
    ----------
    data: Union[npt.NDArray[float], cpt.NDArray[float]]
        input array
    mask: Union[npt.NDArray[float], cpt.NDArray[float]]
        input mask, same dimensions as data
    mean: float
        mean of array in masked region
    mask_weight: Optional[f
        optional weight of mask, if not provided mask.sum() is used to determine weight

    Returns
    -------
    output: Union[float, cpt.NDArray[float]]
        standard deviation of data in the region of the mask
    """
    output = (mean_under_mask(data ** 2, mask, mask_weight=mask_weight) - mean ** 2) ** 0.5
    return output


def normalise(
        data: Union[npt.NDArray[float], cpt.NDArray[float]],
        mask: Optional[Union[npt.NDArray[float], cpt.NDArray[float]]] = None,
        mask_weight: Optional[float] = None
) -> Union[npt.NDArray[float], cpt.NDArray[float]]:
    """Normalise array by subtracting mean and dividing by standard deviation. If a mask is provided the array is
    normalised with the mean and std calculated within the mask.

    data and mask can be cupy or numpy arrays.

    Parameters
    ----------
    data: Union[npt.NDArray[float], cpt.NDArray[float]]
        input array to normalise
    mask: Optional[Union[npt.NDArray[float], cpt.NDArray[float]]]
        optional mask to normalise with mean and std in masked region
    mask_weight: Optional[float]
        optional float specifying mask weight, if not provided mask.sum() is used

    Returns
    -------
    output: Union[npt.NDArray[float], cpt.NDArray[float]]
        normalised array
    """
    if mask is None:
        mean, std = data.mean(), data.std()
    else:
        mean = mean_under_mask(data, mask, mask_weight=mask_weight)
        std = std_under_mask(data, mask, mean, mask_weight=mask_weight)
    output = (data - mean) / std
    return output


def normalised_cross_correlation(
        data1: Union[npt.NDArray[float], cpt.NDArray[float]],
        data2: Union[npt.NDArray[float], cpt.NDArray[float]],
        mask: Optional[Union[npt.NDArray[float], cpt.NDArray[float]]] = None
) -> Union[float, cpt.NDArray[float]]:
    """Calculate normalised cross correlation between two arrays. Optionally only in a masked region.

    data1, data2, and mask can be cupy or numpy arrays.

    Parameters
    ----------
    data1: Union[npt.NDArray[float], cpt.NDArray[float]]
        first array for correlation
    data2: Union[npt.NDArray[float], cpt.NDArray[float]]
        second array for correlation
    mask: Optional[Union[npt.NDArray[float], cpt.NDArray[float]]]
        optional mask to calculate the correlation under

    Returns
    -------
    output: Union[float, cpt.NDArray[float]]
        normalised cross correlation between the arrays
    """
    if mask is None:
        output = (normalise(data1) * normalise(data2)).sum() / data1.size
    else:
        output = (normalise(data1, mask) * mask * normalise(data2, mask)).sum() / mask.sum()
    return output
