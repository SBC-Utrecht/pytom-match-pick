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
    return (data * mask).sum() / (mask_weight if mask_weight is not None else mask.sum())


def std_under_mask(
        data: Union[npt.NDArray[float], cpt.NDArray[float]],
        mask: Union[npt.NDArray[float], cpt.NDArray[float]],
        mean: float, mask_weight=None
) -> Union[float, cpt.NDArray[float]]:
    return (mean_under_mask(data ** 2, mask, mask_weight=mask_weight) - mean ** 2) ** 0.5


def normalise(
        data: Union[npt.NDArray[float], cpt.NDArray[float]],
        mask: Optional[Union[npt.NDArray[float], cpt.NDArray[float]]] = None,
        mask_weight: Optional[float] = None
) -> Union[npt.NDArray[float], cpt.NDArray[float]]:
    new = data.copy()
    if mask is None:
        mean, std = data.mean(), data.std()
    else:
        mean = mean_under_mask(data, mask, mask_weight=mask_weight)
        std = std_under_mask(data, mask, mean, mask_weight=mask_weight)
    return (new - mean) / std


def normalised_cross_correlation(
        data1: Union[npt.NDArray[float], cpt.NDArray[float]],
        data2: Union[npt.NDArray[float], cpt.NDArray[float]],
        mask: Optional[Union[npt.NDArray[float], cpt.NDArray[float]]] = None
) -> Union[float, cpt.NDArray[float]]:
    if mask is None:
        return (normalise(data1) * normalise(data2)).sum() / data1.size
    else:
        return (normalise(data1, mask) * mask * normalise(data2, mask)).sum() / mask.sum()
