

def mean_under_mask(data, mask, mask_weight=None):
    return (data * mask).sum() / (mask_weight if mask_weight is not None else mask.sum())


def std_under_mask(data, mask, mean, mask_weight=None):
    return (mean_under_mask(data ** 2, mask, mask_weight=mask_weight) - mean ** 2) ** 0.5


def normalise(data, mask=None, mask_weight=None):
    new = data.copy()
    if mask is None:
        mean, std = data.mean(), data.std()
    else:
        mean = mean_under_mask(data, mask, mask_weight=mask_weight)
        std = std_under_mask(data, mask, mean, mask_weight=mask_weight)
    return (new - mean) / std


def normalised_cross_correlation(data1, data2, mask=None):
    if mask is None:
        return (normalise(data1) * normalise(data2)).sum() / data1.size
    else:
        return (normalise(data1, mask) * mask * normalise(data2, mask)).sum() / mask.sum()
