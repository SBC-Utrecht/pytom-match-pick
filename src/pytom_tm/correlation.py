

def mean_under_mask(data, mask):
    return (data * mask).sum() / mask.sum()


def std_under_mask(data, mask, mean):
    return (mean_under_mask(data ** 2, mask) - mean ** 2) ** 0.5


def normalise(data, mask=None):
    new = data.copy()
    if mask is None:
        mean, std = data.mean(), data.std()
    else:
        mean = mean_under_mask(data, mask)
        std = std_under_mask(data, mask, mean)
    return (new - mean) / std


def normalised_cross_correlation(data1, data2, mask=None):
    if mask is None:
        return (normalise(data1) * normalise(data2)).sum() / data1.size
    else:
        return (normalise(data1, mask) * mask * normalise(data2, mask)).sum() / mask.sum()
