import numpy.typing as npt
import numpy as np
import voltools as vt
import logging
from scipy.ndimage import center_of_mass, zoom
from scipy.fft import rfftn, irfftn
from typing import Optional
from pytom_tm.weights import create_ctf, create_gaussian_low_pass, radial_average

plotting_available = True
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(context='talk', style='ticks')
except ModuleNotFoundError:
    plotting_available = False


def generate_template_from_map(
        input_map: npt.NDArray[float],
        input_spacing: float,
        output_spacing: float,
        ctf_params: Optional[dict] = None,
        center: bool = False,
        filter_to_resolution: Optional[float] = None,
        output_box_size: Optional[int] = None,
        display_filter: bool = False
) -> npt.NDArray[float]:
    """Generate a template from a density map.

    Parameters
    ----------
    input_map: npt.NDArray[float]
        3D density to use for generating the template, if box is not square it will be padded to square
    input_spacing: float
        voxel size of input map (in A)
    output_spacing: float
        voxel size of output map (in A) the ratio of input to output will be used for downsampling
    ctf_params: Optional[dict], default None
        dictionary with CTF params to apply to the template, see pytom_tm.weights.create_ctf() for details
    center: bool, default False
        set to True to center the template in the box by calculating the center of mass
    filter_to_resolution: Optional[float], default None
        low-pass filter resolution to apply to template, if not provided will be set to 2 * output_spacing
    output_box_size:  Optional[int], default None
        final box size of template
    display_filter: bool, default False
        flag to display a plot of the filter applied to the template

    Returns
    -------
    template: npt.NDArray[float]
        processed template in the specified output box size, box will be square
    """
    # make the map a box with equal dimensions
    if len(set(input_map.shape)) != 1:
        diff = [max(input_map.shape) - s for s in input_map.shape]
        input_map = np.pad(
            input_map,
            tuple([(d // 2, d // 2 + d % 2) for d in diff]),
            mode='edge'
        )

    if filter_to_resolution is None:
        # Set to nyquist resolution
        filter_to_resolution = 2 * output_spacing
    elif filter_to_resolution < (2 * output_spacing):
        warning_text = (f"Filter resolution is too low,"
                        f" setting to {2 * output_spacing}A (2 * output voxel size)")
        logging.warning(warning_text)
        filter_to_resolution = 2 * output_spacing

    # extend volume to the desired output size before applying convolutions!
    if output_box_size is not None:
        logging.debug(f'size check {output_box_size} > {(input_map.shape[0] * input_spacing) // output_spacing}')
        if output_box_size > (input_map.shape[0] * input_spacing) // output_spacing:
            pad = int(output_box_size * (output_spacing / input_spacing)) - input_map.shape[0]
            logging.debug(f'pad with this number of zeros: {pad}')
            input_map = np.pad(
                input_map,
                (pad // 2, pad // 2 + pad % 2),
                mode='edge'
            )
        elif output_box_size < (input_map.shape[0] * input_spacing) // output_spacing:
            logging.warning('Could not set specified box size as the map would need to be cut and this might '
                            'result in loss of information of the structure. Please decrease the box size of the map '
                            'by hand (e.g. chimera)')

    if center:
        volume_center = np.divide(np.subtract(input_map.shape, 1), 2, dtype=np.float32)
        input_center_of_mass = center_of_mass(input_map)
        shift = np.subtract(volume_center, input_center_of_mass)
        input_map = vt.transform(input_map, translation=shift, device='cpu')

        logging.debug(f'center of mass, before was '
                      f'{np.round(input_center_of_mass, 2)} '
                      f'and after {np.round(center_of_mass(input_map), 2)}')

    # create ctf function and low pass gaussian if desired
    # for ctf the spacing of pixels/voxels needs to be in meters (not angstrom)
    ctf = 1 if ctf_params is None else create_ctf(input_map.shape, **ctf_params).astype(np.float32)
    lpf = create_gaussian_low_pass(input_map.shape, input_spacing, filter_to_resolution).astype(np.float32)

    if display_filter and plotting_available:
        q, average = radial_average(ctf * lpf)
        fig, ax = plt.subplots()
        ax.plot(q / len(q), average)
        ax.set_xlabel('Fraction of Nyquist')
        ax.set_ylabel('Contrast transfer')
        plt.show()
    elif display_filter and not plotting_available:
        logging.info('Plotting not possible as matplotlib is not installed')

    logging.info('Convoluting volume with filter and then downsampling.')
    return zoom(
        irfftn(rfftn(input_map) * ctf * lpf),
        input_spacing / output_spacing
    )
