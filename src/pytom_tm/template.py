import numpy.typing as npt
import numpy as np
import voltools as vt
import logging
from scipy.ndimage import center_of_mass, zoom
from scipy.fft import rfftn, irfftn
from typing import Optional
from pytom_tm.weights import (
    create_ctf,
    create_gaussian_low_pass,
    radial_average,
    radial_reduced_grid,
)


def generate_template_from_map(
        input_map: npt.NDArray[float],
        input_spacing: float,
        output_spacing: float,
        center: bool = False,
        filter_to_resolution: Optional[float] = None,
        output_box_size: Optional[int] = None,
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
            mode='constant',
            constant_values=0,
        )

    if filter_to_resolution is None:
        # Set to nyquist resolution
        filter_to_resolution = 2 * output_spacing
    elif filter_to_resolution < (2 * output_spacing):
        warning_text = (f"Filter resolution is too low,"
                        f" setting to {2 * output_spacing}A (2 * output voxel size)")
        logging.warning(warning_text)
        filter_to_resolution = 2 * output_spacing

    if center:
        volume_center = np.divide(np.subtract(input_map.shape, 1), 2, dtype=np.float32)
        # square input to make values positive for center of mass
        input_center_of_mass = center_of_mass(input_map ** 2)
        shift = np.subtract(volume_center, input_center_of_mass)
        input_map = vt.transform(input_map, translation=shift, device='cpu')

        logging.debug(f'center of mass, before was '
                      f'{np.round(input_center_of_mass, 2)} '
                      f'and after {np.round(center_of_mass(input_map ** 2), 2)}')

    # extend volume to the desired output size before applying convolutions!
    if output_box_size is not None:
        logging.debug(
            f'size check {output_box_size} > {(input_map.shape[0] * input_spacing) // output_spacing}')
        if output_box_size > (input_map.shape[0] * input_spacing) // output_spacing:
            pad = (int(output_box_size * (output_spacing / input_spacing)) - 
                   input_map.shape[0])
            logging.debug(f'pad with this number of zeros: {pad}')
            input_map = np.pad(
                input_map,
                (pad // 2, pad // 2 + pad % 2),
                mode='constant',
                constant_values=0
            )
        elif output_box_size < (
                input_map.shape[0] * input_spacing) // output_spacing:
            logging.warning(
                'Could not set specified box size as the map would need to be cut and this might '
                'result in loss of information of the structure. Please decrease the box size of the map '
                'by hand (e.g. chimera)')

    # create low pass filter
    lpf = create_gaussian_low_pass(
        input_map.shape,
        input_spacing,
        filter_to_resolution
    ).astype(np.float32)

    logging.info('Convoluting volume with filter and then downsampling.')
    return zoom(
        irfftn(rfftn(input_map) * lpf, s=input_map.shape),
        input_spacing / output_spacing
    )


def phase_randomize_template(
        template: npt.NDArray[float],
        seed: int = 321,
):
    """Create a version of the template that has its phases randomly
    permuted in Fourier space.

    Parameters
    ----------
    template: npt.NDArray[float]
        input structure
    seed: int, default 321
        seed for random number generator for phase permutation

    Returns
    -------
    result: npt.NDArray[float]
        phase randomized version of the template
    """
    ft = rfftn(template)
    amplitude = np.abs(ft)

    # permute the phases in flattened version of the array
    phase = np.angle(ft).flatten()
    grid = np.fft.ifftshift(
        radial_reduced_grid(template.shape), axes=(0, 1)
    ).flatten()
    relevant_freqs = grid <= 1  # permute only up to Nyquist
    noise = np.zeros_like(phase)
    rng = np.random.default_rng(seed)
    noise[relevant_freqs] = rng.permutation(phase[relevant_freqs])

    # construct the new template
    noise = np.reshape(noise, amplitude.shape)
    result = irfftn(
        amplitude * np.exp(1j * noise), s=template.shape
    )
    return result
