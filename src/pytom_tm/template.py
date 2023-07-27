import numpy.typing as npt
import numpy as np
import voltools as vt
from scipy.ndimage import center_of_mass
from scipy.fft import rfftn, irfftn
from typing import Optional
from pytom_tm.weights import create_ctf, create_gaussian_low_pass


def generate_template_from_map(
        input_map: npt.NDArray[float],
        input_spacing: float,
        output_spacing: float,
        ctf_params: dict,
        filter_to_resolution: Optional[float] = None,
        output_box_size: Optional[int] = None,
        display_filter: bool = False
) -> npt.NDArray[float]:

    # make the map a box with equal dimensions
    if len(set(input_map.shape)) != 1:
        input_map = np.pad(
            input_map,
            tuple([max(input_map.shape) - s for s in input_map.shape]),
            mode='edge'
        )

    if filter_to_resolution < (2 * output_spacing):
        print(f'Invalid resolution specified, changing to {2 * output_spacing}A')
        filter_to_resolution = 2 * output_spacing

    # extend volume to the desired output size before applying convolutions!
    if output_box_size is not None:
        if output_box_size > (input_map.shape[0] * input_spacing) // output_spacing:
            new_size = int(output_box_size * (output_spacing/input_spacing))
            input_map = np.pad(
                input_map,
                tuple([new_size - s for s in input_map.shape]),
                mode='edge'
            )
        elif output_box_size < (input_map.shape[0] * input_spacing) // output_spacing:
            print('Could not set specified box size as the map would need to be cut and this might result in '
                  'loss of information of the structure. Please decrease the box size of the map by hand (e.g. '
                  'chimera)')

    # TODO center the volume with the center of mass
    # vt.transform(translation=center_of_mass(input_map))

    # create ctf function and low pass gaussian if desired
    # for ctf the spacing of pixels/voxels needs to be in meters (not angstrom)
    ctf = 1 if ctf_params is None else create_ctf(input_map.shape, **ctf_params)
    lpf = create_gaussian_low_pass(input_map.shape, input_spacing, filter_to_resolution)

    if display_filter:
        raise NotImplementedError

    return vt.transform(
        irfftn(rfftn(input_map) * ctf * lpf),
        scale=output_spacing / input_spacing,
        interpolation='filt_bspline',
        device='cpu'
    )
