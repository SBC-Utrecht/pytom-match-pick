import numpy.typing as npt
from typing import Optional

# from pytom.agnostic.io import read, read_pixelsize
# from pytom.simulation.microscope import create_ctf, display_microscope_function
# from pytom.agnostic.transform import resize
# from pytom.agnostic.filter import applyFourierFilterFull
# from pytom.agnostic.tools import paste_in_center
# from pytom.simulation.support import create_gaussian_low_pass


def generate_template_from_map(
        input_map: npt.NDArray[float],
        input_spacing: float,
        output_spacing: float,
        apply_ctf_correction: bool = False,
        defocus: float = 3E-6,
        amplitude_contrast: float = 0.07,
        voltage: float = 300E3,
        spherical_aberration: float = 2.7E-3,
        phase_flip: bool = False,
        cut_after_first_zero: bool = False,
        filter_to_resolution: Optional[float] = None,
        output_box_size: Optional[int] = None,
        display_filter: bool = False
) -> npt.NDArray[float]:

    # read map and its pixel size
    template = read(map_file_path, deviceID=device) if 'gpu' in device else read(map_file_path)
    pixel_size = original_spacing if original_spacing is not None else read_pixelsize(map_file_path)

    # ensure pixel size is equal among dimensions
    if type(pixel_size) == list:
        assert pixel_size[0] == pixel_size[1] and pixel_size[0] == pixel_size[2], \
            'pixel size not equal in each dimension'
        # set to a single float
        pixel_size = pixel_size[0]

    if not resolution >= (2 * spacing * binning):
        print(f'Invalid resolution specified, changing to {2*spacing*binning}A')
        resolution = 2 * spacing * binning

    # extend volume to the desired input size before applying convolutions!
    if box_size is not None:
        if box_size > (template.shape[0] * pixel_size) // (spacing * binning):
            new_box = xp.zeros((int(box_size * binning * (spacing/pixel_size)),) * 3)
            template = paste_in_center(template, new_box)
        elif box_size < (template.shape[0] * pixel_size) // (spacing * binning):
            print('Could not set specified box size as the map would need to be cut and this might result in '
                  'loss of information of the structure. Please decrease the box size of the map by hand (e.g. '
                  'chimera)')

    # create ctf function and low pass gaussian if desired
    # for ctf the spacing of pixels/voxels needs to be in meters (not angstrom)
    ctf = create_ctf(template.shape, pixel_size * 1E-10, defocus, amplitude_contrast, voltage, Cs,
                     sigma_decay=ctf_decay, zero_cut=zero_cut) if apply_ctf_correction else 1
    if apply_ctf_correction and phase_flip:
        ctf = xp.abs(ctf) * -1 if defocus > 0 else xp.abs(ctf)
    lpf = create_gaussian_low_pass(template.shape, pixel_size, resolution)

    # print information back to user
    if apply_ctf_correction:
        print(f'Applying ctf correction with defocus {defocus*1e6:.2f} um')
        if zero_cut == 0:
            print(f' --> cutting the ctf after first zero crossing')
    print(f'Applying low pass filter to {resolution}A resolution')
    # apply ctf and low pass in fourier space
    filter = lpf * ctf
    if display_ctf:
        try:
            print('Displaying combined ctf and lpf frequency modulation')
            display_microscope_function(filter[..., filter.shape[2] // 2], form='ctf*lpf')
        except Exception as e:
            print(e)
            print('Skipping plotting due to error.')

    # filter the template
    template = applyFourierFilterFull(template, xp.fft.ifftshift(filter))

    # binning
    if binning > 1:
        print(f'Binning volume {binning} times')
        template = resize(template, pixel_size / (spacing * binning), interpolation='Spline')

    return template