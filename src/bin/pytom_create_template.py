#!/usr/bin/env python

import argparse
import pathlib
import numpy as np
import logging
from pytom_tm.io import read_mrc_meta_data, read_mrc, write_mrc, LargerThanZero, CheckFileExists, ParseLogging
from pytom_tm.template import generate_template_from_map


def main():
    parser = argparse.ArgumentParser(description='Generate template from MRC density. -- Marten Chaillet (@McHaillet)')
    parser.add_argument('-i', '--input-map', type=pathlib.Path, required=True, action=CheckFileExists,
                        help='Map to generate template from; MRC file.')
    parser.add_argument('-o', '--output-file', type=pathlib.Path, required=False,
                        help='Provide path to write output, needs to end in .mrc . If not provided file is written to '
                             'current directory in the following format: template_{input_map.stem}_{voxel_size}A.mrc')
    parser.add_argument('--input-voxel-size-angstrom', type=float, required=False,
                        action=LargerThanZero,
                        help='Voxel size of input map, in Angstrom. If not provided will be read from MRC input (so '
                             'make sure it is annotated correctly!).')
    parser.add_argument('--output-voxel-size-angstrom', type=float, required=True,
                        action=LargerThanZero,
                        help='Output voxel size of the template, in Angstrom. Needs to be equal to the voxel size of '
                             'the tomograms for template matching. Input map will be downsampled to this spacing.')
    parser.add_argument('--center', action='store_true', default=False, required=False,
                        help='Set this flag to automatically center the density in the volume by measuring the center '
                             'of mass.')
    parser.add_argument('-c', '--ctf-correction', action='store_true', default=False, required=False,
                        help='Set this flag to multiply the input map with a CTF. The following parameters are also '
                             'important to specify because the defaults might not apply to your data: --defocus, '
                             '--amplitude-contrast, --voltage, --Cs.')
    parser.add_argument('-z', '--defocus', type=float, required=False, default=3.,
                        help='Defocus in um (negative value is overfocus).')
    parser.add_argument('-a', '--amplitude-contrast', type=float, required=False, default=0.08,
                        help='Fraction of amplitude contrast in the image ctf.')
    parser.add_argument('-v', '--voltage', type=float, required=False, default=300.,
                        help='Acceleration voltage of electrons in keV')
    parser.add_argument('--Cs', type=float, required=False, default=2.7,
                        help='Spherical aberration in mm.')
    parser.add_argument('--cut-after-first-zero', action='store_true', default=False, required=False,
                        help='Set this flag to cut the CTF after the first zero crossing. Generally recommended to '
                             'apply as the simplistic CTF convolution will likely become inaccurate after this point '
                             'due to defocus gradients.')
    parser.add_argument('--flip-phase', action='store_true', default=False, required=False,
                        help='Set this flag to apply a phase flipped CTF. Only required if the CTF is modelled '
                             'beyond the first zero crossing and if the tomograms have been CTF corrected by phase '
                             'flipping.')
    parser.add_argument('--low-pass', type=float, required=False, action=LargerThanZero,
                        help='Apply a low pass filter to this resolution, in Angstrom. By default a low pass filter '
                             'is applied to a resolution of (2 * output_spacing_angstrom) before downsampling the '
                             'input volume.')
    parser.add_argument('-b', '--box-size', type=int, required=False, action=LargerThanZero,
                        help='Specify a desired size for the output box of the template. Only works if it is larger '
                             'than the downsampled box size of the input.')
    parser.add_argument('--invert', action='store_true', default=False, required=False,
                        help='Multiply template by -1. WARNING not needed if ctf with defocus is already applied!'),
    parser.add_argument('-m', '--mirror', action='store_true', default=False, required=False,
                        help='Mirror the final template before writing to disk.')
    parser.add_argument('--display-filter', action='store_true', default=False, required=False,
                        help='Display the combined CTF and low pass filter to the user.')
    parser.add_argument('--log', type=str, required=False, default=20, action=ParseLogging,
                        help='Can be set to `info` or `debug`')
    args = parser.parse_args()
    logging.basicConfig(level=args.log)

    # set input voxel size and give user warning if it does not match with MRC annotation
    input_data = read_mrc(args.input_map)
    input_meta_data = read_mrc_meta_data(args.input_map)
    if args.input_voxel_size_angstrom is not None:
        if args.input_voxel_size_angstrom != round(input_meta_data['voxel_size'], 3):
            logging.warning('Provided voxel size does not match voxel size annotated in input map.')
        map_spacing_angstrom = args.input_voxel_size_angstrom
    else:
        map_spacing_angstrom = input_meta_data['voxel_size']

    # set output path
    output_path = args.output_file if args.output_file is not None else (
        pathlib.Path(f'template_{args.input_map.stem}_{args.output_voxel_size_angstrom}A.mrc'))

    if map_spacing_angstrom > args.output_voxel_size_angstrom:
        raise NotImplementedError('It is assumed the input map has smaller voxel size than the output template.')

    ctf_params = None
    if args.ctf_correction:
        ctf_params = {
            'pixel_size': map_spacing_angstrom * 1E-10,
            'defocus': args.defocus * 1E-6,
            'amplitude_contrast': args.amplitude_contrast,
            'voltage': args.voltage * 1E3,
            'spherical_aberration': args.Cs * 1E-3,
            'cut_after_first_zero': args.cut_after_first_zero,
            'flip_phase': args.flip_phase
        }

    template = generate_template_from_map(
        input_data,
        map_spacing_angstrom,
        args.output_voxel_size_angstrom,
        center=args.center,
        ctf_params=ctf_params,
        filter_to_resolution=args.low_pass,
        output_box_size=args.box_size,
        display_filter=args.display_filter
    ) * (-1 if args.invert else 1)

    logging.debug(f'shape of template after processing is: {template.shape}')

    write_mrc(
        output_path,
        np.flip(template, axis=0) if args.mirror else template,
        args.output_voxel_size_angstrom
    )


if __name__ == '__main__':
    main()
