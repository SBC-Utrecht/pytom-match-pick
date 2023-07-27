#!/usr/bin/env python
import argparse
import mrcfile
from pytom_tm.weights import ctf, create_gaussian_low_pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate template from MRC density. -- Marten Chaillet (@McHaillet)')
    parser.add_argument('-i', '--input-map', type=str, required=True,
                        help='Map to generate template from; MRC file.')
    parser.add_argument('-o', '--output-file', type=str, required=False,
                        help='Provide path to write output, needs to end in .mrc . If not provided file is written to '
                             'current directory in the following format: template_{input_map.stem}_{voxel_size}A.mrc')
    parser.add_argument('--input-spacing-angstrom', type=float, required=False,
                        help='Voxel size of input map, in Angstrom. If not provided will be read from MRC input (so '
                             'make sure it is annotated correctly!).')
    parser.add_argument('--output-spacing-angstrom', type=float, required=True,
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
                        help='Spherical abberration in mm.')
    parser.add_argument('--cut-after-first-zero', action='store_true', default=False, required=False,
                        help='Set this flag to cut the CTF after the first zero crossing. Generally recommended to '
                             'apply as the simplistic CTF convolution will likely become inaccurate after this point '
                             'due to defocus gradients.')
    parser.add_argument('--flip-phase', action='store_true', default=False, required=False,
                        help='Set this flag to apply a phase flipped CTF. Only required if the CTF is modelled '
                             'beyond the first zero crossing and if the tomograms have been CTF corrected by phase '
                             'flipping.')
    parser.add_argument('-l', '--lpf-resolution', type=float, required=False,
                        help='Apply a low pass filter to this resolution, in Angstrom. By default a low pass filter '
                             'is applied to a resolution of (2 * output_spacing_angstrom) before downsampling the '
                             'input volume.')
    parser.add_argument('-x', '--xyz', type=int, required=False,
                        help='Specify a desired size for the output box of the template. Only works if it is larger '
                             'than the downsampled box size of the input.')
    parser.add_argument('--invert',  action='store_true', default=False, required=False,
                        help='Multiply template by -1. WARNING not needed if ctf with defocus is already applied!'),
    parser.add_argument('-m', '--mirror', action='store_true', default=False, required=False,
                        help='Mirror the final template before writing to disk.')
    args = parser.parse_args()

    raise NotImplementedError
