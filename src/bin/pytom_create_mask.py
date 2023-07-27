#!/usr/bin/env python
import argparse
import mrcfile
import pathlib
from pytom_tm.mask import spherical_mask, ellipsoidal_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a mask for template matching. '
                                                 '-- Marten Chaillet (@McHaillet)')
    parser.add_argument('-b', '--box-size', type=int, required=True,
                        help='Shape of square box for the mask.')
    parser.add_argument('-o', '--output-file', type=str, required=False,
                        help='Provided a filename to write the output. Needs to end with .mrc . By default will write '
                             'a file along this format: ./mask_b[box_size]px_r[radius]px.mrc ')
    parser.add_argument('--voxel-size', type=float, required=False, default=1.,
                        help='Provide a voxel size to annotate the MRC (currently not used for any mask calculation).')
    parser.add_argument('-r', '--radius', type=float, required=True,
                        help='Radius of the spherical mask in number of pixels. In case minor1 and minor2 are '
                             'provided, this will be the radius of the ellipsoidal mask along the x-axis.')
    parser.add_argument('--radius-minor1', type=float, required=False,
                        help='Radius of the ellipsoidal mask along the y-axis in number of pixels.')
    parser.add_argument('--radius-minor2', type=float, required=False,
                        help='Radius of the ellipsoidal mask along the z-axis in number of pixels.')
    parser.add_argument('-s', '--sigma', type=float, required=False,
                        help='Sigma of gaussian drop-off around the mask edges in number of pixels. Values in the '
                             'range from 0.5-1.0 are usually sufficient for tomograms with 20A-10A voxel sizes.')
    args = parser.parse_args()

    # generate mask
    if args.radius_minor1 is not None and args.radius_minor2 is not None:
        mask = ellipsoidal_mask(
            args.box_size,
            args.radius,
            args.radius_minor1,
            args.radius_minor2,
            smooth=args.sigma
        )
    else:
        mask = spherical_mask(
            args.box_size,
            args.radius,
            smooth=args.sigma
        )

    # write to disk
    if args.output_file is not None:
        output_path = pathlib.Path(args.output_file)
    else:
        output_path = pathlib.Path('.').joinpath(f'mask_b{args.box_size}px_r{args.radius}px.mrc')
    mrcfile.write(output_path, mask.T, voxel_size=args.voxel_size, overwrite=True)
