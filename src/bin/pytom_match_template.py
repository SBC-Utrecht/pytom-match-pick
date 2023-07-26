#!/usr/bin/env python

import argparse
import pathlib
import sys
import mrcfile
from pytom_tm.tmjob import TMJob
from pytom_tm.parallel import run_job_parallel


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run template matching. -- Marten Chaillet (@McHaillet)')
    parser.add_argument('-t', '--tomogram', type=str, required=True,
                        help='Tomogram MRC file.')
    parser.add_argument('-r', '--template', type=str, required=True,
                        help='Template MRC file.')
    parser.add_argument('-m', '--mask', type=str, required=True,
                        help='Mask MRC file.')
    parser.add_argument('-d', '--destination', type=str, required=False, default='./',
                        help='Folder to store the files produced by template matching.')
    parser.add_argument('-w', '--wedge-angles', nargs=2, type=float, required=True,
                        help='Missing wedge angles for a tilt series collected from +/- 60: --wedge-angles -60 60')
    parser.add_argument('--angular-search', type=str, required=True,
                        help='Options are: [7.00, 35.76, 19.95, 90.00, 18.00, '
                             '12.85, 38.53, 11.00, 17.86, 25.25, 50.00, 3.00]')
    parser.add_argument('-s', '--volume-split', nargs=3, type=int, required=False, default=[1, 1, 1],
                        help='Split the volume into smaller parts for the search, can be relevant if the volume does '
                             'not fit into GPU memory. Format is x y z, e.g. --volume-split 1 2 1')
    parser.add_argument('--search-origin', nargs=3, type=int, required=False,
                        help='Limit the search area of the tomogram, in combination with --search-size. '
                             'Format is x y z, e.g. --search-origin 0 0 100 will skip first 100 voxels in z.')
    parser.add_argument('--search-size', nargs=3, type=int, required=False,
                        help='Limit the search area of the tomogram, in combination with --search-origin. '
                             'Format is x y z, e.g. --search-size 0 0 100 will search only the first 100 voxels from '
                             'the origin in z.')
    parser.add_argument('-v', '--voxel-spacing-angstrom', type=float, required=False,
                        help='Voxel spacing of tomogram/template in angstrom, if not provided will try to read from '
                             'the MRC files. Argument is important for bandpass filtering!')
    parser.add_argument('--bandpass', nargs=2, type=float, required=False,
                        help='Apply a bandpass to the tomogram and template. Option requires two '
                             'arguments: one for the high pass and one for low pass. 0 indicates no cutoff, i.e. '
                             '--bandpass 0 40 will just apply a low pass filter to 40A resolution. Resolution is '
                             'determined from the voxel spacing, so set appropriately if your MRCs are not annotated!.')
    parser.add_argument('-g', '--gpu-ids', nargs='+', type=int, required=True,
                        help='GPU indices to run the program on.')

    args = parser.parse_args()

    # check if all locations are valid
    tomogram_path = pathlib.Path(args.tomogram)
    template_path = pathlib.Path(args.template)
    mask_path = pathlib.Path(args.mask)

    if not (tomogram_path.exists() and template_path.exists() and mask_path.exists()):
        print('One of provided files does not exist. Exiting...')
        sys.exit(0)

    destination = pathlib.Path(args.destination)

    if not (destination.exists() and destination.is_dir()):
        print('Output destination for file writing is invalid. Exiting...')
        sys.exit(0)

    job = TMJob(
        job_key='0',
        tomogram=tomogram_path,
        template=template_path,
        mask=mask_path,
        output_dir=destination,
        angle_increment=args.angular_search,
        mask_is_spherical=True,
        wedge_angles=tuple([90 - abs(w) for w in args.wedge_angles]),
        search_origin=args.search_origin,
        search_size=args.search_size,
        voxel_size=args.voxel_spacing_angstrom,
        bandpass=args.bandpass
    )

    score_volume, angle_volume = run_job_parallel(job, tuple(args.volume_split), args.gpu_ids)

    # set the appropriate headers when writing!
    mrcfile.write(destination.joinpath(f'{job.tomo_id}_scores.mrc'), score_volume.T, voxel_size=job.voxel_size,
                  overwrite=True)
    mrcfile.write(destination.joinpath(f'{job.tomo_id}_angles.mrc'), angle_volume.T, voxel_size=job.voxel_size,
                  overwrite=True)

    # write the job as well
    job.write_to_json(destination.joinpath(f'{job.tomo_id}_job.json'))
