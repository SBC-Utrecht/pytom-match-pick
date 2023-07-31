#!/usr/bin/env python

import argparse
import mrcfile
import pathlib
from pytom_tm.io import LargerThanZero
from pytom_tm.tmjob import TMJob
from pytom_tm.parallel import run_job_parallel
from pytom_tm.io import CheckFileExists, CheckDirExists, SetLogging


def main():
    parser = argparse.ArgumentParser(description='Run template matching. -- Marten Chaillet (@McHaillet)')
    parser.add_argument('-t', '--template', type=pathlib.Path, required=True, action=CheckFileExists,
                        help='Template; MRC file.')
    parser.add_argument('-m', '--mask', type=pathlib.Path, required=True,  action=CheckFileExists,
                        help='Mask with same box size as template; MRC file.')
    parser.add_argument('-v', '--tomogram', type=pathlib.Path, required=True,  action=CheckFileExists,
                        help='Tomographic volume; MRC file.')
    parser.add_argument('-d', '--destination', type=pathlib.Path, required=False,
                        default='./', action=CheckDirExists,
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
    parser.add_argument('--voxel-size-angstrom', type=float,
                        required=False, action=LargerThanZero,
                        help='Voxel spacing of tomogram/template in angstrom, if not provided will try to read from '
                             'the MRC files. Argument is important for bandpass filtering!')
    parser.add_argument('--lowpass', type=float, required=False, action=LargerThanZero,
                        help='Apply a lowpass filter to the tomogram and template. Generally desired if the template '
                             'was already filtered to a certain resolution. Value is the resolution in A.')
    parser.add_argument('--highpass', type=float, required=False, action=LargerThanZero,
                        help='Apply a highpass filter to the tomogram and template to reduce correlation with large '
                             'low frequency variations. Value is a resolution in A, e.g. 500 could be appropriate as '
                             'the CTF is often incorrectly modelled up to 50nm.')
    parser.add_argument('-g', '--gpu-ids', nargs='+', type=int, required=True,
                        help='GPU indices to run the program on.')
    parser.add_argument('--log', type=str, required=False, default=20, action=SetLogging,
                        help='Can be set to `info` or `debug`')

    args = parser.parse_args()

    job = TMJob(
        '0',
        args.tomogram,
        args.template,
        args.mask,
        args.destination,
        args.log,
        angle_increment=args.angular_search,
        mask_is_spherical=True,
        wedge_angles=tuple([90 - abs(w) for w in args.wedge_angles]),
        search_origin=args.search_origin,
        search_size=args.search_size,
        voxel_size=args.voxel_size_angstrom,
        lowpass=args.lowpass,
        highpass=args.highpass
    )

    score_volume, angle_volume = run_job_parallel(job, tuple(args.volume_split), args.gpu_ids)

    # set the appropriate headers when writing!
    mrcfile.write(args.destination.joinpath(f'{job.tomo_id}_scores.mrc'), score_volume.T, voxel_size=job.voxel_size,
                  overwrite=True)
    mrcfile.write(args.destination.joinpath(f'{job.tomo_id}_angles.mrc'), angle_volume.T, voxel_size=job.voxel_size,
                  overwrite=True)

    # write the job as well
    job.write_to_json(args.destination.joinpath(f'{job.tomo_id}_job.json'))


if __name__ == '__main__':
    main()
