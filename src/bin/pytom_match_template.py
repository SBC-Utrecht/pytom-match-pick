#!/usr/bin/env python

import argparse
import pathlib
import logging
from pytom_tm.io import LargerThanZero
from pytom_tm.tmjob import TMJob
from pytom_tm.parallel import run_job_parallel
from pytom_tm.io import (CheckFileExists, CheckDirExists, ParseLogging, ParseSearch, ParseTiltAngles, write_mrc,
                         ParseDoseFile, ParseDefocusFile, BetweenZeroAndOne)


def main():
    parser = argparse.ArgumentParser(description='Run template matching. -- Marten Chaillet (@McHaillet)')
    parser.add_argument('-t', '--template', type=pathlib.Path, required=True, action=CheckFileExists,
                        help='Template; MRC file.')
    parser.add_argument('-m', '--mask', type=pathlib.Path, required=True,  action=CheckFileExists,
                        help='Mask with same box size as template; MRC file.')
    parser.add_argument('--non-spherical-mask', action='store_true', required=False,
                        help='Flag to set when the mask is not spherical. It adds the required computations for '
                             'non-spherical masks and roughly doubles computation time.')
    parser.add_argument('-v', '--tomogram', type=pathlib.Path, required=True,  action=CheckFileExists,
                        help='Tomographic volume; MRC file.')
    parser.add_argument('-d', '--destination', type=pathlib.Path, required=False,
                        default='./', action=CheckDirExists,
                        help='Folder to store the files produced by template matching.')
    parser.add_argument('-a', '--tilt-angles', nargs='+', type=str, required=True, action=ParseTiltAngles,
                        help='Tilt angles of the tilt-series, either the minimum and maximum values of the tilts (e.g. '
                             '--tilt-angles -59.1 60.1) or a .rawtlt/.tlt file with all the angles (e.g. '
                             '--tilt-angles tomo101.rawtlt). In case all the tilt angles are provided a more '
                             'elaborate Fourier space constraint can be used')
    parser.add_argument('--per-tilt-weighting', action='store_true', default=False, required=False,
                        help='Flag to set per tilt weighting, only makes sense if a file with all tilt angles has '
                             'been provided. In case not set when a tilt angle file is provided, the minimum and '
                             'maximum tilt angle are used to create a binary wedge. If dose accumulation and CTF '
                             'params are provided these will all be incorporated in the tilt-weighting.')
    parser.add_argument('--angular-search', type=str, required=True,
                        help='Options are: [7.00, 35.76, 19.95, 90.00, 18.00, '
                             '12.85, 38.53, 11.00, 17.86, 25.25, 50.00, 3.00]')
    parser.add_argument('-s', '--volume-split', nargs=3, type=int, required=False, default=[1, 1, 1],
                        help='Split the volume into smaller parts for the search, can be relevant if the volume does '
                             'not fit into GPU memory. Format is x y z, e.g. --volume-split 1 2 1')
    parser.add_argument('--search-x', nargs=2, type=int, required=False, action=ParseSearch,
                        help='Start and end indices of the search along the x-axis, e.g. --search-x 10 490 ')
    parser.add_argument('--search-y', nargs=2, type=int, required=False, action=ParseSearch,
                        help='Start and end indices of the search along the y-axis, e.g. --search-x 10 490 ')
    parser.add_argument('--search-z', nargs=2, type=int, required=False, action=ParseSearch,
                        help='Start and end indices of the search along the z-axis, e.g. --search-x 30 230 ')
    parser.add_argument('--voxel-size-angstrom', type=float,
                        required=False, action=LargerThanZero,
                        help='Voxel spacing of tomogram/template in angstrom, if not provided will try to read from '
                             'the MRC files. Argument is important for band-pass filtering!')
    parser.add_argument('--low-pass', type=float, required=False, action=LargerThanZero,
                        help='Apply a low-pass filter to the tomogram and template. Generally desired if the template '
                             'was already filtered to a certain resolution. Value is the resolution in A.')
    parser.add_argument('--high-pass', type=float, required=False, action=LargerThanZero,
                        help='Apply a high-pass filter to the tomogram and template to reduce correlation with large '
                             'low frequency variations. Value is a resolution in A, e.g. 500 could be appropriate as '
                             'the CTF is often incorrectly modelled up to 50nm.')
    parser.add_argument('--dose-accumulation', type=str, required=False, action=ParseDoseFile,
                        help='Here you can provide a file that contains the accumulated dose at each tilt angle, '
                             'assuming the same ordering of tilts as the tilt angle file. Format should be a .txt '
                             'file with on each line a dose value in e-/A2 .')
    parser.add_argument('--defocus-file', type=str, required=False, action=ParseDefocusFile,
                        help='Here you can provide an IMOD defocus file, with singular fitted defocus (no '
                             'astigmatism). The values, together with the other ctf params (amplitude, voltage, '
                             'spherical abberation, will be used to create a simplistic 3D CTF weighting function. '
                             'Format should be a .def with the defocus in nm, same ordering as tilt angle list.')
    parser.add_argument('--amplitude-contrast', type=float, required=False, action=BetweenZeroAndOne,
                        help='Amplitude contrast fraction for CTF.')
    parser.add_argument('--spherical-abberation', type=float, required=False, action=LargerThanZero,
                        help='Spherical abberation for CTF in mm.')
    parser.add_argument('--voltage', type=float, required=False, action=LargerThanZero,
                        help='Voltage for CTF in keV.')
    parser.add_argument('--spectral-whitening', action='store_true', default=False, required=False,
                        help='Whiten the power spectra of the template and the tomogram patch, effectively puts more '
                             'weight on high resolution features.')
    parser.add_argument('-g', '--gpu-ids', nargs='+', type=int, required=True,
                        help='GPU indices to run the program on.')
    parser.add_argument('--log', type=str, required=False, default=20, action=ParseLogging,
                        help='Can be set to `info` or `debug`')
    args = parser.parse_args()
    logging.basicConfig(level=args.log)

    # combine ctf values to ctf_params list of dicts
    ctf_params = None
    if args.defocus_file is not None:
        if args.amplitude_contrast is None or args.spherical_abberation is None or args.voltage is None:
            raise ValueError('Cannot create 3D CTF weighting because one or multiple of the required parameters ('
                             'amplitude-contrast, spherical-abberation or voltage) is/are missing.')
        ctf_params = [{
                'defocus': defocus,
                'amplitude': args.amplitude_contrast,
                'voltage': args.voltage,
                'cs': args.spherical_abberation
        } for defocus in args.defocus_file]

    job = TMJob(
        '0',
        args.log,
        args.tomogram,
        args.template,
        args.mask,
        args.destination,
        angle_increment=args.angular_search,
        mask_is_spherical=True if args.non_spherical_mask is None else (not args.non_spherical_mask),
        tilt_angles=args.tilt_angles,
        tilt_weighting=args.per_tilt_weighting,
        search_x=args.search_x,
        search_y=args.search_y,
        search_z=args.search_z,
        voxel_size=args.voxel_size_angstrom,
        low_pass=args.low_pass,
        high_pass=args.high_pass,
        dose_accumulation=args.dose_accumulation,
        ctf_data=ctf_params,
        whiten_spectrum=args.spectral_whitening
    )

    score_volume, angle_volume = run_job_parallel(job, tuple(args.volume_split), args.gpu_ids)

    # set the appropriate headers when writing!
    write_mrc(args.destination.joinpath(f'{job.tomo_id}_scores.mrc'), score_volume, job.voxel_size)
    write_mrc(args.destination.joinpath(f'{job.tomo_id}_angles.mrc'), angle_volume, job.voxel_size)

    # write the job as well
    job.write_to_json(args.destination.joinpath(f'{job.tomo_id}_job.json'))


if __name__ == '__main__':
    main()
