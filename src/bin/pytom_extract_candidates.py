#!/usr/bin/env python

import argparse
import pathlib
import sys
import starfile
from pytom_tm.tmjob import load_json_to_tmjob
from pytom_tm.extract import extract_particles


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run candidate extraction. -- Marten Chaillet (@McHaillet)')
    parser.add_argument('-j', '--job-file', type=str, required=True,
                        help='JSON file that contain all data on the template matching job, written out by '
                             'pytom_match_template.py in the destination path.')
    parser.add_argument('-n', '--number-of-particles', type=int, required=True,
                        help='Maximum number of particles to extract from tomogram.')
    parser.add_argument('-r', '--radius-px', type=int, required=True,
                        help='Particle radius in pixels in the tomogram. It is used during extraction to remove areas '
                             'around peaks preventing double extraction.')
    parser.add_argument('-c', '--cut-off', type=float, required=False,
                        help='Override automated extraction cutoff estimation and instead extract the '
                             'number_of_particles up to this LCCmax value.')
    args = parser.parse_args()

    job_file = pathlib.Path(args.job_file)
    if not job_file.exists():
        print('Job file does not exist, exiting ...')
        sys.exit(0)

    job = load_json_to_tmjob(job_file)
    df = extract_particles(job, args.radius_px, args.number_of_particles, cut_off=args.cut_off)

    starfile.write(df, job.output_dir.joinpath(f'{job.tomo_id}_particles.star'), overwrite=True)