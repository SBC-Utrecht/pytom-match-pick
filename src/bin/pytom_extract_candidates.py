#!/usr/bin/env python

import argparse
import pathlib
import starfile
import logging
from pytom_tm.tmjob import load_json_to_tmjob
from pytom_tm.extract import extract_particles
from pytom_tm.io import CheckFileExists, LargerThanZero, SetLogging


def main():
    parser = argparse.ArgumentParser(description='Run candidate extraction. -- Marten Chaillet (@McHaillet)')
    parser.add_argument('-j', '--job-file', type=pathlib.Path, required=True, action=CheckFileExists,
                        help='JSON file that contain all data on the template matching job, written out by '
                             'pytom_match_template.py in the destination path.')
    parser.add_argument('-n', '--number-of-particles', type=int, required=True, action=LargerThanZero,
                        help='Maximum number of particles to extract from tomogram.')
    parser.add_argument('--number-of-false-positives', type=int, required=False, action=LargerThanZero,
                        help='Number of false positives to determine the false alarm rate. Here one can increase '
                             'the recall of the particle of interest at the expense of more false positives. The '
                             'default value of 1 is recommended for particles that can be distinguished well from '
                             'the background (high specificity).', default=1)
    parser.add_argument('-r', '--radius-px', type=int, required=True, action=LargerThanZero,
                        help='Particle radius in pixels in the tomogram. It is used during extraction to remove areas '
                             'around peaks preventing double extraction.')
    parser.add_argument('-c', '--cut-off', type=float, required=False,
                        help='Override automated extraction cutoff estimation and instead extract the '
                             'number_of_particles down to this LCCmax value. Set to -1 to guarantee extracting '
                             'number_of_particles. Values larger than 1 make no sense as the correlation cannot be '
                             'higher than 1.')
    parser.add_argument('--log', type=str, required=False, default=20, action=SetLogging,
                        help='Can be set to `info` or `debug`')
    args = parser.parse_args()
    logging.basicConfig(level=args.log)

    # load job and extract particles from the volumes
    job = load_json_to_tmjob(args.job_file)
    df = extract_particles(
        job,
        args.radius_px,
        args.number_of_particles,
        cut_off=args.cut_off,
        n_false_positives=args.number_of_false_positives
    )

    # write out as a RELION type starfile
    starfile.write(df, job.output_dir.joinpath(f'{job.tomo_id}_particles.star'), overwrite=True)


if __name__ == '__main__':
    main()
