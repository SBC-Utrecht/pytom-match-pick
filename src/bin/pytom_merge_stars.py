#!/usr/bin/env python

import pathlib
import starfile
import argparse
import logging
import pandas as pd
from pytom_tm.io import CheckDirExists, ParseLogging


def main():
    parser = argparse.ArgumentParser(description='Merge multiple star files in the same directory. -- Marten Chaillet (@McHaillet)')
    parser.add_argument('-i', '--input-dir', type=pathlib.Path, required=False, default='./', action=CheckDirExists,
                        help="Directory with star files, script will try to merge all files that end in '.star'.")
    parser.add_argument('-o', '--output-file', type=pathlib.Path, required=False, default='./particles.star',
                        help='Output star file name.')
    parser.add_argument('--log', type=str, required=False, default=20, action=ParseLogging,
                        help='Can be set to `info` or `debug`')
    args = parser.parse_args()
    logging.basicConfig(level=args.log)
    
    files = [f for f in args.input_dir.iterdir() if f.suffix=='.star']
    
    if len(files) == 0:
        raise ValueError('No starfiles in directory.')
        
    logging.info('Concatting and writing star files')
    
    dataframes = [starfile.read(f) for f in files]
    
    starfile.write(pd.concat(dataframes, ignore_index=True), args.output_file, overwrite=True)


if __name__ == '__main__':
    main()
