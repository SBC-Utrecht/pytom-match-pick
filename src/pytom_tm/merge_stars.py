import pandas as pd
import starfile
import pathlib
import logging


def merge_stars(
    input_dir: pathlib.Path, output_file: pathlib.Path, relion5_compat: bool = False
):
    files = [f for f in input_dir.iterdir() if f.suffix == ".star"]

    if len(files) == 0:
        raise ValueError("No starfiles in directory.")

    logging.info("Concatting and writing star files")

    dataframes = [starfile.read(f) for f in files]

    starfile.write(
        {"particles": pd.concat(dataframes, ignore_index=True)},
        output_file,
        overwrite=True,
    )
