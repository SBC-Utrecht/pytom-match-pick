import pandas as pd
import starfile
import pathlib
import logging


def merge_stars(
    input_star_files: list[pathlib.Path],
    output_file: pathlib.Path,
    relion5_compat: bool = False,
) -> None:
    """Merge particle starfiles together for RELION4 or make an importable RELION5
    starfile pointing to all the starfiles

    Parameters
    ----------
    input_star_file: list[pathlib.Path]
        List of input starfiles
    output_file: pathlib.Path
        Output file to be written
    relion5_compat: Bool, default False
        If True, write the new RELION5 import type starfile instead of just
        concatenating all the starfiles together (default)
    """

    files = set(f.resolve() for f in input_star_files)

    # Warn if we end up with less files (due to symlinks pointing to the same thing
    # or the user giving the same star file multiple times)
    if len(files) != len(input_star_files):
        logging.warning("Found duplicate input, only using unique star files")

    if len(files) == 0:
        raise ValueError("No starfiles in directory.")
    elif len(files) < 2 and not relion5_compat:
        raise ValueError(
            "Only one (unique) starfile given which doesn't make sense to merge"
        )

    dataframes = (starfile.read(f) for f in files)
    if not relion5_compat:
        logging.info("Concatting and writing star files")
        output = pd.concat(dataframes, ignore_index=True)
    else:
        logging.info("Writing out 2-column relion5 star file")
        data = []
        for i, df in enumerate(dataframes):
            if "rlnTomoName" not in df.columns:
                raise ValueError(
                    f"Could not find 'rlnTomoName' column in the file: {files[i]}. "
                    "Are you sure this is a relion5 star file?"
                )
            for name in set(df["rlnTomoName"]):
                data.append((name, files[i]))
        output = pd.DataFrame(
            data, columns=["rlnTomoName", "rlnTomoImportParticleFile"]
        )

    starfile.write({"particles": output}, output_file, overwrite=True)
