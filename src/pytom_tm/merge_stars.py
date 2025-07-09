import pandas as pd
import starfile
import pathlib
import logging


def merge_stars(
    input_dir: pathlib.Path, output_file: pathlib.Path, relion5_compat: bool = False
) -> None:
    """Merge particle starfiles together for RELION4 or make an importable RELION5
    starfile pointing to all the starfiles

    Parameters
    ----------
    input_dir: pathlib.Path
        Input directory that cointains all the particle starfiles
    output_file: pathlib.Path
        Output file to be written
    relion5_compat: Bool, default False
        If True, write the new RELION5 import type starfile instead of just
        concatenating all the starfiles together (default)
    """

    files = [f for f in input_dir.iterdir() if f.suffix == ".star"]

    if len(files) == 0:
        raise ValueError("No starfiles in directory.")

    logging.info("Concatting and writing star files")

    dataframes = (starfile.read(f) for f in files)
    if not relion5_compat:
        output = pd.concat(dataframes, ignore_index=True)
    else:
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
