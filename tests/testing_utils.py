import pathlib
import numpy as np
import pandas as pd
from pytom_tm.io import read_defocus_file, read_dose_file, read_tlt_file


# Dose and ctf params for tomo_104
cs = 2.7
amp = 0.08
vol = 200
defocus_data = read_defocus_file(
    pathlib.Path(__file__).parent.joinpath("Data").joinpath("test_imod.defocus")
)
CTF_PARAMS = []
for d in defocus_data:
    CTF_PARAMS.append(
        {
            "defocus": d,
            "amplitude_contrast": amp,
            "voltage": vol,
            "spherical_aberration": cs,
            "phase_shift_deg": 0.0,
        }
    )

ACCUMULATED_DOSE = read_dose_file(
    pathlib.Path(__file__).parent.joinpath("Data").joinpath("test_dose.txt")
)
TILT_ANGLES = read_tlt_file(
    pathlib.Path(__file__).parent.joinpath("Data").joinpath("test_angles.rawtlt")
)


def make_random_particles(n: int = 10, relion5: bool = False) -> pd.DataFrame:
    """This is a function that outputs completely random data for testing purposses

    Parameters
    ----------
    n: int
        number of particles to generate random data for
    relion5: bool, default False
        output a relion5 compatible dataframe instead of a relion4 one (default)

    Returns
    -------
    dataframe: pd.DataFrame
        dataframe with data that can be written out as a STAR file
    """
    # Some common variables
    cut_off, sigma = np.random.rand(2)
    pixel_size = 0.5 + 2 * np.random.rand()  # between 0.5 and 2.5
    tomogram_id = f"Tomo{np.random.randint(2147483647)}"  # np.int32 max
    common = [cut_off, sigma, pixel_size, tomogram_id]
    data = []
    for i in range(n):
        x, y, z = np.random.randint(-64, 64, size=3)
        rot, psi = 2 * np.pi * np.random.rand(2)
        tilt = np.pi * np.random.rand()
        lcc_max = cut_off + (1 - cut_off) * np.random.rand()
        temp = [x, y, z, rot, tilt, psi, lcc_max] + common
        data.append(temp)

    output = pd.DataFrame(
        data,
        columns=[
            "rlnCoordinateX",
            "rlnCoordinateY",
            "rlnCoordinateZ",
            "rlnAngleRot",
            "rlnAngleTilt",
            "rlnAnglePsi",
            "rlnLCCmax",
            "rlnCutOff",
            "rlnSearchStd",
            "rlnDetectorPixelSize",
            "rlnMicrographName",
        ],
    )
    if relion5:
        # Use same alterations as in extract.py
        center = (64 + 63) / 2
        output["rlnCoordinateX"], output["rlnCoordinateY"], output["rlnCoordinateZ"] = (
            (output["rlnCoordinateX"] - center) * pixel_size,
            (output["rlnCoordinateY"] - center) * pixel_size,
            (output["rlnCoordinateZ"] - center) * pixel_size,
        )
        column_change = {
            "rlnCoordinateX": "rlnCenteredCoordinateXAngst",
            "rlnCoordinateY": "rlnCenteredCoordinateYAngst",
            "rlnCoordinateZ": "rlnCenteredCoordinateZAngst",
            "rlnMicrographName": "rlnTomoName",
            "rlnDetectorPixelSize": "rlnTomoTiltSeriesPixelSize",
        }
        output = output.rename(columns=column_change)
    return output
