import pathlib
import os
import contextlib
import numpy as np
import pandas as pd
import starfile
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


def make_relion5_tomo_stars(
    tomoname: str,
    out_dir: pathlib.Path,
    voltage: int = 200,
    amp_contrast: float = 0.08,
    cs: float = 2.7,
    tomohand: int = 1,
    pixelsize: float = 1.0,
    binning: float = 1.0,
    tilt_angles: list | None = None,
    dose: list | None = None,
    defocus: list | None = None,
):
    """This is a function that outputs a tomogram.star and a tilt_series/tilt_serie.star
    Mostly to test our relion5 processing is (internally) consistent

    Parameters:

    """
    if not out_dir.is_dir():
        out_dir.mkdir(parents=True)
    if tilt_angles is None:
        tilt_angles = TILT_ANGLES
    if dose is None:
        dose = ACCUMULATED_DOSE
    if defocus is None:
        defocus = defocus_data

    tomo_star = out_dir / "tomogram.star"
    tilt_star = out_dir / "tilt_series" / "tilt_serie.star"

    tilt_data = [
        [tilt_angle, do, de, de]
        for do, tilt_angle, de in zip(dose, tilt_angles, defocus)
    ]
    tilt_output = pd.DataFrame(
        tilt_data,
        columns=[
            "rlnTomoNominalStageTiltAngle",
            "rlnMicrographPreExposure",
            "rlnDefocusV",
            "rlnDefocusU",
        ],
    )
    tomo_data = [
        [
            tomoname,
            voltage,
            cs,
            amp_contrast,
            tomohand,
            pixelsize,
            str(tilt_star),
            binning,
        ]
    ]
    tomo_output = pd.DataFrame(
        tomo_data,
        columns=[
            "rlnTomoName",
            "rlnVoltage",
            "rlnSphericalAberration",
            "rlnAmplitudeContrast",
            "rlnTomoHand",
            "rlnTomoTiltSeriesPixelSize",
            "rlnTomoTiltSeriesStarFile",
            "rlnTomoTomogramBinning",
        ],
    )
    starfile.write({"global": tomo_output}, tomo_star, overwrite=True)
    starfile.write({tomoname: tilt_output}, tilt_star, overwrite=True)


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
        x, y, z = np.random.randint(128, size=3)
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
        center = 128 / 2
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


@contextlib.contextmanager
def chdir(directory):
    """This is a function that allows a test to use chdir,
    but not error tests run after it if an error is raised while chdir is active

    Parameters
    ----------
    directory: pathlib.Path
        directory to chdir into

    Returns
    -------
    contextmanager: contextlib.contextmanager
        a contextmanager that deals with unwinding chdir if an
        error is raised while the context is active
    """
    old = os.getcwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(old)
