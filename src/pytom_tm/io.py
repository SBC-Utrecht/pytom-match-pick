import pathlib
import mrcfile
import argparse
import logging
import numpy.typing as npt
import numpy as np
import starfile
from contextlib import contextmanager
from operator import attrgetter
from typing import Optional, Union


class ParseLogging(argparse.Action):
    """argparse.Action subclass to parse logging parameter from input scripts. Users can
    set these to info/debug."""

    def __call__(
        self, parser, namespace, values: str, option_string: Optional[str] = None
    ):
        if values.upper() not in ["INFO", "DEBUG"]:
            parser.error(
                f"{option_string} log got an invalid option, "
                "set either to `info` or `debug` "
            )
        else:
            numeric_level = getattr(logging, values.upper(), None)
            setattr(namespace, self.dest, numeric_level)


class CheckDirExists(argparse.Action):
    """argparse.Action subclass to check if an expected input directory exists."""

    def __call__(
        self,
        parser,
        namespace,
        values: pathlib.Path,
        option_string: Optional[str] = None,
    ):
        if not values.is_dir():
            parser.error(
                "{0} got a file path that does not exist ".format(option_string)
            )

        setattr(namespace, self.dest, values)


class CheckFileExists(argparse.Action):
    """argparse.Action subclass to check if an expected input file exists."""

    def __call__(
        self,
        parser,
        namespace,
        values: pathlib.Path,
        option_string: Optional[str] = None,
    ):
        if not values.exists():
            parser.error(
                "{0} got a file path that does not exist ".format(option_string)
            )

        setattr(namespace, self.dest, values)


class LargerThanZero(argparse.Action):
    """argparse.Action subclass to constrain an input value to larger than zero only."""

    def __call__(
        self,
        parser,
        namespace,
        values: Union[int, float],
        option_string: Optional[str] = None,
    ):
        if values <= 0.0:
            parser.error("{0} must be larger than 0".format(option_string))

        setattr(namespace, self.dest, values)


class BetweenZeroAndOne(argparse.Action):
    """argparse.Action subclass to constrain an input value to a fraction, i.e. between
    0 and 1."""

    def __call__(
        self, parser, namespace, values: float, option_string: Optional[str] = None
    ):
        if 1.0 <= values <= 0.0:
            parser.error(
                "{0} is a fraction and can only range between 0 and 1".format(
                    option_string
                )
            )

        setattr(namespace, self.dest, values)


class ParseSearch(argparse.Action):
    """argparse.Action subclass to restrict the search area of tomogram to these indices
    along an axis. Checks that these value are larger than zero and that the second
    value is larger than the first."""

    def __call__(
        self,
        parser,
        namespace,
        values: list[int, int],
        option_string: Optional[str] = None,
    ):
        if not (0 <= values[0] < values[1]):
            parser.error(
                f"{option_string} start and end indices must be larger than 0 and end "
                "must be larger than start"
            )

        setattr(namespace, self.dest, values)


class ParseTiltAngles(argparse.Action):
    """argparse.Action subclass to parse tilt_angle info. The input can either be two
    floats that specify the tilt range for a continous wedge model. Alternatively can be
    a .tlt/.rawtlt file that specifies all the the tilt angles of the tilt-series to use
    for more refined wedge models."""

    def __call__(
        self,
        parser,
        namespace,
        values: Union[list[str, str], str],
        option_string: Optional[str] = None,
    ):
        if len(values) == 2:  # two wedge angles provided the min and max
            try:
                values = sorted(list(map(float, values)))  # make them floats
                setattr(namespace, self.dest, values)
            except ValueError:
                parser.error(
                    f"{option_string} the two arguments provided could not be parsed "
                    "to floats"
                )
        elif len(values) == 1:
            values = pathlib.Path(values[0])
            if not values.exists() or values.suffix not in [".tlt", ".rawtlt"]:
                parser.error(
                    f"{option_string} provided tilt angle file does not exist or does "
                    "not have the right format"
                )
            setattr(namespace, self.dest, read_tlt_file(values))
        else:
            parser.error("{0} can only take one or two arguments".format(option_string))


class ParseGPUIndices(argparse.Action):
    """argparse.Action subclass to parse gpu indices. The input can either be an int or
    a list of ints that specify the gpu indices to be used.
    """

    def __call__(
        self,
        parser,
        namespace,
        values: list[int, ...],
        option_string: Optional[str] = None,
    ):
        import cupy

        max_value = cupy.cuda.runtime.getDeviceCount()
        for val in values:
            if val < 0 or val >= max_value:
                parser.error(
                    f"{option_string} all gpu indices should be between 0 "
                    "and {max_value-1}"
                )

        setattr(namespace, self.dest, values)


class ParseDoseFile(argparse.Action):
    """argparse.Action subclass to parse a txt file contain information on accumulated
    dose per tilt."""

    def __call__(
        self, parser, namespace, values: str, option_string: Optional[str] = None
    ):
        file_path = pathlib.Path(values)
        if not file_path.exists():
            parser.error(
                "{0} provided dose accumulation file does not exist".format(
                    option_string
                )
            )
        allowed_suffixes = [".txt"]
        if file_path.suffix not in allowed_suffixes:
            parser.error(
                "{0}  provided dose accumulation file does not have the right suffix, "
                "allowed are: {1}".format(option_string, ", ".join(allowed_suffixes))
            )
        setattr(namespace, self.dest, read_dose_file(file_path))


class ParseDefocus(argparse.Action):
    """argparse.Action subclass to read a defocus file, either from IMOD which adheres
    to their file format, or a txt file containing per line the defocus of each tilt."""

    def __call__(
        self, parser, namespace, values: str, option_string: Optional[str] = None
    ):
        if values.endswith((".defocus", ".txt")):
            file_path = pathlib.Path(values)
            if not file_path.exists():
                parser.error(f"{option_string} provided defocus file does not exist")
            # in case of a file the parser attribute becomes a list of defocii
            setattr(namespace, self.dest, read_defocus_file(file_path))
        else:
            try:
                defocus = float(values)
            except ValueError:
                parser.error(f"{option_string} not possible to read defocus as float")
            # pass back as list so defocus always returns a list
            setattr(namespace, self.dest, [defocus])


class UnequalSpacingError(Exception):
    """Exception for an mrc file that has unequal spacing along the xyz dimensions
    annotated in its voxel size metadata."""

    pass


def write_angle_list(
    data: npt.NDArray[float],
    file_name: pathlib.Path,
    order: tuple[int, int, int] = (0, 2, 1),
):
    """Helper function to write angular search list from old PyTom to current module.
    Order had to be changed as old PyTom always stored it as Z1, Z2, X, and here it's
    Z1, X, Z2.

    @todo remove function
    """
    with open(file_name, "w") as fstream:
        for i in range(data.shape[1]):
            fstream.write(
                " ".join([str(x) for x in [data[j, i] for j in order]]) + "\n"
            )


@contextmanager
def _wrap_mrcfile_readers(func, *args, **kwargs):
    """Try to autorecover broken mrcfiles, assumes 'permissive' is a kwarg and not an
    arg"""
    try:
        mrc = func(*args, **kwargs)
    except ValueError as err:
        # see if permissive can safe this
        logging.debug(f"mrcfile raised the following error: {err}, will try to recover")
        kwargs["permissive"] = True
        mrc = func(*args, **kwargs)
        if mrc.data is not None:
            logging.warning(
                f"Loading {args[0]} in strict mode gave an error. "
                "However, loading with 'permissive=True' did generate data, make sure "
                "this is correct!"
            )
        else:
            logging.debug("Could not reasonably recover")
            raise ValueError(
                f"{args[0]} header or data is too corrupt to recover, please fix the "
                "header or data"
            ) from err
    yield mrc
    # this should only be called after the context exists
    mrc.close()


def read_mrc_meta_data(file_name: pathlib.Path) -> dict:
    """Read the metadata of provided MRC file path (using mrcfile) and return as dict.

    If the voxel size along the x, y, and z dimensions differs a lot (not within 3
    decimals) the function will raise an UnequalSpacingError as it could mean template
    matching on these volumes might not be consistent.

    Parameters
    ----------
    file_name: pathlib.Path
        path to an MRC file

    Returns
    -------
    metadata: dict
        a dictionary of the mrc metadata with key 'shape' containing the x,y,z
        dimensions of the file and key 'voxel_size' containing the voxel size along
        x, y, and z and dimensions in Å units
    """
    meta_data = {}
    with _wrap_mrcfile_readers(mrcfile.mmap, file_name) as mrc:
        meta_data["shape"] = tuple(map(int, attrgetter("nx", "ny", "nz")(mrc.header)))
        # allow small numerical inconsistencies in voxel size of MRC headers, sometimes
        # seen in Warp
        if not all(
            [
                np.round(mrc.voxel_size.x, 3) == np.round(s, 3)
                for s in attrgetter("y", "z")(mrc.voxel_size)
            ]
        ):
            raise UnequalSpacingError(
                "Input volume voxel spacing is not identical in each dimension!"
            )
        else:
            if not all(
                [mrc.voxel_size.x == s for s in attrgetter("y", "z")(mrc.voxel_size)]
            ):
                logging.warning(
                    "Voxel size annotation in MRC is slightly different between "
                    f"dimensions, namely {mrc.voxel_size}. It might be a tiny "
                    "numerical inaccuracy, but please ensure this is not problematic."
                )
            meta_data["voxel_size"] = float(mrc.voxel_size.x)
    return meta_data


def write_mrc(
    file_name: pathlib.Path,
    data: npt.NDArray[float],
    voxel_size: float,
    overwrite: bool = True,
    transpose: bool = True,
) -> None:
    """Write data to an MRC file. Data is transposed before writing as pytom internally
    uses xyz ordering and MRCs use zyx.

    Parameters
    ----------
    file_name: pathlib.Path
        path on disk to write the file to
    data: npt.NDArray[float]
        numpy array to write as MRC
    voxel_size: float
        voxel size of array to annotate in MRC header
    overwrite: bool, default True
        True (default) will overwrite current MRC on path, setting to False will error
        when writing to existing file
    transpose: bool, default True
        True (default) transpose array before writing, setting to False prevents this

    Returns
    -------
    """
    if data.dtype not in [np.float32, np.float16]:
        logging.warning(
            "data for mrc writing is not np.float32 or np.float16, will convert to "
            "np.float32"
        )
        data = data.astype(np.float32)
    mrcfile.write(
        file_name,
        data.T if transpose else data,
        voxel_size=voxel_size,
        overwrite=overwrite,
    )


def read_mrc(file_name: pathlib.Path, transpose: bool = True) -> npt.NDArray[float]:
    """Read an MRC file from disk. Data is transposed after reading as pytom internally
    uses xyz ordering and MRCs use zyx.

    Parameters
    ----------
    file_name: pathlib.Path
        path to file on disk
    transpose: bool, default True
        True (default) transposes the volume after reading, setting to False prevents
        transpose but probably not a good idea when using the functions from this module

    Returns
    -------
    data: npt.NDArray[float]
        returns the MRC data as a numpy array
    """
    with _wrap_mrcfile_readers(mrcfile.open, file_name) as mrc:
        data = np.ascontiguousarray(mrc.data.T) if transpose else mrc.data
    return data


def read_txt_file(file_name: pathlib.Path) -> list[float, ...]:
    """Read a txt file from disk with on each line a single float value.

    Parameters
    ----------
    file_name: pathlib.Path
        file on disk to read

    Returns
    -------
    output: list[float, ...]
        list of floats
    """
    with open(file_name, "r") as fstream:
        lines = fstream.readlines()
    return list(map(float, [x.strip() for x in lines if not x.isspace()]))


def read_tlt_file(file_name: pathlib.Path) -> list[float, ...]:
    """Read a txt file from disk using read_txt_file(). File is expected to have tilt
    angles in degrees.

    Parameters
    ----------
    file_name: pathlib.Path
        file on disk to read

    Returns
    -------
    output: list[float, ...]
        list of floats with tilt angles
    """
    return read_txt_file(file_name)


def read_dose_file(file_name: pathlib.Path) -> list[float, ...]:
    """Read a txt file from disk using read_txt_file(). File is expected to have dose
    accumulation in e-/(Å^2).

    Parameters
    ----------
    file_name: pathlib.Path
        file on disk to read

    Returns
    -------
    output: list[float, ...]
        list of floats with accumulated dose
    """
    return read_txt_file(file_name)


def read_imod_defocus_file(file_name: pathlib.Path) -> list[float, ...]:
    """Read an IMOD style defocus file. This function can read version 2 and 3 defocus
    files. For format specification see:
    https://bio3d.colorado.edu/imod/doc/man/ctfphaseflip.html
    (section: Defocus File Format).

    Parameters
    ----------
    file_name: pathlib.Path
        file on disk to read

    Returns
    -------
    output: list[float, ...]
        list of floats with defocus (in μm)
    """
    with open(file_name, "r") as fstream:
        lines = fstream.readlines()
    imod_defocus_version = float(lines[0].strip().split()[5])
    # imod defocus files have the values specified in nm:
    if imod_defocus_version == 2:  # file with one defocus value; data starts on line 0
        return [float(x.strip().split()[4]) * 1e-3 for x in lines]
    elif (
        imod_defocus_version == 3
    ):  # file with astigmatism; line 0 contains metadata that we do not need
        return [
            (float(x.strip().split()[4]) + float(x.strip().split()[5])) / 2 * 1e-3
            for x in lines[1:]
        ]
    else:
        raise ValueError("Invalid IMOD defocus file inversion, can only be 2 or 3.")


def read_defocus_file(file_name: pathlib.Path) -> list[float, ...]:
    """Read a defocus file with values in nm. Output returns defocus in μm.

    Depending on file suffix the function calls:
     - read_imod_defocus_file() for .defocus suffix
     - read_txt_file for .txt suffix

    Parameters
    ----------
    file_name: pathlib.Path
        file on disk to read

    Returns
    -------
    output: list[float, ...]
        list of floats with defocus (in μm)
    """
    if file_name.suffix == ".defocus":
        return read_imod_defocus_file(file_name)
    elif file_name.suffix == ".txt":
        return read_txt_file(file_name)
    else:
        raise ValueError("Defocus file needs to have format .defocus or .txt")


def parse_relion5_star_data(
    tomograms_star_path: pathlib.Path,
    tomogram_path: pathlib.Path,
    phase_flip_correction: bool = False,
    phase_shift: float = 0.0,
) -> tuple[float, list[float, ...], list[float, ...], dict]:
    """Read RELION5 metadata from a project directory.

    Parameters
    ----------
    tomograms_star_path: pathlib.Path
        the tomograms.star from a RELION5 reconstruct job contains invariable metadata
        and points to a tilt series star file with fitted values
    tomogram_path: pathlib.Path
        path to the tomogram for template matching; we use the name to pattern match in
        the RELION5 star file
    phase_flip_correction: bool, default False
    phase_shift: float, default 0.0

    Returns
    -------
    tomogram_voxel_size, tilt_angles, dose_accumulation, ctf_params:
        tuple[float, list[float, ...], list[float, ...], list[dict, ...]]
    """
    tomogram_id = tomogram_path.stem
    tomograms_star_data = starfile.read(tomograms_star_path)

    # match the tomo_id and check if viable
    matches = [
        (i, x)
        for i, x in enumerate(tomograms_star_data["rlnTomoName"])
        if x in tomogram_id
    ]
    if len(matches) == 1:
        tomogram_meta_data = tomograms_star_data.loc[matches[0][0]]
    else:
        raise ValueError(
            "Multiple or zero matches of tomogram id in RELION5 STAR file. Aborting..."
        )

    # grab the path to tilt series star where tilt angles, defocus and dose are
    # annotated
    tilt_series_star_path = pathlib.Path(
        tomogram_meta_data["rlnTomoTiltSeriesStarFile"]
    )
    # update the path to a location we can actually find from CD
    tilt_series_star_path = tomograms_star_path.parent.joinpath("tilt_series").joinpath(
        tilt_series_star_path.name
    )
    tilt_series_star_data = starfile.read(tilt_series_star_path)

    # we extract tilt angles, dose accumulation and ctf params
    # TODO We need to have an internal structure for tilt series meta data
    tilt_angles = list(tilt_series_star_data["rlnTomoNominalStageTiltAngle"])
    dose_accumulation = list(tilt_series_star_data["rlnMicrographPreExposure"])

    # _rlnTomoName  # 1
    # _rlnVoltage  # 2
    # _rlnSphericalAberration  # 3
    # _rlnAmplitudeContrast  # 4
    # _rlnMicrographOriginalPixelSize  # 5
    # _rlnTomoHand  # 6
    # _rlnTomoTiltSeriesPixelSize  # 7
    # _rlnTomoTiltSeriesStarFile  # 8
    # _rlnTomoTomogramBinning  # 9
    # _rlnTomoSizeX  # 10
    # _rlnTomoSizeY  # 11
    # _rlnTomoSizeZ  # 12
    # _rlnTomoReconstructedTomogram  # 13
    tomogram_voxel_size = (
        tomogram_meta_data["rlnTomoTiltSeriesPixelSize"]
        * tomogram_meta_data["rlnTomoTomogramBinning"]
    )

    ctf_params = [
        {
            "defocus": defocus * 1e-6,
            "amplitude_contrast": tomogram_meta_data["rlnAmplitudeContrast"],
            "voltage": tomogram_meta_data["rlnVoltage"] * 1e3,
            "spherical_aberration": tomogram_meta_data["rlnSphericalAberration"] * 1e-3,
            "flip_phase": phase_flip_correction,
            "phase_shift_deg": phase_shift,  # Unclear to me where the phase
            # shifts are annotated in RELION5 tilt series file
        }
        for defocus in (
            tilt_series_star_data.rlnDefocusV + tilt_series_star_data.rlnDefocusU
        )
        / 2
        * 1e-4
    ]

    return tomogram_voxel_size, tilt_angles, dose_accumulation, ctf_params
