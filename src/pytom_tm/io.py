import pathlib
import mrcfile
import argparse
import logging
import numpy.typing as npt
import numpy as np
from operator import attrgetter
from typing import Optional, Union


class ParseLogging(argparse.Action):
    def __call__(self, parser, namespace, values: str, option_string: Optional[str] = None):
        if not values.upper() in ['INFO', 'DEBUG']:
            parser.error("{0} log got an invalid option, set either to `info` or `debug` ".format(option_string))
        else:
            numeric_level = getattr(logging, values.upper(), None)
            setattr(namespace, self.dest, numeric_level)


class CheckDirExists(argparse.Action):
    def __call__(self, parser, namespace, values: pathlib.Path, option_string: Optional[str] = None):
        if not values.is_dir():
            parser.error("{0} got a file path that does not exist ".format(option_string))

        setattr(namespace, self.dest, values)


class CheckFileExists(argparse.Action):
    def __call__(self, parser, namespace, values: pathlib.Path, option_string: Optional[str] = None):
        if not values.exists():
            parser.error("{0} got a file path that does not exist ".format(option_string))

        setattr(namespace, self.dest, values)


class LargerThanZero(argparse.Action):
    def __call__(self, parser, namespace, values: float, option_string: Optional[str] = None):
        if values <= .0:
            parser.error("{0} must be larger than 0".format(option_string))

        setattr(namespace, self.dest, values)


class ParseSearch(argparse.Action):
    def __call__(self, parser, namespace, values: list[int, int], option_string: Optional[str] = None):
        if not (0 <= values[0] < values[1]):
            parser.error("{0} start and end indices must be larger than 0 and end must be larger than start".format(
                option_string))

        setattr(namespace, self.dest, values)


class ParseTiltAngles(argparse.Action):
    def __call__(self, parser, namespace, values: Union[list[str, str], str], option_string: Optional[str] = None):
        if len(values) == 2:  # two wedge angles provided the min and max
            try:
                values = sorted(list(map(float, values)))  # make them floats
                setattr(namespace, self.dest, values)
            except ValueError:
                parser.error("{0} the two arguments provided could not be parsed to floats".format(option_string))
        elif len(values) == 1:
            values = pathlib.Path(values[0])
            if not values.exists() or values.suffix not in ['.tlt', '.rawtlt']:
                parser.error("{0} provided tilt angle file does not exist or does not have the right format".format(
                    option_string))
            setattr(namespace, self.dest, read_tlt_file(values))
        else:
            parser.error("{0} can only take one or two arguments".format(option_string))


class UnequalSpacingError(Exception):
    pass


def write_angle_list(data: npt.NDArray[float], file_name: pathlib.Path, order: tuple[int, int, int] = (0, 2, 1)):
    with open(file_name, 'w') as fstream:
        for i in range(data.shape[1]):
            fstream.write(' '.join([str(x) for x in [data[j, i] for j in order]]) + '\n')


def read_mrc_meta_data(file_name: pathlib.Path, permissive: bool = True) -> dict:
    meta_data = {}
    with mrcfile.mmap(file_name, permissive=permissive) as mrc:
        meta_data['shape'] = tuple(map(int, attrgetter('nx', 'ny', 'nz')(mrc.header)))
        # allow small numerical inconsistencies in voxel size of MRC headers, sometimes seen in Warp
        if not all(
                [np.round(mrc.voxel_size.x, 3) == np.round(s, 3)
                 for s in attrgetter('x', 'y', 'z')(mrc.voxel_size)]
        ):
            raise UnequalSpacingError('Input volume voxel spacing is not identical in each dimension!')
        else:
            if not all([mrc.voxel_size.x == s for s in attrgetter('x', 'y', 'z')(mrc.voxel_size)]):
                logging.warning(f'Voxel size annotation in MRC is slightly different between dimensions, '
                                f'namely {mrc.voxel_size}. It might be a tiny numerical inaccuracy, but '
                                f'please ensure this is not problematic.')
            meta_data['voxel_size'] = float(mrc.voxel_size.x)
    return meta_data


def write_mrc(
        file_name: pathlib.Path,
        data: npt.NDArray[float],
        voxel_size: float,
        overwrite: bool = True,
        transpose: bool = True
) -> None:
    if data.dtype != np.float32:
        logging.warning(f'data for mrc writing is not np.float32 will convert to np.float32')
        data = data.astype(np.float32)
    mrcfile.write(file_name, data.T if transpose else data, voxel_size=voxel_size, overwrite=overwrite)


def read_mrc(
        file_name: pathlib.Path,
        permissive: bool = True,
        transpose: bool = True
) -> npt.NDArray[float]:
    with mrcfile.open(file_name, permissive=permissive) as mrc:
        data = np.ascontiguousarray(mrc.data.T) if transpose else mrc.data
    return data


def read_tlt_file(file_name: pathlib.Path) -> list[float, ...]:
    with open(file_name, 'r') as fstream:
        lines = fstream.readlines()
    return sorted(list(map(float, [x.strip() for x in lines])))

