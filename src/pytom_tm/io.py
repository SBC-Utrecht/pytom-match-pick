import pathlib
import mrcfile
import argparse
import logging
import numpy.typing as npt
from operator import attrgetter
from typing import Optional


class SetLogging(argparse.Action):
    def __call__(self, parser, namespace, values: str, option_string: Optional[str] = None):
        if not values.upper() in ['INFO', 'DEBUG']:
            parser.error("{0} log got an invalid option, set either to `info` or `debug` ".format(option_string))
        else:
            numeric_level = getattr(logging, values.upper(), None)
            logging.basicConfig(level=numeric_level)
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


def write_angle_list(data: npt.NDArray[float], file_name: pathlib.Path, order: tuple[int, int, int] = (0, 2, 1)):
    with open(file_name, 'w') as fstream:
        for i in range(data.shape[1]):
            fstream.write(' '.join([str(x) for x in [data[j, i] for j in order]]) + '\n')


def read_mrc_meta_data(file_name: pathlib.Path) -> dict:
    meta_data = {}
    with mrcfile.mmap(file_name) as mrc:
        meta_data['shape'] = tuple(map(int, attrgetter('nx', 'ny', 'nz')(mrc.header)))
        if not all([mrc.voxel_size.x == s for s in attrgetter('x', 'y', 'z')(mrc.voxel_size)]):
            raise ValueError('Input tomogram voxel spacing is not identical in each dimension!')
        else:
            meta_data['voxel_size'] = float(mrc.voxel_size.x)
    return meta_data
