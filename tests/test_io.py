import unittest
import pathlib
import warnings
import contextlib
from tempfile import TemporaryDirectory
import numpy as np
import mrcfile

from pytom_tm.io import (
    read_mrc,
    read_mrc_meta_data,
    write_mrc,
    parse_relion5_star_data,
    read_tlt_file,
    MultiColumnAngleFileError,
)

FAILING_MRC = pathlib.Path(__file__).parent.joinpath(
    pathlib.Path("Data/human_ribo_mask_32_8_5.mrc")
)
# The below file was made with head -c 1024 human_ribo_mask_32_8_5.mrc > header_only.mrc
CORRUPT_MRC = pathlib.Path(__file__).parent.joinpath(
    pathlib.Path("Data/header_only.mrc")
)
RELION5_TOMOGRAMS_STAR = pathlib.Path(__file__).parent.joinpath(
    "Data/relion5_project_example/Tomograms/job009/tomograms.star"
)
REGULAR_TLT = pathlib.Path(__file__).parent.joinpath(
    pathlib.Path("Data/test_angles.rawtlt")
)
# same file ase regular except an index column is added
MULTI_COLUMN_TLT = pathlib.Path(__file__).parent.joinpath(
    pathlib.Path("Data/test_angles_multi_column.rawtlt")
)


class TestMultiColumnTilt(unittest.TestCase):
    def test_read_multi_column_tilt(self):
        # test we raise on default
        with self.assertRaises(MultiColumnAngleFileError):
            _ = read_tlt_file(MULTI_COLUMN_TLT)
        # allow reading when an override flag is passed
        angles_multi = read_tlt_file(MULTI_COLUMN_TLT, error_on_multi_column=False)
        angles = read_tlt_file(REGULAR_TLT)
        self.assertEqual(angles, angles_multi)


class TestBrokenMRC(unittest.TestCase):
    def setUp(self):
        # Mute the RuntimeWarnings comming from other code-base inside these tests
        # following this SO answer: https://stackoverflow.com/a/45809502
        stack = contextlib.ExitStack()
        _ = stack.enter_context(warnings.catch_warnings())
        warnings.simplefilter("ignore")
        # The follwing line is better, but only works in python >= 3.11
        # _ = stack.enter_context(warnings.catch_warnings(action="ignore"))

        self.addCleanup(stack.close)

        # prep temporary directory
        tempdir = TemporaryDirectory()
        self.tempdirname = tempdir.name
        self.addCleanup(tempdir.cleanup)

    def test_read_mrc_minor_broken(self):
        # Test if this mrc can be read and if the approriate logs are printed
        with self.assertLogs(level="WARNING") as cm:
            mrc = read_mrc(FAILING_MRC)
        self.assertIsNotNone(mrc)
        self.assertEqual(len(cm.output), 1)
        self.assertIn(FAILING_MRC.name, cm.output[0])
        self.assertIn("make sure this is correct", cm.output[0])

    def test_read_mrc_too_broken(self):
        # Test if this mrc raises an error as expected
        with self.assertRaises(ValueError) as err:
            _ = read_mrc(CORRUPT_MRC)
        self.assertIn(CORRUPT_MRC.name, str(err.exception))
        self.assertIn("too corrupt", str(err.exception))

    def test_read_mrc_meta_data(self):
        # Test if this mrc can be read and if the approriate logs are printed
        with self.assertLogs(level="WARNING") as cm:
            mrc = read_mrc_meta_data(FAILING_MRC)
        self.assertIsNotNone(mrc)
        self.assertEqual(len(cm.output), 1)
        self.assertIn(FAILING_MRC.name, cm.output[0])
        self.assertIn("make sure this is correct", cm.output[0])

    def test_half_precision_read_write_cycle(self):
        array = np.random.rand(27).reshape((3, 3, 3)).astype(np.float16)
        fname = pathlib.Path(self.tempdirname) / "test_half.mrc"
        # Make sure no warnings are raised
        with self.assertNoLogs(level="WARNING"):
            write_mrc(fname, array, 1.0)
        # Make sure the file can be read back
        # make sure mode is as expected for float16
        # https://mrcfile.readthedocs.io/en/stable/source/mrcfile.html#mrcfile.utils.dtype_from_mode
        mrc = mrcfile.open(fname)
        self.assertEqual(mrc.header.mode, 12)
        mrc.close()
        # make sure dtype is expected
        mrc = read_mrc(fname)
        self.assertEqual(mrc.dtype, np.float16)
        # make sure data is identical
        np.testing.assert_equal(mrc, array)

    def test_cast_warning(self):
        # make sure a warning is raised when writing an integer based array
        array = np.random.rand(27).reshape((3, 3, 3)).astype(np.int32)
        fname = pathlib.Path(self.tempdirname) / "test_cast.mrc"
        with self.assertLogs(level="WARNING") as cm:
            write_mrc(fname, array, 1.0)
        self.assertEqual(len(cm.output), 1)
        self.assertIn("np.float32", cm.output[0])

    def test_parse_relion5_star_data(self):
        tomogram = pathlib.Path("rec_tomo200528_107.mrc")
        meta_data = parse_relion5_star_data(RELION5_TOMOGRAMS_STAR, tomogram)
        self.assertEqual(len(meta_data), 5)
        self.assertIsInstance(meta_data[0], float)
        self.assertIsInstance(meta_data[1], list)
        self.assertIsInstance(meta_data[2], list)
        self.assertIsInstance(meta_data[3], list)
        self.assertIsInstance(meta_data[3][0], dict)
        self.assertIsInstance(meta_data[4], int)

        tomogram = pathlib.Path("tomogram.mrc")
        with self.assertRaises(
            ValueError, msg="Unmatching tomograms name should raise an error."
        ):
            parse_relion5_star_data(RELION5_TOMOGRAMS_STAR, tomogram)

        tomogram = pathlib.Path("rec_tomogram200528_1077.mrc")
        with self.assertRaises(
            ValueError, msg="Partially matching tomogram name should raise an error."
        ):
            parse_relion5_star_data(RELION5_TOMOGRAMS_STAR, tomogram)

        tomogram = pathlib.Path("rec_tomogram200528_10.mrc")
        with self.assertRaises(
            ValueError, msg="Partially matching tomogram name should raise an error."
        ):
            parse_relion5_star_data(RELION5_TOMOGRAMS_STAR, tomogram)
