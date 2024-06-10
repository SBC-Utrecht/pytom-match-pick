import unittest
from pytom_tm.io import read_mrc, read_mrc_meta_data
import pathlib
import warnings
import contextlib

FAILING_MRC = pathlib.Path(__file__).parent.joinpath(
    pathlib.Path("Data/human_ribo_mask_32_8_5.mrc")
)
# The below file was made with head -c 1024 human_ribo_mask_32_8_5.mrc > header_only.mrc
CORRUPT_MRC = pathlib.Path(__file__).parent.joinpath(
    pathlib.Path("Data/header_only.mrc")
)


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
