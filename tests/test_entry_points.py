import unittest
import pathlib
import numpy as np
from shutil import which
from contextlib import redirect_stdout
from io import StringIO
from pytom_tm import entry_points
from pytom_tm import io

# (command line function, function in entry_points file)
ENTRY_POINTS_TO_TEST = [
    ("pytom_create_mask.py", "pytom_create_mask"),
    ("pytom_create_template.py", "pytom_create_template"),
    ("pytom_match_template.py", "match_template"),
    ("pytom_extract_candidates.py", "extract_candidates"),
    ("pytom_merge_stars.py",  "merge_stars"),
    ]
# Test if optional dependencies are installed
try:
    from pytom_tm import plotting
except:
    pass
else:
    ENTRY_POINTS_TO_TEST.append(("pytom_estimate_roc.py", "estimate_roc"))

# Input files for template matching
TEST_DATA = pathlib.Path(__file__).parent.joinpath('test_data')
TEMPLATE = TEST_DATA.joinpath('template.mrc')
MASK = TEST_DATA.joinpath('mask.mrc')
TOMOGRAM = TEST_DATA.joinpath('tomogram.mrc')
DESTINATION = TEST_DATA.joinpath('output')
TILT_ANGLES = TEST_DATA.joinpath('angles.rawtlt')
DOSE = TEST_DATA.joinpath('test_dose.txt')
DEFOCUS = TEST_DATA.joinpath('defocus.txt')
IMOD_DEFOCUS = pathlib.Path(__file__).parent.joinpath('Data').joinpath('test_imod.defocus')


class TestEntryPoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        io.write_mrc(TEMPLATE, np.zeros((5, 5, 5)), 1)
        io.write_mrc(MASK, np.zeros((5, 5, 5)), 1)
        io.write_mrc(TOMOGRAM, np.zeros((10, 10, 10)), 1)
        np.savetxt(TILT_ANGLES, np.linspace(-50, 50, 35))
        np.savetxt(DOSE, np.linspace(0, 100, 35))
        np.savetxt(DEFOCUS, np.ones(35) * 3000)

    @classmethod
    def tearDownClass(cls) -> None:
        TEMPLATE.unlink()
        MASK.unlink()
        TOMOGRAM.unlink()
        TILT_ANGLES.unlink()
        DOSE.unlink()
        DEFOCUS.unlink()
        for f in DESTINATION.iterdir():
            f.unlink()  # should test specific output?
        DESTINATION.rmdir()
        TEST_DATA.rmdir()

    def test_entry_points_exist(self):
        for cli, fname in ENTRY_POINTS_TO_TEST:
            # test the command line function can be found
            self.assertIsNotNone(which(cli))
            # assert the entry_point be called with -h and exit cleanly
            # catch stdout to prevent shell polution
            func = getattr(entry_points, fname)
            dump = StringIO()
            with self.assertRaises(SystemExit) as ex, redirect_stdout(dump):
                func([cli, '-h'])
            dump.close()
            # check if the system return code is 0 (success)
            self.assertEqual(ex.exception.code, 0)

    # def test_match_template(self):

