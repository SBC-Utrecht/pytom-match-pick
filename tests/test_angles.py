import unittest
import pathlib
from pytom_tm.angles import load_angle_list


TEST_DATA_DIR = pathlib.Path(__file__).parent.joinpath('test_data')
ERRONEOUS_ANGLE_FILE = TEST_DATA_DIR.joinpath('error_angles.txt')


class TestAngles(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        TEST_DATA_DIR.mkdir()
        # create an erroneous angle file
        with open(ERRONEOUS_ANGLE_FILE, 'w') as fstream:
            fstream.write(' '.join(map(str, [1.] * 4)))
            fstream.write(' '.join(map(str, [1.] * 3)))

    @classmethod
    def tearDownClass(cls) -> None:
        ERRONEOUS_ANGLE_FILE.unlink()
        TEST_DATA_DIR.rmdir()

    def test_load_list(self):
        with self.assertRaises(ValueError, msg='Invalid angle file should raise an error'):
            load_angle_list(ERRONEOUS_ANGLE_FILE)

