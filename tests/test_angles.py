import unittest
import pathlib
from pytom_tm.angles import load_angle_list


TEST_DATA_DIR = pathlib.Path(__file__).parent.joinpath('test_data')
ERRONEOUS_ANGLE_FILE = TEST_DATA_DIR.joinpath('error_angles.txt')
UNORDERED_ANGLE_FILE = TEST_DATA_DIR.joinpath('unordered_angles.txt')



class TestAngles(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        TEST_DATA_DIR.mkdir()
        # create an erroneous angle file
        with open(ERRONEOUS_ANGLE_FILE, 'w') as fstream:
            fstream.write(' '.join(map(str, [1.] * 4)) + '\n')
            fstream.write(' '.join(map(str, [1.] * 3)) + '\n')
        # create an unordered angle file    
        with open(UNORDERED_ANGLE_FILE, 'w') as fstream:
            fstream.write(' '.join(['3.', '3.', '1.']) + '\n') 
            fstream.write(' '.join(['3', '2.', '1.']) + '\n') 
            fstream.write(' '.join(['2.', '3.', '1.'])+ '\n') 
            fstream.write(' '.join(['3.', '2.', '2.'])+ '\n') 



    @classmethod
    def tearDownClass(cls) -> None:
        for f in [ERRONEOUS_ANGLE_FILE, UNORDERED_ANGLE_FILE]:
            f.unlink()
        TEST_DATA_DIR.rmdir()

    def test_load_list(self):
        with self.assertRaisesRegex(ValueError, "each line should have 3",
                msg='Invalid angle file should raise an error'):
            load_angle_list(ERRONEOUS_ANGLE_FILE)

    def test_load_sort(self):
        angles = load_angle_list(UNORDERED_ANGLE_FILE, sort_angles=True)
        expected = [(2.,3.,1.), (3.,2.,1.), (3.,2.,2.), (3.,3.,1.)]
        self.assertEqual(angles, expected)
