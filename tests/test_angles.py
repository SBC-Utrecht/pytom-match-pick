import unittest
import pathlib
from pytom_tm.angles import load_angle_list, angle_to_angle_list
import numpy as np
import itertools as itt
import re

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

    def test_angle_to_angle_list(self):
        # ask for a random sample between [1 - 90)
        angle = 1 + np.random.random() * 89
        with self.assertLogs(level='INFO') as cm:
            angles = angle_to_angle_list(angle)

        # Check logs and if all angles are smaller or equal
        self.assertEqual(len(cm.output), 2)
        for out in cm.output:
            # Check if the used_angle <= input angle
            # regex = whitespace -> 0-inf digits -> "." -> 1-inf digits -> whitespace
            possible_match = re.findall(r"\s\d*[.]\d+\s", out)
            self.assertEqual(len(possible_match), 1)
            self.assertLessEqual(float(possible_match[0]), angle)

        # make sure everything is sorted and X is never 0
        for a, b in itt.pairwise(angles):
            # make sure default is sorted
            self.assertLess(a, b)
            # make sure X is never 0
            self.assertNotEqual(a[1], 0)
        # also check the last X
        self.assertNotEqual(b[1], 0)
