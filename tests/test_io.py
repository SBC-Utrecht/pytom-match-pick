import unittest
from pytom_tm.io import read_mrc, read_mrc_meta_data
import pathlib

FAILING_MRC = pathlib.Path(__file__).parent.joinpath(pathlib.Path('Data/human_ribo_mask_32_8_5.mrc'))

class TestBrokenMRC(unittest.TestCase):
    def test_read_mrc(self):
        read_mrc(FAILING_MRC)

    def test_read_mrc_meat_data(self):
        read_mrc_meta_data(FAILING_MRC)

