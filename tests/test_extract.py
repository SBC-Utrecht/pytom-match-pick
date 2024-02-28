import unittest
import numpy as np
from pytom_tm.extract import predict_tophat_mask


class TestExtract(unittest.TestCase):
    def test_predict_tophat_mask(self):
        rng = np.random.default_rng(0)
        volume = rng.normal(loc=0, scale=0.1, size=(50, ) * 3)  # generate random peaks
        tophat_mask = predict_tophat_mask(volume)
        self.assertEqual(tophat_mask.shape, volume.shape, msg='tophat mask should have same size as input')
        self.assertEqual(tophat_mask.dtype, bool, msg='predicted tophat mask should be boolean')
