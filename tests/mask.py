import unittest
import voltools as vt
import cupy as cp
import numpy as np
from importlib_resources import files
from pytom_tm.mask import spherical_mask
from pytom_tm.angles import load_angle_list
from pytom_tm.correlation import normalised_cross_correlation


class TestMask(unittest.TestCase):
    def test_rotational_invariance(self):
        nxcc_offcenter, nxcc_centered = [], []

        angles = load_angle_list(str(files('pytom_tm.angle_lists').joinpath('angles_50.00_100.txt')))

        mask = spherical_mask(12, 4, 0.5)
        mask_rotated = cp.zeros_like(mask)

        mask_texture = vt.StaticVolume(
            mask,
            interpolation='filt_bspline',
            device='gpu'
        )

        for i in range(len(angles)):

            mask_texture.transform(
                rotation=angles[i],
                rotation_units='deg',
                rotation_order='rzxz',
                center=tuple([x // 2 for x in mask.shape]),
                output=mask_rotated
            )

            nxcc_offcenter.append(normalised_cross_correlation(mask, mask_rotated).get())

        for i in range(len(angles)):

            mask_texture.transform(
                rotation=angles[i],
                rotation_units='deg',
                rotation_order='rzxz',
                center=np.divide(np.subtract(mask.shape, 1), 2, dtype=np.float32),
                output=mask_rotated
            )

            nxcc_centered.append(normalised_cross_correlation(mask, mask_rotated).get())

        self.assertTrue(sum(nxcc_centered) > sum(nxcc_offcenter),
                        msg='Center of rotation for mask is incorrect.')
        self.assertTrue(sum(nxcc_centered) > 99.96, msg='Precision of mask rotation is too low.')


if __name__ == '__main__':
    unittest.main()
