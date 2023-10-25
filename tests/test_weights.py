import numpy as np
import unittest
from pytom_tm.weights import (create_wedge, create_ctf, create_gaussian_band_pass, radial_reduced_grid)


class TestWeights(unittest.TestCase):
    def setUp(self):
        self.volume_shape = (10, 10, 10)
        self.tilt_angles = list(range(-51, 54, 3))
        self.voxel_size = 3.34
        self.low_pass = 10
        self.high_pass = 50

        self.expected_output_shape = (10, 10, 6)

    def test_radial_reduced_grid(self):
        self.assertEqual(radial_reduced_grid(self.volume_shape).shape, (10, 10, 6),
                         msg='3D radial reduced grid does not have the correct shape')
        self.assertEqual(radial_reduced_grid(self.volume_shape[:2]).shape, (10, 6),
                         msg='2D radial reduced grid does not have the correct shape')

    def test_band_pass(self):
        with self.assertRaises(ValueError, msg='Bandpass should raise ValueError if both low and high pass are None'):
            create_gaussian_band_pass(
                self.volume_shape,
                self.voxel_size,
                None,
                None
            )
        with self.assertRaises(ValueError, msg='Bandpass should raise ValueError if low pass resolution > high pass '
                                               'resolution'):
            create_gaussian_band_pass(
                self.volume_shape,
                self.voxel_size,
                50,
                10
            )
        band_pass = create_gaussian_band_pass(self.volume_shape, self.voxel_size, self.low_pass, self.high_pass)
        low_pass = create_gaussian_band_pass(self.volume_shape, self.voxel_size, low_pass=self.low_pass)
        high_pass = create_gaussian_band_pass(self.volume_shape, self.voxel_size, high_pass=self.high_pass)

        self.assertEqual(band_pass.shape, self.expected_output_shape,
                         msg='Bandpass filter does not have expected output shape')
        self.assertEqual(band_pass.dtype, np.float64,
                         msg='Bandpass filter does not have expected dtype')
        self.assertEqual(low_pass.shape, self.expected_output_shape,
                         msg='Low-pass filter does not have expected output shape')
        self.assertEqual(low_pass.dtype, np.float64,
                         msg='Low-pass filter does not have expected dtype')
        self.assertEqual(high_pass.shape, self.expected_output_shape,
                         msg='High-pass filter does not have expected output shape')
        self.assertEqual(high_pass.dtype, np.float64,
                         msg='High-pass filter does not have expected dtype')

        self.assertTrue(np.sum((band_pass != low_pass) * 1) != 0,
                        msg='Band-pass and low-pass should be different')
        self.assertTrue(np.sum((band_pass != high_pass) * 1) != 0,
                        msg='Band-pass and low-pass filter should be different')
        self.assertTrue(np.sum((low_pass != high_pass) * 1) != 0,
                        msg='Low-pass and high-pass filter should be different')

    def test_create_wedge(self):
        with self.assertRaises(ValueError, msg='Create wedge should raise ValueError if tilt_angles list does not '
                                               'contain at least two values'):
            create_wedge(
                self.volume_shape,
                [1.],
                1.
            )
        with self.assertRaises(ValueError, msg='Create wedge should raise ValueError if tilt_angles input is not a '
                                               'list'):
            create_wedge(
                self.volume_shape,
                1.,
                1.
            )
        with self.assertRaises(ValueError, msg='Create wedge should raise ValueError if voxel_size is smaller or '
                                               'equal to 0'):
            create_wedge(
                self.volume_shape,
                self.tilt_angles,
                0.
            )
        with self.assertRaises(ValueError, msg='Create wedge should raise ValueError if cut_off_radius is smaller or '
                                               'equal to 0'):
            create_wedge(
                self.volume_shape,
                self.tilt_angles,
                1.,
                cut_off_radius=0.
            )

        # create test wedges
        structured_wedge = create_wedge(self.volume_shape, self.tilt_angles, 1., tilt_weighting=True)
        symmetric_wedge = create_wedge(self.volume_shape, [self.tilt_angles[0], self.tilt_angles[-1]],
                                       1., tilt_weighting=False)
        asymmetric_wedge = create_wedge(self.volume_shape, [self.tilt_angles[0], self.tilt_angles[-2]],
                                        1., tilt_weighting=False)

        self.assertEqual(structured_wedge.shape, self.expected_output_shape,
                         msg='Structured wedge does not have expected output shape')
        self.assertEqual(structured_wedge.dtype, np.float32,
                         msg='Structured wedge does not have expected dtype')

        self.assertEqual(symmetric_wedge.shape, self.expected_output_shape,
                         msg='Symmetric wedge does not have expected output shape')
        self.assertEqual(symmetric_wedge.dtype, np.float32,
                         msg='Symmetric wedge does not have expected dtype')

        self.assertEqual(asymmetric_wedge.shape, self.expected_output_shape,
                         msg='Asymmetric wedge does not have expected output shape')
        self.assertEqual(asymmetric_wedge.dtype, np.float32,
                         msg='Asymmetric wedge does not have expected dtype')

        self.assertTrue(np.sum((symmetric_wedge != asymmetric_wedge) * 1) != 0,
                        msg='Symmetric and asymmetric wedge should be different!')

        structured_wedge = create_wedge(self.volume_shape, self.tilt_angles, self.voxel_size, tilt_weighting=True,
                                        cut_off_radius=1., low_pass=self.low_pass, high_pass=self.high_pass)
        self.assertEqual(structured_wedge.shape, self.expected_output_shape,
                         msg='Wedge with band-pass does not have expected output shape')
        self.assertEqual(structured_wedge.dtype, np.float32,
                         msg='Wedge with band-pass does not have expected dtype')

    def test_ctf(self):
        ctf_raw = create_ctf(
            self.volume_shape,
            self.voxel_size * 1E-10,
            3E-6,
            0.08,
            300E3,
            2.7E-3
        )
        ctf_cut = create_ctf(
            self.volume_shape,
            self.voxel_size * 1E-10,
            3E-6,
            0.08,
            300E3,
            2.7E-3,
            cut_after_first_zero=True
        )
        self.assertEqual(ctf_raw.shape, self.expected_output_shape,
                         msg='CTF does not have expected output shape')
        self.assertTrue(np.sum((ctf_raw != ctf_cut) * 1) != 0,
                        msg='CTF should be different when cutting it off after the first zero crossing')


if __name__ == '__main__':
    unittest.main()
