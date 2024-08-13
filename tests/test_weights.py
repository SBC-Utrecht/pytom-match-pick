import numpy as np
import unittest
from pytom_tm.weights import (
    create_wedge,
    create_ctf,
    create_gaussian_band_pass,
    radial_reduced_grid,
    radial_average,
    power_spectrum_profile,
    profile_to_weighting,
)
from testing_utils import TILT_ANGLES, ACCUMULATED_DOSE, CTF_PARAMS


class TestWeights(unittest.TestCase):
    def setUp(self):
        self.volume_shape_even = (10, 10, 10)
        self.volume_shape_uneven = (11, 11, 11)
        self.volume_shape_irregular = (7, 12, 6)
        self.voxel_size = 3.34
        self.low_pass = 10
        self.high_pass = 50

        self.reduced_even_shape_3d = (10, 10, 6)
        self.reduced_even_shape_2d = (10, 6)
        self.reduced_uneven_shape_3d = (11, 11, 6)
        self.reduced_uneven_shape_2d = (11, 6)
        self.reduced_irregular_shape_3d = (7, 12, 6 // 2 + 1)
        self.reduced_irregular_shape_2d = (7, 12 // 2 + 1)

    def test_radial_reduced_grid(self):
        with self.assertRaises(
            ValueError,
            msg="Radial reduced grid should raise ValueError if the shape is "
            "not 2- or 3-dimensional.",
        ):
            radial_reduced_grid((5,))
        with self.assertRaises(
            ValueError,
            msg="Radial reduced grid should raise ValueError if the shape is "
            "not 2- or 3-dimensional.",
        ):
            radial_reduced_grid((5,) * 4)

        self.assertEqual(
            radial_reduced_grid(self.volume_shape_even).shape,
            self.reduced_even_shape_3d,
            msg="3D radial reduced grid does not have the correct shape",
        )
        self.assertEqual(
            radial_reduced_grid(self.volume_shape_even[:2]).shape,
            self.reduced_even_shape_2d,
            msg="2D radial reduced grid does not have the correct shape",
        )

    def test_band_pass(self):
        with self.assertRaises(
            ValueError,
            msg="Bandpass should raise ValueError if both low and high pass are None",
        ):
            create_gaussian_band_pass(
                self.volume_shape_even, self.voxel_size, None, None
            )
        with self.assertRaises(
            ValueError,
            msg="Bandpass should raise ValueError if low pass resolution > high pass "
            "resolution",
        ):
            create_gaussian_band_pass(self.volume_shape_even, self.voxel_size, 50, 10)
        band_pass = create_gaussian_band_pass(
            self.volume_shape_even, self.voxel_size, self.low_pass, self.high_pass
        )
        low_pass = create_gaussian_band_pass(
            self.volume_shape_even, self.voxel_size, low_pass=self.low_pass
        )
        high_pass = create_gaussian_band_pass(
            self.volume_shape_even, self.voxel_size, high_pass=self.high_pass
        )

        self.assertEqual(
            band_pass.shape,
            self.reduced_even_shape_3d,
            msg="Bandpass filter does not have expected output shape",
        )
        self.assertEqual(
            band_pass.dtype,
            np.float64,
            msg="Bandpass filter does not have expected dtype",
        )
        self.assertEqual(
            low_pass.shape,
            self.reduced_even_shape_3d,
            msg="Low-pass filter does not have expected output shape",
        )
        self.assertEqual(
            low_pass.dtype,
            np.float64,
            msg="Low-pass filter does not have expected dtype",
        )
        self.assertEqual(
            high_pass.shape,
            self.reduced_even_shape_3d,
            msg="High-pass filter does not have expected output shape",
        )
        self.assertEqual(
            high_pass.dtype,
            np.float64,
            msg="High-pass filter does not have expected dtype",
        )

        self.assertTrue(
            np.sum((band_pass != low_pass) * 1) != 0,
            msg="Band-pass and low-pass should be different",
        )
        self.assertTrue(
            np.sum((band_pass != high_pass) * 1) != 0,
            msg="Band-pass and low-pass filter should be different",
        )
        self.assertTrue(
            np.sum((low_pass != high_pass) * 1) != 0,
            msg="Low-pass and high-pass filter should be different",
        )

    def test_create_wedge(self):
        with self.assertRaises(
            ValueError,
            msg="Create wedge should raise ValueError if tilt_angles list does not "
            "contain at least two values",
        ):
            create_wedge(self.volume_shape_even, [1.0], 1.0)
        with self.assertRaises(
            ValueError,
            msg="Create wedge should raise ValueError if tilt_angles input is not a "
            "list",
        ):
            create_wedge(self.volume_shape_even, 1.0, 1.0)
        with self.assertRaises(
            ValueError,
            msg="Create wedge should raise ValueError if voxel_size is smaller or "
            "equal to 0",
        ):
            create_wedge(self.volume_shape_even, TILT_ANGLES, 0.0)
        with self.assertRaises(
            ValueError,
            msg="Create wedge should raise ValueError if cut_off_radius is smaller or "
            "equal to 0",
        ):
            create_wedge(self.volume_shape_even, TILT_ANGLES, 1.0, cut_off_radius=0.0)

        # create test wedges
        structured_wedge = create_wedge(
            self.volume_shape_even,
            TILT_ANGLES,
            1.0,
            tilt_weighting=True,
            ctf_params_per_tilt=CTF_PARAMS,
        )
        symmetric_wedge = create_wedge(
            self.volume_shape_even,
            [TILT_ANGLES[0], TILT_ANGLES[-1]],
            1.0,
            tilt_weighting=False,
            ctf_params_per_tilt=CTF_PARAMS,
        )
        asymmetric_wedge = create_wedge(
            self.volume_shape_even,
            [TILT_ANGLES[0], TILT_ANGLES[-2]],
            1.0,
            tilt_weighting=False,
            ctf_params_per_tilt=CTF_PARAMS,
        )

        self.assertEqual(
            structured_wedge.shape,
            self.reduced_even_shape_3d,
            msg="Structured wedge does not have expected output shape",
        )
        self.assertEqual(
            structured_wedge.dtype,
            np.float32,
            msg="Structured wedge does not have expected dtype",
        )

        self.assertEqual(
            symmetric_wedge.shape,
            self.reduced_even_shape_3d,
            msg="Symmetric wedge does not have expected output shape",
        )
        self.assertEqual(
            symmetric_wedge.dtype,
            np.float32,
            msg="Symmetric wedge does not have expected dtype",
        )

        self.assertEqual(
            asymmetric_wedge.shape,
            self.reduced_even_shape_3d,
            msg="Asymmetric wedge does not have expected output shape",
        )
        self.assertEqual(
            asymmetric_wedge.dtype,
            np.float32,
            msg="Asymmetric wedge does not have expected dtype",
        )

        self.assertTrue(
            np.sum((symmetric_wedge != asymmetric_wedge) * 1) != 0,
            msg="Symmetric and asymmetric wedge should be different!",
        )

        structured_wedge = create_wedge(
            self.volume_shape_even,
            TILT_ANGLES,
            self.voxel_size,
            tilt_weighting=True,
            cut_off_radius=1.0,
            low_pass=self.low_pass,
            high_pass=self.high_pass,
        )
        self.assertEqual(
            structured_wedge.shape,
            self.reduced_even_shape_3d,
            msg="Wedge with band-pass does not have expected output shape",
        )
        self.assertEqual(
            structured_wedge.dtype,
            np.float32,
            msg="Wedge with band-pass does not have expected dtype",
        )

        # test shapes of wedges
        weights = create_wedge(
            self.volume_shape_even,
            TILT_ANGLES,
            self.voxel_size * 3,
            tilt_weighting=True,
            low_pass=40,
            accumulated_dose_per_tilt=ACCUMULATED_DOSE,
            ctf_params_per_tilt=CTF_PARAMS,
        )
        self.assertEqual(
            weights.shape,
            self.reduced_even_shape_3d,
            msg="3D CTF does not have the correct reduced fourier shape.",
        )
        weights = create_wedge(
            self.volume_shape_uneven,
            TILT_ANGLES,
            self.voxel_size * 3,
            tilt_weighting=True,
            low_pass=40,
            accumulated_dose_per_tilt=ACCUMULATED_DOSE,
            ctf_params_per_tilt=CTF_PARAMS,
        )
        self.assertEqual(
            weights.shape,
            self.reduced_uneven_shape_3d,
            msg="3D CTF does not have the correct reduced fourier shape.",
        )

        # test parameter flexibility of tilt_weighted wedge
        weights = create_wedge(
            self.volume_shape_even,
            TILT_ANGLES,
            self.voxel_size * 3,
            tilt_weighting=True,
            low_pass=self.low_pass,
            accumulated_dose_per_tilt=None,
            ctf_params_per_tilt=None,
        )
        self.assertEqual(
            weights.shape,
            self.reduced_even_shape_3d,
            msg="Tilt weighted wedge should also work without defocus and dose info.",
        )
        weights = create_wedge(
            self.volume_shape_even,
            TILT_ANGLES,
            self.voxel_size * 3,
            tilt_weighting=True,
            low_pass=self.low_pass,
            accumulated_dose_per_tilt=None,
            ctf_params_per_tilt=CTF_PARAMS[:1],
        )
        self.assertEqual(
            weights.shape,
            self.reduced_even_shape_3d,
            msg="Tilt weighted wedge should work with single defocus.",
        )

    def test_ctf(self):
        ctf_raw = create_ctf(
            self.volume_shape_even, self.voxel_size * 1e-10, 3e-6, 0.08, 300e3, 2.7e-3
        )
        ctf_cut = create_ctf(
            self.volume_shape_even,
            self.voxel_size * 1e-10,
            3e-6,
            0.08,
            300e3,
            2.7e-3,
            cut_after_first_zero=True,
        )
        self.assertEqual(
            ctf_raw.shape,
            self.reduced_even_shape_3d,
            msg="CTF does not have expected output shape",
        )
        self.assertTrue(
            np.sum((ctf_raw != ctf_cut) * 1) != 0,
            msg="CTF should be different when cutting it off after the first zero "
            "crossing",
        )

    def test_radial_average(self):
        x, y = 100, 50
        with self.assertRaises(
            ValueError,
            msg="Radial average should raise error if something other than 2d/3d "
            "array is provided.",
        ):
            radial_average(np.zeros(x))
        q, m = radial_average(np.zeros((x, y)))
        self.assertEqual(
            m.shape[0],
            x // 2 + 1,
            msg="Radial average shape should equal largest sampling dimension.",
        )
        q, m = radial_average(np.zeros((30, y)))
        self.assertEqual(
            m.shape[0],
            y,
            msg="Radial average shape should equal largest sampling dimension, "
            "considering Fourier reduced form.",
        )
        q, m = radial_average(np.zeros((20, x, y)))
        self.assertEqual(
            m.shape[0],
            x // 2 + 1,
            msg="Radial average shape should equal largest sampling dimension.",
        )
        q, m = radial_average(np.zeros((20, 30, y)))
        self.assertEqual(
            m.shape[0],
            y,
            msg="Radial average shape should equal largest sampling dimension, "
            "considering Fourier reduced form.",
        )

    def test_power_spectrum_profile(self):
        with self.assertRaises(
            ValueError,
            msg="Power spectrum profile should raise ValueError if input image is "
            "not 2- or 3-dimensional.",
        ):
            power_spectrum_profile(np.zeros(5))
        with self.assertRaises(
            ValueError,
            msg="Power spectrum profile should raise ValueError if input image is "
            "not 2- or 3-dimensional.",
        ):
            power_spectrum_profile(np.zeros((5,) * 4))
        profile = power_spectrum_profile(np.zeros(self.volume_shape_irregular))
        self.assertEqual(
            profile.shape,
            (max(self.volume_shape_irregular) // 2 + 1,),
            msg="Power spectrum profile output shape should be a 1-dimensional array "
            "with length equal to max(input_shape) // 2 + 1, corresponding to largest "
            "sampling component in Fourier space.",
        )

    def test_profile_to_weighting(self):
        with self.assertRaises(
            ValueError,
            msg="Profile to weighting should raise a ValueError if the profile is not "
            "1-dimensional.",
        ):
            profile_to_weighting(np.zeros((5, 5)), (5, 5))
        with self.assertRaises(
            ValueError,
            msg="Profile to weighting should raise a ValueError if the output shape "
            "for the weighting is not 2- or 3-dimensional.",
        ):
            profile_to_weighting(np.zeros(5), (5,))
        with self.assertRaises(
            ValueError,
            msg="Profile to weighting should raise a ValueError if the output shape "
            "for the weighting is not 2- or 3-dimensional.",
        ):
            profile_to_weighting(np.zeros(5), (5,) * 4)

        profile = power_spectrum_profile(np.zeros(self.volume_shape_irregular))
        self.assertEqual(
            profile_to_weighting(profile, self.volume_shape_irregular).shape,
            self.reduced_irregular_shape_3d,
            msg="Profile to weighting should return 3D Fourier reduced array.",
        )
        self.assertEqual(
            profile_to_weighting(profile, self.volume_shape_irregular[:2]).shape,
            self.reduced_irregular_shape_2d,
            msg="Profile to weighting should return 2D Fourier reduced array.",
        )
