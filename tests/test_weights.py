import numpy as np
import unittest
from pytom_tm.weights import (create_wedge, create_ctf, create_gaussian_band_pass, radial_reduced_grid,
                              radial_average, power_spectrum_profile, profile_to_weighting)
from pytom_tm.io import write_mrc


# Dose and ctf params for tomo_104
CS = 2.7
AMP = 0.08
VOL = 200
IMOD_CTF = '''1	1	-50.99	-50.99	  3083   2
2	2	-47.99	-47.99	  3084
3	3	-44.99	-44.99	  3147
4	4	-41.99	-41.99	  3083
5	5	-38.99	-38.99	  3164
6	6	-35.99	-35.99	  3087
7	7	-32.99	-32.99	  3116
8	8	-29.99	-29.99	  3071
9	9	-26.99	-26.99	  3089
10	10	-23.99	-23.99	  3078
11	11	-20.99	-20.99	  3082
12	12	-17.99	-17.99	  3007
13	13	-14.99	-14.99	  3047
14	14	-11.99	-11.99	  3089
15	15	-8.99	-8.99	  3055
16	16	-5.99	-5.99	  3133
17	17	-2.99	-2.99	  3206
18	18	 0.01	 0.01	  3323
19	19	 3.01	 3.01	  3233
20	20	 6.01	 6.01	  3166
21	21	 9.01	 9.01	  3181
22	22	12.01	12.01	  3167
23	23	15.00	15.00	  3183
24	24	18.00	18.00	  3194
25	25	21.01	21.01	  3257
26	26	24.01	24.01	  3281
27	27	27.00	27.00	  3274
28	28	30.01	30.01	  3249
29	29	33.00	33.00	  3326
30	30	36.00	36.00	  3319
31	31	39.00	39.00	  3338
32	32	42.00	42.00	  3309
33	33	45.00	45.00	  3326
34	34	48.00	48.00	  3274
35	35	51.00	51.00	  3354
'''
CTF_PARAMS = []
for d in [float(x.split()[4]) * 1e-3 for x in IMOD_CTF.split('\n') if x != '']:
    CTF_PARAMS.append({
        'defocus': d,
        'amplitude_contrast': AMP,
        'voltage': VOL,
        'spherical_aberration': CS,
        'phase_shift_deg': .0,
    })

DOSE_FILE = '''60.165 
58.255 
56.345 
46.795 
44.885 
42.975 
41.065 
35.335 
33.425 
27.695 
25.785 
20.055 
18.145 
12.415 
10.505 
4.775 
2.865 
0.955 
6.685 
8.595 
14.325 
16.235 
21.965 
23.875 
29.605 
31.515 
37.245 
39.155 
48.705 
50.615 
52.525 
54.435 
62.075 
63.985 
65.895 
'''
ACCUMULATED_DOSE = [float(x.strip()) for x in DOSE_FILE.split('\n') if x != '']
TILT_ANGLES = list(range(-51, 54, 3))


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
        with self.assertRaises(ValueError, msg='Radial reduced grid should raise ValueError if the shape is '
                                               'not 2- or 3-dimensional.'):
            radial_reduced_grid((5, ))
        with self.assertRaises(ValueError, msg='Radial reduced grid should raise ValueError if the shape is '
                                               'not 2- or 3-dimensional.'):
 
            radial_reduced_grid((5, ) * 4)

        self.assertEqual(radial_reduced_grid(self.volume_shape_even).shape, self.reduced_even_shape_3d,
                         msg='3D radial reduced grid does not have the correct shape')
        self.assertEqual(radial_reduced_grid(self.volume_shape_even[:2]).shape, self.reduced_even_shape_2d,
                         msg='2D radial reduced grid does not have the correct shape')

    def test_band_pass(self):
        with self.assertRaises(ValueError, msg='Bandpass should raise ValueError if both low and high pass are None'):
            create_gaussian_band_pass(
                self.volume_shape_even,
                self.voxel_size,
                None,
                None
            )
        with self.assertRaises(ValueError, msg='Bandpass should raise ValueError if low pass resolution > high pass '
                                               'resolution'):
            create_gaussian_band_pass(
                self.volume_shape_even,
                self.voxel_size,
                50,
                10
            )
        band_pass = create_gaussian_band_pass(self.volume_shape_even, self.voxel_size, self.low_pass, self.high_pass)
        low_pass = create_gaussian_band_pass(self.volume_shape_even, self.voxel_size, low_pass=self.low_pass)
        high_pass = create_gaussian_band_pass(self.volume_shape_even, self.voxel_size, high_pass=self.high_pass)

        self.assertEqual(band_pass.shape, self.reduced_even_shape_3d,
                         msg='Bandpass filter does not have expected output shape')
        self.assertEqual(band_pass.dtype, np.float64,
                         msg='Bandpass filter does not have expected dtype')
        self.assertEqual(low_pass.shape, self.reduced_even_shape_3d,
                         msg='Low-pass filter does not have expected output shape')
        self.assertEqual(low_pass.dtype, np.float64,
                         msg='Low-pass filter does not have expected dtype')
        self.assertEqual(high_pass.shape, self.reduced_even_shape_3d,
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
                self.volume_shape_even,
                [1.],
                1.
            )
        with self.assertRaises(ValueError, msg='Create wedge should raise ValueError if tilt_angles input is not a '
                                               'list'):
            create_wedge(
                self.volume_shape_even,
                1.,
                1.
            )
        with self.assertRaises(ValueError, msg='Create wedge should raise ValueError if voxel_size is smaller or '
                                               'equal to 0'):
            create_wedge(
                self.volume_shape_even,
                TILT_ANGLES,
                0.
            )
        with self.assertRaises(ValueError, msg='Create wedge should raise ValueError if cut_off_radius is smaller or '
                                               'equal to 0'):
            create_wedge(
                self.volume_shape_even,
                TILT_ANGLES,
                1.,
                cut_off_radius=0.
            )

        # create test wedges
        structured_wedge = create_wedge(self.volume_shape_even, TILT_ANGLES, 1.,
                                        tilt_weighting=True, ctf_params_per_tilt=CTF_PARAMS)
        symmetric_wedge = create_wedge(self.volume_shape_even, [TILT_ANGLES[0], TILT_ANGLES[-1]],
                                       1., tilt_weighting=False,
                                       ctf_params_per_tilt=CTF_PARAMS)
        asymmetric_wedge = create_wedge(self.volume_shape_even, [TILT_ANGLES[0], TILT_ANGLES[-2]],
                                        1., tilt_weighting=False,
                                        ctf_params_per_tilt=CTF_PARAMS)

        self.assertEqual(structured_wedge.shape, self.reduced_even_shape_3d,
                         msg='Structured wedge does not have expected output shape')
        self.assertEqual(structured_wedge.dtype, np.float32,
                         msg='Structured wedge does not have expected dtype')

        self.assertEqual(symmetric_wedge.shape, self.reduced_even_shape_3d,
                         msg='Symmetric wedge does not have expected output shape')
        self.assertEqual(symmetric_wedge.dtype, np.float32,
                         msg='Symmetric wedge does not have expected dtype')

        self.assertEqual(asymmetric_wedge.shape, self.reduced_even_shape_3d,
                         msg='Asymmetric wedge does not have expected output shape')
        self.assertEqual(asymmetric_wedge.dtype, np.float32,
                         msg='Asymmetric wedge does not have expected dtype')

        self.assertTrue(np.sum((symmetric_wedge != asymmetric_wedge) * 1) != 0,
                        msg='Symmetric and asymmetric wedge should be different!')

        structured_wedge = create_wedge(self.volume_shape_even, TILT_ANGLES, self.voxel_size, tilt_weighting=True,
                                        cut_off_radius=1., low_pass=self.low_pass, high_pass=self.high_pass)
        self.assertEqual(structured_wedge.shape, self.reduced_even_shape_3d,
                         msg='Wedge with band-pass does not have expected output shape')
        self.assertEqual(structured_wedge.dtype, np.float32,
                         msg='Wedge with band-pass does not have expected dtype')

        # test shapes of wedges
        weights = create_wedge(self.volume_shape_even, TILT_ANGLES, self.voxel_size * 3,
                               tilt_weighting=True, low_pass=40,
                               accumulated_dose_per_tilt=ACCUMULATED_DOSE,
                               ctf_params_per_tilt=CTF_PARAMS)
        self.assertEqual(weights.shape, self.reduced_even_shape_3d,
                         msg='3D CTF does not have the correct reduced fourier shape.')
        weights = create_wedge(self.volume_shape_uneven, TILT_ANGLES, self.voxel_size * 3,
                               tilt_weighting=True, low_pass=40,
                               accumulated_dose_per_tilt=ACCUMULATED_DOSE,
                               ctf_params_per_tilt=CTF_PARAMS)
        self.assertEqual(weights.shape, self.reduced_uneven_shape_3d,
                         msg='3D CTF does not have the correct reduced fourier shape.')

        # test parameter flexibility of tilt_weighted wedge
        weights = create_wedge(self.volume_shape_even, TILT_ANGLES, self.voxel_size * 3,
                               tilt_weighting=True, low_pass=self.low_pass,
                               accumulated_dose_per_tilt=None,
                               ctf_params_per_tilt=None)
        self.assertEqual(weights.shape, self.reduced_even_shape_3d,
                         msg='Tilt weighted wedge should also work without defocus and dose info.')
        weights = create_wedge(self.volume_shape_even, TILT_ANGLES, self.voxel_size * 3,
                               tilt_weighting=True, low_pass=self.low_pass,
                               accumulated_dose_per_tilt=None,
                               ctf_params_per_tilt=CTF_PARAMS[slice(0, 1)])
        self.assertEqual(weights.shape, self.reduced_even_shape_3d,
                         msg='Tilt weighted wedge should work with single defocus.')

    def test_ctf(self):
        ctf_raw = create_ctf(
            self.volume_shape_even,
            self.voxel_size * 1E-10,
            3E-6,
            0.08,
            300E3,
            2.7E-3
        )
        ctf_cut = create_ctf(
            self.volume_shape_even,
            self.voxel_size * 1E-10,
            3E-6,
            0.08,
            300E3,
            2.7E-3,
            cut_after_first_zero=True
        )
        self.assertEqual(ctf_raw.shape, self.reduced_even_shape_3d,
                         msg='CTF does not have expected output shape')
        self.assertTrue(np.sum((ctf_raw != ctf_cut) * 1) != 0,
                        msg='CTF should be different when cutting it off after the first zero crossing')

    def test_radial_average(self):
        x, y = 100, 50
        with self.assertRaises(ValueError, msg='Radial average should raise error if something other than 2d/3d '
                                               'array is provided.'):
            radial_average(
                np.zeros(x)
            )
        q, m = radial_average(np.zeros((x, y)))
        self.assertEqual(m.shape[0], x // 2 + 1, msg='Radial average shape should equal largest sampling dimension.')
        q, m = radial_average(np.zeros((30, y)))
        self.assertEqual(m.shape[0], y, msg='Radial average shape should equal largest sampling dimension, '
                                            'considering Fourier reduced form.')
        q, m = radial_average(np.zeros((20, x, y)))
        self.assertEqual(m.shape[0], x // 2 + 1, msg='Radial average shape should equal largest sampling dimension.')
        q, m = radial_average(np.zeros((20, 30, y)))
        self.assertEqual(m.shape[0], y, msg='Radial average shape should equal largest sampling dimension, '
                                            'considering Fourier reduced form.')

    def test_power_spectrum_profile(self):
        with self.assertRaises(ValueError, msg='Power spectrum profile should raise ValueError if input image is '
                                               'not 2- or 3-dimensional.'):
            power_spectrum_profile(np.zeros(5))
        with self.assertRaises(ValueError, msg='Power spectrum profile should raise ValueError if input image is '
                                               'not 2- or 3-dimensional.'):

            power_spectrum_profile(np.zeros((5, ) * 4))
        profile = power_spectrum_profile(np.zeros(self.volume_shape_irregular))
        self.assertEqual(profile.shape, (max(self.volume_shape_irregular) // 2 + 1, ),
                         msg='Power spectrum profile output shape should be a 1-dimensional array with '
                             'length equal to max(input_shape) // 2 + 1, corresponding to largest sampling component '
                             'in Fourier space.')

    def test_profile_to_weighting(self):
        with self.assertRaises(ValueError, msg='Profile to weighting should raise a ValueError if the profile is not '
                                               '1-dimensional.'):
            profile_to_weighting(np.zeros((5, 5)), (5, 5))
        with self.assertRaises(ValueError, msg='Profile to weighting should raise a ValueError if the output shape '
                                               'for the weighting is not 2- or 3-dimensional.'):
            profile_to_weighting(np.zeros(5), (5, ))
        with self.assertRaises(ValueError, msg='Profile to weighting should raise a ValueError if the output shape '
                                               'for the weighting is not 2- or 3-dimensional.'):
            profile_to_weighting(np.zeros(5), (5,) * 4)

        profile = power_spectrum_profile(np.zeros(self.volume_shape_irregular))
        self.assertEqual(profile_to_weighting(profile, self.volume_shape_irregular).shape,
                         self.reduced_irregular_shape_3d,
                         msg='Profile to weighting should return 3D Fourier reduced array.')
        self.assertEqual(profile_to_weighting(profile, self.volume_shape_irregular[:2]).shape,
                         self.reduced_irregular_shape_2d,
                         msg='Profile to weighting should return 2D Fourier reduced array.')
