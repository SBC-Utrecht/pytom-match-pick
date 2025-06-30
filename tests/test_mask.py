import unittest
import voltools as vt
import cupy as cp
from pytom_tm.mask import spherical_mask, ellipsoidal_mask
from pytom_tm.angles import angle_to_angle_list
from pytom_tm.correlation import normalised_cross_correlation


class TestMask(unittest.TestCase):
    def setUp(self):
        self.angles = angle_to_angle_list(50.00)

    def test_rotational_invariance_even(self):
        # TEST EVEN MASK
        nxcc_offcenter, nxcc_centered = [], []

        mask = cp.asarray(spherical_mask(12, 4, 0.5))
        mask_rotated = cp.zeros_like(mask)

        mask_texture = vt.StaticVolume(mask, interpolation="filt_bspline", device="gpu")

        for i in range(len(self.angles)):
            mask_texture.transform(
                rotation=self.angles[i],
                rotation_units="rad",
                rotation_order="rzxz",
                center=tuple([x // 2 for x in mask.shape]),
                output=mask_rotated,
            )

            nxcc_offcenter.append(
                normalised_cross_correlation(mask, mask_rotated).get()
            )

        for i in range(len(self.angles)):
            mask_texture.transform(
                rotation=self.angles[i],
                rotation_units="rad",
                rotation_order="rzxz",
                output=mask_rotated,
                # center=np.divide(np.subtract(mask.shape, 1), 2, dtype=np.float32),
            )

            nxcc_centered.append(normalised_cross_correlation(mask, mask_rotated).get())

        self.assertTrue(
            sum(nxcc_centered) > sum(nxcc_offcenter),
            msg="Center of rotation for mask is incorrect.",
        )
        self.assertTrue(
            sum(nxcc_centered) > 99.27, msg="Precision of mask rotation is too low."
        )

    def test_rotational_invariance_uneven(self):
        # TEST UNEVEN MASK
        nxcc_offcenter, nxcc_centered = [], []

        mask = cp.asarray(spherical_mask(13, 4, 0.5))
        mask_rotated = cp.zeros_like(mask)

        mask_texture = vt.StaticVolume(mask, interpolation="filt_bspline", device="gpu")

        for i in range(len(self.angles)):
            mask_texture.transform(
                rotation=self.angles[i],
                rotation_units="rad",
                rotation_order="rzxz",
                center=tuple([x // 2 for x in mask.shape]),
                output=mask_rotated,
            )

            nxcc_offcenter.append(
                normalised_cross_correlation(mask, mask_rotated).get()
            )

        for i in range(len(self.angles)):
            mask_texture.transform(
                rotation=self.angles[i],
                rotation_units="rad",
                rotation_order="rzxz",
                output=mask_rotated,
                # center=np.divide(np.subtract(mask.shape, 1), 2, dtype=np.float32),
            )

            nxcc_centered.append(normalised_cross_correlation(mask, mask_rotated).get())

        self.assertAlmostEqual(
            sum(nxcc_centered),
            sum(nxcc_offcenter),
            places=4,
            msg="Center of rotation for mask is incorrect.",
        )
        self.assertTrue(
            sum(nxcc_centered) > 99.09, msg="Precision of mask rotation is too low."
        )

    def test_ellipsoidal_mask_errors(self):
        # Make sure we error on negative box_size, major, minor1, or minor2
        default = {
            "box_size": 16,
            "major": 12,
            "minor1": 8,
            "minor2": 6,
            "smooth": 3,
            "cutoff_sd": 2,
        }
        for i in ["box_size", "major", "minor1", "minor2"]:
            inp = default.copy()
            inp[i] *= -1
            with self.assertRaisesRegex(ValueError, "box_size or radii"):
                ellipsoidal_mask(**inp)
        # Make sure we error on negative smooth or cutoff_sd
        for i in ["smooth", "cutoff_sd"]:
            inp = default.copy()
            inp[i] *= -1
            with self.assertRaisesRegex(ValueError, "smooth or sd cutoff"):
                ellipsoidal_mask(**inp)
