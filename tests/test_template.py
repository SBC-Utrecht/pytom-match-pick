import unittest

import numpy as np
from scipy.ndimage import center_of_mass

from pytom_tm.template import _phase_randomize_template, generate_template_from_map


class TestTemplate(unittest.TestCase):
    def setUp(self):
        self.template = np.zeros((13, 13, 13), dtype=np.float32)
        self.template[2:5, 2:5, 7:9] = -1
        self.template[3:5, 3:5, 7:9] = 2
        self.template_center = (6, 6, 6)

    def test_template_padding(self):
        uneven_box = np.zeros((13, 13, 7))
        new_template = generate_template_from_map(uneven_box, 1, 1)
        self.assertEqual(
            new_template.shape, self.template.shape, msg="Box should be made square"
        )
        new_template = generate_template_from_map(uneven_box, 1, 2, output_box_size=20)
        self.assertEqual(
            new_template.shape,
            (20,) * 3,
            msg="Template was not padded to output box size",
        )

        with self.assertLogs(logger="pytom_tm", level="WARNING") as cm:
            new_template = generate_template_from_map(
                uneven_box, 1, 2, output_box_size=3
            )
        self.assertEqual(len(cm.output), 1)
        self.assertIn("Could not set specified box size", cm.output[0])

    def test_template_centering(self):
        new_template = generate_template_from_map(
            self.template,
            1,
            1,
            center=False,
        )
        square_sum = np.square(new_template - self.template).sum()
        self.assertTrue(
            square_sum < 10,
            msg="Template should not change strongly without recentering.",
        )
        new_template = generate_template_from_map(
            self.template,
            1,
            1,
            center=True,
        )
        square_sum = np.square(new_template - self.template).sum()
        self.assertTrue(square_sum > 10, msg="Template didnt change after shift")
        diff = np.array(center_of_mass(new_template**2)) - np.array(
            self.template_center
        )
        diff = np.abs(diff).sum()
        self.assertTrue(diff < 1, msg="Total shift difference should be small")
        # absolute of template should be identical due to square
        abs_template = generate_template_from_map(
            np.abs(self.template),
            1,
            1,
            center=True,
        )
        diff = np.array(center_of_mass(new_template**2)) - np.array(
            center_of_mass(abs_template**2)
        )
        diff = np.abs(diff).sum()
        self.assertTrue(diff < 1, msg="Absolute should provide exactly same center")

    def test_lowpass_resolution(self):
        # Test too low filter resolution
        with self.assertLogs(logger="pytom_tm", level="WARNING") as cm:
            _ = generate_template_from_map(
                self.template, 1.0, 1.0, filter_to_resolution=1.5
            )
        self.assertEqual(len(cm.output), 1)
        self.assertIn("Filter resolution", cm.output[0])
        self.assertIn("too low", cm.output[0])

        # Test working filter resolution
        with self.assertNoLogs(logger="pytom_tm", level="WARNING"):
            _ = generate_template_from_map(
                self.template, 1.0, 1.0, filter_to_resolution=2.5
            )

    def test_phase_randomize_template(self):
        full_mask = np.ones_like(self.template)
        randomized = _phase_randomize_template(
            self.template,
            full_mask,  # use default seed
        )
        self.assertEqual(self.template.shape, randomized.shape)
        self.assertGreater(
            (self.template != randomized).sum(),
            0,
            msg="After phase randomization the template should "
            "no longer be equal to the input.",
        )

        randomized_seeded = _phase_randomize_template(
            self.template,
            full_mask,
            seed=11,
        )
        diff = np.abs(randomized_seeded - randomized).sum()
        self.assertNotEqual(
            diff, 0, msg="Different seed should return different randomization"
        )

    def test_phase_randomize_template_support_mask(self):
        # mask covers the structure with some margin, but not the full box, so the
        # Gerchberg-Saxton support constraint actually restricts the result
        support_mask = np.zeros_like(self.template)
        support_mask[1:6, 1:6, 6:10] = 1

        randomized = _phase_randomize_template(self.template, support_mask)
        self.assertEqual(self.template.shape, randomized.shape)
        np.testing.assert_array_equal(
            randomized * (1 - support_mask),
            np.zeros_like(randomized),
            err_msg="Result should be exactly zero outside the support mask.",
        )
        self.assertTrue(
            np.any(randomized[support_mask.astype(bool)] != 0),
            msg="Result should not be all zero inside the support mask.",
        )

    def test_phase_randomize_template_empty_mask(self):
        # an all-zero mask forces a zero amplitude spectrum, so the result stays
        # zero throughout; the sign check (0 vs 0) should not error or alter that
        empty_mask = np.zeros_like(self.template)
        randomized = _phase_randomize_template(self.template, empty_mask)
        np.testing.assert_array_equal(randomized, np.zeros_like(randomized))
