import unittest
import voltools as vt
import numpy as np
from pytom_tm.matching import TemplateMatchingGPU
from pytom_tm.mask import spherical_mask
from pytom_tm.angles import angle_to_angle_list


class TestTM(unittest.TestCase):
    def setUp(self):
        self.t_size = 12
        self.volume = np.zeros((100,) * 3, dtype=float)
        self.template = np.zeros((self.t_size,) * 3, dtype=float)
        self.template[3:8, 4:8, 3:7] = 1.0
        self.template[7, 8, 5:7] = 1.0
        self.mask = spherical_mask(self.t_size, 5, 0.5)
        self.gpu_id = "gpu:0"
        self.angles = angle_to_angle_list(38.53)

    def test_search_spherical_mask(self):
        angle_id = 100
        rotation = self.angles[angle_id]
        loc = (77, 26, 40)
        self.volume[
            loc[0] - self.t_size // 2 : loc[0] + self.t_size // 2,
            loc[1] - self.t_size // 2 : loc[1] + self.t_size // 2,
            loc[2] - self.t_size // 2 : loc[2] + self.t_size // 2,
        ] = vt.transform(
            self.template,
            rotation=rotation,
            rotation_units="rad",
            rotation_order="rzxz",
            device="cpu",
        )

        tm = TemplateMatchingGPU(
            0,
            0,
            self.volume,
            self.template,
            self.mask,
            self.angles,
            list(range(len(self.angles))),
        )
        score_volume, angle_volume, stats = tm.run()

        ind = np.unravel_index(score_volume.argmax(), self.volume.shape)
        self.assertTrue(
            score_volume.max() > 0.99, msg="lcc max value lower than expected"
        )
        self.assertEqual(angle_id, angle_volume[ind])
        self.assertSequenceEqual(loc, ind)
        expected_search_space = len(self.angles) * self.volume.size
        self.assertEqual(
            stats["search_space"],
            expected_search_space,
            msg="Search space should exactly equal this value",
        )
        self.assertAlmostEqual(
            stats["std"],
            0.005006,
            places=5,
            msg="Standard deviation of the search should be almost equal",
        )

    def test_search_non_spherical_mask(self):
        angle_id = 100
        rotation = self.angles[angle_id]
        loc = (77, 26, 40)
        self.volume[
            loc[0] - self.t_size // 2 : loc[0] + self.t_size // 2,
            loc[1] - self.t_size // 2 : loc[1] + self.t_size // 2,
            loc[2] - self.t_size // 2 : loc[2] + self.t_size // 2,
        ] = vt.transform(
            self.template,
            rotation=rotation,
            rotation_units="rad",
            rotation_order="rzxz",
            device="cpu",
        )

        tm = TemplateMatchingGPU(
            0,
            0,
            self.volume,
            self.template,
            self.mask,
            self.angles,
            list(range(len(self.angles))),
            mask_is_spherical=False,
        )
        score_volume, angle_volume, stats = tm.run()

        ind = np.unravel_index(score_volume.argmax(), self.volume.shape)
        self.assertTrue(
            score_volume.max() > 0.99, msg="lcc max value lower than expected"
        )
        self.assertEqual(angle_id, angle_volume[ind])
        self.assertSequenceEqual(loc, ind)

        expected_search_space = len(self.angles) * self.volume.size
        self.assertEqual(
            stats["search_space"],
            expected_search_space,
            msg="Search space should exactly equal this value",
        )
        self.assertAlmostEqual(
            stats["std"],
            0.005015,
            places=4,
            msg="Standard deviation of the search should be almost equal",
        )

    def test_search_noise_correction(self):
        angle_id = 100
        rotation = self.angles[angle_id]
        loc = (77, 26, 40)
        self.volume[
            loc[0] - self.t_size // 2 : loc[0] + self.t_size // 2,
            loc[1] - self.t_size // 2 : loc[1] + self.t_size // 2,
            loc[2] - self.t_size // 2 : loc[2] + self.t_size // 2,
        ] = vt.transform(
            self.template,
            rotation=rotation,
            rotation_units="rad",
            rotation_order="rzxz",
            device="cpu",
        )

        tm = TemplateMatchingGPU(
            0,
            0,
            self.volume,
            self.template,
            self.mask,
            self.angles,
            list(range(len(self.angles))),
            noise_correction=True,
        )
        score_volume, angle_volume, stats = tm.run()

        ind = np.unravel_index(score_volume.argmax(), self.volume.shape)
        # Noise correction subtracts the correlation of a phase-randomized version of
        # the template, built with a Gerchberg-Saxton support constraint using
        # self.mask. Because self.template is a small, simple, compact shape and
        # self.mask covers most of the box, GS phase retrieval partially reconstructs
        # the true template instead of randomizing it, so the noise-corrected peak
        # ends up well below the ~1.0 seen without noise correction (test_search_
        # spherical_mask). A real, structurally complex template would not have its
        # phases retrieved this way and would keep a much smaller noise correlation.
        self.assertAlmostEqual(
            score_volume.max(),
            0.49181,
            places=2,
            msg="lcc max value not almost equal to expected",
        )
        self.assertEqual(angle_id, angle_volume[ind])
        self.assertSequenceEqual(loc, ind)
        expected_search_space = len(self.angles) * self.volume.size
        self.assertEqual(
            stats["search_space"],
            expected_search_space,
            msg="Search space should exactly equal this value",
        )
        self.assertAlmostEqual(
            stats["std"],
            0.005006,
            places=4,
            msg="Standard deviation of the search should be almost equal",
        )
