import unittest
import voltools as vt
import numpy as np
from pytom_tm.matching import TemplateMatchingGPU
from importlib_resources import files
from pytom_tm.mask import spherical_mask
from pytom_tm.angles import load_angle_list


class TestTM(unittest.TestCase):
    def setUp(self):
        self.t_size = 12
        self.volume = np.zeros((100, ) * 3, dtype=float)
        self.template = np.zeros((self.t_size, ) * 3, dtype=float)
        self.template[3:8, 4:8, 3:7] = 1.
        self.mask = spherical_mask(self.t_size, 5, 0.5)
        self.gpu_id = 'gpu:0'
        self.angles = load_angle_list(files('pytom_tm.angle_lists').joinpath('angles_38.53_256.txt'))

    def test_search(self):
        angle_id = 100
        rotation = self.angles[angle_id]
        loc = (77, 26, 40)
        self.volume[loc[0] - self.t_size // 2: loc[0] + self.t_size // 2,
                    loc[1] - self.t_size // 2: loc[1] + self.t_size // 2,
                    loc[2] - self.t_size // 2: loc[2] + self.t_size // 2] = vt.transform(
            self.template,
            rotation=rotation,
            rotation_units='rad',
            rotation_order='rzxz',
            device='cpu'
        )

        tm = TemplateMatchingGPU(0, 0, self.volume, self.template, self.mask, self.angles, list(range(len(
            self.angles))))
        score_volume, angle_volume, stats = tm.run()

        ind = np.unravel_index(score_volume.argmax(), self.volume.shape)
        self.assertTrue(score_volume.max() > 0.99, msg='lcc max value lower than expected')
        self.assertEqual(angle_id, angle_volume[ind])
        self.assertSequenceEqual(loc, ind)
        self.assertEqual(stats['search_space'], 256000000, msg='Search space should exactly equal this value')
        self.assertAlmostEqual(stats['std'], 0.005175, places=5,
                               msg='Standard deviation of the search should be almost equal')
