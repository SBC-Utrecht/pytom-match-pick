import unittest
import voltools as vt
import numpy as np
import time
from pytom_tm.structures import TemplateMatchingGPU, Monitor
from importlib_resources import files
from pytom_tm.mask import spherical_mask
from pytom_tm.angles import load_angle_list

# visuals
# import napari


class TestTM(unittest.TestCase):
    def setUp(self):
        self.t_size = 12
        self.volume = np.zeros((100, ) * 3, dtype=float)
        self.template = np.zeros((self.t_size, ) * 3, dtype=float)
        self.template[3:8, 4:8, 3:7] = 1.
        self.mask = spherical_mask(self.t_size, 5, 0.5).get()
        self.gpu_id = 'gpu:0'
        self.angles = load_angle_list(str(files('pytom_tm.angle_lists').joinpath('angles_38.53_256.txt')))

    def test_search(self):
        # Instantiate monitor with a 1-second delay between updates
        # monitor = Monitor(1)
        # time.sleep(1)

        # intrinsic rotation R.from_euler('ZXZ', self.angles[100], degrees=False)
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

        tm_thread = TemplateMatchingGPU(0, 0, self.volume, self.template, self.mask, self.angles)
        tm_thread.run()
        while tm_thread.active:
            time.sleep(0.5)
        score_volume, angle_volume = tm_thread.plan.scores.get(), tm_thread.plan.angles.get()
        del tm_thread

        ind = np.unravel_index(score_volume.argmax(), self.volume.shape)
        self.assertTrue(score_volume.max() > 0.99, msg='lcc max value lower than expected')
        self.assertEqual(angle_id, angle_volume[ind])
        self.assertSequenceEqual(loc, ind)

        # Close monitor
        # monitor.stop()

        # viewer = napari.Viewer()
        # viewer.add_image(score_volume)
        # viewer.add_image(self.volume)
        # napari.run()


if __name__ == '__main__':
    unittest.main()