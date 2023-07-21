import unittest
import pathlib
import mrcfile
import numpy as np
import voltools as vt
import napari
from importlib_resources import files
from pytom_tm.mask import spherical_mask
from pytom_tm.angles import load_angle_list
from pytom_tm.parallel import TMJob


TOMO_SHAPE = (100, 107, 59)
LOCATION = (77, 26, 40)
ANGLE_ID = 100
ANGULAR_SEARCH = 'angles_38.53_256.txt'
TEST_DATA_DIR = pathlib.Path(__file__).parent.joinpath('test_data')
TEST_TOMOGRAM = TEST_DATA_DIR.joinpath('tomogram.mrc')
TEST_TEMPLATE = TEST_DATA_DIR.joinpath('template.mrc')
TEST_MASK = TEST_DATA_DIR.joinpath('mask.mrc')


class TestTMJob(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        t_size = 13  # template size

        # Create template, mask and tomogram
        volume = np.zeros(TOMO_SHAPE, dtype=np.float32)
        template = np.zeros((t_size,) * 3, dtype=np.float32)
        template[3:8, 4:8, 3:7] = 1.
        mask = spherical_mask(t_size, 5, 0.5).get()
        rotation = load_angle_list(files('pytom_tm.angle_lists').joinpath(ANGULAR_SEARCH))[ANGLE_ID]

        volume[LOCATION[0] - t_size // 2: LOCATION[0] + t_size // 2 + t_size % 2,
               LOCATION[1] - t_size // 2: LOCATION[1] + t_size // 2 + t_size % 2,
               LOCATION[2] - t_size // 2: LOCATION[2] + t_size // 2 + t_size % 2] = vt.transform(
            template,
            rotation=rotation,
            rotation_units='rad',
            rotation_order='rzxz',
            device='cpu'
        )

        TEST_DATA_DIR.mkdir(exist_ok=True)
        mrcfile.write(TEST_MASK, mask.T, overwrite=True, voxel_size=1)
        mrcfile.write(TEST_TEMPLATE, template.T, overwrite=True, voxel_size=1)
        mrcfile.write(TEST_TOMOGRAM, volume.T, overwrite=True, voxel_size=1)

        # do a run without splitting to compare against

    @classmethod
    def tearDownClass(cls) -> None:
        TEST_MASK.unlink()
        TEST_TEMPLATE.unlink()
        TEST_TOMOGRAM.unlink()
        TEST_DATA_DIR.rmdir()

    def setUp(self):
        self.job = TMJob('0', TEST_TOMOGRAM, TEST_TEMPLATE, TEST_MASK, TEST_DATA_DIR, '38.53')

    def test_tm_job_copy(self):
        copy = self.job.copy()
        self.assertIsNot(
            self.job, copy,
            msg='Copying the job should create a new object.'
        )
        self.assertEqual(
            TOMO_SHAPE, copy.tomo_shape,
            msg='Tomogram shape not correct in job, perhaps transpose issue?'
        )

    def test_tm_job_split_volume(self):
        sub_jobs = self.job.split_volume_search((1, 2, 1))
        for x in sub_jobs:
            x.start_job(0)
            job_scores = TEST_DATA_DIR.joinpath(f'scores_{x.job_key}.mrc')
            job_angles = TEST_DATA_DIR.joinpath(f'angles_{x.job_key}.mrc')
            self.assertTrue(
                job_scores.exists(),
                msg='Expected output from job does not exist.'
            )
            self.assertTrue(
                job_angles.exists(),
                msg='Expected output from job does not exist.'
            )
        score, angle = self.job.merge_sub_jobs()
        ind = np.unravel_index(score.argmax(), score.shape)
        self.assertTrue(score.max() > 0.99, msg='lcc max value lower than expected')
        self.assertEqual(ANGLE_ID, angle[ind])
        self.assertSequenceEqual(LOCATION, ind)

        # viewer = napari.Viewer()
        # viewer.add_image(mrcfile.read(TEST_TOMOGRAM))
        # viewer.add_image(score.T)
        # napari.run()

    def test_tm_job_split_angles(self):
        sub_jobs = self.job.split_rotation_search(2)
        for x in sub_jobs:
            x.start_job(0)
            job_scores = TEST_DATA_DIR.joinpath(f'scores_{x.job_key}.mrc')
            job_angles = TEST_DATA_DIR.joinpath(f'angles_{x.job_key}.mrc')
            self.assertTrue(
                job_scores.exists(),
                msg='Expected output from job does not exist.'
            )
            self.assertTrue(
                job_angles.exists(),
                msg='Expected output from job does not exist.'
            )
        score, angle = self.job.merge_sub_jobs()
        ind = np.unravel_index(score.argmax(), score.shape)
        self.assertTrue(score.max() > 0.99, msg='lcc max value lower than expected')
        self.assertEqual(ANGLE_ID, angle[ind])
        self.assertSequenceEqual(LOCATION, ind)


if __name__ == '__main__':
    unittest.main()
