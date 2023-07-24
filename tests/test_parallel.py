import unittest
import pathlib
import mrcfile
import numpy as np
import voltools as vt
from importlib_resources import files
from pytom_tm.mask import spherical_mask
from pytom_tm.angles import load_angle_list
from pytom_tm.parallel import run_job_parallel
from pytom_tm.tmjob import TMJob


TOMO_SHAPE = (100, 107, 59)
TEMPLATE_SIZE = 13
LOCATION = (77, 26, 40)
ANGLE_ID = 100
ANGULAR_SEARCH = 'angles_38.53_256.txt'
TEST_DATA_DIR = pathlib.Path(__file__).parent.joinpath('test_data')
TEST_TOMOGRAM = TEST_DATA_DIR.joinpath('tomogram.mrc')
TEST_TEMPLATE = TEST_DATA_DIR.joinpath('template.mrc')
TEST_MASK = TEST_DATA_DIR.joinpath('mask.mrc')
TEST_SCORES = TEST_DATA_DIR.joinpath('scores.mrc')
TEST_ANGLES = TEST_DATA_DIR.joinpath('angles.mrc')


class TestTMJob(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Create template, mask and tomogram
        volume = np.zeros(TOMO_SHAPE, dtype=np.float32)
        template = np.zeros((TEMPLATE_SIZE,) * 3, dtype=np.float32)
        template[3:8, 4:8, 3:7] = 1.
        mask = spherical_mask(TEMPLATE_SIZE, 5, 0.5).get()
        rotation = load_angle_list(files('pytom_tm.angle_lists').joinpath(ANGULAR_SEARCH))[ANGLE_ID]

        volume[LOCATION[0] - TEMPLATE_SIZE // 2: LOCATION[0] + TEMPLATE_SIZE // 2 + TEMPLATE_SIZE % 2,
               LOCATION[1] - TEMPLATE_SIZE // 2: LOCATION[1] + TEMPLATE_SIZE // 2 + TEMPLATE_SIZE % 2,
               LOCATION[2] - TEMPLATE_SIZE // 2: LOCATION[2] + TEMPLATE_SIZE // 2 + TEMPLATE_SIZE % 2] = vt.transform(
            template,
            rotation=rotation,
            rotation_units='rad',
            rotation_order='rzxz',
            device='cpu'
        )

        # add some noise
        rng = np.random.default_rng(0)
        volume += rng.normal(loc=0, scale=0.1, size=volume.shape)

        TEST_DATA_DIR.mkdir(exist_ok=True)
        mrcfile.write(TEST_MASK, mask.T, overwrite=True, voxel_size=1)
        mrcfile.write(TEST_TEMPLATE, template.T, overwrite=True, voxel_size=1)
        mrcfile.write(TEST_TOMOGRAM, volume.T, overwrite=True, voxel_size=1)

        # do a run without splitting to compare against
        job = TMJob('0', TEST_TOMOGRAM, TEST_TEMPLATE, TEST_MASK, TEST_DATA_DIR, '38.53')
        score, angle = job.start_job(0, return_volumes=True)
        mrcfile.write(
            TEST_SCORES,
            score.T,
            voxel_size=job.voxel_size,
            overwrite=True
        )
        mrcfile.write(
            TEST_ANGLES,
            angle.T,
            voxel_size=job.voxel_size,
            overwrite=True
        )

    @classmethod
    def tearDownClass(cls) -> None:
        TEST_MASK.unlink()
        TEST_TEMPLATE.unlink()
        TEST_TOMOGRAM.unlink()
        TEST_SCORES.unlink()
        TEST_ANGLES.unlink()
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
        sub_jobs = self.job.split_volume_search((1, 3, 1))
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

        self.assertTrue(score.max() > 0.934, msg='lcc max value lower than expected')
        self.assertEqual(ANGLE_ID, angle[ind])
        self.assertSequenceEqual(LOCATION, ind)

        # Small difference in the edge regions of the split dimension. This is because the cross correlation function
        # is not well defined in the boundary area, only a small part of the template is correlated here (and we are
        # not really interested in it). Probably the inaccuracy in this area becomes more apparent when splitting
        # into subvolumes due to a smaller number of sampling points in Fourier space.
        score_diff = np.abs(score[:, TEMPLATE_SIZE // 2: -TEMPLATE_SIZE // 2] -
                     mrcfile.read(TEST_SCORES).T[:, TEMPLATE_SIZE // 2: -TEMPLATE_SIZE // 2]).sum()
        angle_diff = np.abs(angle[:, TEMPLATE_SIZE // 2: -TEMPLATE_SIZE // 2] -
                     mrcfile.read(TEST_ANGLES).T[:, TEMPLATE_SIZE // 2: -TEMPLATE_SIZE // 2]).sum()
        self.assertAlmostEqual(score_diff, 0, places=1, msg='score diff should not be larger than 0.01')
        self.assertTrue(angle_diff == 0, msg='angle diff should not change')

    def test_tm_job_split_angles(self):
        sub_jobs = self.job.split_rotation_search(3)
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

        self.assertTrue(score.max() > 0.934, msg='lcc max value lower than expected')
        self.assertEqual(ANGLE_ID, angle[ind])
        self.assertSequenceEqual(LOCATION, ind)

        self.assertTrue(np.abs(score - mrcfile.read(TEST_SCORES).T).sum() == 0,
                        msg='split rotation search should be identical')
        self.assertTrue(np.abs(angle - mrcfile.read(TEST_ANGLES).T).sum() == 0,
                        msg='split rotation search should be identical')

    def test_parallel_manager(self):
        score, angle = run_job_parallel(self.job, volume_splits=(1, 3, 1), gpu_ids=[0])
        ind = np.unravel_index(score.argmax(), score.shape)

        self.assertTrue(score.max() > 0.934, msg='lcc max value lower than expected')
        self.assertEqual(ANGLE_ID, angle[ind])
        self.assertSequenceEqual(LOCATION, ind)

        # Small difference in the edge regions of the split dimension. This is because the cross correlation function
        # is not well defined in the boundary area, only a small part of the template is correlated here (and we are
        # not really interested in it). Probably the inaccuracy in this area becomes more apparent when splitting
        # into subvolumes due to a smaller number of sampling points in Fourier space.
        score_diff = np.abs(score[:, TEMPLATE_SIZE // 2: -TEMPLATE_SIZE // 2] -
                     mrcfile.read(TEST_SCORES).T[:, TEMPLATE_SIZE // 2: -TEMPLATE_SIZE // 2]).sum()
        angle_diff = np.abs(angle[:, TEMPLATE_SIZE // 2: -TEMPLATE_SIZE // 2] -
                     mrcfile.read(TEST_ANGLES).T[:, TEMPLATE_SIZE // 2: -TEMPLATE_SIZE // 2]).sum()
        self.assertAlmostEqual(score_diff, 0, places=1, msg='score diff should not be larger than 0.01')
        self.assertTrue(angle_diff == 0, msg='angle diff should not change')


if __name__ == '__main__':
    unittest.main()
