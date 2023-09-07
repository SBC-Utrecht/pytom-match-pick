import unittest
import pathlib
import numpy as np
import voltools as vt
import mrcfile
from importlib_resources import files
from pytom_tm.mask import spherical_mask
from pytom_tm.angles import load_angle_list
from pytom_tm.parallel import run_job_parallel
from pytom_tm.tmjob import TMJob, TMJobError
from pytom_tm.io import read_mrc, write_mrc, UnequalSpacingError
from pytom_tm.extract import extract_particles


TOMO_SHAPE = (100, 107, 59)
TEMPLATE_SIZE = 13
LOCATION = (77, 26, 40)
ANGLE_ID = 100
ANGULAR_SEARCH = 'angles_38.53_256.txt'
TEST_DATA_DIR = pathlib.Path(__file__).parent.joinpath('test_data')
TEST_TOMOGRAM = TEST_DATA_DIR.joinpath('tomogram.mrc')
TEST_EXTRACTION_MASK_OUTSIDE = TEST_DATA_DIR.joinpath('extraction_mask_outside.mrc')
TEST_EXTRACTION_MASK_INSIDE = TEST_DATA_DIR.joinpath('extraction_mask_inside.mrc')
TEST_TEMPLATE = TEST_DATA_DIR.joinpath('template.mrc')
TEST_TEMPLATE_UNEQUAL_SPACING = TEST_DATA_DIR.joinpath('template_unequal_spacing.mrc')
TEST_TEMPLATE_WRONG_VOXEL_SIZE = TEST_DATA_DIR.joinpath('template_voxel_error_test.mrc')
TEST_MASK = TEST_DATA_DIR.joinpath('mask.mrc')
TEST_SCORES = TEST_DATA_DIR.joinpath('tomogram_scores.mrc')
TEST_ANGLES = TEST_DATA_DIR.joinpath('tomogram_angles.mrc')


class TestTMJob(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Create template, mask and tomogram
        volume = np.zeros(TOMO_SHAPE, dtype=np.float32)
        template = np.zeros((TEMPLATE_SIZE,) * 3, dtype=np.float32)
        template[3:8, 4:8, 3:7] = 1.
        template[7, 8, 5:7] = 1.
        mask = spherical_mask(TEMPLATE_SIZE, 5, 0.5)
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

        # extraction mask
        extraction_mask_outside = np.zeros(TOMO_SHAPE, dtype=np.float32)
        extraction_mask_outside[20:40, 60:80, 10:30] = 1
        extraction_mask_inside = np.zeros(TOMO_SHAPE, dtype=np.float32)
        extraction_mask_inside[70:90, 15:35, 30:50] = 1

        TEST_DATA_DIR.mkdir(exist_ok=True)
        write_mrc(TEST_EXTRACTION_MASK_OUTSIDE, extraction_mask_outside, 1.)
        write_mrc(TEST_EXTRACTION_MASK_INSIDE, extraction_mask_inside, 1.)
        write_mrc(TEST_MASK, mask, 1.)
        write_mrc(TEST_TEMPLATE, template, 1.)
        write_mrc(TEST_TEMPLATE_WRONG_VOXEL_SIZE, template, 1.5)
        mrcfile.write(TEST_TEMPLATE_UNEQUAL_SPACING, template, voxel_size=(1.5, 1., 2.), overwrite=True)
        write_mrc(TEST_TOMOGRAM, volume, 1.)

        # do a run without splitting to compare against
        job = TMJob(
            '0',
            10,
            TEST_TOMOGRAM,
            TEST_TEMPLATE,
            TEST_MASK,
            TEST_DATA_DIR,
            angle_increment='38.53',
            voxel_size=1.
        )
        score, angle = job.start_job(0, return_volumes=True)
        write_mrc(
            TEST_SCORES,
            score,
            job.voxel_size
        )
        write_mrc(
            TEST_ANGLES,
            angle,
            job.voxel_size
        )

    @classmethod
    def tearDownClass(cls) -> None:
        TEST_MASK.unlink()
        TEST_EXTRACTION_MASK_OUTSIDE.unlink()
        TEST_EXTRACTION_MASK_INSIDE.unlink()
        TEST_TEMPLATE.unlink()
        TEST_TEMPLATE_UNEQUAL_SPACING.unlink()
        TEST_TEMPLATE_WRONG_VOXEL_SIZE.unlink()
        TEST_TOMOGRAM.unlink()
        TEST_SCORES.unlink()
        TEST_ANGLES.unlink()
        TEST_DATA_DIR.rmdir()

    def setUp(self):
        self.job = TMJob('0', 10, TEST_TOMOGRAM, TEST_TEMPLATE, TEST_MASK, TEST_DATA_DIR,
                         angle_increment='38.53', voxel_size=1.)

    def test_tm_job_errors(self):
        with self.assertRaises(ValueError, msg='Different voxel size in tomogram and template and no voxel size '
                                               'provided should raise a ValueError'):
            TMJob('0', 10, TEST_TOMOGRAM, TEST_TEMPLATE_WRONG_VOXEL_SIZE, TEST_MASK, TEST_DATA_DIR)

        with self.assertRaises(UnequalSpacingError,
                               msg='Unequal spacing should raise specific Error'):
            TMJob('0', 10, TEST_TOMOGRAM, TEST_TEMPLATE_UNEQUAL_SPACING, TEST_MASK, TEST_DATA_DIR)

        # test searches raise correct errors
        for param in ['search_x', 'search_y', 'search_z']:
            with self.assertRaises(ValueError, msg='Invalid start index in search should raise ValueError'):
                TMJob('0', 10, TEST_TOMOGRAM, TEST_TEMPLATE, TEST_MASK, TEST_DATA_DIR, voxel_size=1.,
                      **{param: [-10, 100]})
                TMJob('0', 10, TEST_TOMOGRAM, TEST_TEMPLATE, TEST_MASK, TEST_DATA_DIR, voxel_size=1.,
                      **{param: [110, 130]})
            with self.assertRaises(ValueError, msg='Invalid end index in search should raise ValueError'):
                TMJob('0', 10, TEST_TOMOGRAM, TEST_TEMPLATE, TEST_MASK, TEST_DATA_DIR, voxel_size=1.,
                      **{param: [0, 120]})

        with self.assertRaises(TMJobError, msg='Angular increment for which no default list is available should raise '
                                               'a TMJobError'):
            TMJob('0', 10, TEST_TOMOGRAM, TEST_TEMPLATE, TEST_MASK, TEST_DATA_DIR, voxel_size=1.,
                  angle_increment='5.41')

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
            job_scores = TEST_DATA_DIR.joinpath(f'tomogram_scores_{x.job_key}.mrc')
            job_angles = TEST_DATA_DIR.joinpath(f'tomogram_angles_{x.job_key}.mrc')
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
                     read_mrc(TEST_SCORES)[:, TEMPLATE_SIZE // 2: -TEMPLATE_SIZE // 2]).sum()
        angle_diff = np.abs(angle[:, TEMPLATE_SIZE // 2: -TEMPLATE_SIZE // 2] -
                     read_mrc(TEST_ANGLES)[:, TEMPLATE_SIZE // 2: -TEMPLATE_SIZE // 2]).sum()
        self.assertAlmostEqual(score_diff, 0, places=1, msg='score diff should not be larger than 0.01')
        self.assertAlmostEqual(angle_diff, 0, places=1, msg='angle diff should not change')

    def test_tm_job_split_angles(self):
        sub_jobs = self.job.split_rotation_search(3)
        for x in sub_jobs:
            x.start_job(0)
            job_scores = TEST_DATA_DIR.joinpath(f'tomogram_scores_{x.job_key}.mrc')
            job_angles = TEST_DATA_DIR.joinpath(f'tomogram_angles_{x.job_key}.mrc')
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

        self.assertTrue(np.abs(score - read_mrc(TEST_SCORES)).sum() == 0,
                        msg='split rotation search should be identical')
        self.assertTrue(np.abs(angle - read_mrc(TEST_ANGLES)).sum() == 0,
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
                     read_mrc(TEST_SCORES)[:, TEMPLATE_SIZE // 2: -TEMPLATE_SIZE // 2]).sum()
        angle_diff = np.abs(angle[:, TEMPLATE_SIZE // 2: -TEMPLATE_SIZE // 2] -
                     read_mrc(TEST_ANGLES)[:, TEMPLATE_SIZE // 2: -TEMPLATE_SIZE // 2]).sum()
        self.assertAlmostEqual(score_diff, 0, places=1, msg='score diff should not be larger than 0.01')
        self.assertAlmostEqual(angle_diff, 0, places=1, msg='angle diff should not change')

    def test_extraction(self):
        _ = self.job.start_job(0, return_volumes=True)

        # extract particles after running the job
        df, scores = extract_particles(self.job, 5, 100)
        self.assertNotEqual(len(scores), 0, msg='Here we expect to get some annotations.')

        # test extraction mask that does not cover the particle
        df, scores = extract_particles(self.job, 5, 100,
                                       tomogram_mask_path=TEST_EXTRACTION_MASK_OUTSIDE)
        self.assertEqual(len(scores), 0, msg='Length of returned list should be 0 after applying mask where the '
                                             'object is not in the region of interest.')

        # test mask that covers the particle
        df, scores = extract_particles(self.job, 5, 100,
                                       tomogram_mask_path=TEST_EXTRACTION_MASK_INSIDE)
        self.assertNotEqual(len(scores), 0, msg='We expected a detected particle with a extraction mask that '
                                                'covers the object.')


if __name__ == '__main__':
    unittest.main()
