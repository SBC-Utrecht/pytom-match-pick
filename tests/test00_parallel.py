"""
This file name explicitly starts with test00_ to ensure its run first during testing.
Other tests will run jobs on the GPU which keeps the main unittest process lingering on the GPU. When starting the
parallel manager test on a GPU in exclusive process mode, the spawned process from the parallel manager will fail due
the occupation from the other unittests. If the parallel manager is tested first, the spawned process is fully closed
which will allow the remaining test to use the GPU.
"""
import unittest
import pathlib
import numpy as np
import voltools as vt
import multiprocessing
from importlib_resources import files
from pytom_tm.mask import spherical_mask
from pytom_tm.angles import load_angle_list
from pytom_tm.parallel import run_job_parallel
from pytom_tm.tmjob import TMJob
from pytom_tm.io import write_mrc


TOMO_SHAPE = (100, 107, 59)
TEMPLATE_SIZE = 13
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

        TEST_DATA_DIR.mkdir(exist_ok=True)
        write_mrc(TEST_MASK, mask, 1.)
        write_mrc(TEST_TEMPLATE, template, 1.)
        write_mrc(TEST_TOMOGRAM, volume, 1.)

    @classmethod
    def tearDownClass(cls) -> None:
        TEST_MASK.unlink()
        TEST_TEMPLATE.unlink()
        TEST_TOMOGRAM.unlink()
        TEST_DATA_DIR.rmdir()

    def setUp(self):
        self.job = TMJob('0', 10, TEST_TOMOGRAM, TEST_TEMPLATE, TEST_MASK, TEST_DATA_DIR,
                         angle_increment='38.53', voxel_size=1.)

    def test_parallel_breaking(self):
        try:
            _ = run_job_parallel(self.job, volume_splits=(1, 2, 1), gpu_ids=[0, -1], unittest_mute=True)
        except RuntimeError:
            self.assertEqual(len(multiprocessing.active_children()), 0,
                             msg='a process was still lingering after a parallel job with partially invalid resources '
                                 'was started')
        else: # pragma: no cover
            self.fail('This should have given a RuntimeError')

    def test_parallel_manager(self):
        score, angle = run_job_parallel(self.job, volume_splits=(1, 3, 1), gpu_ids=[0])
        ind = np.unravel_index(score.argmax(), score.shape)

        self.assertTrue(score.max() > 0.931, msg='lcc max value lower than expected')
        self.assertEqual(ANGLE_ID, angle[ind])
        self.assertSequenceEqual(LOCATION, ind)
