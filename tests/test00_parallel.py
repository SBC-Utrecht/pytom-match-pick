"""
This file name explicitly starts with test00_ to ensure its run first during testing.
Other tests will run jobs on the GPU which keeps the main unittest process lingering on
the GPU. When starting the parallel manager test on a GPU in exclusive process mode,
the spawned process from the parallel manager will fail due the occupation from the
other unittests. If the parallel manager is tested first, the spawned process is fully
closed which will allow the remaining test to use the GPU.
"""

import unittest
import time
import numpy as np
import voltools as vt
import multiprocessing
import pathlib
from tempfile import TemporaryDirectory
from pytom_tm.mask import spherical_mask
from pytom_tm.angles import angle_to_angle_list
from pytom_tm.dataclass import TiltSeriesMetaData
from pytom_tm.parallel import run_job_parallel, split_job_efficiently
from pytom_tm.tmjob import TMJob
from pytom_tm.io import write_mrc


TOMO_SHAPE = (100, 107, 59)
TEMPLATE_SIZE = 13
LOCATION = (77, 26, 40)
ANGLE_ID = 100
ANGULAR_SEARCH = 38.53
TEMP_DIR = TemporaryDirectory()
TEST_DATA_DIR = pathlib.Path(TEMP_DIR.name)
TEST_TOMOGRAM = TEST_DATA_DIR.joinpath("tomogram.mrc")
TEST_TEMPLATE = TEST_DATA_DIR.joinpath("template.mrc")
TEST_MASK = TEST_DATA_DIR.joinpath("mask.mrc")


class TestTMJob(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Create template, mask and tomogram
        volume = np.zeros(TOMO_SHAPE, dtype=np.float32)
        template = np.zeros((TEMPLATE_SIZE,) * 3, dtype=np.float32)
        template[3:8, 4:8, 3:7] = 1.0
        template[7, 8, 5:7] = 1.0
        mask = spherical_mask(TEMPLATE_SIZE, 5, 0.5)
        rotation = angle_to_angle_list(ANGULAR_SEARCH)[ANGLE_ID]

        volume[
            LOCATION[0] - TEMPLATE_SIZE // 2 : LOCATION[0]
            + TEMPLATE_SIZE // 2
            + TEMPLATE_SIZE % 2,
            LOCATION[1] - TEMPLATE_SIZE // 2 : LOCATION[1]
            + TEMPLATE_SIZE // 2
            + TEMPLATE_SIZE % 2,
            LOCATION[2] - TEMPLATE_SIZE // 2 : LOCATION[2]
            + TEMPLATE_SIZE // 2
            + TEMPLATE_SIZE % 2,
        ] = vt.transform(
            template,
            rotation=rotation,
            rotation_units="rad",
            rotation_order="rzxz",
            device="cpu",
        )

        # add some noise
        rng = np.random.default_rng(0)
        volume += rng.normal(loc=0, scale=0.1, size=volume.shape)

        TEST_DATA_DIR.mkdir(exist_ok=True)
        write_mrc(TEST_MASK, mask, 1.0)
        write_mrc(TEST_TEMPLATE, template, 1.0)
        write_mrc(TEST_TOMOGRAM, volume, 1.0)

    @classmethod
    def tearDownClass(cls) -> None:
        TEMP_DIR.cleanup()

    def setUp(self):
        metadata = TiltSeriesMetaData(tilt_angles=[-90, 90])
        self.job = TMJob(
            "0",
            10,
            TEST_TOMOGRAM,
            TEST_TEMPLATE,
            TEST_MASK,
            TEST_DATA_DIR,
            ts_metadata=metadata,
            angle_increment=38.53,
            voxel_size=1.0,
        )

    def test_parallel_breaking(self):
        try:
            _ = run_job_parallel(
                self.job, volume_splits=(1, 2, 1), gpu_ids=[0, -1], unittest_mute=True
            )
        except RuntimeError:
            # sleep a second to make sure all children are cleaned
            time.sleep(2)
            self.assertEqual(
                len(multiprocessing.active_children()),
                0,
                msg="a process was still lingering after a parallel job with partially "
                "invalid resources was started",
            )
        else:  # pragma: no cover
            self.fail("This should have given a RuntimeError")

    def test_parallel_manager(self):
        score, angle = run_job_parallel(self.job, volume_splits=(1, 3, 1), gpu_ids=[0])
        ind = np.unravel_index(score.argmax(), score.shape)

        self.assertTrue(score.max() > 0.931, msg="lcc max value lower than expected")
        self.assertEqual(ANGLE_ID, angle[ind])
        self.assertSequenceEqual(LOCATION, ind)

    def test_split_job_efficiently(self):
        # test more volume splits than gpus
        # divisable
        job = self.job.copy()
        jobs = split_job_efficiently(job, (2, 2, 2), 4)
        # jobs should be the same as total volume splits
        self.assertEqual(len(jobs), 8)

        # not divisible
        job = self.job.copy()
        jobs = split_job_efficiently(job, (3, 2, 1), 4)
        # 6 jobs don't nicely divide on 4 GPUS,
        # but 12 is the smallest multiple of 6 that does
        self.assertEqual(len(jobs), 12)

        # Test more gpus than jobs
        # divisible
        job = self.job.copy()
        jobs = split_job_efficiently(job, (2, 2, 1), 8)
        # should be the same as the numbers of gpus (8)
        self.assertEqual(len(jobs), 8)

        # not divisble
        job = self.job.copy()
        jobs = split_job_efficiently(job, (2, 2, 1), 6)
        # 12 is the smallest multple of 4 that can be evenly split by 6 gpus
        self.assertEqual(len(jobs), 12)

        # test imposible error
        job = self.job.copy()
        job.n_rotations = 6
        with self.assertRaisesRegex(ValueError, r"gpus: 16.*jobs: 12"):
            _ = split_job_efficiently(job, (2, 1, 1), 16)
        # make sure it does work with more splits
        jobs = split_job_efficiently(job, (2, 2, 1), 16)
        self.assertEqual(len(jobs), 16)

        # test angle split only
        job = self.job.copy()
        jobs = split_job_efficiently(job, (1, 1, 1), 6)
        self.assertEqual(len(jobs), 6)

        # test volume split only
        job = self.job.copy()
        jobs = split_job_efficiently(job, (2, 2, 1), 4)
        self.assertEqual(len(jobs), 6)

        # test no splits
        job = self.job.copy()
        jobs = split_job_efficiently(job, (1, 1, 1), 1)
        self.assertEqual(len(jobs), 1)
        self.assertIs(job, jobs[0])
