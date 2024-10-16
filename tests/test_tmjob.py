import unittest
import pathlib
import numpy as np
import voltools as vt
import mrcfile
from tempfile import TemporaryDirectory
from pytom_tm.mask import spherical_mask
from pytom_tm.angles import angle_to_angle_list
from pytom_tm.tmjob import TMJob, TMJobError, load_json_to_tmjob, get_defocus_offsets
from pytom_tm.io import read_mrc, write_mrc, UnequalSpacingError
from pytom_tm.extract import extract_particles
from testing_utils import CTF_PARAMS, ACCUMULATED_DOSE, TILT_ANGLES


TOMO_SHAPE = (100, 107, 59)
TEMPLATE_SIZE = 13
LOCATION = (77, 26, 40)
ANGLE_ID = 100
ANGULAR_SEARCH = "38.53"
TEMP_DIR = TemporaryDirectory()
TEST_DATA_DIR = pathlib.Path(TEMP_DIR.name)
TEST_TOMOGRAM = TEST_DATA_DIR.joinpath("tomogram.mrc")
TEST_BROKEN_TOMOGRAM_MASK = TEST_DATA_DIR.joinpath("broken_tomogram_mask.mrc")
TEST_WRONG_SIZE_TOMO_MASK = TEST_DATA_DIR.joinpath("wrong_size_tomogram_mask.mrc")
TEST_EXTRACTION_MASK_OUTSIDE = TEST_DATA_DIR.joinpath("extraction_mask_outside.mrc")
TEST_EXTRACTION_MASK_INSIDE = TEST_DATA_DIR.joinpath("extraction_mask_inside.mrc")
TEST_TEMPLATE = TEST_DATA_DIR.joinpath("template.mrc")
TEST_TEMPLATE_UNEQUAL_SPACING = TEST_DATA_DIR.joinpath("template_unequal_spacing.mrc")
TEST_TEMPLATE_WRONG_VOXEL_SIZE = TEST_DATA_DIR.joinpath("template_voxel_error_test.mrc")
TEST_MASK = TEST_DATA_DIR.joinpath("mask.mrc")
TEST_SCORES = TEST_DATA_DIR.joinpath("tomogram_scores.mrc")
TEST_ANGLES = TEST_DATA_DIR.joinpath("tomogram_angles.mrc")
TEST_CUSTOM_ANGULAR_SEARCH = TEST_DATA_DIR.joinpath("custom_angular_search.txt")
TEST_WHITENING_FILTER = TEST_DATA_DIR.joinpath("tomogram_whitening_filter.npy")
TEST_JOB_JSON = TEST_DATA_DIR.joinpath("tomogram_job.json")
TEST_JOB_JSON_WHITENING = TEST_DATA_DIR.joinpath("tomogram_job_whitening.json")
TEST_JOB_OLD_VERSION = TEST_DATA_DIR.joinpath("tomogram_job_old_version.json")


class TestTMJob(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Create template, mask and tomogram
        volume = np.zeros(TOMO_SHAPE, dtype=np.float32)
        template = np.zeros((TEMPLATE_SIZE,) * 3, dtype=np.float32)
        template[3:8, 4:8, 3:7] = 1.0
        template[7, 8, 5:7] = 1.0
        mask = spherical_mask(TEMPLATE_SIZE, 5, 0.5)
        rotation = angle_to_angle_list(float(ANGULAR_SEARCH))[ANGLE_ID]

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

        # extraction mask
        extraction_mask_outside = np.zeros(TOMO_SHAPE, dtype=np.float32)
        extraction_mask_outside[20:40, 60:80, 10:30] = 1
        extraction_mask_inside = np.zeros(TOMO_SHAPE, dtype=np.float32)
        extraction_mask_inside[70:90, 15:35, 30:50] = 1

        TEST_DATA_DIR.mkdir(exist_ok=True)
        write_mrc(TEST_EXTRACTION_MASK_OUTSIDE, extraction_mask_outside, 1.0)
        write_mrc(TEST_EXTRACTION_MASK_INSIDE, extraction_mask_inside, 1.0)
        write_mrc(TEST_MASK, mask, 1.0)
        write_mrc(TEST_TEMPLATE, template, 1.0)
        write_mrc(TEST_TEMPLATE_WRONG_VOXEL_SIZE, template, 1.5)
        mrcfile.write(
            TEST_TEMPLATE_UNEQUAL_SPACING,
            template,
            voxel_size=(1.5, 1.0, 2.0),
            overwrite=True,
        )
        write_mrc(TEST_TOMOGRAM, volume, 1.0)

        # do a run without splitting to compare against
        job = TMJob(
            "0",
            10,
            TEST_TOMOGRAM,
            TEST_TEMPLATE,
            TEST_MASK,
            TEST_DATA_DIR,
            angle_increment=ANGULAR_SEARCH,
            voxel_size=1.0,
        )
        score, angle = job.start_job(0, return_volumes=True)
        write_mrc(TEST_SCORES, score, job.voxel_size)
        write_mrc(TEST_ANGLES, angle, job.voxel_size)
        job.write_to_json(TEST_JOB_JSON)

        np.savetxt(TEST_CUSTOM_ANGULAR_SEARCH, np.random.rand(100, 3))

        # create job with spectrum whitening
        job = TMJob(
            "0",
            10,
            TEST_TOMOGRAM,
            TEST_TEMPLATE,
            TEST_MASK,
            TEST_DATA_DIR,
            angle_increment=90.00,
            voxel_size=1.0,
            whiten_spectrum=True,
        )
        job.write_to_json(TEST_JOB_JSON_WHITENING)

        # write broken tomogram mask
        broken_tomogram_mask = np.zeros(TOMO_SHAPE, dtype=np.float32)
        write_mrc(TEST_BROKEN_TOMOGRAM_MASK, broken_tomogram_mask, 1.0)

        # write wrong size tomogram mask
        size = list(TOMO_SHAPE)
        size[0] += 1
        wrong_size_tomogram_mask = np.ones(tuple(size), dtype=np.float32)
        write_mrc(TEST_WRONG_SIZE_TOMO_MASK, wrong_size_tomogram_mask, 1.0)

    @classmethod
    def tearDownClass(cls) -> None:
        TEMP_DIR.cleanup()

    def setUp(self):
        self.job = TMJob(
            "0",
            10,
            TEST_TOMOGRAM,
            TEST_TEMPLATE,
            TEST_MASK,
            TEST_DATA_DIR,
            angle_increment=ANGULAR_SEARCH,
            voxel_size=1.0,
        )

    def test_tm_job_errors(self):
        with self.assertRaises(
            ValueError,
            msg="Different voxel size in tomogram and template and no voxel size "
            "provided should raise a ValueError",
        ):
            TMJob(
                "0",
                10,
                TEST_TOMOGRAM,
                TEST_TEMPLATE_WRONG_VOXEL_SIZE,
                TEST_MASK,
                TEST_DATA_DIR,
            )

        with self.assertRaises(
            UnequalSpacingError, msg="Unequal spacing should raise specific Error"
        ):
            TMJob(
                "0",
                10,
                TEST_TOMOGRAM,
                TEST_TEMPLATE_UNEQUAL_SPACING,
                TEST_MASK,
                TEST_DATA_DIR,
            )

        # test searches raise correct errors
        for param in ["search_x", "search_y", "search_z"]:
            with self.assertRaises(
                ValueError, msg="Invalid start index in search should raise ValueError"
            ):
                TMJob(
                    "0",
                    10,
                    TEST_TOMOGRAM,
                    TEST_TEMPLATE,
                    TEST_MASK,
                    TEST_DATA_DIR,
                    voxel_size=1.0,
                    **{param: [-10, 100]},
                )
            with self.assertRaises(
                ValueError, msg="Invalid start index in search should raise ValueError"
            ):
                TMJob(
                    "0",
                    10,
                    TEST_TOMOGRAM,
                    TEST_TEMPLATE,
                    TEST_MASK,
                    TEST_DATA_DIR,
                    voxel_size=1.0,
                    **{param: [110, 130]},
                )
            with self.assertRaises(
                ValueError, msg="Invalid end index in search should raise ValueError"
            ):
                TMJob(
                    "0",
                    10,
                    TEST_TOMOGRAM,
                    TEST_TEMPLATE,
                    TEST_MASK,
                    TEST_DATA_DIR,
                    voxel_size=1.0,
                    **{param: [0, 120]},
                )
        # Test broken angle input
        with self.assertRaisesRegex(TMJobError, "Invalid angular search"):
            TMJob(
                "0",
                10,
                TEST_TOMOGRAM,
                TEST_TEMPLATE,
                TEST_MASK,
                TEST_DATA_DIR,
                angle_increment="1.2.3",
                voxel_size=1.0,
            )

        # Test broken template mask
        with self.assertRaisesRegex(ValueError, str(TEST_BROKEN_TOMOGRAM_MASK)):
            TMJob(
                "0",
                10,
                TEST_TOMOGRAM,
                TEST_TEMPLATE,
                TEST_MASK,
                TEST_DATA_DIR,
                angle_increment=ANGULAR_SEARCH,
                voxel_size=1.0,
                tomogram_mask=TEST_BROKEN_TOMOGRAM_MASK,
            )
        # Test wrong size template mask
        with self.assertRaisesRegex(ValueError, str(TOMO_SHAPE)):
            TMJob(
                "0",
                10,
                TEST_TOMOGRAM,
                TEST_TEMPLATE,
                TEST_MASK,
                TEST_DATA_DIR,
                angle_increment=ANGULAR_SEARCH,
                voxel_size=1.0,
                tomogram_mask=TEST_WRONG_SIZE_TOMO_MASK,
            )

    def test_tm_job_copy(self):
        copy = self.job.copy()
        self.assertIsNot(
            self.job, copy, msg="Copying the job should create a new object."
        )
        self.assertEqual(
            TOMO_SHAPE,
            copy.tomo_shape,
            msg="Tomogram shape not correct in job, perhaps transpose issue?",
        )

    def test_tm_job_weighting_options(self):
        # run with all options
        job = TMJob(
            "0",
            10,
            TEST_TOMOGRAM,
            TEST_TEMPLATE,
            TEST_MASK,
            TEST_DATA_DIR,
            angle_increment=90.00,
            voxel_size=1.0,
            low_pass=10,
            high_pass=100,
            dose_accumulation=ACCUMULATED_DOSE,
            ctf_data=CTF_PARAMS,
            tilt_angles=TILT_ANGLES,
            whiten_spectrum=True,
            tilt_weighting=True,
            defocus_handedness=1,
        )
        score, angle = job.start_job(0, return_volumes=True)
        self.assertEqual(
            score.shape, job.tomo_shape, msg="TMJob with all options failed"
        )

        # run with only tilt weighting
        # (in test_weights different options are tested for create_wedge)
        job = TMJob(
            "0",
            10,
            TEST_TOMOGRAM,
            TEST_TEMPLATE,
            TEST_MASK,
            TEST_DATA_DIR,
            angle_increment=90.00,
            voxel_size=1.0,
            dose_accumulation=ACCUMULATED_DOSE,
            ctf_data=CTF_PARAMS,
            tilt_angles=TILT_ANGLES,
            tilt_weighting=True,
        )
        score, angle = job.start_job(0, return_volumes=True)
        self.assertEqual(
            score.shape, job.tomo_shape, msg="TMJob with only wedge creation failed"
        )

        # run with only bandpass (in test_weights bandpass option are tested)
        job = TMJob(
            "0",
            10,
            TEST_TOMOGRAM,
            TEST_TEMPLATE,
            TEST_MASK,
            TEST_DATA_DIR,
            angle_increment=90.00,
            voxel_size=1.0,
            low_pass=10,
            high_pass=100,
        )
        score, angle = job.start_job(0, return_volumes=True)
        self.assertEqual(
            score.shape, job.tomo_shape, msg="TMJob with only band-pass failed"
        )

        # run with only spectrum whitening
        # (in test_weights the whitening filter is tested)
        job = TMJob(
            "0",
            10,
            TEST_TOMOGRAM,
            TEST_TEMPLATE,
            TEST_MASK,
            TEST_DATA_DIR,
            angle_increment=90.00,
            voxel_size=1.0,
            whiten_spectrum=True,
        )
        score, angle = job.start_job(0, return_volumes=True)
        self.assertEqual(
            score.shape, job.tomo_shape, msg="TMJob with only whitening filter failed"
        )

        # load the whitening filter from previous job to compare against
        whitening_filter = np.load(TEST_WHITENING_FILTER)
        job = TMJob(
            "0",
            10,
            TEST_TOMOGRAM,
            TEST_TEMPLATE,
            TEST_MASK,
            TEST_DATA_DIR,
            angle_increment=90.00,
            voxel_size=1.0,
            whiten_spectrum=True,
            search_y=[10, 90],
        )
        new_whitening_filter = np.load(TEST_WHITENING_FILTER)
        self.assertNotEqual(
            whitening_filter.shape,
            new_whitening_filter.shape,
            msg="After reducing the search region along the largest dimension the "
            "whitening filter should have less sampling points",
        )
        self.assertEqual(
            new_whitening_filter.shape,
            (max(job.search_size) // 2 + 1,),
            msg="The whitening filter does not have the expected size, it should be "
            "equal (x // 2) + 1, where x is the largest dimension of the search box.",
        )

        # TMJob with none of these weighting options is tested in all other runs
        # in this file.

    def test_load_json_to_tmjob(self):
        # check base job loading
        job = load_json_to_tmjob(TEST_JOB_JSON)
        self.assertIsInstance(
            job, TMJob, msg="TMJob could not be properly loaded from disk."
        )

        # check job loading and preventing whitening filter recalculation
        with self.assertNoLogs(level="INFO"):
            _ = load_json_to_tmjob(TEST_JOB_JSON_WHITENING, load_for_extraction=True)
        with self.assertLogs(level="INFO") as cm:
            _ = load_json_to_tmjob(TEST_JOB_JSON_WHITENING, load_for_extraction=False)
        self.assertIn("Estimating whitening filter...", "".join(cm.output))

        # turn current job into 0.6.0 job with ctf params
        job.pytom_tm_version_number = "0.6.0"
        job.ctf_data = []
        for ctf in CTF_PARAMS:
            job.ctf_data.append(ctf.copy())
            del job.ctf_data[-1]["phase_shift_deg"]
        job.write_to_json(TEST_JOB_OLD_VERSION)

        # test backward compatibility with the update to 0.6.1
        job = load_json_to_tmjob(TEST_JOB_OLD_VERSION)
        self.assertEqual(job.ctf_data[0]["phase_shift_deg"], 0.0)

    def test_custom_angular_search(self):
        with TemporaryDirectory() as data_dir:
            data_dir = pathlib.Path(data_dir)
            job = TMJob(
                "0",
                10,
                TEST_TOMOGRAM,
                TEST_TEMPLATE,
                TEST_MASK,
                data_dir,
                angle_increment=TEST_CUSTOM_ANGULAR_SEARCH,
                voxel_size=1.0,
            )
            self.assertEqual(
                job.rotation_file,
                TEST_CUSTOM_ANGULAR_SEARCH,
                msg="Passing a custom angular search file to TMJob failed.",
            )

            # Also test extraction works with custom angle file

            scores, angles = job.start_job(0, return_volumes=True)
            write_mrc(data_dir / "tomogram_scores.mrc", scores, job.voxel_size)
            write_mrc(data_dir / "tomogram_angles.mrc", angles, job.voxel_size)
            df, scores = extract_particles(job, 5, 100, create_plot=False)
            self.assertNotEqual(
                len(scores), 0, msg="Here we expect to get some annotations."
            )

    def test_tm_job_split_volume(self):
        # Splitting the volume into smaller boxes than the template
        # should not raise an error
        _ = self.job.split_volume_search((10, 3, 2))
        # Reset
        self.job.sub_jobs = []
        # Make sure that asking for more splits than pixels results in
        # just a pixel number of jobs
        with self.assertWarnsRegex(RuntimeWarning, "More splits than pixels"):
            self.job.split_volume_search((TOMO_SHAPE[0] + 42, 1, 1))
        self.assertEqual(len(self.job.sub_jobs), TOMO_SHAPE[0])
        # Reset
        self.job.sub_jobs = []
        # Negative splits should fail
        with self.assertRaisesRegex(RuntimeError, "splits=-42"):
            self.job.split_volume_search((-42, 1, 1))
        sub_jobs = self.job.split_volume_search((2, 3, 2))
        stats = []
        for x in sub_jobs:
            stats.append(x.start_job(0))
            job_scores = TEST_DATA_DIR.joinpath(f"tomogram_scores_{x.job_key}.mrc")
            job_angles = TEST_DATA_DIR.joinpath(f"tomogram_angles_{x.job_key}.mrc")
            self.assertTrue(
                job_scores.exists(), msg="Expected output from job does not exist."
            )
            self.assertTrue(
                job_angles.exists(), msg="Expected output from job does not exist."
            )
        score, angle = self.job.merge_sub_jobs(stats)
        ind = np.unravel_index(score.argmax(), score.shape)

        self.assertTrue(score.max() > 0.931, msg="lcc max value lower than expected")
        self.assertEqual(ANGLE_ID, angle[ind])
        self.assertSequenceEqual(LOCATION, ind)

        # Small difference in the edge regions of the split dimension. This is because
        # the cross correlation function is not well defined in the boundary area, only
        # a small part of the template is correlated here (and we are not really
        # interested in it). Probably the inaccuracy in this area becomes more apparent
        # when splitting into subvolumes due to a smaller number of sampling points in
        # Fourier space.
        ok_region = slice(TEMPLATE_SIZE // 2, -TEMPLATE_SIZE // 2)
        score_diff = np.abs(
            score[ok_region, ok_region, ok_region]
            - read_mrc(TEST_SCORES)[ok_region, ok_region, ok_region]
        ).sum()

        self.assertAlmostEqual(
            score_diff, 0, places=1, msg="score diff should not be larger than 0.01"
        )
        # There is some race condition that sometimes gives
        # a different angle for 1 specific point.
        # This point is deterministic but different per machine
        # We suspect it has something to do with the FFT padding
        # See https://github.com/SBC-Utrecht/pytom-match-pick/pull/163

        # angle_diff = np.abs(
        #    angle[ok_region, ok_region, ok_region] -
        #    read_mrc(TEST_ANGLES)[ok_region, ok_region, ok_region]
        #    ).sum()

        # self.assertAlmostEqual(angle_diff, 0, places=1,
        #    msg='angle diff should not change')

        # get search statistics before and after splitting
        split_stats = self.job.job_stats
        reference_stats = load_json_to_tmjob(TEST_JOB_JSON).job_stats
        self.assertEqual(
            split_stats["search_space"],
            reference_stats["search_space"],
            msg="Search space should remain identical upon subvolume splitting.",
        )
        self.assertAlmostEqual(
            split_stats["std"],
            reference_stats["std"],
            places=3,
            msg="Standard deviation over template matching with subvolume splitting "
            "should be almost identical.",
        )

    def test_splitting_with_tomogram_mask(self):
        job = self.job.copy()
        job.tomogram_mask = TEST_EXTRACTION_MASK_INSIDE
        job.split_volume_search((10, 10, 10))
        self.assertLess(len(job.sub_jobs), 10 * 10 * 10)

    def test_splitting_with_offsets(self):
        # check if subjobs have correct offsets for the main job,
        # the last sub job will have the largest errors
        job = TMJob(
            "0",
            10,
            TEST_TOMOGRAM,
            TEST_TEMPLATE,
            TEST_MASK,
            TEST_DATA_DIR,
            angle_increment=ANGULAR_SEARCH,
            voxel_size=1.0,
            search_x=[9, 90],
            search_y=[25, 102],
            search_z=[19, 54],
        )
        # split along each dimension and get only the last sub job
        last_sub_job = job.split_volume_search((2, 3, 2))[-1]
        # Make sure the start of the data + size of the last subjob
        # result in the correct size
        final_size = [
            i + j for i, j in zip(last_sub_job.whole_start, last_sub_job.sub_step)
        ]
        self.assertEqual(
            final_size,
            job.search_size,
            msg="the last subjobs (unique) start position plus its size should equal "
            "the search size of the main job",
        )

    def test_tm_job_split_angles(self):
        sub_jobs = self.job.split_rotation_search(3)
        stats = []
        for x in sub_jobs:
            stats.append(x.start_job(0))
            job_scores = TEST_DATA_DIR.joinpath(f"tomogram_scores_{x.job_key}.mrc")
            job_angles = TEST_DATA_DIR.joinpath(f"tomogram_angles_{x.job_key}.mrc")
            self.assertTrue(
                job_scores.exists(), msg="Expected output from job does not exist."
            )
            self.assertTrue(
                job_angles.exists(), msg="Expected output from job does not exist."
            )
        score, angle = self.job.merge_sub_jobs(stats)
        ind = np.unravel_index(score.argmax(), score.shape)

        self.assertTrue(score.max() > 0.931, msg="lcc max value lower than expected")
        self.assertEqual(ANGLE_ID, angle[ind])
        self.assertSequenceEqual(LOCATION, ind)

        self.assertTrue(
            np.abs(score - read_mrc(TEST_SCORES)).sum() == 0,
            msg="split rotation search should be identical",
        )
        self.assertTrue(
            np.abs(angle - read_mrc(TEST_ANGLES)).sum() == 0,
            msg="split rotation search should be identical",
        )

        # get search statistics before and after splitting
        split_stats = self.job.job_stats
        reference_stats = load_json_to_tmjob(TEST_JOB_JSON).job_stats
        self.assertEqual(
            split_stats["search_space"],
            reference_stats["search_space"],
            msg="Search space should remain identical upon angular search splitting.",
        )
        self.assertAlmostEqual(
            split_stats["std"],
            reference_stats["std"],
            places=6,
            msg="Standard deviation of template matching with angular search split "
            "should be almost identical.",
        )

    def test_tm_job_half_precision(self):
        job = TMJob(
            "0",
            10,
            TEST_TOMOGRAM,
            TEST_TEMPLATE,
            TEST_MASK,
            TEST_DATA_DIR,
            angle_increment=ANGULAR_SEARCH,
            voxel_size=1.0,
            output_dtype=np.float16,
        )
        s, a = job.start_job(0, return_volumes=True)
        self.assertEqual(s.dtype, np.float16)
        self.assertEqual(a.dtype, np.float32)

    def test_extractions(self):
        self.job.tomo_id = "rec_" + self.job.tomo_id
        scores, angles = self.job.start_job(0, return_volumes=True)
        # set the appropriate headers when writing!
        write_mrc(
            TEST_DATA_DIR.joinpath(f"{self.job.tomo_id}_scores.mrc"),
            scores,
            self.job.voxel_size,
        )
        write_mrc(
            TEST_DATA_DIR.joinpath(f"{self.job.tomo_id}_angles.mrc"),
            angles,
            self.job.voxel_size,
        )

        # extract particles after running the job
        df, scores = extract_particles(self.job, 5, 100, create_plot=False)
        self.assertNotEqual(
            len(scores), 0, msg="Here we expect to get some annotations."
        )

        # extract particles in relion5 style
        df_rel5, scores = extract_particles(
            self.job, 5, 100, create_plot=False, relion5_compat=True
        )
        for column in (
            "rlnCenteredCoordinateXAngst",
            "rlnCenteredCoordinateYAngst",
            "rlnCenteredCoordinateZAngst",
            "rlnTomoName",
            "rlnTomoTiltSeriesPixelSize",
        ):
            self.assertTrue(
                column in df_rel5.columns,
                msg=f"Expected {column} in relion5 dataframe.",
            )
        centered_location = (
            np.array(LOCATION) - (np.array(TOMO_SHAPE) / 2 - 1)
        ) * self.job.voxel_size
        diff = np.abs(np.array(df_rel5.iloc[0, 0:3]) - centered_location).sum()
        self.assertEqual(
            diff,
            0,
            msg="relion5 compat mode should return a centered "
            "location of the object",
        )
        self.assertNotIn("rec_", df_rel5["rlnTomoName"][0])

        # test extraction mask that does not cover the particle
        df, scores = extract_particles(
            self.job,
            5,
            100,
            tomogram_mask_path=TEST_EXTRACTION_MASK_OUTSIDE,
            create_plot=False,
        )
        self.assertEqual(
            len(scores),
            0,
            msg="Length of returned list should be 0 after applying mask where the "
            "object is not in the region of interest.",
        )
        # test if the extraction mask can be grabbed from the job instead
        job = self.job.copy()
        job.tomogram_mask = TEST_EXTRACTION_MASK_OUTSIDE
        df, scores = extract_particles(
            job,
            5,
            100,
            create_plot=False,
        )
        self.assertEqual(
            len(scores),
            0,
            msg="Length of returned list should be 0 after applying mask where the "
            "object is not in the region of interest.",
        )
        # test if all masks are ignored if ignore_tomogram_mask=True
        # and that a warning is raised
        with self.assertLogs(level="WARNING") as cm:
            df, scores = extract_particles(
                job,
                5,
                100,
                tomogram_mask_path=TEST_EXTRACTION_MASK_OUTSIDE,
                create_plot=False,
                ignore_tomogram_mask=True,
            )
        # Test if expected warning is logged
        for o in cm.output:
            if "Ignoring tomogram mask" in o:
                break
        else:
            # break is not hit
            self.fail("expected warning is not logged")  # pragma: no cover
        self.assertNotEqual(
            len(scores),
            0,
            msg="We would expect some annotations if all tomogram masks are ignored",
        )

        # test mask that covers the particle
        # and should override the one now attached to the job
        df, scores = extract_particles(
            job,
            5,
            100,
            tomogram_mask_path=TEST_EXTRACTION_MASK_INSIDE,
            create_plot=False,
        )
        self.assertNotEqual(
            len(scores),
            0,
            msg="We expected a detected particle with a extraction mask that "
            "covers the object.",
        )

        # test mask that is the wrong size raises an error
        with self.assertRaisesRegex(ValueError, str(TOMO_SHAPE)):
            _, _ = extract_particles(
                job,
                5,
                100,
                tomogram_mask_path=TEST_WRONG_SIZE_TOMO_MASK,
                create_plot=False,
            )

        # Also test the raise if it somehow got attached to the job
        job = self.job.copy()
        job.tomogram_mask = TEST_WRONG_SIZE_TOMO_MASK
        with self.assertRaisesRegex(ValueError, str(TOMO_SHAPE)):
            _, _ = extract_particles(
                job,
                5,
                100,
                create_plot=False,
            )

        # Test exraction with tophat filter and plotting
        df, scores = extract_particles(
            job,
            5,
            100,
            tomogram_mask_path=TEST_EXTRACTION_MASK_INSIDE,
            create_plot=True,
            tophat_filter=True,
        )
        self.assertNotEqual(
            len(scores),
            0,
            msg="We expected a detected particle with a extraction mask that "
            "covers the object.",
        )
        # We don't look for the plots, they might be skipped if no plotting is available

    def test_get_defocus_offsets(self):
        tilt_angles = list(range(-51, 54, 3))
        x_offset_um = 200 * 13.79 * 1e-4
        z_offset_um = 100 * 13.79 * 1e-4
        defocus_offsets = get_defocus_offsets(x_offset_um, z_offset_um, tilt_angles)
        self.assertEqual(
            len(defocus_offsets),
            len(tilt_angles),
            msg="get_defocus_offsets did not return a list with the same length as "
            "the number of tilt_angles",
        )
        defocus_offsets_inverted = get_defocus_offsets(
            x_offset_um, z_offset_um, tilt_angles, invert_handedness=True
        )
        # only the offset at the 0 degrees tilt is expected to be identical,
        # so the test checks if exactly one element is the same
        self.assertTrue(
            (defocus_offsets == defocus_offsets_inverted).sum() == 1,
            msg="inverted handedness should have one identical offset",
        )
