import contextlib
import pathlib
import unittest
import warnings
from tempfile import TemporaryDirectory

import mrcfile
import numpy as np
from lxml import etree

from pytom_tm.dataclass import CtfData, RelionTiltSeriesMetaData
from pytom_tm.io import (
    MultiColumnAngleFileError,
    parse_relion5_star_data,
    parse_warp_xml_data,
    read_mrc,
    read_mrc_meta_data,
    read_tlt_file,
    write_mrc,
)

FAILING_MRC = pathlib.Path(__file__).parent.joinpath(
    pathlib.Path("Data/human_ribo_mask_32_8_5.mrc")
)
# The below file was made with head -c 1024 human_ribo_mask_32_8_5.mrc > header_only.mrc
CORRUPT_MRC = pathlib.Path(__file__).parent.joinpath(
    pathlib.Path("Data/header_only.mrc")
)
RELION5_TOMOGRAMS_STAR = pathlib.Path(__file__).parent.joinpath(
    "Data/relion5_project_example/Tomograms/job009/tomograms.star"
)
REGULAR_TLT = pathlib.Path(__file__).parent.joinpath(
    pathlib.Path("Data/test_angles.rawtlt")
)
# same file ase regular except an index column is added
MULTI_COLUMN_TLT = pathlib.Path(__file__).parent.joinpath(
    pathlib.Path("Data/test_angles_multi_column.rawtlt")
)
WARP_XML = pathlib.Path(__file__).parent.joinpath(
    pathlib.Path("Data/warptools_xml_example/gs04_ts_003.xml")
)
TEST_DATA = pathlib.Path(__file__).parent.joinpath("test_data")
TEST_TOMOGRAM = TEST_DATA.joinpath("rec_tomo200528_107.mrc")


class TestMultiColumnTilt(unittest.TestCase):
    def test_read_multi_column_tilt(self):
        # test we raise on default
        with self.assertRaises(MultiColumnAngleFileError):
            _ = read_tlt_file(MULTI_COLUMN_TLT)
        # allow reading when an override flag is passed
        angles_multi = read_tlt_file(MULTI_COLUMN_TLT, error_on_multi_column=False)
        angles = read_tlt_file(REGULAR_TLT)
        self.assertEqual(angles, angles_multi)


class TestWarpXMLParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        TEST_DATA.mkdir(parents=True)
        write_mrc(TEST_TOMOGRAM, np.zeros((10, 10, 10), dtype=np.float32), 1)

    @classmethod
    def tearDownClass(cls) -> None:
        TEST_TOMOGRAM.unlink()
        TEST_DATA.rmdir()

    def test_correct_defocus_units(self):
        # prevent issue 325 by testing if all defocus values are sane
        # (between 100 and 0.1 μm)
        _voxel_size, ts_metadata = parse_warp_xml_data(WARP_XML, TEST_TOMOGRAM)
        for ctf in ts_metadata.ctf_data:
            self.assertTrue(10e-6 >= ctf.defocus >= 0.1e-6)

    def test_phase_flip_default_on(self):
        # warp/AreTomo reconstructions are always CTF-corrected, so phase flip
        # correction should default to on for warp metadata
        _voxel_size, ts_metadata = parse_warp_xml_data(WARP_XML, TEST_TOMOGRAM)
        for ctf in ts_metadata.ctf_data:
            self.assertTrue(ctf.flip_phase)

    def test_correct_angle_sign(self):
        _voxel_size, ts_metadata = parse_warp_xml_data(WARP_XML, TEST_TOMOGRAM)
        # grab raw xml data
        tree = etree.parse(WARP_XML)
        tilt_angle_nodes = tree.findall(".//Angles")
        angles = [
            float(j)
            for i in tilt_angle_nodes
            for j in i.text.split("\n")
            if i.text.strip()
        ]
        for a, b in zip(ts_metadata.tilt_angles, angles):
            self.assertEqual(a, -b)

    def test_level_angles_default_to_zero(self):
        # the fixture xml predates LevelAngleX/LevelAngleY, so they should default
        # to 0.0 instead of raising
        _, ts_metadata = parse_warp_xml_data(WARP_XML, TEST_TOMOGRAM)
        self.assertEqual(ts_metadata.level_angle_x, 0.0)
        self.assertEqual(ts_metadata.level_angle_y, 0.0)

    def test_level_angle_sign(self):
        # LevelAngleY is negated the same way tilt angles are, to swap from warp's
        # internal convention to pytom's (see PR #334). LevelAngleX is kept as-is:
        # it composes as a separate rotation about a different axis than the tilt
        # angle, so it does not follow the same sign convention (verified against
        # a real WarpTools reconstruction via warpylib)
        raw_xml = WARP_XML.read_text(encoding="utf-8-sig")
        raw_xml = raw_xml.replace(
            'AreAnglesInverted="False"',
            'AreAnglesInverted="False" LevelAngleX="1.5" LevelAngleY="3.2"',
            1,
        )
        with TemporaryDirectory() as tmp_dir:
            level_angle_xml = pathlib.Path(tmp_dir).joinpath("level_angle.xml")
            level_angle_xml.write_text(raw_xml, encoding="utf-8")
            _, ts_metadata = parse_warp_xml_data(level_angle_xml, TEST_TOMOGRAM)
        self.assertEqual(ts_metadata.level_angle_x, 1.5)
        # Warp tilt angles are inverted on loading to match our convention
        self.assertEqual(ts_metadata.level_angle_y, -3.2)

    def test_defocus_handedness_default(self):
        # the fixture xml has AreAnglesInverted="False", which should give the
        # default WarpTools defocus handedness of -1
        _, ts_metadata = parse_warp_xml_data(WARP_XML, TEST_TOMOGRAM)
        self.assertEqual(ts_metadata.defocus_handedness, -1)

    def test_defocus_handedness_inverted(self):
        # AreAnglesInverted="True" should flip the defocus handedness to 1
        raw_xml = WARP_XML.read_text(encoding="utf-8-sig")
        raw_xml = raw_xml.replace(
            'AreAnglesInverted="False"', 'AreAnglesInverted="True"', 1
        )
        with TemporaryDirectory() as tmp_dir:
            inverted_xml = pathlib.Path(tmp_dir).joinpath("inverted.xml")
            inverted_xml.write_text(raw_xml, encoding="utf-8")
            _, ts_metadata = parse_warp_xml_data(inverted_xml, TEST_TOMOGRAM)
        self.assertEqual(ts_metadata.defocus_handedness, 1)


class TestBrokenMRC(unittest.TestCase):
    def setUp(self):
        # Mute the RuntimeWarnings comming from other code-base inside these tests
        # following this SO answer: https://stackoverflow.com/a/45809502
        stack = contextlib.ExitStack()
        _ = stack.enter_context(warnings.catch_warnings())
        warnings.simplefilter("ignore")
        # The follwing line is better, but only works in python >= 3.11
        # _ = stack.enter_context(warnings.catch_warnings(action="ignore"))

        self.addCleanup(stack.close)

        # prep temporary directory
        tempdir = TemporaryDirectory()
        self.tempdirname = tempdir.name
        self.addCleanup(tempdir.cleanup)

    def test_read_mrc_minor_broken(self):
        # Test if this mrc can be read and if the approriate logs are printed
        with self.assertLogs(logger="pytom_tm", level="WARNING") as cm:
            mrc = read_mrc(FAILING_MRC)
        self.assertIsNotNone(mrc)
        self.assertEqual(len(cm.output), 1)
        self.assertIn(FAILING_MRC.name, cm.output[0])
        self.assertIn("make sure this is correct", cm.output[0])

    def test_read_mrc_too_broken(self):
        # Test if this mrc raises an error as expected
        with self.assertRaises(ValueError) as err:
            _ = read_mrc(CORRUPT_MRC)
        self.assertIn(CORRUPT_MRC.name, str(err.exception))
        self.assertIn("too corrupt", str(err.exception))

    def test_read_mrc_meta_data(self):
        # Test if this mrc can be read and if the approriate logs are printed
        with self.assertLogs(logger="pytom_tm", level="WARNING") as cm:
            mrc = read_mrc_meta_data(FAILING_MRC)
        self.assertIsNotNone(mrc)
        self.assertEqual(len(cm.output), 1)
        self.assertIn(FAILING_MRC.name, cm.output[0])
        self.assertIn("make sure this is correct", cm.output[0])

    def test_half_precision_read_write_cycle(self):
        array = np.random.rand(27).reshape((3, 3, 3)).astype(np.float16)
        fname = pathlib.Path(self.tempdirname) / "test_half.mrc"
        # Make sure no warnings are raised
        with self.assertNoLogs(logger="pytom_tm", level="WARNING"):
            write_mrc(fname, array, 1.0)
        # Make sure the file can be read back
        # make sure mode is as expected for float16
        # https://mrcfile.readthedocs.io/en/stable/source/mrcfile.html#mrcfile.utils.dtype_from_mode
        mrc = mrcfile.open(fname)
        self.assertEqual(mrc.header.mode, 12)
        mrc.close()
        # make sure dtype is expected
        mrc = read_mrc(fname)
        self.assertEqual(mrc.dtype, np.float16)
        # make sure data is identical
        np.testing.assert_equal(mrc, array)

    def test_cast_warning(self):
        # make sure a warning is raised when writing an integer based array
        array = np.random.rand(27).reshape((3, 3, 3)).astype(np.int32)
        fname = pathlib.Path(self.tempdirname) / "test_cast.mrc"
        with self.assertLogs(logger="pytom_tm", level="WARNING") as cm:
            write_mrc(fname, array, 1.0)
        self.assertEqual(len(cm.output), 1)
        self.assertIn("np.float32", cm.output[0])

    def test_almost_equal_voxel_warning(self):
        array = np.random.rand(27).reshape((3, 3, 3)).astype(np.float32)
        fname = pathlib.Path(self.tempdirname) / "test_almost_equal_voxels.mrc"
        # Make sure no warnings are raised
        with self.assertNoLogs(logger="pytom_tm", level="WARNING"):
            write_mrc(fname, array, voxel_size=(1.0, 1.0, 1.0001))
        # Make sure a warning is raised when reading
        with self.assertLogs(logger="pytom_tm", level="WARNING") as cm:
            _ = read_mrc_meta_data(fname)
        self.assertEqual(len(cm.output), 1)
        self.assertIn(
            "Voxel size annotation in MRC is slightly different", cm.output[0]
        )

    def test_parse_relion5_star_data(self):
        tomogram = pathlib.Path("rec_tomo200528_107.mrc")
        voxel_size, metadata = parse_relion5_star_data(RELION5_TOMOGRAMS_STAR, tomogram)
        self.assertIsInstance(voxel_size, float)
        self.assertIsInstance(metadata, RelionTiltSeriesMetaData)
        self.assertIsInstance(metadata.tilt_angles, list)
        self.assertIsInstance(metadata.ctf_data, list)
        self.assertIsInstance(metadata.ctf_data[0], CtfData)
        self.assertIsInstance(metadata.defocus_handedness, int)
        self.assertIsInstance(metadata.binning, float)
        self.assertIsInstance(metadata.tilt_series_pixel_size, float)
        self.assertAlmostEqual(
            voxel_size, metadata.tilt_series_pixel_size * metadata.binning
        )

        tomogram = pathlib.Path("tomogram.mrc")
        with self.assertRaises(
            ValueError, msg="Unmatching tomograms name should raise an error."
        ):
            parse_relion5_star_data(RELION5_TOMOGRAMS_STAR, tomogram)

        tomogram = pathlib.Path("rec_tomogram200528_1077.mrc")
        with self.assertRaises(
            ValueError, msg="Partially matching tomogram name should raise an error."
        ):
            parse_relion5_star_data(RELION5_TOMOGRAMS_STAR, tomogram)

        tomogram = pathlib.Path("rec_tomogram200528_10.mrc")
        with self.assertRaises(
            ValueError, msg="Partially matching tomogram name should raise an error."
        ):
            parse_relion5_star_data(RELION5_TOMOGRAMS_STAR, tomogram)
