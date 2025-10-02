import unittest
from pytom_tm.dataclass import TiltSeriesMetaData


class TestTiltSeriesDataclass(unittest.TestCase):
    def test_sanity_errors(self):
        with self.assertRaisesRegex(ValueError, "at least 2 tilt angles"):
            _ = TiltSeriesMetaData(tilt_angles=[1.0])
        with self.assertRaisesRegex(ValueError, "at least 2 tilt angles"):
            _ = TiltSeriesMetaData(tilt_angles=1.0)

    def test_sanity_on_replace(self):
        # test that the sanity checks happen when replacing values
        temp = TiltSeriesMetaData(tilt_angles=[1.0, 2.0], dose_accumulation=[1, 2])
        with self.assertRaisesRegex(
            ValueError, "same number of doses as tilt angles (3)"
        ):
            _ = temp.replace(tilt_angles=[1, 2, 3])
