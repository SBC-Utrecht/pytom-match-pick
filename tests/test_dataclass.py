import unittest
from pytom_tm.dataclass import CtfData, TiltSeriesMetaData


class TestCtfDataclass(unittest.TestCase):
    def test_replace(self):
        # test that replace exists and can be used
        temp = CtfData(
            defocus=1, amplitude_contrast=0.1, voltage=300, spherical_aberration=1e-6
        )
        temp2 = temp.replace(defocus=2)
        self.assertNotEqual(temp.defocus, temp2.defocus)
        self.assertEqual(temp.amplitude_contrast, temp2.amplitude_contrast)
        self.assertEqual(temp.voltage, temp2.voltage)
        self.assertEqual(temp.spherical_aberration, temp2.spherical_aberration)


class TestTiltSeriesDataclass(unittest.TestCase):
    def test_sanity_errors(self):
        with self.assertRaisesRegex(ValueError, "at least 2 tilt angles"):
            _ = TiltSeriesMetaData(tilt_angles=[1.0])
        with self.assertRaisesRegex(ValueError, "at least 2 tilt angles"):
            _ = TiltSeriesMetaData(tilt_angles=1.0)

        a = CtfData(
            defocus=1, amplitude_contrast=0.1, voltage=300, spherical_aberration=1e-6
        )
        with self.assertRaisesRegex(ValueError, "a single CtfData or the same number"):
            _ = TiltSeriesMetaData(tilt_angles=[1, 2, 3], ctf_data=[a, a])

        with self.assertRaisesRegex(ValueError, "invalid defocus handedness"):
            _ = TiltSeriesMetaData(tilt_angles=[1, 2, 3], defocus_handedness=0.5)

    def test_sanity_on_replace(self):
        # test that the sanity checks happen when replacing values
        temp = TiltSeriesMetaData(tilt_angles=[1.0, 2.0], dose_accumulation=[1, 2])
        with self.assertRaisesRegex(
            ValueError, r"same number of doses as tilt angles \(3\)"
        ):
            _ = temp.replace(tilt_angles=[1, 2, 3])

    def test_ctf_expansion(self):
        a = CtfData(
            defocus=1, amplitude_contrast=0.1, voltage=300, spherical_aberration=1e-6
        )
        ts_metadata = TiltSeriesMetaData(tilt_angles=[1, 2, 3], ctf_data=[a])
        self.assertEqual(len(ts_metadata.tilt_angles), len(ts_metadata.ctf_data))
        for ctf in ts_metadata.ctf_data:
            self.assertIs(ctf, a)
