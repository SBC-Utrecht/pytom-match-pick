import unittest
from pytom_tm.utils import get_defocus_offsets


class TestUtils(unittest.TestCase):
    def test_get_defocus_offsets(self):
        tilt_angles = list(range(-51, 54, 3))
        x_offset_um = 200 * 13.79 * 1e-4
        z_offset_um = 100 * 13.79 * 1e-4
        defocus_offsets = get_defocus_offsets(x_offset_um, z_offset_um, tilt_angles)
        self.assertEquals(
            len(defocus_offsets),
            len(tilt_angles),
            msg='get_defocus_offsets did not return a list with the same length as the number of tilt_angles'
        )


if __name__ == '__main__':
    unittest.main()
