import unittest
import numpy as np
from pytom_tm.template import generate_template_from_map

class TestLogging(unittest.TestCase):
    def test_lowpass_resolution(self):
        template = np.zeros((13, 13, 13), dtype=np.float32)

        # Test missing filter resolution
        with self.assertLogs(level='WARNING') as cm:
            _ = generate_template_from_map(template, 1., 1.)
        self.assertEqual(len(cm.output) == 1)
        self.assertIn('Filter resolution', cm.output[0])
        self.assertIn(' not specified ',cm.output[0])
        self.assertNotIn(' too low ', cm.output[0])

        # Test too low filter resolution
        with self.assertLogs(level='WARNING') as cm:
            _ = generate_template_from_map(template, 1., 1., filter_to_resolution=1.5)
        self.assertEqual(len(cm.output) == 1)
        self.assertIn('Filter resolution', cm.output[0])
        self.assertNotIn(' not specified ',cm.output[0])
        self.assertIn(' too low ', cm.output[0])

        # Test working filter resolution
        with self.assertNoLogs(level='WARNING'):
            _ = generate_template_from_map(template, 1., 1., filter_to_resolution=2.5)
 
