import unittest
from shutil import which
from contextlib import redirect_stdout
from io import StringIO
from pytom_tm import entry_points

# (command line function, function in entry_points file)
ENTRY_POINTS_TO_TEST = [
    ("pytom_create_mask.py", "pytom_create_mask"),
    ("pytom_create_template.py", "pytom_create_template"),
    ("pytom_match_template.py", "match_template"),
    ("pytom_extract_candidates.py", "extract_candidates"),
    ("pytom_merge_stars.py",  "merge_stars"),
    ]
# Test if optional dependencies are installed
try:
    from pytom_tm import plotting
except:
    pass
else:
    ENTRY_POINTS_TO_TEST.append(("pytom_estimate_roc.py", "estimate_roc"))

class TestEntryPoints(unittest.TestCase):
    def test_entry_points_exist(self):
        for cli, fname in ENTRY_POINTS_TO_TEST:
            # test the command line function can be found
            self.assertIsNotNone(which(cli))
            # assert the entry_point be called with -h and exit cleanly
            # catch stdout to prevent shell polution
            func = getattr(entry_points, fname)
            dump = StringIO()
            with self.assertRaises(SystemExit) as ex, redirect_stdout(dump):
                func([cli, '-h'])
            dump.close()
            # check if the system return code is 0 (success)
            self.assertEqual(ex.exception.code, 0)
