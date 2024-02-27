# No imports of pytom_tm outside of the methods
import unittest
# Mock out installed dependencies
orig_import = __import__

def module_not_found_mock(missing_name):
    def import_mock(name, *args):
        if name == missing_name:
            raise ModuleNotFoundError(f"No module named '{name}'")
        return orig_import(name, *args)
    return import_mock

def cupy_import_error_mock(name, *args):
    if name == 'cupy':
        raise ImportError("Failed to import cupy")
    return orig_import(name, *args)

matplotlib_not_found = module_not_found_mock('matplotlib')
seaborn_not_found = module_not_found_mock('seaborn')

class TestMissingDependencies(unittest.TestCase):
    def test_missing_cupy(self):
        cupy_not_found = module_not_found_mock('cupy')
        # Test missing cupy
        with unittest.mock.patch('builtins.__import__', side_effect=cupy_not_found):
            with self.assertLogs(level='WARNING') as cm:
                import pytom_tm
            self.assertEqual(len(cm.output), 1)
            self.assertIn("cupy installation not found or not functional", cm.output[0])

    def test_broken_cupy(self):
        # Test cupy ImportError
        with unittest.mock.patch('builtins.__import__', side_effect=cupy_import_error_mock):
            with self.assertLogs(level='WARNING') as cm:
                import pytom_tm
            self.assertEqual(len(cm.output), 1)
            self.assertIn("cupy installation not found or not functional", cm.output[0])

    def test_missing_matplotlib(self):
        with unittest.mock.patch('builtins.__import__', side_effect=matplotlib_not_found):
            # check if we can still import pytom_tm
            import pytom_tm
            # check if plotting is indeed disbaled
            self.assertFalse(pytom_tm.template.plotting_available)
            self.assertFalse(pytom_tm.extract.plotting_available)
            # assert that importing the plotting module fails completely
            with self.assertRaisesRegex(RuntimeError, "matplotlib and seaborn"):
                import pytom_tm.plotting

    def test_missing_seaborn(self):
        with unittest.mock.patch('builtins.__import__', side_effect=seaborn_not_found):
            # check if we can still import pytom_tm
            import pytom_tm
            # check if plotting is indeed disbaled
            self.assertFalse(pytom_tm.template.plotting_available)
            self.assertFalse(pytom_tm.extract.plotting_available)
            # assert that importing the plotting module fails completely
            with self.assertRaisesRegex(RuntimeError, "matplotlib and seaborn"):
                import pytom_tm.plotting



