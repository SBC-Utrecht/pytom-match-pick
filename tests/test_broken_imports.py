# No imports of pytom_tm outside of the methods
import unittest
from importlib import reload
# Mock out installed dependencies
orig_import = __import__

# skip tests if optional stuff is not installed
SKIP_PLOT = False
try:
    import pytom_tm.plotting
except:
    SKIP_PLOT = True

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


class TestMissingDependencies(unittest.TestCase):

    def test_missing_cupy(self):
        # assert working import
        with self.assertNoLogs(level='WARNING'):
            import pytom_tm
        cupy_not_found = module_not_found_mock('cupy')
        # Test missing cupy
        with unittest.mock.patch('builtins.__import__', side_effect=cupy_not_found):
            with self.assertLogs(level='WARNING') as cm:
                reload(pytom_tm)
            self.assertEqual(len(cm.output), 1)
            self.assertIn("cupy installation not found or not functional", cm.output[0])

    def test_broken_cupy(self):
        # assert working import
        with self.assertNoLogs(level='WARNING'):
            import pytom_tm
        # Test cupy ImportError
        with unittest.mock.patch('builtins.__import__', side_effect=cupy_import_error_mock):
            with self.assertLogs(level='WARNING') as cm:
                reload(pytom_tm)
            self.assertEqual(len(cm.output), 1)
            self.assertIn("cupy installation not found or not functional", cm.output[0])

    @unittest.skipIf(SKIP_PLOT, "plotting module not installed")
    def test_missing_matplotlib(self):
        # assert working import
        import pytom_tm

        matplotlib_not_found = module_not_found_mock('matplotlib.pyplot')
        with unittest.mock.patch('builtins.__import__', side_effect=matplotlib_not_found):
            with self.assertRaisesRegex(ModuleNotFoundError, 'matplotlib'):
                # only pyplot is directly imported so this should be tested
                import matplotlib.pyplot as plt
            # force reload 
            # check if we can still import pytom_tm
            reload(pytom_tm)
            
            # check if plotting is indeed disabled after reload
            # (reload is needed to prevent python import caching)
            self.assertFalse(reload(pytom_tm.template).plotting_available)
            self.assertFalse(reload(pytom_tm.extract).plotting_available)
            # assert that importing the plotting module fails completely
            with self.assertRaisesRegex(RuntimeError, "matplotlib and seaborn"):
                reload(pytom_tm.plotting)

    @unittest.skipIf(SKIP_PLOT, "plotting module not installed")
    def test_missing_seaborn(self):
        # assert working import
        import pytom_tm
        
        seaborn_not_found = module_not_found_mock('seaborn')
        with unittest.mock.patch('builtins.__import__', side_effect=seaborn_not_found):
            with self.assertRaisesRegex(ModuleNotFoundError, 'seaborn'):
                import seaborn
            # check if we can still import pytom_tm
            reload(pytom_tm)
            # check if plotting is indeed disabled
            # (reload is needed to prevent python import caching)
            self.assertFalse(reload(pytom_tm.template).plotting_available)
            self.assertFalse(reload(pytom_tm.extract).plotting_available)
            # assert that importing the plotting module fails completely
            with self.assertRaisesRegex(RuntimeError, "matplotlib and seaborn"):
                reload(pytom_tm.plotting)



