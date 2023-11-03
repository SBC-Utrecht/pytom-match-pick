from ._version import __version__

try:
    import cupy
except (ModuleNotFoundError, ImportError):
    print('Error for template matching: cupy installation not found or not functional.')
