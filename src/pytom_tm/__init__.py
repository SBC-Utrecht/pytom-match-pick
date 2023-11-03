from importlib import metadata
__version__ = metadata.version('pytom-template-matching-gpu')

try:
    import cupy
except (ModuleNotFoundError, ImportError):
    print('Error for template matching: cupy installation not found or not functional.')
