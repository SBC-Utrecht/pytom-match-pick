import logging
from importlib import metadata

__version__ = metadata.version("pytom-match-pick")


try:
    import cupy  # noqa: F401
except (ModuleNotFoundError, ImportError):
    logging.warning(
        "Error for template matching: cupy installation not found or not functional."
    )
