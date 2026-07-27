import logging
from importlib import metadata

__version__ = metadata.version("pytom-match-pick")

# Package-level logger. Library code should never configure or log through the
# root logger; instead each module gets its own `logging.getLogger(__name__)`
# logger which is a child of this one. Per Python logging best practices for
# libraries, we attach a NullHandler here so that nothing is emitted unless the
# application (e.g. the pytom_tm CLI entry points) explicitly configures
# handlers on this logger.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


try:
    import cupy  # noqa: F401
except (ModuleNotFoundError, ImportError):
    logger.warning(
        "Error for template matching: cupy installation not found or not functional."
    )


def configure_logging(level: int) -> None:
    """Configure logging for the pytom_tm package only, without touching the root
    logger. This should be called once by an application entry point (e.g. one of
    the pytom_tm CLI scripts) rather than by library code.

    Parameters
    ----------
    level: int
        the logging level to set for the pytom_tm package logger, for example
        logging.INFO or logging.DEBUG
    """
    package_logger = logging.getLogger(__name__)
    # remove any handlers that might have been added by a previous call, this
    # mimics the `force=True` behaviour of logging.basicConfig() but scoped to
    # the pytom_tm logger instead of the root logger
    for handler in list(package_logger.handlers):
        if not isinstance(handler, logging.NullHandler):
            package_logger.removeHandler(handler)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    package_logger.addHandler(handler)
    package_logger.setLevel(level)
    # Note: we deliberately leave `propagate` at its default (True). This means
    # records still bubble up to the root logger's handlers if any are ever
    # added elsewhere (e.g. by a test harness using unittest.assertLogs, or by
    # another library) - but pytom_tm itself never configures or logs directly
    # through the root logger.
