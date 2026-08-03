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
    package_logger = logger
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
    # Now that a real handler is attached directly to the pytom_tm logger, stop
    # propagating records up to the root logger. Otherwise anything else that
    # happens to have a handler on the root logger (e.g. pytest's automatic log
    # capture, or another library's basicConfig() call) would emit the same
    # record a second time. Note this only takes effect once configure_logging()
    # has actually been called; until then `propagate` stays at its default
    # (True).
    package_logger.propagate = False
