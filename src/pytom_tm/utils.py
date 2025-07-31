import os
import sys
import contextlib


@contextlib.contextmanager
def mute_stdout_stderr():
    """Context manager to redirect stdout and stderr to devnull. Only used to prevent
    terminal flooding in unittests."""

    outnull = open(os.devnull, "w")
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = outnull
    sys.stderr = outnull
    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        outnull.close()
