import os
import sys
import contextlib


@contextlib.contextmanager
def mute_stdout_stderr():
    """Context manager to redirect stdout and stderr to devnull. Only used to prevent
    terminal flooding in unittests. If an error is raised and not caught before, this
    will hard-exit out"""

    fail = False
    outnull = open(os.devnull, "w")
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = outnull
    sys.stderr = outnull
    try:
        yield
    except Exception:  # Bare exception to exit without printing anything
        fail = True
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        outnull.close()
        if fail:
            sys.exit(2)
