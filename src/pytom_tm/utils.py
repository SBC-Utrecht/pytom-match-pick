import contextlib
import os
import sys


@contextlib.contextmanager
def mute_stdout_stderr():
    """Context manager to redirect stdout and stderr to devnull. Only used to prevent
    terminal flooding in unittests. If an error is raised and not caught before, this
    will hard-exit out"""

    fail = False
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    with open(os.devnull, "w") as outnull:
        sys.stdout = outnull
        sys.stderr = outnull
        try:
            yield
        except Exception:  # noqa: BLE001
            # -- Bare exception to exit without printing anything
            fail = True
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    if fail:
        sys.exit(2)
