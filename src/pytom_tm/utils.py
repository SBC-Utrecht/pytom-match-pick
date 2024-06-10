import os
import sys


class mute_stdout_stderr(object):
    """Context manager to redirect stdout and stderr to devnull. Only used to prevent terminal flooding in unittests."""

    def __enter__(self):
        self.outnull = open(os.devnull, "w")
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = self.outnull
        sys.stderr = self.outnull
        return self

    def __exit__(self):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.outnull.close()
