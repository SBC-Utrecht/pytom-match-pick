import os
import sys
import numpy as np


class mute_stdout_stderr(object):
    """Context manager to redirect stdout and stderr to devnull. Only used to prevent
    terminal flooding in unittests."""

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


def get_defocus_offsets(
        patch_center_x: float,
        patch_center_z: float,
        tilt_angles: list[float, ...],
        angles_in_degrees: bool = True
) -> list[float]:
    n_tilts = len(tilt_angles)
    x_centers = np.array([patch_center_x, ] * n_tilts)
    z_centers = np.array([patch_center_z, ] * n_tilts)
    ta_array = np.array(tilt_angles)
    # if upside_down:  # invert the tilt_angles if tomogram is reconstructed upside down? but then template should
    #   ta_array *= -1  # also be mirrored
    if angles_in_degrees:
        ta_array = np.deg2rad(ta_array)
    z_offsets = list(z_centers * np.cos(ta_array) + x_centers * np.sin(ta_array))
    return z_offsets
