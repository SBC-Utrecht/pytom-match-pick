import os
import sys
import numpy as np
import numpy.typing as npt


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
    angles_in_degrees: bool = True,
    invert_handedness: bool = False,
) -> npt.NDArray[float]:
    """Calculate the defocus offsets for a subvolume
    based on the tilt geometry.

    I used the definition from Pyle & Zianetti (https://doi.org/10.1042/BCJ20200715)
    for the default setting of the defocus handedness. It assumes the defocus
    increases for positive tilt angles on the right side of the sample (positive X
    coordinate relative to the center).

    The offset is calculated as follows:
        z_offset = z_center * np.cos(tilt_angle) + x_center * np.sin(tilt_angle)

    Parameters
    ----------
    patch_center_x: float
        x center of subvolume relative to tomogram center
    patch_center_z: float
        z center of subvolume relative to tomogram center
    tilt_angles: list[float, ...]
        list of tilt angles
    angles_in_degrees: bool, default True
        whether tilt angles are in degrees or radians
    invert_handedness: bool, default False
        invert defocus handedness geometry

    Returns
    -------
    z_offsets: npt.NDArray[float]
        an array of defocus offsets for each tilt angle
    """
    n_tilts = len(tilt_angles)
    x_centers = np.array(
        [
            patch_center_x,
        ]
        * n_tilts
    )
    z_centers = np.array(
        [
            patch_center_z,
        ]
        * n_tilts
    )
    ta_array = np.array(tilt_angles)
    if angles_in_degrees:
        ta_array = np.deg2rad(ta_array)
    if invert_handedness:
        ta_array *= -1
    z_offsets = z_centers * np.cos(ta_array) + x_centers * np.sin(ta_array)
    return z_offsets
