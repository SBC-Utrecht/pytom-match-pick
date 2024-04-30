import pathlib
from importlib_resources import files
from scipy.spatial.transform import Rotation
import numpy as np
import healpix as hp
import logging

def angle_to_angle_list(angle_diff: float, sort_angles: bool = True) -> list[tuple[float, float, float]]:
    """Auto generate an angle list for a given maximum angle difference. 

    The code uses healpix to determine Z1 and X and splits Z2 linearly.

    Parameters
    ----------
    angle_diff: float
        maximum difference (in degrees) for the angle list

    sort_angles: bool, default True
        sort the list, using python default angle_list.sort(), sorts first on Z1, then X, then Z2

    Returns
    -------
    angle_list: list[tuple[float, float, float]]
        a list where each element is a tuple of 3 floats containing 
        an anti-clockwise ZXZ Euler rotation in degrees
    """
    # We use an approximation of the square root of the area as the median angle diff 
    # This works reasonably well and is based on the following formula:
    # angle_diff = (4*np.pi/npix)**0.5 * 360/(2*np.pi)
    npix = 4 * np.pi / (angle_diff * np.pi / 180) ** 2
    nside = 0
    while hp.nside2npix(nside) < npix:
        nside += 1
    used_npix = hp.nside2npix(nside)
    used_angle_diff = (4*np.pi/used_npix)**0.5 * (180/np.pi)
    logging.info(f"Using an angle difference of {used_angle_diff:.4f} for Z1 and X")
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    theta *= 180/np.pi
    phi *= 180/np.pi
    # Now for psi
    n_psi_angles = int(np.ceil(360/angle_diff))
    psi, used_psi_diff = np.linspace(0,360, n_psi_angles, endpoint=False, retstep=True)
    logging.info(f"Using an angle difference of {used_psi_diff:.4f} for Z2")
    angle_list = [(ph, th, ps) for ph, th in zip(phi, theta) for ps in psi]
    if sort_angles:
        angle_list.sort()
    return angle_list

def load_angle_list(file_name: pathlib.Path, sort_angles: bool = True) -> list[tuple[float, float, float]]:
    """Load an angular search list from disk.

    Parameters
    ----------
    file_name: pathlib.Path
        path to text file containing angular search, each line should contain 3 floats of anti-clockwise ZXZ
    sort_angles: bool, default True
        sort the list, using python default angle_list.sort(), sorts first on Z1, then X, then Z2

    Returns
    -------
    angle_list: list[tuple[float, float, float]]
        a list where each element is a tuple of 3 floats containing an anti-clockwise ZXZ Euler rotation in degrees
    """
    with open(str(file_name)) as fstream:
        lines = fstream.readlines()
    angle_list = [tuple(map(float, x.strip().split(' '))) for x in lines]
    if not all([len(a) == 3 for a in angle_list]):
        raise ValueError('Invalid angle file provided, each line should have 3 ZXZ Euler angles!')
    if sort_angles:
        angle_list.sort()  # angle list needs to be sorted otherwise symmetry reduction cannot be used!
    return angle_list


def convert_euler(
        angles: tuple[float, float, float],
        order_in: str = 'ZXZ',
        order_out: str = 'ZXZ',
        degrees_in: bool = True,
        degrees_out: bool = True
) -> tuple[float, float, float]:
    """Convert a single set of Euler angles from one Euler notation to another. This function makes use of
    scipy.spatial.transform.Rotation meaning that capital letters (i.e. ZXZ) specify intrinsic rotations (commonly
    used in cryo-EM) and small letters (i.e. zxz) specific extrinsic rotations.

    Parameters
    ----------
    angles: tuple[float, float, float]
        tuple of three angles
    order_in: str, default 'ZXZ'
        Euler rotation axis of input angles
    order_out: str, default 'ZXZ'
        Euler rotation axis of output angles
    degrees_in: bool, default True
        whether the input angles are in degrees
    degrees_out: bool, default True
        whether the output angles should be in degrees

    Returns
    -------
    output: tuple[float, float, float]
        tuple of three angles
    """
    r = Rotation.from_euler(order_in, angles, degrees=degrees_in)
    return tuple(r.as_euler(order_out, degrees=degrees_out))
