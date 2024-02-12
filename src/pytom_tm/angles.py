import pathlib
from importlib_resources import files
from scipy.spatial.transform import Rotation


ANGLE_LIST_DIR = files('pytom_tm.angle_lists')
AVAILABLE_ROTATIONAL_SAMPLING = {
    '7.00': ['angles_7.00_45123.txt', 45123],
    '35.76': ['angles_35.76_320.txt', 320],
    '19.95': ['angles_19.95_1944.txt', 1944],
    '90.00': ['angles_90.00_26.txt', 26],
    '18.00': ['angles_18.00_3040.txt', 3040],
    '12.85': ['angles_12.85_7112.txt', 7112],
    '38.53': ['angles_38.53_256.txt', 256],
    '11.00': ['angles_11.00_15192.txt', 15192],
    '17.86': ['angles_17.86_3040.txt', 3040],
    '25.25': ['angles_25.25_980.txt', 980],
    '50.00': ['angles_50.00_100.txt', 100],
    '3.00': ['angles_3.00_553680.txt', 553680],
}
for v in AVAILABLE_ROTATIONAL_SAMPLING.values():
    v[0] = ANGLE_LIST_DIR.joinpath(v[0])


def load_angle_list(file_name: pathlib.Path, sort_angles: bool = True) -> list[tuple[float, float, float]]:
    """Load an angular search list from disk.

    Parameters
    ----------
    file_name: pathlib.Path
        path to text file containing angular search, each line should contain 3 floats of anti-clockwise ZXZ
    sort_angles: bool
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
    order_in: str
        Euler rotation axis of input angles
    order_out: str
        Euler rotation axis of output angles
    degrees_in: bool
        whether the input angles are in degrees
    degrees_out
        whether the output angles should be in degrees

    Returns
    -------
    output: tuple[float, float, float]
        tuple of three angles
    """
    r = Rotation.from_euler(order_in, angles, degrees=degrees_in)
    return tuple(r.as_euler(order_out, degrees=degrees_out))
