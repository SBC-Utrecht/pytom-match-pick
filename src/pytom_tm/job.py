import pathlib
from pytom_tm.angles import AVAILABLE_ROTATIONAL_SAMPLING


class Job:
    def __init__(self, subregion, angle_increment):

        self.tomogram = pathlib.Path()
        self.reference = pathlib.Path()
        self.mask = pathlib.Path()

        self.subregion = subregion
        self.rotation_file = pathlib.Path(ROTATIONAL_SAMPLING[angle_increment][0])
        self.n_rotations = ROTATIONAL_SAMPLING[angle_increment][1]
        self.rotation_slice = slice(0, self.n_rotations, 1)

    def split_rotation_search(self, slices):
        pass
