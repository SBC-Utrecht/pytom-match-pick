import pathlib
import copy
from pytom_tm.angles import AVAILABLE_ROTATIONAL_SAMPLING


class Job:
    def __init__(
            self,
            job_id: int,
            tomogram: pathlib.Path,
            template: pathlib.Path,
            mask: pathlib.Path,
            subregion: list[tuple[int]],
            angle_increment: str
    ):
        self.job_id = job_id
        self.tomogram = tomogram
        self.reference = template
        self.mask = mask

        self.subregion = subregion
        self.rotation_file = AVAILABLE_ROTATIONAL_SAMPLING[angle_increment][0]
        self.n_rotations = AVAILABLE_ROTATIONAL_SAMPLING[angle_increment][1]
        self.rotation_slice = slice(0, self.n_rotations, 1)

    def split_rotation_search(self, slices):
        pass

    def copy(self):
        return copy.deepcopy(self)
