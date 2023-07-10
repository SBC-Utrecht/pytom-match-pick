import pathlib


ROTATIONAL_SAMPLING = {
    '11': ('11_15192.txt', 15192),
    '7': ('7_45123.txt', 45123),
    '3': ('3_553680.txt', 553680)
}


class Job:
    def __init__(self, subregion, angle_increment):

        self.tomogram = pathlib.Path
        self.reference = pathlib.Path
        self.mask = pathlib.Path

        self.subregion = subregion
        self.rotation_file = pathlib.Path(ROTATIONAL_SAMPLING[angle_increment][0])
        self.n_rotations = ROTATIONAL_SAMPLING[angle_increment][1]
        self.rotation_slice = slice(0, self.n_rotations, 1)

    def split_rotation_search(self, slices):
