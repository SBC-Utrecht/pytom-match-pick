import pathlib
import copy
import numpy as np
import numpy.typing as npt
import mrcfile
from operator import attrgetter
from typing import Optional
from pytom_tm.angles import AVAILABLE_ROTATIONAL_SAMPLING, load_angle_list
from pytom_tm.matching import TemplateMatchingGPU
from pytom_tm.weights import create_wedge
from functools import reduce


def read_mrc_meta_data(file_name: pathlib.Path) -> dict:
    meta_data = {}
    with mrcfile.mmap(file_name) as mrc:
        meta_data['shape'] = tuple(map(int, attrgetter('nx', 'ny', 'nz')(mrc.header)))
        if not all([mrc.voxel_size.x == s for s in attrgetter('x', 'y', 'z')(mrc.voxel_size)]):
            raise ValueError('Input tomogram voxel spacing is not identical in each dimension!')
        else:
            meta_data['voxel_size'] = float(mrc.voxel_size.x)
    return meta_data


class TMJobError(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


class TMJob:
    def __init__(
            self,
            job_key: str,
            tomogram: pathlib.Path,
            template: pathlib.Path,
            mask: pathlib.Path,
            output_dir: pathlib.Path,
            angle_increment: str,
            mask_is_spherical: bool = True,
            wedge_angles: Optional[tuple[float, float]] = None,
            search_origin: Optional[tuple[int, int, int]] = None,
            search_size: Optional[tuple[int, int, int]] = None,
            voxel_size: Optional[float] = None
    ):
        self.mask = mask
        self.mask_is_spherical = mask_is_spherical
        self.output_dir = output_dir

        self.tomogram = tomogram
        self.template = template

        meta_data_tomo = read_mrc_meta_data(self.tomogram)
        meta_data_template = read_mrc_meta_data(self.template)

        self.tomo_shape = meta_data_tomo['shape']
        self.template_shape = meta_data_template['shape']

        if voxel_size is not None:
            if voxel_size <= 0:
                raise ValueError('Invalid voxel size provided, smaller or equal to zero.')
            self.voxel_size = voxel_size
            if (
                    self.voxel_size != meta_data_tomo['voxel_size'] or
                    self.voxel_size != meta_data_template['voxel_size']
            ):
                print('WARNING: Provided voxel size does not match voxel size annotated in tomogram/template mrc.')
        elif meta_data_tomo['voxel_size'] == meta_data_template['voxel_size'] and meta_data_tomo['voxel_size'] > 0:
            self.voxel_size = meta_data_tomo['voxel_size']
        else:
            raise ValueError('Voxel size could not be assigned, either a mismatch between tomogram and template or'
                             ' annotated as 0.')

        # Check if tomogram origin is valid
        if search_origin is None:
            self.search_origin = (0, 0, 0)
        elif all([0 <= x < y for x, y in zip(search_origin, self.tomo_shape)]):
            self.search_origin = search_origin
        else:
            raise ValueError('Invalid input provided for search origin of tomogram.')

        # Check if search size is valid
        if search_size is None:
            self.search_size = tuple([x - y for x, y in zip(self.tomo_shape, self.search_origin)])
        elif all([(x + y <= z) and (x > 0) for x, y, z in zip(search_size, self.search_origin, self.tomo_shape)]):
            self.search_size = search_size
        else:
            raise ValueError('Invalid input provided for search size in the tomogram.')

        self.whole_start = None
        self.sub_start, self.sub_step = None, None

        # Rotations to search
        try:
            self.rotation_file = AVAILABLE_ROTATIONAL_SAMPLING[angle_increment][0]
            self.n_rotations = AVAILABLE_ROTATIONAL_SAMPLING[angle_increment][1]
        except KeyError:
            raise TMJobError('Provided angular search is not available in  the default lists.')

        self.start_slice = 0
        self.steps_slice = 1

        # missing wedge
        self.wedge_angles = wedge_angles

        # Job details
        self.job_key = job_key
        self.leader = None  # the job that spawned this job
        self.sub_jobs = []  # if this job had no sub jobs it should be executed
        self.finished_run = False

    def copy(self):
        return copy.deepcopy(self)

    def split_rotation_search(self, n: int):
        if len(self.sub_jobs) > 0:
            raise TMJobError('Could not further split this job as it already has subjobs assigned!')

        sub_jobs = []
        for i in range(n):
            new_job = self.copy()
            new_job.start_slice = i
            new_job.steps_slice = n
            new_job.leader = self.job_key
            new_job.job_key = self.job_key + str(i)
            sub_jobs.append(new_job)

        self.sub_jobs = sub_jobs

        return self.sub_jobs

    def split_volume_search(self, split: tuple[int, int, int]):
        if len(self.sub_jobs) > 0:
            raise TMJobError('Could not further split this job as it already has subjobs assigned!')

        # size of sub-volumes after splitting
        split_size = tuple([x // y for x, y in zip(self.search_size, split)])

        # shape of template for overhang
        template_shape = self.template_shape  # tranposed

        # check if valid
        if any([x > y for x, y in zip(template_shape, split_size)]):
            raise RuntimeError("Not big enough volume to split!")

        # size of subvolume with overhang (underscore indicates not final tuple)
        _size = tuple([x + y for x, y in zip(template_shape, split_size)])

        number_of_pieces = reduce((lambda x, y: x * y), split)

        sub_jobs = []

        for i in range(number_of_pieces):
            stride_z, stride_y = split[0] * split[1], split[0]
            inc_z, inc_y, inc_x = i // stride_z, (i % stride_z) // stride_y, i % stride_y

            _start = tuple([
                -template_shape[0] // 2 + self.search_origin[0] + inc_x * split_size[0],
                -template_shape[1] // 2 + self.search_origin[1] + inc_y * split_size[1],
                -template_shape[2] // 2 + self.search_origin[2] + inc_z * split_size[2]
            ])

            _end = tuple([_start[j] + _size[j] for j in range(len(_start))])

            # if start location is smaller than search origin we need to reset it to search origin
            start = tuple([o if s < o else s for s, o in zip(_start, self.search_origin)])
            # if end location is larger than origin + search size it needs to be set to origin + search size
            end = tuple([s + o if e > s + o else e for e, s, o in zip(_end, self.search_size, self.search_origin)])
            size = tuple([end[j] - start[j] for j in range(len(start))])

            # for reassembling the result
            whole_start = list(start[:])  # whole_start and sub_start should also be job attributes
            sub_start = [0, ] * 3
            sub_step = list(split_size)
            if start[0] != self.search_origin[0]:
                whole_start[0] = start[0] + template_shape[0] // 2 - self.search_origin[0]
                sub_start[0] = template_shape[0] // 2
            if start[1] != self.search_origin[1]:
                whole_start[1] = start[1] + template_shape[1] // 2 - self.search_origin[0]
                sub_start[1] = template_shape[1] // 2
            if start[2] != self.search_origin[2]:
                whole_start[2] = start[2] + template_shape[2] // 2 - self.search_origin[0]
                sub_start[2] = template_shape[2] // 2
            if end[0] == self.search_origin[0] + self.search_size[0]:
                sub_step[0] = size[0] - sub_start[0]
            if end[1] == self.search_origin[1] + self.search_size[1]:
                sub_step[1] = size[1] - sub_start[1]
            if end[2] == self.search_origin[2] + self.search_size[2]:
                sub_step[2] = size[2] - sub_start[2]

            # create a split volume job
            new_job = self.copy()
            new_job.leader = self.job_key
            new_job.job_key = self.job_key + str(i)
            new_job.search_origin = (start[0], start[1], start[2])
            new_job.search_size = (size[0], size[1], size[2])
            new_job.whole_start = tuple(whole_start)
            new_job.sub_start = tuple(sub_start)
            new_job.sub_step = tuple(sub_step)
            new_job.split_size = split_size
            sub_jobs.append(new_job)

        self.sub_jobs = sub_jobs

        return self.sub_jobs

    def merge_sub_jobs(self) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
        if len(self.sub_jobs) == 0:
            # read the volumes, remove them and return them
            score_file, angle_file = (
                self.output_dir.joinpath(f'scores_{self.job_key}.mrc'),
                self.output_dir.joinpath(f'angles_{self.job_key}.mrc')
            )
            result = (
                np.ascontiguousarray(mrcfile.read(score_file).T),
                np.ascontiguousarray(mrcfile.read(angle_file).T)
            )
            (score_file.unlink(), angle_file.unlink())
            return result

        is_subvolume_split = np.all(np.array([x.start_slice for x in self.sub_jobs]) == 0)

        # print(f'Is job split into sub-volumes? : {is_subvolume_split}')

        score_volumes, angle_volumes = [], []
        for x in self.sub_jobs:
            result = x.merge_sub_jobs()
            score_volumes.append(result[0])
            angle_volumes.append(result[1])

        if not is_subvolume_split:
            scores, angles = np.zeros_like(score_volumes[0]) - 1., np.zeros_like(angle_volumes[0]) - 1.
            for s, a in zip(score_volumes, angle_volumes):
                angles = np.where(s > scores, a, angles)
                scores = np.where(s > scores, s, scores)
        else:
            scores, angles = (
                np.zeros(self.search_size, dtype=np.float32),
                np.zeros(self.search_size, dtype=np.float32)
            )
            for job, s, a in zip(self.sub_jobs, score_volumes, angle_volumes):

                sub_scores = s[
                             job.sub_start[0]: job.sub_start[0] + job.sub_step[0],
                             job.sub_start[1]: job.sub_start[1] + job.sub_step[1],
                             job.sub_start[2]: job.sub_start[2] + job.sub_step[2]]
                sub_angles = a[
                             job.sub_start[0]: job.sub_start[0] + job.sub_step[0],
                             job.sub_start[1]: job.sub_start[1] + job.sub_step[1],
                             job.sub_start[2]: job.sub_start[2] + job.sub_step[2]]

                # Then the corrected sub part needs to be placed back into the full volume
                scores[
                    job.whole_start[0]: job.whole_start[0] + sub_scores.shape[0],
                    job.whole_start[1]: job.whole_start[1] + sub_scores.shape[1],
                    job.whole_start[2]: job.whole_start[2] + sub_scores.shape[2]
                ] = sub_scores
                angles[
                    job.whole_start[0]: job.whole_start[0] + sub_scores.shape[0],
                    job.whole_start[1]: job.whole_start[1] + sub_scores.shape[1],
                    job.whole_start[2]: job.whole_start[2] + sub_scores.shape[2]
                ] = sub_angles
        return scores, angles

    def start_job(self, gpu_id: int, return_volumes: bool = False):
        # load the (sub)volume
        search_volume = np.ascontiguousarray(mrcfile.read(self.tomogram).T)[
            self.search_origin[0]: self.search_origin[0] + self.search_size[0],
            self.search_origin[1]: self.search_origin[1] + self.search_size[1],
            self.search_origin[2]: self.search_origin[2] + self.search_size[2]
        ]

        # load template and mask
        template, mask = (
            np.ascontiguousarray(mrcfile.read(self.template).T),
            np.ascontiguousarray(mrcfile.read(self.mask).T)
        )

        # TODO apply optionally a bandpass filter to the volume

        template_wedge = None
        if self.wedge_angles is not None:
            # convolute tomo with wedge
            tomo_wedge = create_wedge(search_volume.shape, self.wedge_angles, 1.).astype(np.float32)
            search_volume = np.real(np.fft.irfftn(np.fft.rfftn(search_volume) * tomo_wedge, s=search_volume.shape))

            mrcfile.write(
                self.output_dir.joinpath(f'wedge_{self.job_key}.mrc'),
                tomo_wedge.T,
                voxel_size=self.voxel_size,
                overwrite=True
            )

            # get template wedge
            template_wedge = create_wedge(self.template_shape, self.wedge_angles, 1.).astype(np.float32)

        # load rotation search
        angle_ids = list(range(self.start_slice, self.n_rotations, self.steps_slice))
        angle_list = load_angle_list(self.rotation_file)[slice(self.start_slice, self.n_rotations, self.steps_slice)]

        tm = TemplateMatchingGPU(
            job_id=self.job_key,
            device_id=gpu_id,
            volume=search_volume,
            template=template,
            mask=mask,
            angle_list=angle_list,
            angle_ids=angle_ids,
            mask_is_spherical=self.mask_is_spherical,
            wedge=template_wedge
        )

        tm.run()

        score_volume, angle_volume = tm.plan.scores.get(), tm.plan.angles.get()
        # stats = tm_thread.get_std_stats()
        del tm

        if return_volumes:
            return score_volume, angle_volume
        else:  # otherwise write them out with job_key
            # TODO files should be written with tomogram name in it
            mrcfile.write(
                self.output_dir.joinpath(f'scores_{self.job_key}.mrc'),
                score_volume.T,
                voxel_size=self.voxel_size,
                overwrite=True
            )
            mrcfile.write(
                self.output_dir.joinpath(f'angles_{self.job_key}.mrc'),
                angle_volume.T,
                voxel_size=self.voxel_size,
                overwrite=True
            )
