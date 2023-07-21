import pathlib
import copy
import time
import numpy as np
import numpy.typing as npt
import mrcfile
from typing import Optional
from pytom_tm.angles import AVAILABLE_ROTATIONAL_SAMPLING, load_angle_list
from pytom_tm.structures import TemplateMatchingGPU
from functools import reduce


class TMJobError(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


class Job:
    def __init__(
            self,
            job_key: str,
            tomogram: pathlib.Path,
            template: pathlib.Path,
            mask: pathlib.Path,
            mask_is_spherical: bool,
            output_dir: pathlib.Path,
            angle_increment: str,
            search_origin: Optional[tuple[int, int, int]] = None,
            search_size: Optional[tuple[int, int, int]] = None
    ):
        self.tomogram = mrcfile.mmap(tomogram, mode='r+')
        self.template = mrcfile.mmap(template, mode='r+')
        self.mask = mrcfile.mmap(mask, mode='r+')
        self.mask_is_spherical = mask_is_spherical
        self.output_dir = output_dir

        # Check if tomogram origin is valid
        tomo_shape = self.tomogram.shape[::-1]
        if search_origin is None:
            self.search_origin = (0, 0, 0)
        elif all([0 <= x < y for x, y in zip(search_origin, tomo_shape)]):
            self.search_origin = search_origin
        else:
            raise ValueError('Invalid input provided for search origin of tomogram.')

        # Check if search size is valid
        if search_size is None:
            self.search_size = tuple([x - y for x, y in zip(tomo_shape, self.search_origin)])
        elif all([(x + y <= z) and (x > 0) for x, y, z in zip(search_size, self.search_origin, tomo_shape)]):
            self.search_size = search_size
        else:
            raise ValueError('Invalid input provided for search size in the tomogram.')

        self.whole_start, self.sub_start, self.split_size = None, None, None

        # Rotations to search
        self.rotation_file = AVAILABLE_ROTATIONAL_SAMPLING[angle_increment][0]
        self.n_rotations = AVAILABLE_ROTATIONAL_SAMPLING[angle_increment][1]
        self.start_slice = 0
        self.steps_slice = 1

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

        for i in range(n):
            new_job = self.copy()
            new_job.start_slice = i
            new_job.steps_slice = n
            new_job.leader = self.job_key
            new_job.job_id = self.job_key + str(i)
            self.sub_jobs.append(new_job)

        return self.sub_jobs

    def split_volume_search(self, split: list[int, int, int]):
        if len(self.sub_jobs) > 0:
            raise TMJobError('Could not further split this job as it already has subjobs assigned!')

        # size of sub-volumes after splitting
        split_size = tuple([x // y for x, y in zip(self.search_size, split)])

        # shape of template for overhang
        template_shape = self.template.shape[::-1]  # tranposed

        # check if valid
        if any([x > y for x, y in zip(template_shape, split_size)]):
            raise RuntimeError("Not big enough volume to split!")

        # size of subvolume with overhang (underscore indicates not final tuple)
        _size = tuple([x + y for x, y in zip(template_shape, split_size)])

        number_of_pieces = reduce((lambda x, y: x * y), split)

        for i in range(number_of_pieces):
            stride_z, stride_y = split[0] * split[1], split[0]
            inc_z, inc_y, inc_x = i // stride_z, (i % stride_z) // stride_y, i % stride_y

            _start = tuple([
                -template_shape[0] // 2 + self.search_origin[0] + inc_x * split_size[0],
                -template_shape[1] // 2 + self.search_origin[1] + inc_y * split_size[1],
                -template_shape[2] // 2 + self.search_origin[2] + inc_z * split_size[2]
            ])

            _end = tuple([_start[j] + _size[j] for j in range(len(_start))])

            # adjust boundaries if they go outside the box
            start = tuple([o if s < o else s for s, o in zip(_start, self.search_origin)])
            end = tuple([s + o if e > s + o else e for e, s, o in zip(_end, self.search_size, self.search_origin)])
            size = tuple([end[j] - start[j] for j in range(len(start))])

            # for reassembling the result
            whole_start = list(start[:])  # whole_start and sub_start should also be job attributes
            sub_start = [0, 0, 0]
            if start[0] != self.search_origin[0]:
                whole_start[0] = start[0] + template_shape[0] // 2
                sub_start[0] = template_shape[0] // 2
            if start[1] != self.search_origin[1]:
                whole_start[1] = start[1] + template_shape[1] // 2
                sub_start[1] = template_shape[1] // 2
            if start[2] != self.search_origin[2]:
                whole_start[2] = start[2] + template_shape[2] // 2
                sub_start[2] = template_shape[2] // 2

            # create a split volume job
            new_job = self.copy()
            new_job.leader = self.job_key
            new_job.job_id = self.job_key + str(i)
            new_job.search_origin = (start[0], start[1], start[2])
            new_job.search_size = (size[0], size[1], size[2])
            new_job.whole_start = tuple(whole_start)
            new_job.sub_start = tuple(sub_start)
            new_job.split_size = split_size
            self.sub_jobs.append(new_job)

        return self.sub_jobs

    def merge_sub_jobs(self) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
        if len(self.sub_jobs) == 0:
            # read the volumes and return them
            return (
                read(self.output_dir.joinpath(f'scores_{self.job_key}.mrc')),
                read(self.output_dir.joinpath(f'angles_{self.job_key}.mrc'))
            )

        is_rotation_split = np.all(np.array([x.start_slice for x in self.sub_jobs]) == 0)
        score_volumes, angle_volumes = [], []
        for x in self.sub_jobs:
            result = x.merge_sub_jobs()
            score_volumes.append(result[0])
            angle_volumes.append(result[1])

        if is_rotation_split:
            scores, angles = np.zeros_like(score_volumes[0]) - 1., np.zeros_like(angle_volumes[0]) - 1.
            for s, a in zip(score_volumes, angle_volumes):
                angles = np.where(s > scores, a, angles)
                scores = np.where(s > scores, s, scores)
        else:
            scores, angles = (
                np.zeros(self.search_size, dtype=np.float32),
                np.zeros(self.search_size, dtype=np.float32)
            )
            for x, s, a in zip(self.sub_jobs, score_volumes, angle_volumes):
                [vsize_x, vsize_y, vsize_z] = self.search_size
                [size_x, size_y, size_z] = x.split_size
                sub_start = x.sub_start
                start = x.whole_start

                step_size_x = min(vsize_x - sub_start[0], size_x)
                step_size_y = min(vsize_y - sub_start[1], size_y)
                step_size_z = min(vsize_z - sub_start[2], size_z)

                sub_scores = s[sub_start[0]: step_size_x, sub_start[1], step_size_y, sub_start[2], step_size_z]
                sub_angles = a[sub_start[0]: step_size_x, sub_start[1], step_size_y, sub_start[2], step_size_z]

                scores[
                    start[0] - self.search_origin[0]: start[0] - self.search_origin[0] + sub_scores.shape[0],
                    start[1] - self.search_origin[1]: start[1] - self.search_origin[1] + sub_scores.shape[1],
                    start[2] - self.search_origin[2]: start[2] - self.search_origin[2] + sub_scores.shape[2]
                ] = sub_scores
                angles[
                    start[0] - self.search_origin[0]: start[0] - self.search_origin[0] + sub_scores.shape[0],
                    start[1] - self.search_origin[1]: start[1] - self.search_origin[1] + sub_scores.shape[1],
                    start[2] - self.search_origin[2]: start[2] - self.search_origin[2] + sub_scores.shape[2]
                ] = sub_angles
        return scores, angles

    def start_job(self, gpu_id: int):
        # load (sub)volume, template, mask
        search_volume = self.tomogram.data[:].transpose()  # index with subregion

        # load wedge, apply to volume and make wedge for template

        # load rotation search
        angle_ids = list(range(self.start_slice, self.n_rotations, self.steps_slice))
        angle_list = load_angle_list(self.rotation_file)[angle_ids]

        tm_thread = TemplateMatchingGPU(
            job_id=self.job_key,
            device_id=gpu_id,
            volume=volume,
            template=template,
            mask=mask,
            angle_list=angle_list,
            angle_ids=angle_ids,
            mask_is_spherical=self.mask_is_spherical
        )
        tm_thread.run()
        while tm_thread.active:
            time.sleep(0.5)

        if tm_thread.completed:
            self.finished_run = True
        else:
            raise TMJobError(f'job {self.job_key} stopped but dit not complete')

        score_volume, angle_volume = tm_thread.plan.scores.get(), tm_thread.plan.angles.get()
        del tm_thread

        # write out scores and angles volume
        self.output_dir.joinpath(f'scores_{self.job_key}.mrc')
        self.output_dir.joinpath(f'angles_{self.job_key}.mrc')
