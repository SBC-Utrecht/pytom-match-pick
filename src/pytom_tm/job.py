import pathlib
import copy
import time
import numpy as np
import numpy.typing as npt
import mrcfile
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
            subregion: list[tuple[int]],
            angle_increment: str
    ):
        self.tomogram = mrcfile.mmap(tomogram, mode='r+')
        self.template = mrcfile.mmap(template, mode='r+')
        self.mask = mrcfile.mmap(mask, mode='r+')
        self.mask_is_spherical = mask_is_spherical
        self.output_dir = output_dir

        # Region of the tomogram to search
        self.subregion = subregion
        # self.indices => for placing subvolume back in large volume?

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

        # check whether this job already has a subregion
        if self.subregion == [0, 0, 0, 0, 0, 0]:
            origin = [0, 0, 0]
            tomo_shape = self.tomogram.shape[::-1]  # tranposed
        else:
            origin = self.subregion[0:3]
            tomo_shape = self.subregion[3:6]

        # size of sub-volumes after splitting
        split_size = [x // y for x, y in zip(tomo_shape, split)]

        # shape of template for overhang
        template_shape = self.template.shape[::-1]  # tranposed

        # check if valid
        if any([x > y for x, y in zip(template_shape, split_size)]):
            raise RuntimeError("Not big enough volume to split!")

        # size of subvolume with overhang
        _size = [x + y for x, y in zip(template_shape, split_size)]

        number_of_pieces = reduce((lambda x, y: x * y), split)

        for i in range(number_of_pieces):
            stride_z, stride_y = split[0] * split[1], split[0]
            inc_z, inc_y, inc_x = i // stride_z, (i % stride_z) // stride_y, i % stride_y

            start = [
                -template_shape[0] // 2 + origin[0] + inc_x * split_size[0],
                -template_shape[1] // 2 + origin[1] + inc_y * split_size[1],
                -template_shape[2] // 2 + origin[2] + inc_z * split_size[2]
            ]

            end = [start[j] + _size[j] for j in range(len(start))]

            # adjust boundaries if they go outside the box
            start = [o if s < o else s for s, o in zip(start, origin)]
            end = [t + o if e > t + o else e for e, t, o in zip(end, tomo_shape, origin)]
            size = [end[j] - start[j] for j in range(len(start))]

            # for reassembling the result
            whole_start = start[:]  # whole_start and sub_start should also be job attributes
            sub_start = [0, 0, 0]
            if start[0] != origin[0]:
                whole_start[0] = start[0] + template_shape[0] // 2
                sub_start[0] = template_shape[0] // 2
            if start[1] != origin[1]:
                whole_start[1] = start[1] + template_shape[1] // 2
                sub_start[1] = template_shape[1] // 2
            if start[2] != origin[2]:
                whole_start[2] = start[2] + template_shape[2] // 2
                sub_start[2] = template_shape[2] // 2

            new_job = self.copy()
            new_job.leader = self.job_key
            new_job.job_id = self.job_key + str(i)
            new_job.subregion = [start[0], start[1], start[2], size[0], size[1], size[2]]
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
            scores, angles = np.zeros(self.subregion), np.zeros(self.subregion)  # or something like this
            for i, x in enumerate(self.sub_jobs):
                pass

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
