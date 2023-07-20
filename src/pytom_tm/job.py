import pathlib
import copy
import time
from pytom_tm.angles import AVAILABLE_ROTATIONAL_SAMPLING, load_angle_list
from pytom_tm.structures import TemplateMatchingGPU


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
        self.tomogram = tomogram
        self.reference = template
        self.mask = mask
        self.mask_is_spherical = mask_is_spherical
        self.output_dir = output_dir

        # Region of the tomogram to search
        self.subregion = subregion

        # Rotations to search
        self.rotation_file = AVAILABLE_ROTATIONAL_SAMPLING[angle_increment][0]
        self.n_rotations = AVAILABLE_ROTATIONAL_SAMPLING[angle_increment][1]
        self.start_slice = 0
        self.steps_slice = 1

        # Job details
        self.job_key = job_key
        self.leader = None  # the job that spawned this job
        self.sub_jobs = []  # if this job had no sub jobs it should be executed

    def split_rotation_search(self, n):
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

    def copy(self):
        return copy.deepcopy(self)

    def start_job(self, gpu_id):
        # load (sub)volume, template, mask
        ...

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
        score_volume, angle_volume = tm_thread.plan.scores.get(), tm_thread.plan.angles.get()
        del tm_thread
