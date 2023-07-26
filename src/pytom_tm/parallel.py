import numpy.typing as npt
import multiprocessing as mp
from functools import reduce
from pytom_tm.tmjob import TMJob
from itertools import cycle

try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass


def start_single_job(job: TMJob, gpu_id: int) -> dict:  # function for starting each process
    print('{}: got assigned GPU {}'.format(mp.current_process().ident, gpu_id))
    return job.start_job(gpu_id, return_volumes=False)


def run_job_parallel(
        main_job: TMJob, volume_splits: tuple[int, int, int], gpu_ids: list[int, ...]
) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """
    @param main_job: a TMJob object from pytom_tm that contains all data for a search
    @param volume_splits: tuple of len 3 with splits in x, y, and z
    @param gpu_ids: list of gpu indices available for the job
    @return: the volumes with the LCCmax and angle ids
    """

    n_pieces = reduce(lambda x, y: x * y, volume_splits)
    jobs = []

    # =================== Splitting into subjobs ===============
    if n_pieces == 1:
        if len(gpu_ids) > 1:  # split rotation search

            jobs = main_job.split_rotation_search(len(gpu_ids))

        else:  # we just run the whole tomo on a single gpu

            jobs.append(main_job)

    elif n_pieces > 1:

        rotation_split_factor = len(gpu_ids) % n_pieces

        if rotation_split_factor >= 2:  # we can split the rotation search for the subvolumes

            for j in main_job.split_volume_search(volume_splits):

                jobs.append(j.split_rotation_search(rotation_split_factor))

        else:  # only split the subvolume search

            jobs = main_job.split_volume_search(volume_splits)

    else:
        raise ValueError('Invalid number of pieces in split volume')

    # ================== Execution of jobs =========================
    if len(jobs) == 1:

        score_volume, angle_volume = main_job.start_job(gpu_ids[0], return_volumes=True)

    elif len(jobs) >= len(gpu_ids):

        # map the pool onto all the subjobs
        with mp.Pool(len(gpu_ids)) as pool:  # TODO need to prevent new job starting on an already used GPU
            results = pool.starmap(start_single_job, zip(jobs, cycle(gpu_ids)))

        # merge split jobs; pass the stats from the sub job to annotate them in main_job
        score_volume, angle_volume = main_job.merge_sub_jobs(stats=results)

    else:
        ValueError('For some reason there are more gpu_ids than split job, this should never happen.')

    return score_volume, angle_volume

