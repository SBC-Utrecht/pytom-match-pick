import numpy.typing as npt
import multiprocessing as mp
import logging
from multiprocessing.managers import BaseProxy
from contextlib import closing
from functools import reduce
from pytom_tm.tmjob import TMJob

try:  # new processes need to be spawned in order to set cupy to use the correct GPU
    mp.set_start_method('spawn')
except RuntimeError:
    pass


def start_single_job(job: TMJob, gpu_queue: BaseProxy) -> dict:  # function for starting each process
    gpu_id = gpu_queue.get()  # get gpu_id from the queue to run the job on

    # logging needs to be reset for spawned child processes
    logging.basicConfig(level=job.log_level)
    logging.debug('{}: got assigned GPU {}'.format(mp.current_process().ident, gpu_id))
    # print('{}: got assigned GPU {}'.format(mp.current_process().ident, gpu_id))

    # run the template matching, result contains search statistics (variance, std, search space)
    result = job.start_job(gpu_id, return_volumes=False)

    # the gpu is free to use again so place it back on the queue
    gpu_queue.put(gpu_id)

    return result


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

        rotation_split_factor = len(gpu_ids) // n_pieces

        if rotation_split_factor >= 2:  # we can split the rotation search for the subvolumes

            for j in main_job.split_volume_search(volume_splits):

                jobs.append(j.split_rotation_search(rotation_split_factor))

        else:  # only split the subvolume search

            jobs = main_job.split_volume_search(volume_splits)

    else:
        raise ValueError('Invalid number of pieces in split volume')

    # ================== Execution of jobs =========================
    if len(jobs) == 1:

        return main_job.start_job(gpu_ids[0], return_volumes=True)

    elif len(jobs) >= len(gpu_ids):

        with mp.Manager() as manager:
            # create the shared queue
            queue = manager.Queue()
            # add gpu_ids to the queue
            [queue.put(g) for g in gpu_ids]

            # map the pool onto all the subjobs; use closing to correctly close the Pool at the end
            with closing(mp.Pool(len(gpu_ids))) as pool:
                results = pool.starmap(start_single_job, zip(jobs, [queue, ] * len(jobs)))

        # merge split jobs; pass along the list of stats to annotate them in main_job
        return main_job.merge_sub_jobs(stats=results)

    else:
        ValueError('For some reason there are more gpu_ids than split job, this should never happen.')
