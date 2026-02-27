import numpy.typing as npt
import multiprocessing as mp
import logging
import queue
import time
import contextlib
from math import lcm
from multiprocessing.managers import BaseProxy
from functools import reduce
from pytom_tm.tmjob import TMJob
from pytom_tm.utils import mute_stdout_stderr

try:  # new processes need to be spawned in order to set cupy to use the correct GPU
    mp.set_start_method("spawn")
except RuntimeError:
    pass


def gpu_runner(
    gpu_id: int,
    task_queue: BaseProxy,
    result_queue: BaseProxy,
    log_level: int,
    unittest_mute: bool,
) -> None:
    """Start a GPU runner, each runner should be initialized to a
    multiprocessing.Process() and manage running jobs on a single GPU. Each runner will
    grab jobs from the task_queue and assign jobs to the result_queue once they finish.
    When the task_queue is empty the gpu_runner will stop.

    Parameters
    ----------
    gpu_id: int
        a GPU index to assign to the runner
    task_queue: mp.managers.BaseProxy
        shared queue from multiprocessing with jobs to run
    result_queue: mp.manager.BaseProxy
        shared queue from multiprocessing for finished jobs
    log_level: int
        log level for logging
    unittest_mute: Bool
        optional muting of runner to prevent unittests flooding the terminal, only use
        for development
    """
    if unittest_mute:
        mute_context = mute_stdout_stderr
    else:
        mute_context = contextlib.nullcontext
    with mute_context():
        logging.basicConfig(level=log_level)
        while True:
            try:
                job = task_queue.get_nowait()
                result_queue.put_nowait(job.start_job(gpu_id, return_volumes=False))
            except queue.Empty:
                break


def split_job_efficiently(
    job: TMJob, volume_splits: tuple[int, int, int], gpus: int
) -> list[TMJob, ...]:
    """Do volume and angle splits in such a way that we can fill any number of gpus
    with the same number of jobs for each gpu and the least amount of tomogram loading

    Parameters
    ----------
    job: pytom_tm.tmjob.TMJob
      a TM job to split into subjobs
    volume_splits: tuple[int, int, int]
      tuple of len 3 with number of splits in x, y, and z
    gpus: int
      number of gpus to split for
    """
    n_pieces = reduce(lambda x, y: x * y, volume_splits)
    least_total_jobs = lcm(n_pieces, gpus)
    n_angle_splits = least_total_jobs // n_pieces

    if n_angle_splits > job.n_rotations:
        # This should not happen for any sane setup,
        # as the number of angles is normally in the 1000s
        raise ValueError(
            "Can't fill the assigned nuber of GPUs!"
            f" number of assigned gpus: {gpus}, "
            f" maximum number of possible jobs: {n_pieces * job.n_rotations}"
        )

    # do volume splits
    if n_pieces > 1:
        jobs = job.split_volume_search(volume_splits)
    else:
        jobs = [job]

    # do angle splits
    if n_angle_splits > 1:
        out = []
        for j in jobs:
            out += j.split_rotation_search(n_angle_splits)
    else:
        out = jobs
    return out


def run_job_parallel(
    main_job: TMJob,
    volume_splits: tuple[int, int, int],
    gpu_ids: list[int, ...],
    unittest_mute: bool = False,
) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
    """Run a job in parallel over a single or multiple GPUs. If no volume_splits are
    given the search is parallelized by splitting the angular search. If volume_splits
    are provided the job will first be split by volume, if there are still more GPUs
    available, the subvolume jobs are still further split by angular search.

    Parameters
    ----------
    main_job: pytom_tm.tmjob.TMJob
        a TMJob object from pytom_tm that contains all data for a search
    volume_splits: tuple[int, int, int]
        tuple of len 3 with splits in x, y, and z
    gpu_ids: list[int, ...]
        list of gpu indices to spread jobs over
    unittest_mute: bool, default False
        boolean to mute spawned process terminal output, only set to True for
        unittesting

    Returns
    -------
    result: tuple[npt.NDArray[float], npt.NDArray[float]]
        the volumes with the LCCmax and angle ids
    """
    jobs = split_job_efficiently(main_job, volume_splits, len(gpu_ids))

    # ================== Execution of jobs =========================
    if len(jobs) == 1:
        return main_job.start_job(gpu_ids[0], return_volumes=True)

    elif len(jobs) >= len(gpu_ids):
        results = []

        with mp.Manager() as manager:
            task_queue = (
                manager.Queue()
            )  # the list of tasks where processes can get there next task from
            result_queue = (
                manager.Queue()
            )  # this will accumulate results from the processes

            [task_queue.put_nowait(j) for j in jobs]  # put all tasks

            # set the processes and start them!
            procs = [
                mp.Process(
                    target=gpu_runner,
                    args=(
                        g,
                        task_queue,
                        result_queue,
                        main_job.log_level,
                        unittest_mute,
                    ),
                )
                for g in gpu_ids
            ]
            [p.start() for p in procs]

            while True:
                while not result_queue.empty():
                    results.append(result_queue.get_nowait())

                if len(results) == len(
                    jobs
                ):  # its done if all the results from the spawn were send back
                    logging.debug("Got all results from the child processes")
                    break

                for p in procs:
                    # if one of the processes is no longer alive and has a failed exit
                    # we should error
                    if not p.is_alive() and p.exitcode != 0:  # to prevent a deadlock
                        [
                            x.terminate() for x in procs
                        ]  # kill all spawned processes if something broke
                        raise RuntimeError(
                            "One or more of the processes stopped unexpectedly."
                        )

                time.sleep(1)

            [p.join() for p in procs]
            logging.debug("Terminated the processes")

        # merge split jobs; pass along the list of stats to annotate them in main_job
        return main_job.merge_sub_jobs(stats=results)
