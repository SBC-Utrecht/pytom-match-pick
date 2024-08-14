import numpy.typing as npt
import multiprocessing as mp
import logging
import queue
import time
import contextlib
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

        if (
            rotation_split_factor >= 2
        ):  # we can split the rotation search for the subvolumes
            for j in main_job.split_volume_search(volume_splits):
                jobs += j.split_rotation_search(rotation_split_factor)

        else:  # only split the subvolume search
            jobs = main_job.split_volume_search(volume_splits)

    else:
        raise ValueError("Invalid number of pieces in split volume")

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
                    if not p.is_alive() and p.exitcode == 1:  # to prevent a deadlock
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

    else:
        ValueError(
            "For some reason there are more gpu_ids than split job, this should never "
            "happen."
        )
