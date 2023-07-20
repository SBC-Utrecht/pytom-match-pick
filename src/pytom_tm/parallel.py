from functools import reduce
import math


def parallel_start(job, volume_splits, gpu_ids):
    """

    :param job:
    :param volume_splits: tuple of len 3 with splits in x, y, and z
    :param gpu_ids: list of gpu indices available for the job
    :return:
    """
    pieces = reduce(lambda x, y: x * y, volume_splits)
    jobs = []

    # Split the tomograms into subvolumes

    if len(gpu_ids) > len(jobs):  # We need to queue jobs by splitting the rotation search
        splits = math.ceil(len(gpu_ids) / pieces)
        rotation_slices = [slice(x, len(job.n_rotations), splits) for x in range(splits)]
        # Apply the rotation slice to each subjob
    elif True:  # We need to queue subvolumes for the gpu's
        # jobs ...
        pass


def splitVolumes(self, job, splitX, splitY, splitZ, verbose=True):
    """
    splitVolumes: Split the job in the "volume" way (sequentially)
    @param job: job
    @type job: L{pytom.localization.peak_job.PeakJob}
    @param splitX: split part along the x dimension
    @type splitX: integer
    @param splitY: split part along the y dimension
    @type splitY: integer
    @param splitZ: split part along the z dimension
    @type splitZ: integer
    """
    # check if the split is feasible
    if job.volume.subregion == [0, 0, 0, 0, 0, 0]:
        v = job.volume.getVolume()
        origin = [0 ,0 ,0]
        vsizeX = v.sizeX(); vsizeY = v.sizeY(); vsizeZ = v.sizeZ()
    else:
        origin = job.volume.subregion[0:3]
        vsizeX = job.volume.subregion[3]
        vsizeY = job.volume.subregion[4]
        vsizeZ = job.volume.subregion[5]

    sizeX = vsizeX // splitX; sizeY = vsizeY // splitY; sizeZ = vsizeZ // splitZ
    r = job.reference.getVolume()
    rsizeX = r.sizeX(); rsizeY = r.sizeY(); rsizeZ = r.sizeZ()
    if rsizeX >sizeX or rsizeY >sizeY or rsizeZ >sizeZ:
        raise RuntimeError("Not big enough volume to split!")

    # initialize the jobInfo structure
    originalJobID = job.jobID

    # read the target volume, calculate the respective subregion
    from pytom.localization.peak_job import PeakJob
    from pytom.localization.structures import Volume
    _start = [-rsize X/ / 2 +origin[0] ,-rsize Y/ / 2 +origin[1] ,-rsize Z/ / 2 +origin[2]]
    _size = [size X +rsizeX, size Y +rsizeY, size Z +rsizeZ]

    numPieces = split X *split Y *splitZ
    totalMem = self.members
    numMemEach = totalMe m/ /numPieces
    targetID = self.mpi_id

    for i in range(numPieces):
        strideZ = split X *splitY; strideY = splitX
        incZ = i// strideZ;
        incY = (i % strideZ) // strideY;
        incX = i % strideY
        _start = [-rsizeX // 2 + origin[0] + incX * sizeX, -rsizeY // 2 + origin[1] + incY * sizeY,
                  -rsizeZ // 2 + origin[2] + incZ * sizeZ]

        start = _start[:]
        end = [start[j] + _size[j] for j in range(len(start))]

        if start[0] < origin[0]:
            start[0] = origin[0]
        if start[1] < origin[1]:
            start[1] = origin[1]
        if start[2] < origin[2]:
            start[2] = origin[2]
        if end[0] > vsizeX + origin[0]:
            end[0] = vsizeX + origin[0]
        if end[1] > vsizeY + origin[1]:
            end[1] = vsizeY + origin[1]
        if end[2] > vsizeZ + origin[2]:
            end[2] = vsizeZ + origin[2]

        size = [end[j] - start[j] for j in range(len(start))]

        # make sure that the last dimension is not odd
        # if size[2]%2 == 1:
        #     size[2] = size[2] - 1
        #     end[2] = end[2] - 1

        # for reassembling the result
        whole_start = start[:]
        sub_start = [0, 0, 0]
        if start[0] != origin[0]:
            whole_start[0] = start[0] + rsizeX // 2
            sub_start[0] = rsizeX // 2
        if start[1] != origin[1]:
            whole_start[1] = start[1] + rsizeY // 2
            sub_start[1] = rsizeY // 2
        if start[2] != origin[2]:
            whole_start[2] = start[2] + rsizeZ // 2
            sub_start[2] = rsizeZ // 2

        subJobID = job.jobID + i + 1
        subVol = Volume(job.volume.getFilename(),
                        [start[0], start[1], start[2], size[0], size[1], size[2]])
        if i == 0:
            numMem = totalMem - (numPieces - 1) * numMemEach
        else:
            numMem = numMemEach
        subJob = PeakJob(subVol, job.reference, job.mask, job.wedge, job.rotations, job.score, subJobID, numMem,
                         self.dstDir, job.bandpass)

        from pytom.localization.peak_job import JobInfo
        info = JobInfo(subJob.jobID, originalJobID, "Vol")
        info.sub_start = sub_start
        info.whole_start = whole_start
        info.originalSize = [vsizeX, vsizeY, vsizeZ]
        info.splitSize = [sizeX, sizeY, sizeZ]
        info.origin = origin
        self.jobInfoPool[subJobID] = info

        if targetID == self.mpi_id:
            self.setJob(subJob)
            if self.members > 1:
                self.splitAngles(subJob, verbose)
        else:
            if verbose == True:
                print(self.name + ' : send part of the volume to ' + str(targetID))
            subJob.send(self.mpi_id, targetID)

        targetID = targetID + numMem
        self.jobInfoPool["numJobsV"] = self.jobInfoPool["numJobsV"] + 1