from __future__ import annotations
from importlib import metadata
from packaging import version
import pathlib
import os
import copy
import numpy as np
import numpy.typing as npt
import json
import logging
from typing import Optional, Union
from functools import reduce
from scipy.fft import next_fast_len, rfftn, irfftn
from pytom_tm.angles import load_angle_list
from pytom_tm.matching import TemplateMatchingGPU
from pytom_tm.weights import create_wedge, power_spectrum_profile, profile_to_weighting, create_gaussian_band_pass
from pytom_tm.io import read_mrc_meta_data, read_mrc, write_mrc, UnequalSpacingError
from pytom_tm import __version__ as PYTOM_TM_VERSION


def load_json_to_tmjob(file_name: pathlib.Path, load_for_extraction: bool = True) -> TMJob:
    """Load a previous job that was stored with TMJob.write_to_json().

    Parameters
    ----------
    file_name: pathlib.Path
        path to TMJob json file
    load_for_extraction: bool, default True
        whether a finished job is loaded form disk for extraction, default is True as this function is currently only
        called for pytom_extract_candidates and pytom_estimate_roc which run on previously finished jobs

    Returns
    -------
    job: TMJob
        initialized TMJob
    """
    with open(file_name, 'r') as fstream:
        data = json.load(fstream)

    job = TMJob(
        data['job_key'],
        data['log_level'],
        pathlib.Path(data['tomogram']),
        pathlib.Path(data['template']),
        pathlib.Path(data['mask']),
        pathlib.Path(data['output_dir']),
        angle_increment=data.get('angle_increment', data['rotation_file']),
        mask_is_spherical=data['mask_is_spherical'],
        tilt_angles=data['tilt_angles'],
        tilt_weighting=data['tilt_weighting'],
        search_x=data['search_x'],
        search_y=data['search_y'],
        search_z=data['search_z'],
        voxel_size=data['voxel_size'],
        low_pass=data['low_pass'],
        # Use 'get' for backwards compatibility
        high_pass=data.get('high_pass', None),
        dose_accumulation=data.get('dose_accumulation', None),
        ctf_data=data.get('ctf_data', None),
        whiten_spectrum=data.get('whiten_spectrum', False),
        rotational_symmetry=data.get('rotational_symmetry', 1),
        # if version number is not in the .json, it must be 0.3.0 or older
        pytom_tm_version_number=data.get('pytom_tm_version_number', '0.3.0'),
        job_loaded_for_extraction=load_for_extraction,
    )
    # if the file originates from an old version set the phase shift for compatibility
    if (
        version.parse(job.pytom_tm_version_number) < version.parse('0.6.1') and
        job.ctf_data is not None
    ):
        for tilt in job.ctf_data:
            tilt['phase_shift_deg'] = .0
    job.whole_start = data['whole_start']
    job.sub_start = data['sub_start']
    job.sub_step = data['sub_step']
    job.n_rotations = data['n_rotations']
    job.start_slice = data['start_slice']
    job.steps_slice = data['steps_slice']
    job.job_stats = data['job_stats']
    return job


class TMJobError(Exception):
    """TMJob Exception with provided message."""
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


class TMJob:
    def __init__(
            self,
            job_key: str,
            log_level: int,
            tomogram: pathlib.Path,
            template: pathlib.Path,
            mask: pathlib.Path,
            output_dir: pathlib.Path,
            angle_increment: Union[str, float] = 7.00,
            mask_is_spherical: bool = True,
            tilt_angles: Optional[list[float, ...]] = None,
            tilt_weighting: bool = False,
            search_x: Optional[list[int, int]] = None,
            search_y: Optional[list[int, int]] = None,
            search_z: Optional[list[int, int]] = None,
            voxel_size: Optional[float] = None,
            low_pass: Optional[float] = None,
            high_pass: Optional[float] = None,
            dose_accumulation: Optional[list[float, ...]] = None,
            ctf_data: Optional[list[dict, ...]] = None,
            whiten_spectrum: bool = False,
            rotational_symmetry: int = 1,
            pytom_tm_version_number: str = PYTOM_TM_VERSION,
            job_loaded_for_extraction: bool = False,
    ):
        """
        Parameters
        ----------
        job_key: str
            job identifier
        log_level: int
            log level for logging module
        tomogram: pathlib.Path
            path to tomogram MRC
        template: pathlib.Path
            path to template MRC
        mask: pathlib.Path
            path to mask MRC
        output_dir: pathlib.Path
            path to output directory
        angle_increment: Union[str, float]; default 7.00
            angular increment of template search
        mask_is_spherical: bool, default True
            whether template mask is spherical, reduces computation complexity
        tilt_angles: Optional[list[float, ...]], default None
            tilt angles of tilt-series used to reconstruct tomogram, if only two floats will be used to generate a
            continuous wedge model
        tilt_weighting: bool, default False
            use advanced tilt weighting options, can be supplemented with CTF parameters and accumulated dose
        search_x: Optional[list[int, int]], default None
            restrict tomogram search region along the x-axis
        search_y: Optional[list[int, int]], default None
            restrict tomogram search region along the y-axis
        search_z: Optional[list[int, int]], default None
            restrict tomogram search region along the z-axis
        voxel_size: Optional[float], default None
            voxel size of tomogram and template (in A) if not provided will be read from template/tomogram MRCs
        low_pass: Optional[float], default None
            optional low-pass filter (resolution in A) to apply to tomogram and template
        high_pass: Optional[float], default None
            optional high-pass filter (resolution in A) to apply to tomogram and template
        dose_accumulation: Optional[list[float, ...]], default None
            list with dose accumulation per tilt image
        ctf_data: Optional[list[dict, ...]], default None
            list of dictionaries with CTF parameters per tilt image, see pytom_tm.weight.create_ctf() for parameter
            definition
        whiten_spectrum: bool, default False
            whether to apply spectrum whitening
        rotational_symmetry: int, default 1
            specify a rotational symmetry around the z-axis, is only valid if the symmetry axis of the template is
            aligned with the z-axis
        pytom_tm_version_number: str, default current version
            a string with the version number of pytom_tm for backward compatibility
        job_loaded_for_extraction: bool, default False
            flag to set for finished template matching jobs that are loaded back for extraction, it prevents
            recalculation of the whitening filter which is unnecessary at this stage
        """
        self.mask = mask
        self.mask_is_spherical = mask_is_spherical
        self.output_dir = output_dir

        self.tomogram = tomogram
        self.template = template
        self.tomo_id = self.tomogram.stem

        try:
            meta_data_tomo = read_mrc_meta_data(self.tomogram)
        except UnequalSpacingError:  # add information that the problem is the tomogram
            raise UnequalSpacingError('Input tomogram voxel spacing is not equal in each dimension!')

        try:
            meta_data_template = read_mrc_meta_data(self.template)
        except UnequalSpacingError:  # add information that the problem is the template
            raise UnequalSpacingError('Input template voxel spacing is not equal in each dimension!')

        self.tomo_shape = meta_data_tomo['shape']
        self.template_shape = meta_data_template['shape']

        if voxel_size is not None:
            if voxel_size <= 0:
                raise ValueError('Invalid voxel size provided, smaller or equal to zero.')
            self.voxel_size = voxel_size
            if (  # allow tiny numerical differences that are not relevant for template matching
                    round(self.voxel_size, 3) != round(meta_data_tomo['voxel_size'], 3) or
                    round(self.voxel_size, 3) != round(meta_data_template['voxel_size'], 3)
            ):
                logging.debug(f"provided {self.voxel_size} tomogram {meta_data_tomo['voxel_size']} "
                              f"template {meta_data_template['voxel_size']}")
                print('WARNING: Provided voxel size does not match voxel size annotated in tomogram/template mrc.')
        elif (round(meta_data_tomo['voxel_size'], 3) == round(meta_data_template['voxel_size'], 3) and
              meta_data_tomo['voxel_size'] > 0):
            self.voxel_size = round(meta_data_tomo['voxel_size'], 3)
        else:
            raise ValueError('Voxel size could not be assigned, either a mismatch between tomogram and template or'
                             ' annotated as 0.')

        search_origin = [x[0] if x is not None else 0 for x in (search_x, search_y, search_z)]
        # Check if tomogram origin is valid
        if all([0 <= x < y for x, y in zip(search_origin, self.tomo_shape)]):
            self.search_origin = search_origin
        else:
            raise ValueError('Invalid input provided for search origin of tomogram.')

        # if end not valid raise and error
        search_end = []
        for x, s in zip([search_x, search_y, search_z], self.tomo_shape):
            if x is not None:
                if not x[1] <= s:
                    raise ValueError('One of search end indices is larger than the tomogram dimension.')
                search_end.append(x[1])
            else:
                search_end.append(s)
        self.search_size = [end - start for end, start in zip(search_end, self.search_origin)]

        logging.debug(f'origin, size = {self.search_origin}, {self.search_size}')

        self.whole_start = None
        # For the main job these are always [0,0,0] and self.search_size, for sub_jobs these will differ from
        # self.search_origin and self.search_size. The main job only uses them to calculate the search_volume_roi for
        # statistics. Sub jobs also use these to extract and place back the relevant region in the master job.
        self.sub_start, self.sub_step = [0, 0, 0], self.search_size.copy()

        # Rotation parameters
        self.start_slice = 0
        self.steps_slice = 1
        self.rotational_symmetry = rotational_symmetry
        self.angle_increment = angle_increment
        succesfull_angle = False
        try:
            angle_increment = float(angle_increment)
            angle_is_float = True
        except ValueError:
            angle_is_float = False
        if angle_is_float:
            # Generate angle list on the fly
            logging.info("Will generate an angle list with a maximum increment of {angle_increment}")
            self.rotation_file = None
            self.angle_list = angle_to_angle_list(angle_increment, sort_angles=False, log=True)
            self.n_rotaions = len(self.angle_list)
            succesfull_angle = True
        elif isinstance(angle_increment, (str, os.PathLike)):
            possible_file_path = pathlib.Path(angle_increment)
            if possible_file_path.exists() and possible_file_path.suffix == '.txt':
                logging.info(
                    'Custom file provided for the angular search. Checking if it can be read...'
                )
                # load_angle_list will throw an error if it does not encounter three inputs per line
                self.n_rotations = len(load_angle_list(possible_file_path, sort_angles=False))
                self.rotation_file = possible_file_path
                succesfull_angle = True

        if not succesfull_angle:
            raise TMJobError('Invalid angular search provided.')

        # missing wedge
        self.tilt_angles = tilt_angles
        self.tilt_weighting = tilt_weighting
        # set the band-pass resolution shells
        self.low_pass = low_pass
        self.high_pass = high_pass

        # set dose and ctf
        self.dose_accumulation = dose_accumulation
        self.ctf_data = ctf_data
        self.whiten_spectrum = whiten_spectrum
        self.whitening_filter = self.output_dir.joinpath(f'{self.tomo_id}_whitening_filter.npy')
        if self.whiten_spectrum and not job_loaded_for_extraction:
            logging.info('Estimating whitening filter...')
            weights = 1 / np.sqrt(
                power_spectrum_profile(
                    read_mrc(self.tomogram)[
                        self.search_origin[0]: self.search_origin[0] + self.search_size[0],
                        self.search_origin[1]: self.search_origin[1] + self.search_size[1],
                        self.search_origin[2]: self.search_origin[2] + self.search_size[2]
                    ]
                )
            )
            weights /= weights.max()  # scale to 1
            np.save(self.whitening_filter, weights)

        # Job details
        self.job_key = job_key
        self.leader = None  # the job that spawned this job
        self.sub_jobs = []  # if this job had no sub jobs it should be executed

        # dict to keep track of job statistics
        self.job_stats = None

        self.log_level = log_level

        # version number of the job
        self.pytom_tm_version_number = pytom_tm_version_number

    def copy(self) -> TMJob:
        """Create a copy of the TMJob

        Returns
        -------
        job: TMJob
            copied TMJob instance
        """
        return copy.deepcopy(self)

    def write_to_json(self, file_name: pathlib.Path) -> None:
        """Write job to .json file.

        Parameters
        ----------
        file_name: pathlib.Path
            path to the output file
        """
        d = self.__dict__.copy()
        d.pop('sub_jobs')
        d.pop('search_origin')
        d.pop('search_size')
        d['search_x'] = [self.search_origin[0], self.search_origin[0] + self.search_size[0]]
        d['search_y'] = [self.search_origin[1], self.search_origin[1] + self.search_size[1]]
        d['search_z'] = [self.search_origin[2], self.search_origin[2] + self.search_size[2]]
        for key, value in d.items():
            if isinstance(value, pathlib.Path):
                d[key] = str(value)
        with open(file_name, 'w') as fstream:
            json.dump(d, fstream, indent=4)

    def split_rotation_search(self, n: int) -> list[TMJob, ...]:
        """Split the search into sub_jobs by dividing the rotations. Sub jobs will obtain the key
        self.job_key + str(i) when looping over range(n).

        Parameters
        ----------
        n: int
            number of times to split the angular search

        Returns
        -------
        sub_jobs: list[TMJob, ...]
            a list of TMJobs that were split from self, the jobs are also assigned as the TMJob.sub_jobs attribute
        """
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

    def split_volume_search(self, split: tuple[int, int, int]) -> list[TMJob, ...]:
        """Split the search into sub_jobs by dividing into subvolumes. Final number of subvolumes is obtained by
        multiplying all the split together, e.g. (2, 2, 1) results in 4 subvolumes. Sub jobs will obtain the key
        self.job_key + str(i) when looping over range(n).

        The sub jobs search area of the full tomogram is defined by: new_job.search_origin and new_job.search_size.
        They are used when loading the search volume from the full tomogram.

        The attribute new_job.whole_start defines how the volume maps back to the score volume of the parent job
        (which can be a different size than the tomogram when the search is restricted along x, y or z).

        Finally, new_job.sub_start and new_job.sub_step, extract the score and angle map without the template
        overhang from the subvolume.

        Parameters
        ----------
        split: tuple[int, int, int]
            tuple that defines how many times the search volume should be split into subvolumes along each axis

        Returns
        -------
        sub_jobs: list[TMJob, ...]
            a list of TMJobs that were split from self, the jobs are also assigned as the TMJob.sub_jobs attribute
        """
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
                -(template_shape[0] // 2) + self.search_origin[0] + inc_x * split_size[0],
                -(template_shape[1] // 2) + self.search_origin[1] + inc_y * split_size[1],
                -(template_shape[2] // 2) + self.search_origin[2] + inc_z * split_size[2]
            ])

            _end = tuple([_start[j] + _size[j] for j in range(len(_start))])

            # if start location is smaller than search origin we need to reset it to search origin
            start = tuple([o if s < o else s for s, o in zip(_start, self.search_origin)])
            # if end location is larger than origin + search size it needs to be set to origin + search size
            end = tuple([s + o if e > s + o else e for e, s, o in zip(_end, self.search_size, self.search_origin)])
            size = tuple([end[j] - start[j] for j in range(len(start))])

            # for reassembling the result
            whole_start = [0, ] * 3  # whole_start and sub_start should also be job attributes
            sub_start = [0, ] * 3
            sub_step = list(split_size)
            if start[0] != self.search_origin[0]:
                whole_start[0] = start[0] + template_shape[0] // 2 - self.search_origin[0]
                sub_start[0] = template_shape[0] // 2
            if start[1] != self.search_origin[1]:
                whole_start[1] = start[1] + template_shape[1] // 2 - self.search_origin[1]
                sub_start[1] = template_shape[1] // 2
            if start[2] != self.search_origin[2]:
                whole_start[2] = start[2] + template_shape[2] // 2 - self.search_origin[2]
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

    def merge_sub_jobs(self, stats: Optional[list[dict, ...]] = None) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
        """Merge the sub jobs present in self.sub_jobs together to create the final output score and angle maps.

        Parameters
        ----------
        stats: Optional[list[dict, ...]], default None
            optional list of sub job statistics to merge together

        Returns
        -------
        output: tuple[npt.NDArray[float], npt.NDArray[float]]
            the merged score and angle maps from the subjobs
        """
        if len(self.sub_jobs) == 0:
            # read the volumes, remove them and return them
            score_file, angle_file = (
                self.output_dir.joinpath(f'{self.tomo_id}_scores_{self.job_key}.mrc'),
                self.output_dir.joinpath(f'{self.tomo_id}_angles_{self.job_key}.mrc')
            )
            result = (
                read_mrc(score_file),
                read_mrc(angle_file)
            )
            (score_file.unlink(), angle_file.unlink())
            return result

        if stats is not None:
            search_space = sum([s['search_space'] for s in stats])
            variance = sum([s['variance'] for s in stats]) / len(stats)
            self.job_stats = {
                'search_space': search_space,
                'variance': variance,
                'std': np.sqrt(variance)
            }

        is_subvolume_split = np.all(np.array([x.start_slice for x in self.sub_jobs]) == 0)

        score_volumes, angle_volumes = [], []
        for x in self.sub_jobs:
            result = x.merge_sub_jobs()
            score_volumes.append(result[0])
            angle_volumes.append(result[1])

        if not is_subvolume_split:
            scores, angles = np.zeros_like(score_volumes[0]) - 1., np.zeros_like(angle_volumes[0]) - 1.
            for s, a in zip(score_volumes, angle_volumes):
                angles = np.where(s > scores, a, angles)
                # prevents race condition due to slicing
                angles = np.where(s == scores, np.minimum(a, angles), angles)
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

    def start_job(
            self,
            gpu_id: int,
            return_volumes: bool = False
    ) -> Union[tuple[npt.NDArray[float], npt.NDArray[float]], dict]:
        """Run this template matching job on the specified GPU. Search statistics of the job will always be assigned
        to the self.job_stats.

        Parameters
        ----------
        gpu_id: int
            index of the GPU to run the job on
        return_volumes: bool, default False
            False (default) does not return volumes but instead writes them to disk, set to True to instead directly
            return the score and angle volumes

        Returns
        -------
        output: Union[tuple[npt.NDArray[float], npt.NDArray[float]], dict]
            when volumes are returned the output consists of two numpy arrays (score and angle map), when no volumes
            are returned the output consists of a dictionary with search statistics
        """
        # next fast fft len
        logging.debug(f'Next fast fft shape: {tuple([next_fast_len(s, real=True) for s in self.search_size])}')
        search_volume = np.zeros(tuple([next_fast_len(s, real=True) for s in self.search_size]), dtype=np.float32)

        # load the (sub)volume
        search_volume[:self.search_size[0],
                      :self.search_size[1],
                      :self.search_size[2]] = np.ascontiguousarray(
            read_mrc(self.tomogram)[self.search_origin[0]: self.search_origin[0] + self.search_size[0],
                                    self.search_origin[1]: self.search_origin[1] + self.search_size[1],
                                    self.search_origin[2]: self.search_origin[2] + self.search_size[2]]
        )

        # load template and mask
        template, mask = (
            read_mrc(self.template),
            read_mrc(self.mask)
        )
        # apply mask directly to prevent any wedge convolution with weird edges
        template *= mask

        # init tomogram and template weighting
        tomo_filter, template_wedge = 1, 1
        # first generate bandpass filters
        if not (self.low_pass is None and self.high_pass is None):
            tomo_filter *= create_gaussian_band_pass(
                    search_volume.shape,
                    self.voxel_size,
                    self.low_pass,
                    self.high_pass
            ).astype(np.float32)
            template_wedge *= create_gaussian_band_pass(
                self.template_shape,
                self.voxel_size,
                self.low_pass,
                self.high_pass
            ).astype(np.float32)

        # then multiply with optional whitening filters
        if self.whiten_spectrum:
            tomo_filter *= profile_to_weighting(
                np.load(self.whitening_filter),
                search_volume.shape
            ).astype(np.float32)
            template_wedge *= profile_to_weighting(
                np.load(self.whitening_filter),
                self.template_shape
            ).astype(np.float32)

        # if tilt angles are provided we can create wedge filters
        if self.tilt_angles is not None:
            # for the tomogram a binary wedge is generated to explicitly set the missing wedge region to 0
            tomo_filter *= create_wedge(
                search_volume.shape,
                self.tilt_angles,
                self.voxel_size,
                cut_off_radius=1.,
                angles_in_degrees=True,
                tilt_weighting=False
            ).astype(np.float32)
            # for the template a binary or per-tilt-weighted wedge is generated depending on the options
            template_wedge *= create_wedge(
                self.template_shape,
                self.tilt_angles,
                self.voxel_size,
                cut_off_radius=1.,
                angles_in_degrees=True,
                tilt_weighting=self.tilt_weighting,
                accumulated_dose_per_tilt=self.dose_accumulation,
                ctf_params_per_tilt=self.ctf_data
            ).astype(np.float32)

        # apply the optional band pass and whitening filter to the search region
        search_volume = np.real(irfftn(rfftn(search_volume) * tomo_filter, s=search_volume.shape))

        # load rotation search
        angle_ids = list(range(
            self.start_slice,
            int(np.ceil(self.n_rotations / self.rotational_symmetry)),
            self.steps_slice
        ))
        if self.rotation_file is not None:
            angle_list = load_angle_list(
                    self.rotation_file,
                    sort_angles=version.parse(self.pytom_tm_version_number) > version.parse('0.3.0')
            )
        else:
            # pre-generated list
            angle_list = self.angle_list

        angle_list = angle_list[slice(
            self.start_slice,
            int(np.ceil(self.n_rotations / self.rotational_symmetry)),
            self.steps_slice
        )]

        # slices for relevant part for job statistics
        search_volume_roi = (
            slice(self.sub_start[0], self.sub_start[0] + self.sub_step[0]),
            slice(self.sub_start[1], self.sub_start[1] + self.sub_step[1]),
            slice(self.sub_start[2], self.sub_start[2] + self.sub_step[2])
        )

        tm = TemplateMatchingGPU(
            job_id=self.job_key,
            device_id=gpu_id,
            volume=search_volume,
            template=template,
            mask=mask,
            angle_list=angle_list,
            angle_ids=angle_ids,
            mask_is_spherical=self.mask_is_spherical,
            wedge=template_wedge,
            stats_roi=search_volume_roi
        )
        results = tm.run()
        score_volume = results[0][:self.search_size[0], :self.search_size[1], :self.search_size[2]]
        angle_volume = results[1][:self.search_size[0], :self.search_size[1], :self.search_size[2]]
        self.job_stats = results[2]

        del tm  # delete the template matching plan

        if return_volumes:
            return score_volume, angle_volume
        else:  # otherwise write them out with job_key
            write_mrc(
                self.output_dir.joinpath(f'{self.tomo_id}_scores_{self.job_key}.mrc'),
                score_volume,
                self.voxel_size
            )
            write_mrc(
                self.output_dir.joinpath(f'{self.tomo_id}_angles_{self.job_key}.mrc'),
                angle_volume,
                self.voxel_size
            )
            return self.job_stats
