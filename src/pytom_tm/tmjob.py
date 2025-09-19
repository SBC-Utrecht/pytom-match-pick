from __future__ import annotations
from packaging import version
import pathlib
import warnings
import copy
import itertools as itt
import numpy as np
import numpy.typing as npt
import json
import logging
from scipy.fft import next_fast_len, rfftn, irfftn
from pytom_tm.angles import get_angle_list
from pytom_tm.matching import TemplateMatchingGPU
from pytom_tm.weights import (
    create_wedge,
    power_spectrum_profile,
    profile_to_weighting,
    create_gaussian_band_pass,
)
from pytom_tm.io import read_mrc_meta_data, read_mrc, write_mrc, UnequalSpacingError
from pytom_tm.json import CustomJSONEncoder, CustomJSONDecoder
from pytom_tm.dataclass import CtfData, TiltSeriesMetaData, RelionTiltSeriesMetaData
from pytom_tm import __version__ as PYTOM_TM_VERSION


def load_json_to_tmjob(
    file_name: pathlib.Path, load_for_extraction: bool = True
) -> TMJob:
    """Load a previous job that was stored with TMJob.write_to_json().

    Parameters
    ----------
    file_name: pathlib.Path
        path to TMJob json file
    load_for_extraction: bool, default True
        whether a finished job is loaded form disk for extraction, default is True as
        this function is currently only called for pytom_extract_candidates and
        pytom_estimate_roc which run on previously finished jobs

    Returns
    -------
    job: TMJob
        initialized TMJob
    """
    with open(file_name) as fstream:
        data = json.load(fstream, cls=CustomJSONDecoder)

    # Deal with loading ctfdata that was a stored before 0.11.0
    ctf_data = data.get("ctf_data", None)
    if ctf_data is not None and type(ctf_data[0]) is dict:
        data["ctf_data"] = [CtfData(**ctf) for ctf in ctf_data]

    # Deal with loading ts_metadata that was stored before 0.11.0
    if "ts_metadata" not in data:
        kw_dict = {
            "tilt_angles": data["tilt_angles"],
            "ctf_data": data["ctf_data"],
            "dose_accumulation": data.get("dose_accumulation", None),
            "defocus_handedness": data.get("defocus_handedness", 0),
            "per_tilt_weighting": data.get("tilt_weighting"),
        }
        if "metadata" in data:
            # relion5 metadata instead
            kw_dict["binning"] = data["metadata"]["relion5_binning"]
            kw_dict["tilt_series_pixel_size"] = data["metadata"]["relion5_ts_ps"]
            cls = RelionTiltSeriesMetaData
        else:
            cls = TiltSeriesMetaData
        data["ts_metadata"] = cls(**kw_dict)

    # wrangle dtypes
    output_dtype = data.get("output_dtype", "float32")
    output_dtype = np.dtype(output_dtype)

    job = TMJob(
        data["job_key"],
        data["log_level"],
        pathlib.Path(data["tomogram"]),
        pathlib.Path(data["template"]),
        pathlib.Path(data["mask"]),
        pathlib.Path(data["output_dir"]),
        angle_increment=data.get("angle_increment", data["rotation_file"]),
        mask_is_spherical=data["mask_is_spherical"],
        voxel_size=data["voxel_size"],
        ts_metadata=data["ts_metadata"],
        search_x=data["search_x"],
        search_y=data["search_y"],
        search_z=data["search_z"],
        # Use 'get' for backwards compatibility
        tomogram_mask=data.get("tomogram_mask", None),
        low_pass=data["low_pass"],
        # Use 'get' for backwards compatibility
        high_pass=data.get("high_pass", None),
        whiten_spectrum=data.get("whiten_spectrum", False),
        rotational_symmetry=data.get("rotational_symmetry", 1),
        # if version number is not in the .json, it must be 0.3.0 or older
        pytom_tm_version_number=data.get("pytom_tm_version_number", "0.3.0"),
        job_loaded_for_extraction=load_for_extraction,
        particle_diameter=data.get("particle_diameter", None),
        random_phase_correction=data.get("random_phase_correction", False),
        rng_seed=data.get("rng_seed", 321),
        output_dtype=output_dtype,
    )
    job.whole_start = data["whole_start"]
    job.sub_start = data["sub_start"]
    job.sub_step = data["sub_step"]
    job.n_rotations = data["n_rotations"]
    job.start_slice = data["start_slice"]
    job.steps_slice = data["steps_slice"]
    job.job_stats = data["job_stats"]
    return job


def get_defocus_offsets(
    patch_center_x: float,
    patch_center_z: float,
    tilt_angles: list[float, ...],
    angles_in_degrees: bool = True,
    invert_handedness: bool = False,
) -> npt.NDArray[float]:
    """Calculate the defocus offsets for a subvolume
    based on the tilt geometry.

    I used the definition from Pyle & Zianetti (https://doi.org/10.1042/BCJ20200715)
    for the default setting of the defocus handedness. It assumes the defocus
    increases for positive tilt angles on the right side of the sample (positive X
    coordinate relative to the center).

    The offset is calculated as follows:
        z_offset = z_center * np.cos(tilt_angle) + x_center * np.sin(tilt_angle)

    Parameters
    ----------
    patch_center_x: float
        x center of subvolume relative to tomogram center
    patch_center_z: float
        z center of subvolume relative to tomogram center
    tilt_angles: list[float, ...]
        list of tilt angles
    angles_in_degrees: bool, default True
        whether tilt angles are in degrees or radians
    invert_handedness: bool, default False
        invert defocus handedness geometry

    Returns
    -------
    z_offsets: npt.NDArray[float]
        an array of defocus offsets for each tilt angle
    """
    n_tilts = len(tilt_angles)
    x_centers = np.full(n_tilts, patch_center_x)
    z_centers = np.full(n_tilts, patch_center_z)
    ta_array = np.array(tilt_angles)
    if angles_in_degrees:
        ta_array = np.deg2rad(ta_array)
    if invert_handedness:
        ta_array *= -1
    z_offsets = z_centers * np.cos(ta_array) + x_centers * np.sin(ta_array)
    return z_offsets


def _determine_1D_fft_splits(
    length: int, splits: int, overhang: int = 0
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Split a 1D length into FFT optimal sizes taking into account overhangs

    Parameters
    ----------
    length: int
        Total 1D length to split
    splits: int
        Number of splits to make
    overhang: int, default 0
        Minimal overhang/overlap to consider between splits

    Returns
    -------
    output: list[tuple[tuple[int, int], tuple[int,int]]]:
        A list splits where every split gets two tuples meaning:
          [start, end) of the tomogram data in this split
          [start, end) of the unique datapoints in this split
        If a datapoint exists in 2 splits, we add it as unique to
        either the split with the most data or the left one if both
        splits have the same size
    """
    # Everything in this code assumes default slices of [x,y) so including x but
    # excluding y
    data_slices = []
    valid_data_slices = []
    sub_len = []
    # if single split return early
    if splits == 1:
        return [((0, length), (0, length))]
    if splits > length:
        warnings.warn(
            "More splits than pixels were asked, will default to 1 split per pixel",
            RuntimeWarning,
        )
        splits = length
    # Ceil to guarantee that we map the whole length with enough buffer
    min_len = int(np.ceil(length / splits)) + overhang
    min_unique_len = min_len - overhang
    no_overhang_left = 0
    while True:
        if no_overhang_left == 0:
            # Treat first split specially, only right overhang
            split_length = next_fast_len(min_len)
            data_slices.append((0, split_length))
            valid_data_slices.append((0, split_length - overhang))
            no_overhang_left = split_length - overhang
            sub_len.append(split_length)
        elif no_overhang_left + min_unique_len >= length:
            # Last slice, only overhang to the left
            split_length = next_fast_len(min_len)
            data_slices.append((length - split_length, length))
            valid_data_slices.append((length - split_length + overhang, length))
            sub_len.append(split_length)
            break
        else:
            # Any other slice
            split_length = next_fast_len(min_len + overhang)
            left_overhang = (split_length - min_unique_len) // 2
            temp_left = no_overhang_left - left_overhang
            temp_right = temp_left + split_length
            data_slices.append((temp_left, temp_right))
            valid_data_slices.append((temp_left + overhang, temp_right - overhang))
            sub_len.append(split_length)
            no_overhang_left = temp_right - overhang
        if split_length <= 0 or no_overhang_left <= 0:
            raise RuntimeError(
                f"Cannot generate legal splits for {length=}, {splits=}, {overhang=}"
            )
    # Now generate the best unique data point,
    # we always pick the bigest data subset or the left one
    unique_data = []
    unique_left = 0
    for i, (len1, len2) in enumerate(itt.pairwise(sub_len)):
        if len1 >= len2:
            right = valid_data_slices[i][1]
        else:
            right = valid_data_slices[i + 1][0]
        unique_data.append((unique_left, right))
        unique_left = right
    # Add final part
    if unique_left != length:
        unique_data.append((unique_left, length))
    # Make sure unique slices are unique and within valid data
    last_right = 0
    for (vd_left, vd_right), (ud_left, ud_right) in zip(valid_data_slices, unique_data):
        if (
            ud_left < vd_left
            or ud_right > vd_right
            or ud_right > length
            or ud_left != last_right
        ):  # pragma: no cover
            raise RuntimeError(
                f"We produced inconsistent slices for {length=}, {splits=}, {overhang=}"
            )
        last_right = ud_right
    return list(zip(data_slices, unique_data))


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
        ts_metadata: TiltSeriesMetaData,
        voxel_size: float | None = None,
        angle_increment: str | float | None = None,
        mask_is_spherical: bool = True,
        search_x: list[int, int] | None = None,
        search_y: list[int, int] | None = None,
        search_z: list[int, int] | None = None,
        tomogram_mask: pathlib.Path | None = None,
        low_pass: float | None = None,
        high_pass: float | None = None,
        whiten_spectrum: bool = False,
        rotational_symmetry: int = 1,
        pytom_tm_version_number: str = PYTOM_TM_VERSION,
        job_loaded_for_extraction: bool = False,
        particle_diameter: float | None = None,
        random_phase_correction: bool = False,
        rng_seed: int = 321,
        output_dtype: np.dtype = np.float32,
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
        voxel_size: float | None, default None
            voxel size of tomogram and template (in A) if not provided will be read from
            template/tomogram MRCs
        ts_metadata: TiltSeriesMetaData | None, default None
            tilt series metadata of the tomogram
        angle_increment: Union[str, float]; default 7.00
            angular increment of template search
        mask_is_spherical: bool, default True
            whether template mask is spherical, reduces computation complexity
        search_x: Optional[list[int, int]], default None
            restrict tomogram search region along the x-axis
        search_y: Optional[list[int, int]], default None
            restrict tomogram search region along the y-axis
        search_z: Optional[list[int, int]], default None
            restrict tomogram search region along the z-axis
        tomogram_mask: Optional[pathlib.Path], default None
            when volume splitting tomograms, only subjobs where any(mask > 0) will be
            generated
        low_pass: Optional[float], default None
            optional low-pass filter (resolution in A) to apply to tomogram and template
        high_pass: Optional[float], default None
            optional high-pass filter (resolution in A) to apply to tomogram and
            template
        whiten_spectrum: bool, default False
            whether to apply spectrum whitening
        rotational_symmetry: int, default 1
            specify a rotational symmetry around the z-axis, is only valid if the
            symmetry axis of the template is aligned with the z-axis
        pytom_tm_version_number: str, default current version
            a string with the version number of pytom_tm for backward compatibility
        job_loaded_for_extraction: bool, default False
            flag to set for finished template matching jobs that are loaded back for
            extraction, it prevents recomputation of the whitening filter
        particle_diameter: Optional[float], default None
            particle diameter (in Angstrom) to calculate angular search
        random_phase_correction: bool, default False,
            run matching with a phase randomized version of the template to correct
            scores for noise
        rng_seed: int, default 321
            set a seed for the rng for phase randomization
        output_dtype: np.dtype, default np.float32
            output score volume dtype, options are np.float32 and np.float16
        """
        self.mask = mask
        self.mask_is_spherical = mask_is_spherical
        self.output_dir = output_dir

        self.tomogram = tomogram
        self.template = template
        self.tomo_id = self.tomogram.stem

        self.ts_metadata = ts_metadata

        try:
            meta_data_tomo = read_mrc_meta_data(self.tomogram)
        except UnequalSpacingError:  # add information that the problem is the tomogram
            raise UnequalSpacingError(
                "Input tomogram voxel spacing is not equal in each dimension!"
            )

        try:
            meta_data_template = read_mrc_meta_data(self.template)
        except UnequalSpacingError:  # add information that the problem is the template
            raise UnequalSpacingError(
                "Input template voxel spacing is not equal in each dimension!"
            )

        try:
            meta_data_mask = read_mrc_meta_data(self.mask)
        except UnequalSpacingError:  # add information that the problem is the template
            raise UnequalSpacingError(
                "Input mask voxel spacing is not equal in each dimension!"
            )

        self.tomo_shape = meta_data_tomo["shape"]
        self.template_shape = meta_data_template["shape"]
        self.mask_shape = meta_data_mask["shape"]

        if self.template_shape != self.mask_shape:
            raise ValueError(
                "Template and mask have a different shape in pixels. "
                f"Found template shape: {self.template_shape}. "
                f"Found maks shape: {self.mask_shape}"
            )

        if voxel_size is not None:
            if voxel_size <= 0:
                raise ValueError(
                    "Invalid voxel size provided, smaller or equal to zero."
                )
            self.voxel_size = voxel_size
            if (  # allow tiny numerical differences that are not relevant for
                # template matching
                round(self.voxel_size, 3) != round(meta_data_tomo["voxel_size"], 3)
                or round(self.voxel_size, 3)
                != round(meta_data_template["voxel_size"], 3)
            ):
                logging.debug(
                    f"provided {self.voxel_size} tomogram "
                    f"{meta_data_tomo['voxel_size']} "
                    f"template {meta_data_template['voxel_size']}"
                )
                print(
                    "WARNING: Provided voxel size does not match voxel size annotated "
                    "in tomogram/template mrc."
                )
        elif (
            round(meta_data_tomo["voxel_size"], 3)
            == round(meta_data_template["voxel_size"], 3)
            and meta_data_tomo["voxel_size"] > 0
        ):
            self.voxel_size = round(meta_data_tomo["voxel_size"], 3)
        else:
            raise ValueError(
                "Voxel size could not be assigned, either a mismatch between tomogram "
                "and template or annotated as 0."
            )

        search_origin = [
            x[0] if x is not None else 0 for x in (search_x, search_y, search_z)
        ]
        # Check if tomogram origin is valid
        if all([0 <= x < y for x, y in zip(search_origin, self.tomo_shape)]):
            self.search_origin = search_origin
        else:
            raise ValueError("Invalid input provided for search origin of tomogram.")

        # if end not valid raise and error
        search_end = []
        for x, s in zip([search_x, search_y, search_z], self.tomo_shape):
            if x is not None:
                if not x[1] <= s:
                    raise ValueError(
                        "One of search end indices is larger than the tomogram "
                        "dimension."
                    )
                search_end.append(x[1])
            else:
                search_end.append(s)
        self.search_size = [
            end - start for end, start in zip(search_end, self.search_origin)
        ]

        logging.debug(f"origin, size = {self.search_origin}, {self.search_size}")
        self.tomogram_mask = tomogram_mask
        if tomogram_mask is not None:
            temp = read_mrc(tomogram_mask)
            if temp.shape != self.tomo_shape:
                raise ValueError(
                    "Tomogram mask does not have the same number of pixels as the "
                    "tomogram.\n"
                    f"Tomogram mask shape: {temp.shape}, "
                    f"tomogram shape: {self.tomo_shape}"
                )
            if np.all(temp <= 0):
                raise ValueError(
                    "No values larger than 0 found in the tomogram mask: "
                    f"{tomogram_mask}"
                )

        self.whole_start = None
        # For the main job these are always [0,0,0] and self.search_size, for sub_jobs
        # these will differ from self.search_origin and self.search_size. The main job
        # only uses them to calculate the search_volume_roi for statistics. Sub jobs
        # also use these to extract and place back the relevant region in the master
        # job.
        self.sub_start, self.sub_step = [0, 0, 0], self.search_size.copy()

        # Rotation parameters
        self.start_slice = 0
        self.steps_slice = 1
        self.rotational_symmetry = rotational_symmetry
        self.particle_diameter = particle_diameter
        # calculate increment from particle diameter
        if angle_increment is None:
            if particle_diameter is not None:
                max_res = max(
                    2 * self.voxel_size, low_pass if low_pass is not None else 0
                )
                angle_increment = np.rad2deg(max_res / particle_diameter)
            else:
                angle_increment = 7.0
        self.rotation_file = angle_increment
        try:
            angle_list = get_angle_list(
                angle_increment,
                sort_angles=False,
                symmetry=rotational_symmetry,
                # This log_level is different from self.log_level that is
                # assigned later. The TMJob.log_level refers to the user provided
                # logging setting, while the log_level here is to control the output
                # of the job during candidate extraction/template matching.
                log_level=logging.DEBUG if job_loaded_for_extraction else logging.INFO,
            )
        except ValueError:
            raise TMJobError("Invalid angular search provided.")

        self.n_rotations = len(angle_list)

        # set the band-pass resolution shells
        self.low_pass = low_pass
        self.high_pass = high_pass

        self.whiten_spectrum = whiten_spectrum
        self.whitening_filter = self.output_dir.joinpath(
            f"{self.tomo_id}_whitening_filter.npy"
        )
        if self.whiten_spectrum and not job_loaded_for_extraction:
            logging.info("Estimating whitening filter...")
            weights = 1 / np.sqrt(
                power_spectrum_profile(
                    read_mrc(self.tomogram)[
                        self.search_origin[0] : self.search_origin[0]
                        + self.search_size[0],
                        self.search_origin[1] : self.search_origin[1]
                        + self.search_size[1],
                        self.search_origin[2] : self.search_origin[2]
                        + self.search_size[2],
                    ]
                )
            )
            weights /= weights.max()  # scale to 1
            np.save(self.whitening_filter, weights)

        # phase randomization options
        self.random_phase_correction = random_phase_correction
        self.rng_seed = rng_seed

        # Job details
        self.job_key = job_key
        self.leader = None  # the job that spawned this job
        self.sub_jobs = []  # if this job had no sub jobs it should be executed

        # dict to keep track of job statistics
        self.job_stats = None

        self.log_level = log_level

        # version number of the job
        self.pytom_tm_version_number = pytom_tm_version_number

        # output dtype
        self.output_dtype = output_dtype

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

        Note: This has to be run from the same cwd as where `self` was initiated
              otherwise the path resolving doesn't make sense

        Parameters
        ----------
        file_name: pathlib.Path
            path to the output file
        """
        d = self.__dict__.copy()
        d.pop("sub_jobs")
        d.pop("search_origin")
        d.pop("search_size")
        d["search_x"] = [
            self.search_origin[0],
            self.search_origin[0] + self.search_size[0],
        ]
        d["search_y"] = [
            self.search_origin[1],
            self.search_origin[1] + self.search_size[1],
        ]
        d["search_z"] = [
            self.search_origin[2],
            self.search_origin[2] + self.search_size[2],
        ]
        for key, value in d.items():
            if isinstance(value, pathlib.Path):
                d[key] = str(value.absolute())
        # wrangle dtype conversion
        d["output_dtype"] = str(np.dtype(d["output_dtype"]))
        with open(file_name, "w") as fstream:
            json.dump(d, fstream, indent=4, cls=CustomJSONEncoder)

    def split_rotation_search(self, n: int) -> list[TMJob, ...]:
        """Split the search into sub_jobs by dividing the rotations. Sub jobs will
        obtain the key self.job_key + str(i) when looping over range(n).

        Parameters
        ----------
        n: int
            number of times to split the angular search

        Returns
        -------
        sub_jobs: list[TMJob, ...]
            a list of TMJobs that were split from self, the jobs are also assigned as
            the TMJob.sub_jobs attribute
        """
        if len(self.sub_jobs) > 0:
            raise TMJobError(
                "Could not further split this job as it already has subjobs assigned!"
            )

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
        """Split the search into sub_jobs by dividing into subvolumes. Final number of
        subvolumes is obtained by multiplying all the split together, e.g. (2, 2, 1)
        results in 4 subvolumes. Sub jobs will obtain the key self.job_key + str(i) when
        looping over range(n).

        The sub jobs search area of the full tomogram is defined by:
        new_job.search_origin and new_job.search_size.
        They are used when loading the search volume from the full tomogram.

        The attribute new_job.whole_start defines how the volume maps back to the score
        volume of the parent job (which can be a different size than the tomogram when
        the search is restricted along x, y or z).

        Finally, new_job.sub_start and new_job.sub_step, extract the score and angle map
        without the template overhang from the subvolume.

        If self.tomogram_mask is set, we will skip subjobs where all(mask <= 0).

        Parameters
        ----------
        split: tuple[int, int, int]
            tuple that defines how many times the search volume should be split into
            subvolumes along each axis

        Returns
        -------
        sub_jobs: list[TMJob, ...]
            a list of TMJobs that were split from self, the jobs are also assigned as
            the TMJob.sub_jobs attribute
        """
        if len(self.sub_jobs) > 0:
            raise TMJobError(
                "Could not further split this job as it already has subjobs assigned!"
            )

        search_size = self.search_size
        if self.tomogram_mask is not None:
            # This should have some positve values after the check in the __init__
            tomogram_mask = read_mrc(self.tomogram_mask)
        else:
            tomogram_mask = None
        # shape of template for overhang
        overhang = self.template_shape
        # use overhang//2 (+1 for odd sizes)
        overhang = tuple(sum(divmod(o, 2)) for o in overhang)

        x_splits = _determine_1D_fft_splits(search_size[0], split[0], overhang[0])
        y_splits = _determine_1D_fft_splits(search_size[1], split[1], overhang[1])
        z_splits = _determine_1D_fft_splits(search_size[2], split[2], overhang[2])

        sub_jobs = []
        for i, data_3D in enumerate(itt.product(x_splits, y_splits, z_splits)):
            # each data point for each dim is slice(left, right) of the search space
            # and slice(left,right) of the unique data point in the search space
            # Look at the comments in the new_job.attribute for the meaning of each
            # attribute

            search_origin = tuple(
                data_3D[d][0][0] + self.search_origin[d] for d in range(3)
            )
            search_size = tuple(dim_data[0][1] - dim_data[0][0] for dim_data in data_3D)
            whole_start = tuple(dim_data[1][0] for dim_data in data_3D)
            sub_start = tuple(dim_data[1][0] - dim_data[0][0] for dim_data in data_3D)
            sub_step = tuple(dim_data[1][1] - dim_data[1][0] for dim_data in data_3D)

            # check if this contains any of the unique data points are where
            # tomo_mask > 0
            if tomogram_mask is not None:
                slices = [
                    slice(origin, origin + step)
                    for origin, step in zip(whole_start, sub_step)
                ]
                if np.all(tomogram_mask[*slices] <= 0):
                    # No non-masked unique data-points, skipping
                    continue
            new_job = self.copy()
            new_job.leader = self.job_key
            new_job.job_key = self.job_key + str(i)

            # search origin with respect to the complete tomogram
            new_job.search_origin = search_origin
            # search size TODO: should be combined with the origin into slices
            new_job.search_size = search_size

            # whole start is the start of the unique data within the complete searched
            # array
            new_job.whole_start = whole_start
            # sub_start is where the unique data starts inside the split array
            new_job.sub_start = sub_start
            # sub_step is the step of unique data inside the split array.
            # TODO: should be slices instead
            new_job.sub_step = sub_step
            sub_jobs.append(new_job)

        self.sub_jobs = sub_jobs

        return self.sub_jobs

    def merge_sub_jobs(
        self, stats: list[dict, ...] | None = None
    ) -> tuple[npt.NDArray[float], npt.NDArray[float]]:
        """Merge the sub jobs present in self.sub_jobs together to create the final
        output score and angle maps.

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
                self.output_dir.joinpath(f"{self.tomo_id}_scores_{self.job_key}.mrc"),
                self.output_dir.joinpath(f"{self.tomo_id}_angles_{self.job_key}.mrc"),
            )
            result = (read_mrc(score_file), read_mrc(angle_file))
            (score_file.unlink(), angle_file.unlink())
            return result

        if stats is not None:
            search_space = sum([s["search_space"] for s in stats])
            variance = sum([s["variance"] for s in stats]) / len(stats)
            self.job_stats = {
                "search_space": search_space,
                "variance": variance,
                "std": np.sqrt(variance),
            }

        is_subvolume_split = np.all(
            np.array([x.start_slice for x in self.sub_jobs]) == 0
        )

        score_volumes, angle_volumes = [], []
        for x in self.sub_jobs:
            result = x.merge_sub_jobs()
            score_volumes.append(result[0])
            angle_volumes.append(result[1])

        if not is_subvolume_split:
            scores, angles = (
                np.zeros_like(score_volumes[0]) - 1.0,
                np.zeros_like(angle_volumes[0]) - 1.0,
            )
            for s, a in zip(score_volumes, angle_volumes):
                angles = np.where(s > scores, a, angles)
                # prevents race condition due to slicing
                angles = np.where(s == scores, np.minimum(a, angles), angles)
                scores = np.where(s > scores, s, scores)
        else:
            scores, angles = (
                np.zeros(self.search_size, dtype=np.float32),
                np.zeros(self.search_size, dtype=np.float32),
            )
            for job, s, a in zip(self.sub_jobs, score_volumes, angle_volumes):
                sub_scores = s[
                    job.sub_start[0] : job.sub_start[0] + job.sub_step[0],
                    job.sub_start[1] : job.sub_start[1] + job.sub_step[1],
                    job.sub_start[2] : job.sub_start[2] + job.sub_step[2],
                ]
                sub_angles = a[
                    job.sub_start[0] : job.sub_start[0] + job.sub_step[0],
                    job.sub_start[1] : job.sub_start[1] + job.sub_step[1],
                    job.sub_start[2] : job.sub_start[2] + job.sub_step[2],
                ]
                # Then the corrected sub part needs to be placed back into the full
                # volume
                scores[
                    job.whole_start[0] : job.whole_start[0] + sub_scores.shape[0],
                    job.whole_start[1] : job.whole_start[1] + sub_scores.shape[1],
                    job.whole_start[2] : job.whole_start[2] + sub_scores.shape[2],
                ] = sub_scores
                angles[
                    job.whole_start[0] : job.whole_start[0] + sub_scores.shape[0],
                    job.whole_start[1] : job.whole_start[1] + sub_scores.shape[1],
                    job.whole_start[2] : job.whole_start[2] + sub_scores.shape[2],
                ] = sub_angles
        return scores.astype(self.output_dtype), angles

    def start_job(
        self, gpu_id: int, return_volumes: bool = False
    ) -> tuple[npt.NDArray[float], npt.NDArray[float]] | dict:
        """Run this template matching job on the specified GPU. Search statistics of the
        job will always be assigned to the self.job_stats.

        Parameters
        ----------
        gpu_id: int
            index of the GPU to run the job on
        return_volumes: bool, default False
            False (default) does not return volumes but instead writes them to disk, set
            to True to instead directly return the score and angle volumes

        Returns
        -------
        output: Union[tuple[npt.NDArray[float], npt.NDArray[float]], dict]
            when volumes are returned the output consists of two numpy arrays (score and
            angle map), when no volumes are returned the output consists of a dictionary
            with search statistics
        """
        # next fast fft len
        logging.debug(
            "Next fast fft shape: "
            f"{tuple([next_fast_len(s, real=True) for s in self.search_size])}"
        )
        search_volume = np.zeros(
            tuple([next_fast_len(s, real=True) for s in self.search_size]),
            dtype=np.float32,
        )

        # load the (sub)volume
        search_volume[
            : self.search_size[0], : self.search_size[1], : self.search_size[2]
        ] = np.ascontiguousarray(
            read_mrc(self.tomogram)[
                self.search_origin[0] : self.search_origin[0] + self.search_size[0],
                self.search_origin[1] : self.search_origin[1] + self.search_size[1],
                self.search_origin[2] : self.search_origin[2] + self.search_size[2],
            ]
        )

        # load template and mask
        template, mask = (read_mrc(self.template), read_mrc(self.mask))

        # init tomogram and template weighting
        tomo_filter, template_wedge = 1, 1
        # first generate bandpass filters
        if not (self.low_pass is None and self.high_pass is None):
            tomo_filter *= create_gaussian_band_pass(
                search_volume.shape, self.voxel_size, self.low_pass, self.high_pass
            ).astype(np.float32)
            template_wedge *= create_gaussian_band_pass(
                self.template_shape, self.voxel_size, self.low_pass, self.high_pass
            ).astype(np.float32)

        # then multiply with optional whitening filters
        if self.whiten_spectrum:
            tomo_filter *= profile_to_weighting(
                np.load(self.whitening_filter), search_volume.shape
            ).astype(np.float32)
            template_wedge *= profile_to_weighting(
                np.load(self.whitening_filter), self.template_shape
            ).astype(np.float32)

        # create wedge filters
        if (
            self.ts_metadata.per_tilt_weighting
            and self.ts_metadata.defocus_handedness != 0
        ):
            # adjust ctf parameters for this specific patch in the tomogram
            full_tomo_center = np.array(self.tomo_shape) / 2
            patch_center = np.array(self.search_origin) + np.array(self.search_size) / 2
            relative_patch_center_angstrom = (
                patch_center - full_tomo_center
            ) * self.voxel_size
            defocus_offsets = get_defocus_offsets(
                relative_patch_center_angstrom[0],  # x-coordinate
                relative_patch_center_angstrom[2],  # z-coordinate
                self.tilt_angles,
                angles_in_degrees=True,
                invert_handedness=self.defocus_handedness < 0,
            )
            # TODO: make sure this doesn't lead to weird race conditions
            for ctf, defocus_shift in zip(self.ctf_data, defocus_offsets):
                ctf.defocus = ctf.defocus + defocus_shift * 1e-10
            logging.debug(
                "Patch center (nr. of voxels): "
                f"{np.array_str(relative_patch_center_angstrom, precision=2)}"
            )
            logging.debug(
                "Defocus values (um): "
                f"{[round(ctf.defocus * 1e6, 2) for ctf in self.ctf_data]}",
            )

        # for the tomogram a binary wedge is generated to explicitly set the missing
        # wedge region to 0
        tomo_filter *= create_wedge(
            search_volume.shape,
            self.ts_metadata,
            self.voxel_size,
            cut_off_radius=1.0,
            angles_in_degrees=True,
            per_tilt_weighting=False,
        ).astype(np.float32)
        # for the template a binary or per-tilt-weighted wedge is generated
        # depending on the options
        template_wedge *= create_wedge(
            self.template_shape,
            self.ts_metadata,
            self.voxel_size,
            cut_off_radius=1.0,
            angles_in_degrees=True,
        ).astype(np.float32)

        if logging.DEBUG >= logging.root.level:
            write_mrc(
                self.output_dir.joinpath("template_psf.mrc"),
                template_wedge,
                self.voxel_size,
            )
            write_mrc(
                self.output_dir.joinpath("template_convolved.mrc"),
                irfftn(rfftn(template) * template_wedge, s=template.shape),
                self.voxel_size,
            )

        # apply the optional band pass and whitening filter to the search region
        search_volume = np.real(
            irfftn(rfftn(search_volume) * tomo_filter, s=search_volume.shape)
        )

        # load rotation search
        angle_ids = list(range(self.start_slice, self.n_rotations, self.steps_slice))
        angle_list = get_angle_list(
            self.rotation_file,
            sort_angles=version.parse(self.pytom_tm_version_number)
            > version.parse("0.3.0"),
            symmetry=self.rotational_symmetry,
        )

        angle_list = angle_list[
            slice(self.start_slice, self.n_rotations, self.steps_slice)
        ]

        # slices for relevant part for job statistics
        search_volume_roi = (
            slice(self.sub_start[0], self.sub_start[0] + self.sub_step[0]),
            slice(self.sub_start[1], self.sub_start[1] + self.sub_step[1]),
            slice(self.sub_start[2], self.sub_start[2] + self.sub_step[2]),
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
            stats_roi=search_volume_roi,
            noise_correction=self.random_phase_correction,
            rng_seed=self.rng_seed,
        )
        results = tm.run()
        score_volume = results[0][
            : self.search_size[0], : self.search_size[1], : self.search_size[2]
        ]
        angle_volume = results[1][
            : self.search_size[0], : self.search_size[1], : self.search_size[2]
        ]
        self.job_stats = results[2]

        del tm  # delete the template matching plan

        # cast to correct dtype
        score_volume = score_volume.astype(self.output_dtype)
        angle_volume = angle_volume

        if return_volumes:
            return score_volume, angle_volume
        else:  # otherwise write them out with job_key
            write_mrc(
                self.output_dir.joinpath(f"{self.tomo_id}_scores_{self.job_key}.mrc"),
                score_volume,
                self.voxel_size,
            )
            write_mrc(
                self.output_dir.joinpath(f"{self.tomo_id}_angles_{self.job_key}.mrc"),
                angle_volume,
                self.voxel_size,
            )
            return self.job_stats
