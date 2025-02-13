from packaging import version
import pandas as pd
import numpy as np
import numpy.typing as npt
import logging
import scipy.ndimage as ndimage
import pathlib
from pytom_tm.tmjob import TMJob
from pytom_tm.mask import spherical_mask
from pytom_tm.angles import get_angle_list, convert_euler
from pytom_tm.io import read_mrc
from scipy.special import erfcinv
from scipy.optimize import curve_fit
from tqdm import tqdm


plotting_available = False
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(context="talk", style="ticks")
    plotting_available = True
except ModuleNotFoundError:
    pass


def predict_tophat_mask(
    score_volume: npt.NDArray[float],
    output_path: pathlib.Path | None = None,
    n_false_positives: float = 1.0,
    create_plot: bool = True,
    tophat_connectivity: int = 1,
    bins: int = 50,
) -> npt.NDArray[bool]:
    """This function gets as input a score map and returns a peak mask as determined
    with a tophat transform.

    It does the following things:
     - calculate a tophat transform using scipy.ndimage.white_tophat() and a kernel
     ndimage.generate_binary_structure(rank=3, connectivity=1).
     - calculate a histogram of the transformed score map and take its log to focus more
        on small values
     - take second derivative of log(histogram) to find the region for fitting a
        Gaussian, where the second derivative switches from negative to positive the
        background noise likely breaks
     - use formula from excellent work of Rickgauer et al. (2017, eLife) which uses the
        error function to find the likelihood of false positives on the background
        Gaussian distribution: N**(-1) = erfc( theta / ( sigma * sqrt(2) ) ) / 2

    Parameters
    ----------
    score_volume: npt.NDArray[float]
        template matching score map
    output_path: Optional[pathlib.Path], default None
        if provided (and plotting is available), write a figure of the fit to the output
        folder
    n_false_positives: float, default 1.0
        number of false positive for error function cutoff calculation
    create_plot: bool, default True
        whether to plot the gaussian fit and cut-off estimation
    tophat_connectivity: int, default 1
        connectivity of binary structure
    bins: int, default 50
        number of bins to use for the historgram for estimation and plotting
    Returns
    -------
    peak_mask: npt.NDArray[bool]
        boolean mask with tophat filtered peak locations
    """
    if score_volume.dtype == np.float16:
        score_volume = score_volume.astype(np.float32)

    tophat = ndimage.white_tophat(
        score_volume,
        structure=ndimage.generate_binary_structure(
            rank=3, connectivity=tophat_connectivity
        ),
    )
    y, bins = np.histogram(tophat.flatten(), bins=bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    x_raw, y_raw = (
        bin_centers[2:],
        y[2:],
    )  # discard first two points because of over-representation of zeros

    # take second derivative and discard inaccurate boundary value (hence the [2:])
    with np.errstate(divide="ignore"):
        y_log = np.log(y_raw)
        y_log[np.isinf(y_log)] = 0
    second_derivative = np.gradient(np.gradient(y_log))[2:]
    m1 = second_derivative[:-1] < 0  # where the derivative is negative
    sign = np.sign(second_derivative[1:] * second_derivative[:-1])
    # if there was a 0 in second derivative, there are now two 0s in sign
    # replace the 0,0 with the next value and previous value instead
    sign = np.where(sign == 0, np.roll(sign, -1), sign)
    sign = np.where(sign == 0, np.roll(sign, 1), sign)
    m2 = sign == -1  # switches from neg. to pos. and vice versa
    idx = (
        int(np.argmax(m1 & m2))  # first switch from negative to positive
        + 2  # +2 for 2nd derivative discarded part
        + 1  # +1 to include last value
    )
    x_fit, y_fit = x_raw[:idx], y_raw[:idx]

    def gauss(x, amp, mu, sigma):  # gaussian for fitting
        return amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    def log_gauss(x, amp, mu, sigma):  # log of gaussian for fitting
        return np.log(gauss(x, amp, mu, sigma))

    guess = np.array(
        [y.max(), 0, score_volume.std()]
    )  # rough initial guess for amp, mu and sigma
    coeff = curve_fit(gauss, x_fit, y_fit, p0=guess)[
        0
    ]  # first fit regular gauss to get better guesses
    coeff_log = curve_fit(log_gauss, x_fit, np.log(y_fit), p0=coeff)[
        0
    ]  # now go for accurate fit to log of gauss
    search_space = coeff_log[0] / (coeff_log[2] * np.sqrt(2 * np.pi))
    # formula Rickgauer et al. (2017, eLife):
    # N**(-1) = erfc( theta / ( sigma * sqrt(2) ) ) / 2
    # we need to find theta (i.e. the cut-off)
    cut_off = (
        erfcinv((2 * n_false_positives) / search_space) * np.sqrt(2) * coeff_log[2]
        + coeff_log[1]
    )

    if plotting_available and output_path is not None and create_plot:
        fig, ax = plt.subplots()
        ax.scatter(x_raw, y_raw, label="tophat", marker="o")
        ax.plot(x_raw, gauss(x_raw, *coeff_log), label="pred", color="tab:orange")
        ax.axvline(cut_off, color="gray", linestyle="dashed", label="cut-off")
        ax.axvspan(x_fit[0], x_fit[-1], alpha=0.25, color="gray", label="fitted data")
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.1)
        ax.set_ylabel("Occurence")
        ax.set_xlabel("Tophat scores")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=600, transparent=False, bbox_inches="tight")

    peak_mask = tophat > cut_off

    return peak_mask


def extract_particles(
    job: TMJob,
    particle_radius_px: int,
    n_particles: int,
    cut_off: float | None = None,
    n_false_positives: float = 1.0,
    tomogram_mask_path: pathlib.Path | None = None,
    tophat_filter: bool = False,
    create_plot: bool = True,
    tophat_connectivity: int = 1,
    relion5_compat: bool = False,
    ignore_tomogram_mask: bool = False,
    tophat_bins: int = 50,
    plot_bins: int = 20,
) -> tuple[pd.DataFrame, list[float, ...]]:
    """
    Parameters
    ----------
    job: pytom_tm.tmjob.TMJob
        template matching job to annotate particles from
    particle_radius_px: int
        particle radius to remove peaks with after a score has been annotated
    n_particles: int
        maximum number of particles to extract
    cut_off: Optional[float]
        manually override the automated score cut-off estimation, value between 0 and 1
    n_false_positives: float, default 1.0
        tune the number of false positives to be included for automated error function
        cut-off estimation: should be a float > 0
    tomogram_mask_path: Optional[pathlib.Path]
        path to a tomographic binary mask for extraction, will override
        job.tomogram_mask
    tophat_filter: bool
        attempt to only select sharp peaks with the tophat filter
    create_plot: bool, default True
        flag for creating extraction plots
    tophat_connectivity: int, default 1
        connectivity of kernel for tophat transform
    relion5_compat: bool, default False
        relion5 compatibility writes coordinates relative to center and in Angstrom
        center definition should be: tomo_shape / 2 - 1
    ignore_tomogram_mask: bool, default False
        Debug option to force the code to ignore job.tomogram_mask and input mask.
        Allows for re-exctraction without rerunning the TM job
        (assuming the scores volume seems reasonable)
    tophat_bins: int, default 50
        The numbers of bins to use in the tophat histogram
    plot_bins: int, default 20
        The numbers of bins to use for the plot histogram of occurences

    Returns
    -------
    dataframe, scores: tuple[pd.DataFrame, list[float, ...]]
        dataframe with annotations that can be written out as a STAR file and a list of
        the selected scores
    """

    score_volume = read_mrc(job.output_dir.joinpath(f"{job.tomo_id}_scores.mrc"))
    angle_volume = read_mrc(job.output_dir.joinpath(f"{job.tomo_id}_angles.mrc"))
    angle_list = get_angle_list(
        job.rotation_file,
        sort_angles=version.parse(job.pytom_tm_version_number) > version.parse("0.3.0"),
        symmetry=job.rotational_symmetry,
    )

    if tophat_filter:  # constrain the extraction with a tophat filter
        predicted_peaks = predict_tophat_mask(
            score_volume,
            output_path=job.output_dir.joinpath(f"{job.tomo_id}_tophat_filter.svg"),
            n_false_positives=n_false_positives,
            create_plot=create_plot,
            tophat_connectivity=tophat_connectivity,
            bins=tophat_bins,
        )
        score_volume *= (
            predicted_peaks  # multiply with predicted peaks to keep only those
        )

    # apply tomogram mask if provided
    tomogram_mask = None
    if ignore_tomogram_mask:
        logging.warning("Ignoring tomogram mask")
    elif tomogram_mask_path is not None:
        tomogram_mask = read_mrc(tomogram_mask_path)
    elif job.tomogram_mask is not None:
        tomogram_mask = read_mrc(job.tomogram_mask)

    if tomogram_mask is not None:
        if tomogram_mask.shape != job.tomo_shape:
            raise ValueError(
                "Tomogram mask does not have the same number of pixels as the "
                f"tomogram.\n Tomogram mask shape: {tomogram_mask.shape}, "
                f"tomogram shape: {job.tomo_shape}"
            )
        slices = [
            slice(origin, origin + size)
            for origin, size in zip(job.search_origin, job.search_size)
        ]
        tomogram_mask = tomogram_mask[*slices]
        # mask should be larger than zero in regions of interest!
        score_volume[tomogram_mask <= 0] = 0

    # mask edges of score volume
    score_volume[0:particle_radius_px, :, :] = 0
    score_volume[:, 0:particle_radius_px, :] = 0
    score_volume[:, :, 0:particle_radius_px] = 0
    score_volume[-particle_radius_px:, :, :] = 0
    score_volume[:, -particle_radius_px:, :] = 0
    score_volume[:, :, -particle_radius_px:] = 0

    sigma = job.job_stats["std"]
    search_space = job.job_stats["search_space"]
    if cut_off is None:
        # formula Rickgauer et al. (2017, eLife):
        # N**(-1) = erfc( theta / ( sigma * sqrt(2) ) ) / 2
        # we need to find theta (i.e. the cut off)
        cut_off = erfcinv((2 * n_false_positives) / search_space) * np.sqrt(2) * sigma
        logging.info(f"cut off for particle extraction: {cut_off}")
    elif cut_off < 0:
        logging.warning(
            "Provided extraction score cut-off is smaller than 0. Changing to 0 as "
            "that is smallest allowed value."
        )
        cut_off = 0

    # mask for iteratively selecting peaks
    cut_box = int(particle_radius_px) * 2 + 1
    cut_mask = (spherical_mask(cut_box, particle_radius_px, cut_box // 2) == 0) * 1

    # data for star file
    pixel_size = job.voxel_size
    tomogram_id = job.tomo_id

    # remove relion5 reconstructed tomogram name as it messes with linking the tilt
    # series id when extracting subtomos
    if relion5_compat and tomogram_id.startswith("rec_"):
        tomogram_id = tomogram_id[4:]

    data = []
    scores = []

    for _ in tqdm(range(n_particles)):
        ind = np.unravel_index(score_volume.argmax(), score_volume.shape)

        lcc_max = score_volume[ind]

        if lcc_max <= cut_off:
            break

        scores.append(lcc_max)

        # assumed that here also need to multiply with -1
        rotation = convert_euler(
            [-1 * a for a in angle_list[int(angle_volume[ind])]],
            order_in="ZXZ",
            order_out="ZYZ",
            degrees_in=False,
            degrees_out=True,
        )

        location = [i + o for i, o in zip(job.search_origin, ind)]

        data.append(
            (
                location[0],  # CoordinateX
                location[1],  # CoordinateY
                location[2],  # CoordinateZ
                rotation[0],  # AngleRot
                rotation[1],  # AngleTilt
                rotation[2],  # AnglePsi
                lcc_max,  # LCCmax
                cut_off,  # Extraction cut off
                sigma,  # Add sigma of template matching search, LCCmax/sigma = SNR
                pixel_size,  # DetectorPixelSize
                tomogram_id,  # MicrographName
            )
        )

        # box out the particle
        start = [i - particle_radius_px for i in ind]
        score_volume[
            start[0] : start[0] + cut_box,
            start[1] : start[1] + cut_box,
            start[2] : start[2] + cut_box,
        ] *= cut_mask

    output = pd.DataFrame(
        data,
        columns=[
            "rlnCoordinateX",
            "rlnCoordinateY",
            "rlnCoordinateZ",
            "rlnAngleRot",
            "rlnAngleTilt",
            "rlnAnglePsi",
            "rlnLCCmax",
            "rlnCutOff",
            "rlnSearchStd",
            "rlnDetectorPixelSize",
            "rlnMicrographName",
        ],
    )

    if relion5_compat:
        dims = np.array(job.tomo_shape)
        center = dims / 2  # we approximate the center, the correct way is to use
        # relion5's unbinned dimensions and calculate: center = (ubinned_dims / 2) - 1
        output["rlnCoordinateX"], output["rlnCoordinateY"], output["rlnCoordinateZ"] = (
            (output["rlnCoordinateX"] - center[0]) * job.voxel_size,
            (output["rlnCoordinateY"] - center[1]) * job.voxel_size,
            (output["rlnCoordinateZ"] - center[2]) * job.voxel_size,
        )
        column_change = {
            "rlnCoordinateX": "rlnCenteredCoordinateXAngst",
            "rlnCoordinateY": "rlnCenteredCoordinateYAngst",
            "rlnCoordinateZ": "rlnCenteredCoordinateZAngst",
            "rlnMicrographName": "rlnTomoName",
            "rlnDetectorPixelSize": "rlnTomoTiltSeriesPixelSize",
        }
        output = output.rename(columns=column_change)

    if plotting_available and create_plot:
        y, bins = np.histogram(scores, bins=plot_bins)
        x = (bins[1:] + bins[:-1]) / 2
        hist_step = bins[1] - bins[0]
        # add more starting values for background Gaussian
        x_ext = np.concatenate((np.linspace(x[0] - 5 * hist_step, x[0], 10), x))
        # calculate amplitude of Gaussian and correct for histogram step size
        noise_amplitude = (search_space / (sigma * np.sqrt(2 * np.pi))) * hist_step
        y_background = noise_amplitude * np.exp(-(x_ext**2) / (2 * sigma**2))

        fig, ax = plt.subplots()
        ax.scatter(x, y, label="extracted", marker="o")
        ax.plot(x_ext, y_background, label="background", color="tab:orange")
        ax.axvline(cut_off, color="gray", linestyle="dashed", label="cut-off")
        ax.set_ylim(bottom=0, top=2 * max(y))
        ax.set_ylabel("Occurence")
        ax.set_xlabel(r"${LCC}_{max}$")
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            job.output_dir.joinpath(f"{job.tomo_id}_extraction_graph.svg"),
            dpi=600,
            transparent=False,
            bbox_inches="tight",
        )

    return output, scores
