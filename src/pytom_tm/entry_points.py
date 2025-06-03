import argparse
import sys
import pathlib
import logging
import numpy as np
import starfile
from pytom_tm.extract import extract_particles
from pytom_tm.io import (
    LargerThanZero,
    write_mrc,
    read_mrc_meta_data,
    read_mrc,
    CheckFileExists,
    ParseLogging,
    CheckDirExists,
    ParseSearch,
    ParseTiltAngles,
    ParseDoseFile,
    ParseDefocus,
    BetweenZeroAndOne,
    ParseGPUIndices,
    parse_relion5_star_data,
    parse_warp_xml_data,
)
from pytom_tm.tmjob import load_json_to_tmjob
from os import urandom


def _parse_argv(argv=None):
    if argv is None:
        return sys.argv[1:]
    return argv


def pytom_create_mask(argv=None):
    from pytom_tm.mask import spherical_mask, ellipsoidal_mask

    argv = _parse_argv(argv)

    # entry_point strings cannot use '\n' characters as this will break the website
    # snippet that displays the CLI help message
    # ---8<--- [start:create_mask_usage]

    parser = argparse.ArgumentParser(
        prog="pytom_create_mask.py",
        description="Create a mask for template matching. "
        "-- Marten Chaillet (@McHaillet)",
    )
    parser.add_argument(
        "-b",
        "--box-size",
        type=int,
        required=True,
        action=LargerThanZero,
        help="Shape of square box for the mask.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=pathlib.Path,
        required=False,
        help="Provide path to write output, needs to end in .mrc ."
        "If not provided file is written to current directory in the following format: "
        "./mask_b[box_size]px_r[radius]px.mrc ",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        required=False,
        default=1.0,
        action=LargerThanZero,
        help="Provide a voxel size to annotate the MRC (currently not used for any "
        "mask calculation).",
    )
    parser.add_argument(
        "-r",
        "--radius",
        type=float,
        required=True,
        action=LargerThanZero,
        help="Radius of the spherical mask in number of pixels. In case minor1 and "
        "minor2 are provided, this will be the radius of the ellipsoidal mask along "
        "the x-axis.",
    )
    parser.add_argument(
        "--radius-minor1",
        type=float,
        required=False,
        action=LargerThanZero,
        help="Radius of the ellipsoidal mask along the y-axis in number of pixels.",
    )
    parser.add_argument(
        "--radius-minor2",
        type=float,
        required=False,
        action=LargerThanZero,
        help="Radius of the ellipsoidal mask along the z-axis in number of pixels.",
    )
    parser.add_argument(
        "-s",
        "--sigma",
        type=float,
        required=False,
        action=LargerThanZero,
        help="Sigma of gaussian drop-off around the mask edges in number of pixels. "
        "Values in the range from 0.5-1.0 are usually sufficient for tomograms with "
        "20A-10A voxel sizes.",
    )

    # ---8<--- [end:create_mask_usage]

    argv = _parse_argv(argv)
    args = parser.parse_args(argv)

    # generate mask
    if args.radius_minor1 is not None and args.radius_minor2 is not None:
        mask = ellipsoidal_mask(
            args.box_size,
            args.radius,
            args.radius_minor1,
            args.radius_minor2,
            smooth=args.sigma,
        )
    else:
        mask = spherical_mask(args.box_size, args.radius, smooth=args.sigma)

    # write to disk
    output_path = (
        args.output_file
        if args.output_file is not None
        else (pathlib.Path(f"mask_b{args.box_size}px_r{args.radius}px.mrc"))
    )
    write_mrc(output_path, mask, args.voxel_size)


def pytom_create_template(argv=None):
    from pytom_tm.template import generate_template_from_map

    argv = _parse_argv(argv)

    # entry_point strings cannot use '\n' characters as this will break the website
    # snippet that displays the CLI help message
    # ---8<--- [start:create_template_usage]

    parser = argparse.ArgumentParser(
        prog="pytom_create_template.py",
        description="Generate template from MRC density. "
        "-- Marten Chaillet (@McHaillet)",
    )
    parser.add_argument(
        "-i",
        "--input-map",
        type=pathlib.Path,
        required=True,
        action=CheckFileExists,
        help="Map to generate template from; MRC file.",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=pathlib.Path,
        required=False,
        help="Provide path to write output, needs to end in .mrc . If not provided "
        "file is written to current directory in the following format: "
        "template_{input_map.stem}_{voxel_size}A.mrc",
    )
    parser.add_argument(
        "--input-voxel-size-angstrom",
        type=float,
        required=False,
        action=LargerThanZero,
        help="Voxel size of input map, in Angstrom. If not provided will be read from "
        "MRC input (so make sure it is annotated correctly!).",
    )
    parser.add_argument(
        "--output-voxel-size-angstrom",
        type=float,
        required=True,
        action=LargerThanZero,
        help="Output voxel size of the template, in Angstrom. Needs to be equal to the "
        "voxel size of the tomograms for template matching. Input map will be "
        "downsampled to this spacing.",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        default=False,
        required=False,
        help="Set this flag to automatically center the density in the volume by "
        "measuring the center of mass.",
    )
    parser.add_argument(
        "--low-pass",
        type=float,
        required=False,
        action=LargerThanZero,
        help="Apply a low pass filter to this resolution, in Angstrom. By default a "
        "low pass filter is applied to a resolution of (2 * output_spacing_angstrom) "
        "before downsampling the input volume.",
    )
    parser.add_argument(
        "-b",
        "--box-size",
        type=int,
        required=False,
        action=LargerThanZero,
        help="Specify a desired size for the output box of the template. "
        "Only works if it is larger than the downsampled box size of the input.",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        default=False,
        required=False,
        help="Multiply template by -1. "
        "WARNING: not needed if ctf with defocus is already applied!",
    )
    parser.add_argument(
        "-m",
        "--mirror",
        action="store_true",
        default=False,
        required=False,
        help="Mirror the final template before writing to disk.",
    )
    parser.add_argument(
        "--log",
        type=str,
        required=False,
        default=20,
        action=ParseLogging,
        help="Can be set to `info` or `debug`",
    )

    # ---8<--- [end:create_template_usage]

    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log, force=True)

    # set input voxel size and give user warning if it does not match
    # with MRC annotation
    input_data = read_mrc(args.input_map)
    input_meta_data = read_mrc_meta_data(args.input_map)
    if args.input_voxel_size_angstrom is not None:
        if round(args.input_voxel_size_angstrom, 3) != round(
            input_meta_data["voxel_size"], 3
        ):
            logging.warning(
                "Provided voxel size does not match voxel size annotated in input map."
            )
        map_spacing_angstrom = args.input_voxel_size_angstrom
    else:
        map_spacing_angstrom = input_meta_data["voxel_size"]

    # set output path
    output_path = (
        args.output_file
        if args.output_file is not None
        else (
            pathlib.Path(
                f"template_{args.input_map.stem}_{args.output_voxel_size_angstrom}A.mrc"
            )
        )
    )

    if map_spacing_angstrom > args.output_voxel_size_angstrom:
        raise NotImplementedError(
            "It is assumed the input map has smaller voxel size than the output "
            "template."
        )

    template = generate_template_from_map(
        input_data,
        map_spacing_angstrom,
        args.output_voxel_size_angstrom,
        center=args.center,
        filter_to_resolution=args.low_pass,
        output_box_size=args.box_size,
    ) * (-1 if args.invert else 1)

    logging.debug(f"shape of template after processing is: {template.shape}")

    write_mrc(
        output_path,
        np.flip(template, axis=0) if args.mirror else template,
        args.output_voxel_size_angstrom,
    )


def estimate_roc(argv=None):
    argv = _parse_argv(argv)
    from pytom_tm.plotting import plist_quality_gaussian_fit

    # entry_point strings cannot use '\n' characters as this will break the website
    # snippet that displays the CLI help message
    # ---8<--- [start:estimate_roc_usage]

    parser = argparse.ArgumentParser(
        prog="pytom_estimate_roc.py",
        description="Estimate ROC curve from TMJob file. "
        "-- Marten Chaillet (@McHaillet)",
    )
    parser.add_argument(
        "-j",
        "--job-file",
        type=pathlib.Path,
        required=True,
        action=CheckFileExists,
        help="JSON file that contain all data on the template matching job, written "
        "out by pytom_match_template.py in the destination path.",
    )
    parser.add_argument(
        "-n",
        "--number-of-particles",
        type=int,
        required=True,
        action=LargerThanZero,
        help="The number of particles to extract and estimate the ROC on, recommended "
        "is to multiply the expected number of particles by 3.",
    )
    parser.add_argument(
        "--particle-diameter",
        type=float,
        required=False,
        action=LargerThanZero,
        help="Particle diameter of the template in Angstrom. It is used during "
        "extraction to remove areas around peaks to prevent double extraction. "
        "Minimal peak-to-peak distance after extraction will be diameter/2."
        "If not previously specified, this option is required. If "
        "specified in pytom_match_template, this is optional and "
        "can be used to overwrite it, which might be relevant for strongly "
        "elongated particles--where the angular sampling should be "
        "determined using its long axis but the extraction mask should use its "
        "short axis.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        required=False,
        action=LargerThanZero,
        default=20,
        help="Number of bins for the histogram to fit Gaussians on.",
    )
    parser.add_argument(
        "--gaussian-peak",
        type=int,
        required=False,
        action=LargerThanZero,
        help="Expected index of the histogram peak of the Gaussian fitted to the "
        "particle population.",
    )
    parser.add_argument(
        "--force-peak",
        action="store_true",
        default=False,
        required=False,
        help="Force the particle peak to the provided peak index.",
    )
    parser.add_argument(
        "--crop-plot",
        action="store_true",
        default=False,
        required=False,
        help="Flag to crop the plot relative to the height of the particle population.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        default=False,
        required=False,
        help="Flag to use a pop-up window for the plot instead of writing it to the "
        "location of the job file.",
    )
    parser.add_argument(
        "--log",
        type=str,
        required=False,
        default=20,
        action=ParseLogging,
        help="Can be set to `info` or `debug`",
    )
    parser.add_argument(
        "--ignore_tomogram_mask",
        action="store_true",
        default=False,
        required=False,
        help="Flag to ignore the TM job tomogram mask. "
        "Useful if the scores mrc looks reasonable, but this finds 0 particles",
    )

    # ---8<--- [end:estimate_roc_usage]

    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log, force=True)

    template_matching_job = load_json_to_tmjob(args.job_file)
    # Set cut off to -1 to ensure the number of particles gets extracted
    _, lcc_max_values = extract_particles(
        template_matching_job,
        args.number_of_particles,
        particle_diameter=args.particle_diameter,
        cut_off=0,
        create_plot=False,
        ignore_tomogram_mask=args.ignore_tomogram_mask,
    )

    score_volume = read_mrc(
        template_matching_job.output_dir.joinpath(
            f"{template_matching_job.tomo_id}_scores.mrc"
        )
    )

    plist_quality_gaussian_fit(
        lcc_max_values,
        score_volume,
        args.bins // 2 if args.gaussian_peak is None else args.gaussian_peak,
        force_peak=args.force_peak,
        output_figure_name=(
            None
            if args.show_plot
            else template_matching_job.output_dir.joinpath(
                f"{template_matching_job.tomo_id}_roc.svg"
            )
        ),
        crop_hist=args.crop_plot,
        num_bins=args.bins,
        n_tomograms=1,
    )


def extract_candidates(argv=None):
    argv = _parse_argv(argv)

    # entry_point strings cannot use '\n' characters as this will break the website
    # snippet that displays the CLI help message
    # ---8<--- [start:extract_candidates_usage]

    parser = argparse.ArgumentParser(
        prog="pytom_extract_candidates.py",
        description="Run candidate extraction. -- Marten Chaillet (@McHaillet)",
    )
    parser.add_argument(
        "-j",
        "--job-file",
        type=pathlib.Path,
        required=True,
        action=CheckFileExists,
        help="JSON file that contain all data on the template matching job, written "
        "out by pytom_match_template.py in the destination path.",
    )
    parser.add_argument(
        "--tomogram-mask",
        type=pathlib.Path,
        required=False,
        action=CheckFileExists,
        help="Here you can provide a mask for the extraction with dimensions "
        "(in pixels) equal to the tomogram. All values in the mask that are smaller or "
        "equal to 0 will be removed, all values larger than 0 are considered regions "
        "of interest. It can be used to extract annotations only within a specific "
        "cellular region. If the job was run with a tomogram mask, this file will be "
        "used instead of the job mask",
    )
    parser.add_argument(
        "--ignore_tomogram_mask",
        action="store_true",
        default=False,
        required=False,
        help="Flag to ignore the input and TM job tomogram mask. Useful if the scores "
        "mrc looks reasonable, but this finds 0 particles to extract",
    )
    parser.add_argument(
        "-n",
        "--number-of-particles",
        type=int,
        required=True,
        action=LargerThanZero,
        help="Maximum number of particles to extract from tomogram.",
    )
    parser.add_argument(
        "--number-of-false-positives",
        type=float,
        required=False,
        action=LargerThanZero,
        default=1.0,
        help="Number of false positives to determine the false alarm rate. Here one "
        "can increase the recall of the particle of interest at the expense "
        "of more false positives. The default value of 1 is recommended for "
        "particles that can be distinguished well from the background (high "
        "specificity). The value can also be set between 0 and 1 to make "
        "the cut-off more restrictive.",
    )
    parser.add_argument(
        "--particle-diameter",
        type=float,
        required=False,
        action=LargerThanZero,
        help="Particle diameter of the template in Angstrom. It is used during "
        "extraction to remove areas around peaks to prevent double extraction. "
        "Minimal peak-to-peak distance after extraction will be diameter/2."
        "If not previously specified, this option is required. If "
        "specified in pytom_match_template, this is optional and "
        "can be used to overwrite it, which might be relevant for strongly "
        "elongated particles--where the angular sampling should be "
        "determined using its long axis but the extraction mask should use its "
        "short axis.",
    )
    parser.add_argument(
        "-c",
        "--cut-off",
        type=float,
        required=False,
        help="Override automated extraction cutoff estimation and instead extract the "
        "number-of-particles down to this LCCmax value. Setting to 0 will keep "
        "extracting until number-of-particles, or until there are no positive values "
        "left in the score map. Values larger than 1 make no sense as the correlation "
        "cannot be higher than 1.",
    )
    parser.add_argument(
        "--tophat-filter",
        action="store_true",
        default=False,
        required=False,
        help="Attempt to filter only sharp correlation peaks with a tophat transform",
    )
    parser.add_argument(
        "--tophat-connectivity",
        type=int,
        required=False,
        default=1,
        action=LargerThanZero,
        help="Set kernel connectivity for ndimage binary structure used for the "
        "tophat transform. Integer value in range 1-3. 1 is the most "
        "restrictive, 3 the least restrictive. Generally recommended to "
        "leave at 1.",
    )
    parser.add_argument(
        "--relion5-compat",
        action="store_true",
        default=False,
        required=False,
        help="Write out centered coordinates in Angstrom for RELION5.",
    )
    parser.add_argument(
        "--log",
        type=str,
        required=False,
        default=20,
        action=ParseLogging,
        help="Can be set to `info` or `debug`",
    )
    parser.add_argument(
        "--tophat-bins",
        type=int,
        required=False,
        default=50,
        action=LargerThanZero,
        help="Number of bins to use in the histogram of occurences in the "
        "tophat transform code (for both the estimation and the plotting).",
    )
    parser.add_argument(
        "--plot-bins",
        type=int,
        required=False,
        default=20,
        action=LargerThanZero,
        help="Number of bins to use for the occurences vs LCC_max plot.",
    )

    # ---8<--- [end:extract_candidates_usage]

    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log, force=True)

    # load job and extract particles from the volumes
    job = load_json_to_tmjob(args.job_file)
    df, _ = extract_particles(
        job,
        args.number_of_particles,
        particle_diameter=args.particle_diameter,
        cut_off=args.cut_off,
        n_false_positives=args.number_of_false_positives,
        tomogram_mask_path=args.tomogram_mask,
        tophat_filter=args.tophat_filter,
        tophat_connectivity=args.tophat_connectivity,
        relion5_compat=args.relion5_compat,
        ignore_tomogram_mask=args.ignore_tomogram_mask,
        tophat_bins=args.tophat_bins,
        plot_bins=args.plot_bins,
    )

    # write out as a RELION type starfile
    starfile.write(
        {"particles": df},
        job.output_dir.joinpath(f"{job.tomo_id}_particles.star"),
        overwrite=True,
    )


def match_template(argv=None):
    from pytom_tm.tmjob import TMJob
    from pytom_tm.parallel import run_job_parallel

    argv = _parse_argv(argv)

    # entry_point strings cannot use '\n' characters as this will break the website
    # snippet that displays the CLI help message
    # ---8<--- [start:match_template_usage]

    parser = argparse.ArgumentParser(
        prog="pytom_match_template.py",
        description="Run template matching. -- Marten Chaillet (@McHaillet)",
    )
    io_group = parser.add_argument_group("Template, search volume, and output")
    io_group.add_argument(
        "-t",
        "--template",
        type=pathlib.Path,
        required=True,
        action=CheckFileExists,
        help="Template; MRC file. Object should match the contrast of the tomogram: "
        "if the tomogram has black ribosomes, the reference should be black. "
        "(pytom_create_template.py has an option to invert contrast) ",
    )
    io_group.add_argument(
        "-v",
        "--tomogram",
        type=pathlib.Path,
        required=True,
        action=CheckFileExists,
        help="Tomographic volume; MRC file.",
    )
    io_group.add_argument(
        "-d",
        "--destination",
        type=pathlib.Path,
        required=False,
        default="./",
        action=CheckDirExists,
        help="Folder to store the files produced by template matching.",
    )
    mask_group = parser.add_argument_group("Mask")
    mask_group.add_argument(
        "-m",
        "--mask",
        type=pathlib.Path,
        required=True,
        action=CheckFileExists,
        help="Mask with same box size as template; MRC file.",
    )
    mask_group.add_argument(
        "--non-spherical-mask",
        action="store_true",
        required=False,
        help="Flag to set when the mask is not spherical. It adds the required "
        "computations for non-spherical masks and roughly doubles computation time.",
    )
    rotation_group = parser.add_argument_group("Angular search")
    rotation_group.add_argument(
        "--particle-diameter",
        type=float,
        required=False,
        action=LargerThanZero,
        help="Provide a particle diameter (in Angstrom) to automatically determine the "
        "angular sampling using the Crowther criterion. For the max resolution, "
        "(2 * pixel size) is used unless a low-pass filter is specified, "
        "in which case the low-pass resolution is used. For non-globular "
        "macromolecules choose the diameter along the longest axis.",
    )
    rotation_group.add_argument(
        "--angular-search",
        type=str,
        required=False,
        help="This option overrides the angular search calculation from the particle "
        "diameter. If given a float it will generate an angle list with healpix "
        "for Z1 and X1 and linear search for Z2. The provided angle will be used "
        "as the maximum for the "
        "linear search and for the mean angle difference from healpix."
        "Alternatively, a .txt file can be provided with three Euler angles "
        "(in radians) per line that define the angular search. "
        "Angle format is ZXZ anti-clockwise (see: "
        "https://www.ccpem.ac.uk/user_help/rotation_conventions.php).",
    )
    rotation_group.add_argument(
        "--z-axis-rotational-symmetry",
        type=int,
        required=False,
        action=LargerThanZero,
        default=1,
        help="Integer value indicating the rotational symmetry of the template around "
        "the z-axis. The length of the rotation search will be shortened through "
        "division by this value. Only works for template symmetry around the z-axis.",
    )
    volume_group = parser.add_argument_group("Volume control")
    volume_group.add_argument(
        "-s",
        "--volume-split",
        nargs=3,
        type=int,
        required=False,
        default=[1, 1, 1],
        help="Split the volume into smaller parts for the search, "
        "can be relevant if the volume does not fit into GPU memory. "
        "Format is x y z, e.g. --volume-split 1 2 1",
    )
    volume_group.add_argument(
        "--search-x",
        nargs=2,
        type=int,
        required=False,
        action=ParseSearch,
        help="Start and end indices of the search along the x-axis, "
        "e.g. --search-x 10 490 ",
    )
    volume_group.add_argument(
        "--search-y",
        nargs=2,
        type=int,
        required=False,
        action=ParseSearch,
        help="Start and end indices of the search along the y-axis, "
        "e.g. --search-x 10 490 ",
    )
    volume_group.add_argument(
        "--search-z",
        nargs=2,
        type=int,
        required=False,
        action=ParseSearch,
        help="Start and end indices of the search along the z-axis, "
        "e.g. --search-x 30 230 ",
    )
    volume_group.add_argument(
        "--tomogram-mask",
        type=pathlib.Path,
        required=False,
        action=CheckFileExists,
        help="Here you can provide a mask for matching with dimensions (in pixels) "
        "equal to the tomogram. If a subvolume only has values <= 0 for this mask it "
        "will be skipped.",
    )

    filter_group = parser.add_argument_group("Filter control")
    filter_group.add_argument(
        "-a",
        "--tilt-angles",
        nargs="+",
        type=str,
        required=False,
        action=ParseTiltAngles,
        help="Tilt angles of the tilt-series, either the minimum and maximum values of "
        "the tilts (e.g. --tilt-angles -59.1 60.1) or a .rawtlt/.tlt file with all the "
        "angles (e.g. --tilt-angles tomo101.rawtlt). In case all the tilt angles are "
        "provided a more elaborate Fourier space constraint can be used",
    )
    filter_group.add_argument(
        "--tilt-angles-first-column",
        nargs=1,
        type=str,
        required=False,
        action=ParseTiltAngles,
        error_on_multi_column=False,
        help="A .rawtlt/.tlt file with multiple columns where all the angles are in "
        "the first column (e.g. --tilt-angles-first-column tomo101.rawtlt). In case "
        "all the tilt angles are provided a more elaborate Fourier space constraint "
        "can be used",
    )
    filter_group.add_argument(
        "--per-tilt-weighting",
        action="store_true",
        default=False,
        required=False,
        help="Flag to activate per-tilt-weighting, only makes sense if a file with all "
        "tilt angles have been provided. In case not set, while a tilt angle file is "
        "provided, the minimum and maximum tilt angle are used to create a binary "
        "wedge. The base functionality creates a fanned wedge where each tilt is "
        "weighted by cos(tilt_angle). If dose accumulation and CTF parameters are "
        "provided these will all be incorporated in the tilt-weighting.",
    )
    filter_group.add_argument(
        "--voxel-size-angstrom",
        type=float,
        required=False,
        action=LargerThanZero,
        help="Voxel spacing of tomogram/template in angstrom, if not provided will "
        "try to read from the MRC files. Argument is important for band-pass "
        "filtering!",
    )
    filter_group.add_argument(
        "--low-pass",
        type=float,
        required=False,
        action=LargerThanZero,
        help="Apply a low-pass filter to the tomogram and template. Generally desired "
        "if the template was already filtered to a certain resolution. "
        "Value is the resolution in A.",
    )
    filter_group.add_argument(
        "--high-pass",
        type=float,
        required=False,
        action=LargerThanZero,
        help="Apply a high-pass filter to the tomogram and template to reduce "
        "correlation with large low frequency variations. Value is a resolution in A, "
        "e.g. 500 could be appropriate as the CTF is often incorrectly modelled "
        "up to 50nm.",
    )
    filter_group.add_argument(
        "--dose-accumulation",
        type=str,
        required=False,
        action=ParseDoseFile,
        help="Here you can provide a file that contains the accumulated dose at each "
        "tilt angle, assuming the same ordering of tilts as the tilt angle file. "
        "Format should be a .txt file with on each line a dose value in e-/A2.",
    )
    filter_group.add_argument(
        "--defocus",
        type=str,
        required=False,
        action=ParseDefocus,
        help="Here you can provide an IMOD defocus (.defocus) file (version 2 or 3) "
        ", a text (.txt) file with a single defocus value per line (in μm), "
        "or a single "
        "defocus value (in μm). "
        "The value(s), together with the other ctf "
        "parameters (amplitude contrast, voltage, spherical abberation), "
        "will be used to create a 3D CTF weighting function. IMPORTANT: if "
        "you provide this, the input template should not be modulated with a CTF "
        "beforehand. If it is a reconstruction it should ideally be Wiener filtered.",
    )
    filter_group.add_argument(
        "--amplitude-contrast",
        type=float,
        required=False,
        action=BetweenZeroAndOne,
        help="Amplitude contrast fraction for CTF.",
    )
    filter_group.add_argument(
        "--spherical-aberration",
        type=float,
        required=False,
        action=LargerThanZero,
        help="Spherical aberration for CTF in mm.",
    )
    filter_group.add_argument(
        "--voltage",
        type=float,
        required=False,
        action=LargerThanZero,
        help="Voltage for CTF in keV.",
    )
    filter_group.add_argument(
        "--phase-shift",
        type=float,
        required=False,
        default=0.0,
        action=LargerThanZero,
        help="Phase shift (in degrees) for the CTF to model phase plates.",
    )
    filter_group.add_argument(
        "--tomogram-ctf-model",
        required=False,
        choices=["phase-flip"],  # possible wiener filter mode to come?
        help="Optionally, you can specify if and how the CTF was corrected during "
        "reconstruction of the input tomogram. This allows "
        "match-pick to match the weighting of the template to the tomogram. "
        "Not using this option is appropriate if the CTF was left uncorrected in "
        "the tomogram. Option 'phase-flip' : appropriate for IMOD's strip-based "
        "phase flipping or reconstructions generated with "
        "novaCTF/3dctf.",
    )
    filter_group.add_argument(
        "--defocus-handedness",
        required=False,
        choices=[-1, 0, 1],
        type=int,
        default=0,
        help="Specify the defocus handedness for defocus gradient correction of the "
        "CTF in each subvolumes. The more subvolumes in x and z, "
        "the finer the defocus gradient will be corrected, at the cost of "
        "increased computing time. It will only have effect for very clean and "
        "high-resolution data, such as isolated macromolecules. IMPORTANT: only "
        "works in combination with --volume-split ! "
        "A value of 0 means no defocus gradient correction (default), 1 means "
        "correction assuming correct handedness (as specified in Pyle and "
        "Zianetti (2021)), -1 means the handedness will be inverted. If uncertain "
        "better to leave off as an inverted correction might hamper results.",
    )
    filter_group.add_argument(
        "--spectral-whitening",
        action="store_true",
        default=False,
        required=False,
        help="Calculate a whitening filtering from the power spectrum of the tomogram; "
        "apply it to the tomogram patch and template. Effectively puts more weight on "
        "high resolution features and sharpens the correlation peaks.",
    )
    additional_group = parser.add_argument_group("Additional options")
    additional_group.add_argument(
        "-r",
        "--random-phase-correction",
        action="store_true",
        default=False,
        required=False,
        help="Run template matching simultaneously with a phase randomized version of "
        "the template, and subtract this 'noise' map from the final score map. "
        "For this method please see STOPGAP as a reference: "
        "https://doi.org/10.1107/S205979832400295X .",
    )
    additional_group.add_argument(
        "--half-precision",
        action="store_true",
        default=False,
        required=False,
        help="Return and save all output in float16 instead of the default float32",
    )
    additional_group.add_argument(
        "--rng-seed",
        type=int,
        action=LargerThanZero,
        default=int.from_bytes(urandom(8)),
        required=False,
        help="Specify a seed for the random number generator used for phase "
        "randomization for consistent results!",
    )
    additional_group.add_argument(
        "--relion5-tomograms-star",
        type=pathlib.Path,
        action=CheckFileExists,
        required=False,
        help="Here, you can provide a path to a RELION5 tomograms.star file (for "
        "example "
        "from a tomogram reconstruction job). pytom-match-pick will fetch all "
        "the tilt-series metadata from this file and overwrite all other "
        "metadata options.",
    )
    additional_group.add_argument(
        "--warp-xml-file",
        type=pathlib.Path,
        action=CheckFileExists,
        required=False,
        help="Here, you can provide a Warp xml file that has the metadata "
        "for that tiltseries."
        "This xml metadata file will be in the tiltseries processing dir "
        "eg. <cwd>/warp_tiltseries/)",
    )
    device_group = parser.add_argument_group("Device control")
    device_group.add_argument(
        "-g",
        "--gpu-ids",
        nargs="+",
        type=int,
        action=ParseGPUIndices,
        required=True,
        help="GPU indices to run the program on.",
    )
    debug_group = parser.add_argument_group("Logging/debugging")
    debug_group.add_argument(
        "--log",
        type=str,
        required=False,
        default=20,
        action=ParseLogging,
        help="Can be set to `info` or `debug`",
    )

    # ---8<--- [end:match_template_usage]

    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log, force=True)

    # set correct tilt angles
    if args.tilt_angles_first_column is not None:
        if args.tilt_angles is not None:
            raise ValueError(
                "Only one of '--tilt-angles' or '--tilt-angles-first-column' is allowed"
            )
        tilt_angles = args.tilt_angles_first_column
    else:
        tilt_angles = args.tilt_angles

    # parse CTF phase correction
    phase_flip_correction = False
    if args.tomogram_ctf_model is not None and args.tomogram_ctf_model == "phase-flip":
        phase_flip_correction = True

    # combine ctf values to ctf_params list of dicts
    ctf_params = None
    if args.defocus is not None:
        if (
            args.amplitude_contrast is None
            or args.spherical_aberration is None
            or args.voltage is None
        ):
            raise ValueError(
                "Cannot create 3D CTF weighting because one or multiple of "
                "the required parameters (amplitude-contrast, "
                "spherical-abberation or voltage) is/are missing."
            )
        ctf_params = [
            {
                "defocus": defocus * 1e-6,
                "amplitude_contrast": args.amplitude_contrast,
                "voltage": args.voltage * 1e3,
                "spherical_aberration": args.spherical_aberration * 1e-3,
                "flip_phase": phase_flip_correction,
                "phase_shift_deg": args.phase_shift,
            }
            for defocus in args.defocus
        ]

    if args.relion5_tomograms_star is not None:
        voxel_size, tilt_angles, dose_accumulation, ctf_params, defocus_handedness = (
            parse_relion5_star_data(
                args.relion5_tomograms_star,
                args.tomogram,
                phase_flip_correction=phase_flip_correction,
                phase_shift=args.phase_shift,
            )
        )
        per_tilt_weighting = True

    elif args.warp_xml_file is not None:
        voxel_size, tilt_angles, dose_accumulation, ctf_params = parse_warp_xml_data(
            args.warp_xml_file,
            args.tomogram,
            phase_flip_correction=phase_flip_correction,
        )
        defocus_handedness = args.defocus_handedness
        per_tilt_weighting = True

    else:
        if tilt_angles is None:
            raise ValueError(
                "Without tilt angles the missing wedge cannot be calculated. A "
                "minimal run requires tilt angles."
            )
        voxel_size = args.voxel_size_angstrom
        defocus_handedness = args.defocus_handedness
        dose_accumulation = args.dose_accumulation
        per_tilt_weighting = args.per_tilt_weighting

    if args.angular_search is None and args.particle_diameter is None:
        raise ValueError(
            "Either the angular search should be specifically set or a particle "
            "diameter should be provided to infer the angular search!"
        )

    job = TMJob(
        "0",
        args.log,
        args.tomogram,
        args.template,
        args.mask,
        args.destination,
        angle_increment=args.angular_search,
        mask_is_spherical=True
        if args.non_spherical_mask is None
        else (not args.non_spherical_mask),
        tilt_angles=tilt_angles,
        tilt_weighting=per_tilt_weighting,
        search_x=args.search_x,
        search_y=args.search_y,
        search_z=args.search_z,
        tomogram_mask=args.tomogram_mask,
        voxel_size=voxel_size,
        low_pass=args.low_pass,
        high_pass=args.high_pass,
        dose_accumulation=dose_accumulation,
        ctf_data=ctf_params,
        whiten_spectrum=args.spectral_whitening,
        rotational_symmetry=args.z_axis_rotational_symmetry,
        particle_diameter=args.particle_diameter,
        random_phase_correction=args.random_phase_correction,
        rng_seed=args.rng_seed,
        defocus_handedness=defocus_handedness,
        output_dtype=np.float16 if args.half_precision else np.float32,
    )

    score_volume, angle_volume = run_job_parallel(
        job, tuple(args.volume_split), args.gpu_ids
    )

    # set the appropriate headers when writing!
    write_mrc(
        args.destination.joinpath(f"{job.tomo_id}_scores.mrc"),
        score_volume,
        job.voxel_size,
    )
    write_mrc(
        args.destination.joinpath(f"{job.tomo_id}_angles.mrc"),
        angle_volume,
        job.voxel_size,
    )

    # write the job as well
    job.write_to_json(args.destination.joinpath(f"{job.tomo_id}_job.json"))


def merge_stars(argv=None):
    import pandas as pd

    # entry_point strings cannot use '\n' characters as this will break the website
    # snippet that displays the CLI help message
    # ---8<--- [start:merge_stars_usage]

    parser = argparse.ArgumentParser(
        prog="pytom_merge_stars.py",
        description=(
            "Merge multiple star files in the same directory. "
            "-- Marten Chaillet (@McHaillet)"
        ),
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=pathlib.Path,
        required=False,
        default="./",
        action=CheckDirExists,
        help=(
            "Directory with star files, "
            "script will try to merge all files that end in '.star'."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=pathlib.Path,
        required=False,
        default="./particles.star",
        help="Output star file name.",
    )
    parser.add_argument(
        "--log",
        type=str,
        required=False,
        default=20,
        action=ParseLogging,
        help="Can be set to `info` or `debug`",
    )

    # ---8<--- [end:merge_stars_usage]

    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log, force=True)

    files = [f for f in args.input_dir.iterdir() if f.suffix == ".star"]

    if len(files) == 0:
        raise ValueError("No starfiles in directory.")

    logging.info("Concatting and writing star files")

    dataframes = [starfile.read(f) for f in files]

    starfile.write(
        {"particles": pd.concat(dataframes, ignore_index=True)},
        args.output_file,
        overwrite=True,
    )
