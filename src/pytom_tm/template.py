import numpy.typing as npt
import numpy as np
import voltools as vt
import logging
from scipy.ndimage import center_of_mass, zoom
from scipy.fft import rfftn, irfftn
from pytom_tm.weights import create_gaussian_low_pass


def generate_template_from_map(
    input_map: npt.NDArray[float],
    input_spacing: float,
    output_spacing: float,
    center: bool = False,
    filter_to_resolution: float | None = None,
    output_box_size: int | None = None,
) -> npt.NDArray[float]:
    """Generate a template from a density map.

    Parameters
    ----------
    input_map: npt.NDArray[float]
        3D density to use for generating the template, if box is not square it will be
        padded to square
    input_spacing: float
        voxel size of input map (in A)
    output_spacing: float
        voxel size of output map (in A) the ratio of input to output will be used for
        downsampling
    center: bool, default False
        set to True to center the template in the box by calculating the center of mass
    filter_to_resolution: Optional[float], default None
        low-pass filter resolution to apply to template, if not provided will be set to
        2 * output_spacing
    output_box_size:  Optional[int], default None
        final box size of template
    display_filter: bool, default False
        flag to display a plot of the filter applied to the template

    Returns
    -------
    template: npt.NDArray[float]
        processed template in the specified output box size, box will be square
    """
    # make the map a box with equal dimensions
    if len(set(input_map.shape)) != 1:
        diff = [max(input_map.shape) - s for s in input_map.shape]
        input_map = np.pad(
            input_map,
            tuple([(d // 2, d // 2 + d % 2) for d in diff]),
            mode="constant",
            constant_values=0,
        )

    if filter_to_resolution is None:
        # Set to nyquist resolution
        filter_to_resolution = 2 * output_spacing
    elif filter_to_resolution < (2 * output_spacing):
        warning_text = (
            f"Filter resolution is too low,"
            f" setting to {2 * output_spacing}A (2 * output voxel size)"
        )
        logging.warning(warning_text)
        filter_to_resolution = 2 * output_spacing

    if center:
        volume_center = np.divide(np.subtract(input_map.shape, 1), 2, dtype=np.float32)
        # square input to make values positive for center of mass
        input_center_of_mass = center_of_mass(input_map**2)
        shift = np.subtract(volume_center, input_center_of_mass)
        input_map = vt.transform(input_map, translation=shift, device="cpu")

        logging.debug(
            f"center of mass, before was "
            f"{np.round(input_center_of_mass, 2)} "
            f"and after {np.round(center_of_mass(input_map**2), 2)}"
        )

    # extend volume to the desired output size before applying convolutions!
    if output_box_size is not None:
        logging.debug(
            f"size check {output_box_size} > "
            f"{(input_map.shape[0] * input_spacing) // output_spacing}"
        )
        if output_box_size > (input_map.shape[0] * input_spacing) // output_spacing:
            pad = (
                int(output_box_size * (output_spacing / input_spacing))
                - input_map.shape[0]
            )
            logging.debug(f"pad with this number of zeros: {pad}")
            input_map = np.pad(
                input_map,
                (pad // 2, pad // 2 + pad % 2),
                mode="constant",
                constant_values=0,
            )
        elif output_box_size < (input_map.shape[0] * input_spacing) // output_spacing:
            logging.warning(
                "Could not set specified box size as the map would need to be cut and "
                "this might result in loss of information of the structure. Please "
                "decrease the box size of the map by hand (e.g. chimera)"
            )

    # create low pass filter
    lpf = create_gaussian_low_pass(
        input_map.shape, input_spacing, filter_to_resolution
    ).astype(np.float32)

    logging.info("Convoluting volume with filter and then downsampling.")
    return zoom(
        irfftn(rfftn(input_map) * lpf, s=input_map.shape),
        input_spacing / output_spacing,
    )


def phase_randomize_template(
    template: npt.NDArray[float],
    mask: npt.NDArray[float] | None = None,
    n_iter: int = 40,
    seed: int = 321,
) -> npt.NDArray[float]:
    """Create a phase-randomized version of `template` that preserves its
    amplitude spectrum.

    Random phases are taken from the rfftn of a random real-valued field
    instead of drawn independently per Fourier voxel. This guarantees they
    satisfy Hermitian symmetry by construction, including at the
    self-conjugate DC/Nyquist points, which independent (e.g. permuted)
    phases would violate.

    If a `mask` is provided, a Gerchberg-Saxton iteration alternates the
    amplitude constraint in Fourier space with a real-space support constraint
    for `n_iter` iterations, so the resulting noise stays compact instead of
    delocalizing over the full box.

    Parameters
    ----------
    template: npt.NDArray[float]
        input structure
    mask: Optional[npt.NDArray[float]], default None
        if provided, real-space support constraint used in a Gerchberg-Saxton
        iteration; same dimensions as template
    n_iter: int, default 40
        number of Gerchberg-Saxton iterations, only used if mask is provided
    seed: int, default 321
        seed for the random number generator

    Returns
    -------
    result: npt.NDArray[float]
        phase randomized version of the template
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(template, dtype=np.float64)
    amplitude = np.abs(rfftn(t))

    # Hermitian-valid random phases: phases of the rfftn of a random real field
    phase = np.angle(rfftn(rng.standard_normal(t.shape)))
    result = irfftn(amplitude * np.exp(1j * phase), s=t.shape)

    if mask is not None:
        for _ in range(n_iter):
            result = result * mask
            phase = np.angle(rfftn(result))
            result = irfftn(amplitude * np.exp(1j * phase), s=t.shape)
        result = result * mask
        if result.sum() > 0:  # match total mass under the (possibly soft) mask
            result = result * ((t * mask).sum() / (result * mask).sum())

    return result.astype(np.float32)
