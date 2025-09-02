from dataclasses import dataclass
from pytom_tm.json import JsonSerializable


@dataclass(kw_only=True)
class CtfData(JsonSerializable):
    """A data class to combine all CTF data for a single tilt

    Attributes
    ----------
    defocus: float
        defocus in m
    amplitude_contrast: float
        fraction of amplitude contrast between 0 and 1
    voltage: float
        voltage in eV
    spherical_aberration: float
        spherical aberration in m
    phase_shift_deg: float, default 0.0
        phase shift for phase plates in degree
    flip_phase: bool, default False
        wether we should apply the phase-flip ctf correction
    """

    defocus: float
    amplitude_contrast: float
    voltage: float
    spherical_aberration: float
    phase_shift_deg: float = 0.0
    flip_phase: bool = False


@dataclass(kw_only=True)
class TiltSeriesMetaData(JsonSerializable):
    """A dataclass for to keep all the meta-data for the tilt-series together

    Attributes
    ----------
    tilt_angles: list[float]
        list of tilt angles of the tilt-series in degrees
    ctf_data: list[CtfData] | None, default None
        list of CtfData per tilt, should either be length 1 if it is identical for all
        tilts or the same length as tilt_angles
    dose_accumulation: list[float] | None, default None
        list of accumulated doses per tilt in electrons per Å^2,
        if given should have the same length as tilt_angles
    defocus_handedness: int from {-1, 0, 1}, default 0
        defocus handesness for gradient correction
        as specified in Pyle and Zianetti (2021)
        0: no defocus gradient correction (default),
        1: correction assuming correct handedness
       -1: the handedness will be inverted
    per_tilt_weighting: bool, default False
        if we want to do per-tilt weighting to create a fanned wedge instead
        of a default binary one
    """

    tilt_angles: list[float]
    ctf_data: list[CtfData] | None = None
    dose_accumulation: list[float] | None = None
    defocus_handedness: int = 0
    per_tilt_weighting: bool = False

    def __post_init__(self):
        if not isinstance(self.tilt_angles, list) or len(self.tilt_angles) < 2:
            raise ValueError("TiltSeriesMetaData requires at least 2 tilt angles")
        n_angles = len(self.tilt_angles)
        # Make sure all lists have the same value
        if self.ctf_data is not None and len(self.ctf_data) == 1:
            self.ctf_data = self.ctf_data * n_angles
        elif self.ctf_data is not None and len(self.ctf_data) != n_angles:
            raise ValueError(
                "Expected either a list with a single CtfData or the "
                f"same number as tilt_angles ({n_angles}). Got {len(self.ctf_data)} "
                "instead."
            )
        if (
            self.dose_accumulation is not None
            and len(self.dose_accumulation) != n_angles
        ):
            raise ValueError(
                "Expected a list with the same number of doses as tilt "
                f"angles ({n_angles}). Got {len(self.dose_accumulation)} instead."
            )

        if self.defocus_handedness not in {-1, 0, 1}:
            raise ValueError(
                "Got an invalid defocus handedness, "
                f"expected -1, 0, or 1. Got: {self.defocus_handedness}"
            )

    def __len__(self):
        return len(self.tilt_angles)


@dataclass(kw_only=True)
class WarpTiltSeriesMetaData(TiltSeriesMetaData):
    """Extension of TiltSeriesMetaData that has per_tilt_weighting default as True"""

    per_tilt_weighting: bool = True


@dataclass(kw_only=True)
class RelionTiltSeriesMetaData(TiltSeriesMetaData):
    """Extension of the TiltSeriesMetaData that has per_tilt_weighting default as True
    and has the following extra attributes

    Attributes
    ----------
    binning: float
        binning of the tomogram
    tilt_series_pixel_size: float
        pixel size of the original tilt series in Å
    """

    binning: float
    tilt_series_pixel_size: float
    per_tilt_weighting: bool = True


# TODO: possible TomogramMetaData dataclass
# @dataclass(kw_only=True)
# class TomogramMetaData(JsonSerializable):
#    """A dataclass for to keep all the meta-data for the tomogram together
#
#    """
#    path: pathlib.Path
#    voxel_size: float | None = None
#    ts_metadata: TiltSeriesMetaData = None
#    mask: pathlib.Path | None = None
#    #TODO: add mrc header info??
#
#    def __post_init__(self):
#        # Set and check the voxel size if not given
