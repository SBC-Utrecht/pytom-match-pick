from dataclasses import dataclass
from pytom_tm.json import JsonSerializable
import copy


@dataclass(kw_only=True)
class CtfData(JsonSerializable):
    """A data class to combine all CTF data for a single tilt

    Attributes
    ----------
    defocus: float
        defocus in µm
    amplitude_contrast: float
        fraction of amplitude contrast between 0 and 1
    voltage: float
        voltage in KeV
    spherical_aberration: float
        spherical aberration in mm
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

    def copy(self):
        return copy.deepcopy(self)


# @dataclass(kw_only=True)
# class TiltSeriesMetaData:
#    tilt_angles: list[float]
#    dose_accumulation: list[float] | None = None
#    ctf_param: list[CtfData] | None = None
#    per_tilt_weighting: bool = False
#
#    def __post_init__(self):
#        """Check that if something is given it has the same length as tilt_angles"""
#        expected = len(self.tilt_angles)
#        for key, val in asdict(self).items():
#            if type(val) is list and len(val) != expected:
#                raise ValueError(
#                    f"{key} should have the same lenght as the list of "
#                    f"tilt_angles. Expected {expected}, found {len(val)}"
#                )
#
#
# @dataclass(kw_only=True)
# class Relion5MetaData(TiltSeriesMetaData):
#    binning: float
#    tilt_series_pixel_size: float
