from dataclasses import dataclass, asdict


@dataclass
class CtfData:
    defocus: float
    amplitude_contrast: float
    voltage: float
    spherical_aberration: float
    flip_phase: bool = False
    phase_shift_deg: float = 0.0


@dataclass
class TiltSeriesMetaData:
    tilt_angles: list[float]
    dose_accumulation: list[float] | None = None
    ctf_param: list[CtfData] | None = None
    per_tilt_weighting: bool = False

    def __post_init__(self):
        """Check that if something is given it has the same length as tilt_angles"""
        expected = len(self.tilt_angles)
        for key, val in asdict(self).items():
            if type(val) is list and len(val) != expected:
                raise ValueError(
                    f"{key} should have the same lenght as the list of "
                    f"tilt_angles. Expected {expected}, found {len(val)}"
                )


@dataclass
class Relion5MetaData(TiltSeriesMetaData):
    binning: float
    tilt_series_pixel_size: float
