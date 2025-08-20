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
    voxel_size: float
    tilt_angles: list[float]
    per_tilt_weighting: bool = False
    defocus_handedness: int = 0
    ctf_data: list[CtfData] | None = None
    dose_accumulation: list[float] | None = None

    def __post_init__(self):
        n_angles = len(self.tilt_angles)
        # Make sure all lists have the same value
        if self.ctf_data is not None and len(self.ctf_data) == 1:
            self.ctf_data = self.ctf_data * n_angles
        elif self.ctf_data is not None and not len(self.ctf_data) == n_angles:
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

        if self.defocus_handedness not in (-1, 0, 1):
            raise ValueError(
                "Got an invalid defocus handedness, "
                f"expected -1, 0, or 1. Got: {self.defocus_handedness}"
            )
