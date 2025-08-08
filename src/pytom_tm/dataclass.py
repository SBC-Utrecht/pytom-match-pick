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
