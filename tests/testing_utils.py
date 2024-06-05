import pathlib
from pytom_tm.io import read_defocus_file, read_dose_file, read_tlt_file


# Dose and ctf params for tomo_104
cs = 2.7
amp = 0.08
vol = 200
defocus_data = read_defocus_file(
    pathlib.Path(__file__).parent.joinpath('Data').joinpath('test_imod.defocus')
)
CTF_PARAMS = []
for d in defocus_data:
    CTF_PARAMS.append({
        'defocus': d,
        'amplitude_contrast': amp,
        'voltage': vol,
        'spherical_aberration': cs,
        'phase_shift_deg': .0,
    })

ACCUMULATED_DOSE = read_dose_file(
    pathlib.Path(__file__).parent.joinpath('Data').joinpath('test_dose.txt')
)
TILT_ANGLES = read_tlt_file(
    pathlib.Path(__file__).parent.joinpath('Data').joinpath('test_angles.rawtlt')
)
