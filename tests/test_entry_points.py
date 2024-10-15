import unittest
import pathlib
import numpy as np
import cupy as cp
import logging
from shutil import which
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pytom_tm import entry_points
from pytom_tm import io

# (command line function, function in entry_points file)
ENTRY_POINTS_TO_TEST = [
    ("pytom_create_mask.py", "pytom_create_mask"),
    ("pytom_create_template.py", "pytom_create_template"),
    ("pytom_match_template.py", "match_template"),
    ("pytom_extract_candidates.py", "extract_candidates"),
    ("pytom_merge_stars.py", "merge_stars"),
]
# Test if optional dependencies are installed
try:
    from pytom_tm import plotting  # noqa: F401
except RuntimeError:
    pass
else:
    ENTRY_POINTS_TO_TEST.append(("pytom_estimate_roc.py", "estimate_roc"))

# Input files for command line match_template
TEST_DATA = pathlib.Path(__file__).parent.joinpath("test_data")
TEMPLATE = TEST_DATA.joinpath("template.mrc")
MASK = TEST_DATA.joinpath("mask.mrc")
TOMOGRAM = TEST_DATA.joinpath("tomogram.mrc")
DESTINATION = TEST_DATA.joinpath("output")
TILT_ANGLES = TEST_DATA.joinpath("angles.rawtlt")
DOSE = TEST_DATA.joinpath("test_dose.txt")
DEFOCUS = TEST_DATA.joinpath("defocus.txt")
DEFOCUS_IMOD = (
    pathlib.Path(__file__).parent.joinpath("Data").joinpath("test_imod.defocus")
)
RELION5_TOMOGRAMS_STAR = pathlib.Path(__file__).parent.joinpath(
    "Data/relion5_project_example/Tomograms/job009/tomograms.star"
)
RELION5_TOMOGRAM = TEST_DATA.joinpath("rec_tomo200528_107.mrc")

# Initial logging level
LOG_LEVEL = logging.getLogger().level


def prep_argv(arg_dict):
    argv = []
    [
        argv.extend([k] + v.split()) if v != "" else argv.append(k)
        for k, v in arg_dict.items()
    ]
    return argv


class TestEntryPoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        DESTINATION.mkdir(parents=True)
        io.write_mrc(TEMPLATE, np.zeros((5, 5, 5), dtype=np.float32), 1)
        io.write_mrc(MASK, np.zeros((5, 5, 5), dtype=np.float32), 1)
        io.write_mrc(TOMOGRAM, np.zeros((10, 10, 10), dtype=np.float32), 1)
        io.write_mrc(RELION5_TOMOGRAM, np.zeros((10, 10, 10), dtype=np.float32), 1)
        np.savetxt(TILT_ANGLES, np.linspace(-50, 50, 35))
        np.savetxt(DOSE, np.linspace(0, 100, 35))
        np.savetxt(DEFOCUS, np.ones(35) * 3000)

    @classmethod
    def tearDownClass(cls) -> None:
        TEMPLATE.unlink()
        MASK.unlink()
        TOMOGRAM.unlink()
        RELION5_TOMOGRAM.unlink()
        TILT_ANGLES.unlink()
        DOSE.unlink()
        DEFOCUS.unlink()
        for f in DESTINATION.iterdir():
            f.unlink()  # should test specific output?
        DESTINATION.rmdir()
        TEST_DATA.rmdir()

    def test_entry_points_exist(self):
        for cli, fname in ENTRY_POINTS_TO_TEST:
            # test the command line function can be found
            self.assertIsNotNone(which(cli))
            # assert the entry_point be called with -h and exit cleanly
            # catch stdout to prevent shell polution
            func = getattr(entry_points, fname)
            dump = StringIO()
            with self.assertRaises(SystemExit) as ex, redirect_stdout(dump):
                func([cli, "-h"])
            dump.close()
            # check if the system return code is 0 (success)
            self.assertEqual(ex.exception.code, 0)

    def test_match_template(self):
        defaults = {
            "-t": str(TEMPLATE),
            "-m": str(MASK),
            "-v": str(TOMOGRAM),
            "-d": str(DESTINATION),
            "--angular-search": "35",
            "--tilt-angles": str(TILT_ANGLES),
            "--per-tilt-weighting": "",
            "--dose-accumulation": str(DOSE),
            "--defocus": str(DEFOCUS_IMOD),
            "--amplitude-contrast": "0.08",
            "--spherical-aberration": "2.7",
            "--voltage": "300",
            "--tomogram-ctf-model": "phase-flip",
            "-g": "0",
        }

        def start(arg_dict):  # simplify run
            entry_points.match_template(prep_argv(arg_dict))

        # test valid defocus arguments
        for z in [str(DEFOCUS_IMOD), str(DEFOCUS), "3000"]:
            arguments = defaults.copy()
            arguments["--defocus"] = z
            start(arguments)

        # test faulty args
        for z in ["asdf.txt", "asdf"]:
            dump = StringIO()
            with (
                self.assertRaises(SystemExit) as ex,
                redirect_stdout(dump),
                redirect_stderr(dump),
            ):
                arguments = defaults.copy()
                arguments["--defocus"] = z
                start(arguments)
            dump.close()
            # check if the system return code is 0 (success)
            self.assertEqual(ex.exception.code, 2)

        # remove per-tilt-weighting
        arguments = defaults.copy()
        arguments.pop("--per-tilt-weighting")
        start(arguments)

        with self.assertRaises(
            ValueError, msg="Missing CTF params should produce " "error"
        ):
            arguments = defaults.copy()
            arguments.pop("--voltage")
            start(arguments)

        # test angular search and particle diameter options
        with self.assertRaises(
            ValueError, msg="Missing angular search should raise " "an error."
        ):
            arguments = defaults.copy()
            arguments.pop("--angular-search")
            start(arguments)

        arguments = defaults.copy()
        arguments.pop("--angular-search")
        arguments["--particle-diameter"] = "50"
        # set low-pass to tune the search to lower degree
        arguments["--low-pass"] = "50"
        start(arguments)

        # phase randomization test
        arguments = defaults.copy()
        arguments["-r"] = ""
        start(arguments)
        # test if we can set the rng seed, see issue #194
        arguments["--rng-seed"] = "42"
        start(arguments)

        # test debug files
        arguments = defaults.copy()
        arguments["--log"] = "debug"
        start(arguments)
        # these files will only exist if the test managed to set the logging correctly
        self.assertTrue(
            DESTINATION.joinpath("template_psf.mrc").exists(),
            msg="File should exist in debug mode",
        )
        self.assertTrue(
            DESTINATION.joinpath("template_convolved.mrc").exists(),
            msg="File should exist in debug mode",
        )

        # reset the log level after the entry point modified it
        logging.basicConfig(level=LOG_LEVEL, force=True)

        # test providing invalid gpu indices
        n_devices = cp.cuda.runtime.getDeviceCount()
        for indices in ["-1", f"0 {n_devices}"]:
            dump = StringIO()
            with (
                self.assertRaises(SystemExit) as ex,
                redirect_stdout(dump),
                redirect_stderr(dump),
            ):
                arguments = defaults.copy()
                arguments["-g"] = indices
                start(arguments)
            self.assertIn("gpu indices", dump.getvalue())
            dump.close()

        # test relion5 metadata reading
        arguments = defaults.copy()
        [
            arguments.pop(x)
            for x in [
                "--tilt-angles",
                "--per-tilt-weighting",
                "--dose-accumulation",
                "--defocus",
                "--amplitude-contrast",
                "--spherical-aberration",
                "--voltage",
            ]
        ]
        arguments["-v"] = str(RELION5_TOMOGRAM)
        arguments["--relion5-tomograms-star"] = str(RELION5_TOMOGRAMS_STAR)
        start(arguments)

        with self.assertRaises(
            ValueError, msg="Missing tilt angles should raise an error."
        ):
            arguments.pop("--relion5-tomograms-star")
            start(arguments)
