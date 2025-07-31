import unittest
import pathlib
import starfile
import glob
from tempfile import TemporaryDirectory
from pytom_tm.entry_points import merge_stars
from pytom_tm.utils import mute_stdout_stderr
from testing_utils import make_random_particles


class TestMergeStars(unittest.TestCase):
    def setUp(self):
        tempdir = TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        self.dirname = tempdir.name
        self.tempdir = pathlib.Path(tempdir.name)
        self.n = 10
        particles1 = make_random_particles(n=self.n)
        particles2 = make_random_particles(n=self.n)
        self.particles = [particles1, particles2]

    def test_error_function_on_empty(self):
        from pytom_tm.merge_stars import merge_stars as ms

        # make sure we raise if we don't find any starfiles
        with self.assertRaisesRegex(ValueError, "No starfiles"):
            ms([], self.tempdir / "not_relevant.star")

    def test_relion4_mode(self):
        # write 2 starfiles
        for particle in self.particles:
            tomo_id = particle["rlnMicrographName"][0]
            starfile.write(
                {"particles": particle}, self.tempdir / f"{tomo_id}_particles.star"
            )

        outfile = str(self.tempdir / "test.star")
        # Make a joined file via the entry point
        # mimick star expansion on a bash shell
        in_files = glob.glob(f"{self.dirname}/*.star")
        merge_stars(["-i"] + in_files + ["-o", outfile])

        # make sure we can read the output starfile
        out = starfile.read(outfile)

        # make sure we have the number of expected lines
        self.assertEqual(self.n * 2, len(out))

        # check that both tomo IDs are in the new starfile
        for particle in self.particles:
            for name in set(particle["rlnMicrographName"]):
                self.assertIn(name, set(out["rlnMicrographName"]))

        # test that we fail if we just give one starfile in this mode
        with self.assertRaisesRegex(ValueError, "doesn't make sense to merge"):
            merge_stars(["-i", in_files[0], "-o", outfile])

    def test_fail_on_incompatible_starfiles(self):
        # Make sure we fail if we try to combine RELION4 starfiles
        # with a RELION5 flag

        # write 2 relion 4 starfiles
        for particle in self.particles:
            tomo_id = particle["rlnMicrographName"][0]
            starfile.write(
                {"particles": particle}, self.tempdir / f"{tomo_id}_particles.star"
            )

        outfile = str(self.tempdir / "test.star")
        # Make sure entry point fails if trying to do relion5 merge
        in_files = glob.glob(f"{self.dirname}/*.star")
        with self.assertRaisesRegex(ValueError, "rlnTomoName"):
            merge_stars(["-i"] + in_files + ["-o", outfile, "--relion5-compat"])

    def test_relion5_mode(self):
        particles1 = make_random_particles(relion5=True)
        particles2 = make_random_particles(relion5=True)
        for particle in [particles1, particles2]:
            tomo_id = particle["rlnTomoName"][0]
            starfile.write(
                {"particles": particle}, self.tempdir / f"{tomo_id}_particles.star"
            )

        outfile = str(self.tempdir / "test.star")
        # Make sure entry point fails if trying to do relion5 merge
        in_files = glob.glob(f"{self.dirname}/*.star")

        # Make a joined file via the entry point
        merge_stars(["-i"] + in_files + ["-o", outfile, "--relion5-compat"])

        # make sure we can read the output starfile
        out = starfile.read(outfile)
        # make sure that the outfile is as expected:
        # - 2 columns with names _rlnTomoName and _rlnTomoImportParticleFile
        self.assertEqual(2, len(out.columns))
        for col in ["rlnTomoName", "rlnTomoImportParticleFile"]:
            self.assertIn(col, out.columns)

        # check that both tomo IDs are in the new starfile and the correct filename
        for particle in [particles1, particles2]:
            for name in set(particle["rlnTomoName"]):
                self.assertIn(name, set(out["rlnTomoName"]))

        # Check that all star files can be read and contains the expected tomoname
        for _, (tomoname, filename) in out.iterrows():
            temp = starfile.read(filename)
            temp_tomoname = set(temp["rlnTomoName"])
            self.assertIn(tomoname, temp_tomoname)

        # test that we pass if we just give one starfile in this mode
        outfile2 = str(self.tempdir / "single_test.star")
        merge_stars(["-i", in_files[0], "-o", outfile])

        # make sure we only written a single line
        out2 = starfile.read(outfile2)
        self.assertEqual(1, len(out2))

    def test_fail_on_dir_input(self):
        with mute_stdout_stderr(), self.assertRaises(SystemExit) as ex:
            merge_stars(["-i", f"{self.dirname}", "-o", "irrelevant.star"])
        self.assertEqual(ex.exception.code, 2)

    def test_multi_dir_input(self):
        # write 2 starfiles
        directories = [self.tempdir / f"test{i}" for i in range(2)]
        for particle, directory in zip(self.particles, directories):
            directory.mkdir()
            tomo_id = particle["rlnMicrographName"][0]
            starfile.write(
                {"particles": particle}, directory / f"{tomo_id}_particles.star"
            )

        outfile = str(self.tempdir / "test.star")

        # Mimick inputting `output_dir/test*/*.star`
        star_files = glob.glob(f"{self.dirname}/test*/*.star")

        merge_stars(["-i"] + star_files + ["-o", outfile])

        # make sure we can read the output starfile
        out = starfile.read(outfile)

        # make sure we have the number of particles we expect
        self.assertEqual(len(out), 2 * self.n)

    def test_non_unique_input(self):
        # write 2 starfiles
        for particle in self.particles:
            tomo_id = particle["rlnMicrographName"][0]
            starfile.write(
                {"particles": particle}, self.tempdir / f"{tomo_id}_particles.star"
            )

        outfile = str(self.tempdir / "test.star")
        # Make a joined file via the entry point
        # mimick star expansion on a bash shell
        in_files = glob.glob(f"{self.dirname}/*.star")
        with self.assertLogs(level="Warning") as cm:
            # Give all the files twice
            merge_stars(["-i"] + in_files + in_files + ["-o", outfile])
        for o in cm.output:
            if "duplicate input" in o:
                break
            else:
                self.fail("expected warning is not logged")  # pragma: no cover

        # make sure we can read the output starfile
        out = starfile.read(outfile)

        # make sure we only have the number of expected lines from the unique files
        self.assertEqual(self.n * 2, len(out))
