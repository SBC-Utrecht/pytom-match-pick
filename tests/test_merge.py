import unittest
import starfile
from tempfile import TemporaryDirectory
from pytom_tm.entry_points import merge_stars
from .testing_utils import make_random_particles


class TestMergeStars(unittest.TestCase):
    def setUp(self):
        self.tempdir = TemporaryDirectory()
        self.dirname = self.tempdir.name
        self.addCleanup(self.tempdir.cleanup)

    def test_error_on_empty(self):
        # make sure we raise if we don't find any starfiles
        with self.assertRaisesRegex(ValueError, "No starfiles"):
            merge_stars(["self.dirname"])

    def test_relion4_mode(self):
        # write 2 starfiles
        particles1 = make_random_particles()
        particles2 = make_random_particles()
        for particle in [particles1, particles2]:
            tomo_id = particle["rlnMicrographName"][0]
            starfile.write(
                {"particles": particle}, self.tempdir / f"{tomo_id}_particles.star"
            )

        outfile = str(self.tempdir / "test.star")
        # Make a joined file via the entry point
        merge_stars([f"{self.dirname}", "-o", outfile])

        # make sure we can read the output starfile
        out = starfile.read(outfile)

        # check that both tomo IDs are in the new starfile
        for particle in [particles1, particles2]:
            self.assertIn(
                particle["rlnMicrographName"], out["particles"]["rlnMicrographName"]
            )

    def test_fail_on_incompatible_starfiles(self):
        # Make sure we fail if we try to combine RELION4 starfiles
        # with a RELION5 flag
        self.fail()

    def test_relion5_mode(self):
        particles1 = make_random_particles(relion5=True)
        particles2 = make_random_particles(relion5=True)
        for particle in [particles1, particles2]:
            tomo_id = particle["rlnTomoName"][0]
            starfile.write(
                {"particles": particle}, self.tempdir / f"{tomo_id}_particles.star"
            )

        outfile = str(self.tempdir / "test.star")
        # Make a joined file via the entry point
        merge_stars([f"{self.dirname}", "-o", outfile, "--relion5-compat"])

        # make sure we can read the output starfile
        out = starfile.read(outfile)
        out = out["particles"]
        # make sure that the outfile is as expected:
        # - 2 columns with names _rlnTomoName and _rlnTomoImportParticleFile
        self.assertEqual(2, len(out.columns))
        for col in ["rlnTomoName", "rlnTomoImportParticleFile"]:
            self.assertIn(col, out.columns)

        # check that both tomo IDs are in the new starfile and the correct filename
        for particle in [particles1, particles2]:
            self.assertIn(particle["rlnTomoName"], out["rlnTomoName"])

        # Check that all star files can be read and contains the expected tomoname
        for _, (tomoname, filename) in out.iterrows():
            temp = starfile.read(filename)
            temp_tomoname = set(temp["particles"]["rlnTomoName"])
            self.assertIn(tomoname, temp_tomoname)
