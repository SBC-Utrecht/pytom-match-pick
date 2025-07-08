import unittest
import starfile
from tempfile import TemporaryDirectory
from pytom.entry_points import merge_stars
from .testing_utils import make_random_particles


class TestMergeStars(unittest.TestCase):
    def setUp(self):
        self.tempdir = TemporaryDirectory()
        self.dirname = self.tempdir.name
        self.addCleanup(self.tempdir.cleanup)

    def test_relion_4_mode(self):
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
