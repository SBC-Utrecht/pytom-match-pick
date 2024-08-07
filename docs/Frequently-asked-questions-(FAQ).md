## How to deal with gold beads?

Gold beads (and other artifacts) can often interfer in template matching due to their high electron scattering potential. The easiest way to deal with them is by removing them on the micrograph level prior to reconstruction. The following IMOD commands can do the trick:

```
imodfindbeads -size [GOLD_BEAD_DIAMETER_IN_PIXELS] -input TS_ID.st -output  TS_ID.fid  -spacing 0.8
ccderaser --input TS_ID.st --output TS_ID_erased.st -model TS_ID.fid -order 0 -circle / -better [1.5 * GOLD_BEAD_RADIUS_IN_PIXELS] -merge 1 -exclude -skip 1 -expand 3
```

## What template box size should I use?

For the simple missing wedge model a box size that tightly fits the template and mask is easiest (and slightly faster).

For the full per-tilt-weighting model its better to have some overhang, a rought estimate could be a box size of `4 * particle_diameter`. This aids in sampling the CTF function and the individual tilts in Fourier space. However, the mask radius should still snuggly fit the template. Although the larger box size slows down rotations slightly, the search benefits more from better sampling of the weighting function.

## Is my particle handedness correct?

If template matching is giving unexpectedly poor results for the particle of interest, it might that the template has the wrong handedness. In that case, the `pytom_create_template.py` has the option `--mirror` to produce a mirrored version of the template. We advice to create a mirrored and non-mirrored version of the template and to run template matching with both. After extracting ~1000 particle from both jobs, while setting the `--cut-off` to -1 (in `pytom_extract_candidates.py`). You can plot the results with the following python code:

```
import starfile
import matplotlib.pyplot as plt

raw = starfile.read('[TOMO_ID]_particles.star')
mirror = starfile.read('[TOMO_ID]_mirror_particles.star')

fig, ax = plt.subplots(figsize=(5,5))
ax.plot(raw['ptmLCCmax'], label='raw')
ax.plot(mirror['ptmLCCmax'], label='mirror')
ax.set_xlabel('Particle index')
ax.set_ylabel('LCCmax')
ax.legend()
plt.show()
```

## What tomogram reconstruction method should I use?

Have a look at this thread on our Discussions page: <https://github.com/SBC-Utrecht/pytom-match-pick/discussions/206>! Feel free to join in with the disucssion if you have another opinion or questions.
