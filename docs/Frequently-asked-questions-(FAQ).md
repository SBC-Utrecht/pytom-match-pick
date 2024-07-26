## How to deal with gold beads?

Gold beads (and other artifacts) can often interfer in template matching due to their high electron scattering potential. The easiest way to deal with them is by removing them on the micrograph level prior to reconstruction. The following IMOD commands can do the trick:

```
imodfindbeads -size [GOLD_BEAD_DIAMETER_IN_PIXELS] -input TS_ID.st -output  TS_ID.fid  -spacing 0.8
ccderaser --input TS_ID.st --output TS_ID_erased.st -model TS_ID.fid -order 0 -circle / -better [1.5 * GOLD_BEAD_RADIUS_IN_PIXELS] -merge 1 -exclude -skip 1 -expand 3
```

## What template box size should I use?

For the simple missing wedge model a box size that tightly fits the template and mask is easiest (and slightly faster).

For the full per-tilt-weighting model its better to have some overhang, a rought estimate could be a box size of `4 * particle_diameter`. This aids in sampling the CTF function and the individual tilts in Fourier space. However, the mask radius should still snuggly fit the template. Although the larger box size slows down rotations slightly, the search benefits more from better sampling of the weighting function.
