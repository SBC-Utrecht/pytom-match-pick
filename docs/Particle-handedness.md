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

