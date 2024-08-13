## Methods

### Requirements

* AreTomo 1.3 
* IMOD 4.11.24
* pytom-match-pick 0.7.3

### Tilt series preprocessing

Download tilt-series 27 from EMPIAR-10985:

* 9x9_ts_27_sort.mrc
* 9x9_ts_27_sort.mrc.mdoc
* 9x9_ts_27_sort_dose.txt

I put the data in the following directory structure:

```
empiar-10985/
+- raw_data/
¦  +- 9x9_ts_27_sort.mrc
¦  +- 9x9_ts_27_sort.mrc.mdoc
¦  +- 9x9_ts_27_sort_dose.txt
+- templates/
+- metadata/
+- tomogram/
+- tm_init/
+- tm_patch_def_reg/
+- tm_patch_def_inv/
```

Create a file with accumulated dose and a rawtlt file:

```commandline
grep '^TiltAngle' raw_data/9x9_ts_27_sort.mrc.mdoc | awk -F'=' '{print $2}' > metadata/9x9_ts_27_sort.rawtlt
```

```commandline
awk '{print $1}' raw_data/9x9_ts_27_sort_dose.txt > metadata/9x9_ts_27_sort_accumulated_dose.txt
```

Use AreTomo (1.3) to align and reconstruct the tilt-series. The .mdoc file does not specify a tilt-axis angle, so I let AreTomo find it.

```commandline
aretomo -InMrc raw_data/9x9_ts_27_sort.mrc -OutMrc tomogram/9x9_ts_27.mrc -AngFile metadata/9x9_ts_27_sort.rawtlt -AlignZ 400 -VolZ 600 -OutBin 8  -Gpu 0  -Wbp 1 -FlipVol 1
```

Fit ctf with IMOD; in the GUI select `fit each view separately` and then `autofit all 
views`. Tilt-axis angle is set to '-3.2323' as optimized by AreTomo (check the .aln file).

```commandline
ctfplotter -input raw_data/9x9_ts_27_sort.mrc -angleFn metadata/9x9_ts_27_sort.rawtlt -aAngle -3.2323 -defFn metadata/9x9_ts_27_sort.defocus -pixelSize 0.107 -volt 300 -cs 2.7  -expDef 3000 -range -60,60
```

### Creating a template and mask

Prepare the template and mask. I downloaded the subtomogram average (EMD-33115) calculated for the EMPIAR dataset. NOTE: The reference is mirrored as I noticed in the first run without mirroring that the scores were unexpectedly poor. Mirroring fixes this issue.

```commandline
pytom_create_template.py -i templates/emd_33115.map -o templates/70S.mrc --output-voxel 8.56 -b 60 --invert --mirror
```

```commandline
pytom_create_mask.py -b 60 -o templates/mask.mrc --voxel-size 8.56 -r 14 -s 1
```

### Running template matching
(Optional) Run a simple job with low-pass to confirm if localization is working. Low-pass filtering reduces available resolution and therefore the angular search.

```commandline
pytom_match_template.py -t templates/70S.mrc -m templates/mask.mrc -v tomogram/9x9_ts_27.mrc -d tm_init --particle-diameter 250 -a metadata/9x9_ts_27_sort.rawtlt --per-tilt-weighting --voxel-size 8.56 --dose metadata/9x9_ts_27_sort_accumulated_dose.txt --defocus metadata/9x9_ts_27_sort.defocus  --amplitude 0.07 --spherical 2.7 --voltage 300 -g 0 --log debug --low-pass 30
```

```commandline
pytom_extract_candidates.py -j tm_init/9x9_ts_27_job.json -n 1000 -r 5 --cut-off 0.4
```

Run with default defocus handedness and a full rotation search. NOTE: defocus 
handedness is calculated for subvolumes so 
the `-s` option needs to be used to split the tomogram into multiple subvolumes. Only the splits along the x-axis influence the defocus values in this case as that is perpendicular to the tilt axis. (i.e. `-s 3 1 1` should produce the same results.).

```commandline
pytom_match_template.py -t templates/70S.mrc -m templates/mask.mrc -v tomogram/9x9_ts_27.mrc -d tm_patch_def_reg --particle-diameter 250 -a metadata/9x9_ts_27_sort.rawtlt --per-tilt-weighting --voxel-size 8.56 --dose metadata/9x9_ts_27_sort_accumulated_dose.txt --defocus metadata/9x9_ts_27_sort.defocus  --amplitude 0.07 --spherical 2.7 --voltage 300 -g 0 -s 3 3 1 --defocus-handedness 1
```

```commandline
pytom_extract_candidates.py -j tm_patch_def_reg/9x9_ts_27_job.json -n 1000 -r 5 --cut-off 0.3
```

Run with inverted defocus handedness.

```commandline
pytom_match_template.py -t templates/70S.mrc -m templates/mask.mrc -v tomogram/9x9_ts_27.mrc -d tm_patch_def_inv --particle-diameter 250 -a metadata/9x9_ts_27_sort.rawtlt --per-tilt-weighting --voxel-size 8.56 --dose metadata/9x9_ts_27_sort_accumulated_dose.txt --defocus metadata/9x9_ts_27_sort.defocus  --amplitude 0.07 --spherical 2.7 --voltage 300 -g 0 -s 3 3 1 --defocus-handedness -1
```

```commandline
pytom_extract_candidates.py -j tm_patch_def_inv/9x9_ts_27_job.json -n 1000 -r 5 --cut-off 0.3
```

### Plotting the results

## Results and conclusions

Right now, the inverted handedness seems to be the correct one. However, if the tilt-axis angle would have been set to 180 for AreTomo, the template would not need to be mirrored and the defocus handedness would also not have to be inverted. I decided not to rerun this analysis with that setting as the point here was to examine the effect of the regular/inverted defocus handedness.





