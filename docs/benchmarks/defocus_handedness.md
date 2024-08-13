Although it has been shown multiple times that correcting defocus gradients is very 
important for subtomogram averaging in tilt-series data, the effects on particle 
localization have not been investigated (to my knowledge). The software Warp 
introduced a detailed correction of defocus gradients during template matching by 
splitting the tomogram into many small sub-boxes where the template can be corrected 
by tilt-dependent defocus offsets 
that adhere to the sample geometry. For an untilted image these 
offsets are a function of the z-coordinate in the tomogram, while for tilted image the 
gradient is a function of both the x- and z-coordinate in the tomogram (assuming the 
tilt-axis is aligned with the y-axis). The defocus gradient are therefore expected 
to be the strongest for the images collected a high sample tilts. Considering that the 
resolution in tomograms is generally not considered to be high due to alignment errors 
and that the high tilt angles usually have an additional drop-off in resolution due to 
beam damage, I wondered how much effect defocus gradient correction actually has 
on the template matching scores. To test I tried to measure the defocus handedness 
of a tomogram in this benchmark.

## Approach

To properly measure the defocus handedness, I selected a dataset of isolated 
ribosomes in thin ice 
(EMPIAR-10985). Isolated macromolecules provide the highest resolution, while _in situ_ 
data would be more limited in resolution. I used an approach, similar to Warp, 
that calculates the defocus offsets in each subvolume of a tomogram. To measure the 
effects of defocus gradients, I 
ran template matching assuming both a default and inverted defocus handedness of the 
tilt-series. This inverted handedness comes down to 
inverting the tilt angles during 
calculation of the defocus offsets. Either one of these two handedness' should be 
correct and hopefully influence the results sufficiently to see a difference. 

<figure markdown="span">
<figure markdown="span">
  ![annotations](defocus_handedness_figures/blik_view.png){ width="400" }
  <figcaption>Initial view of the data: annotations (turqoise sphere) made by 
pytom-match-pick on tomogram 27 of EMPIAR-10985.</figcaption>
</figure>

## Results

To assess the effects of the defocus offsets, I analyzed the results as a 
function of the x-coordinate. As this tomogram had a very thin ice 
layer (~50nm), the defocus offsets are primarily influenced by the position in x and 
the tilt angle. 

<figure markdown="span">
  ![annotations](defocus_handedness_figures/x_vs_defocus.svg)
  <figcaption>LCC<sub>max</sub> scores (normalized by the standard deviation from 
template matching plotted as a function 
of the x-coordinate. In blue the results are shown that assumes the default defocus 
handedness, while orange shows the results of inverted defocus handedness. The left 
figure shows a scatter plot, while the right shows fitted quadratic functions to 
both sets of points. The gray areas indicate the 95% confidence interval of the fit.  
</figcaption>
</figure>

Right now, the inverted handedness seems to be the correct one. However, if the tilt-axis angle would have been set to 180 for AreTomo, the template would not need to be mirrored and the defocus handedness would also not have to be inverted. I decided not to rerun this analysis with that setting as the point here was to examine the effect of the regular/inverted defocus handedness.


If you found these results useful, please cite our repository: https://zenodo.org/records/12667665

## How-to-reproduce

### Requirements

* AreTomo 1.3 
* IMOD 4.11.24
* pytom-match-pick 0.7.3 (and should work with higher versions)

### Implementation of defocus gradient

An implementation for the defocus gradient calculation is available in our 
repository in `src/pytom_tm/TMJob.py`, see function 
`get_defocus_offsets()`. 

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

Our repository contains the file  
docs/benchmarks/defocus_gradient_analysis.ipynb that contains the exact code to 
reproduce the plots shown here. In brief:

* The notebook reads the starfiles containing the regular and inverted defocus 
  handedness template matching annotations.
* A check is made that the set of coordinates overlaps between the two jobs is the same.
* The results are plotted as x-coordinate versus the score (after normalizing by the 
  standard deviation).
* A quadratic curve is fit to both these sets of points to better visualize any 
  changes between them.

### (Optional) Visualization (with Blik)

Using Blik version 0.9.

```commandline
napari -w blik -- tm_patch_def_inv/9x9_ts_27_particles.star tomogram/9x9_ts_27.mrc
```

Then do the following steps:

* Set the `Slice thickness A` on the right to 40
* From the `Experiment` dropdown menu on the right select the tomogram
* Click in the center
* Then do `Ctrl + Y` to toggle 3D view
* Select the points layer (ends with `- particle positions`)
* In layer controls select the icon `Select points` 
* Do `Ctrl + A` and then set the `Point size` to 10
* Click in the center (not on a point) to deselect the points
* Press `Ctrl + Y` again to switch back to 2D view

Now you can scroll through the z-axis of the tomogram and the template matching 
annotations with the slider on the bottom.
