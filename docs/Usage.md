
Features in a tomogram that resemble a structural 'template' can be localized in an automated fashion using 'template matching'. In this approach a 3D template is correlated with a given tomogram. In this procedure the different possible rotations and translations are sampled exhaustively using the algorithm described in [Förster et al. (2010)](http://dx.doi.org/10.1016/S0076-6879(10)83011-3).

## Requirements

For usage you need at least a set of reconstructed tomograms in the MRC format and a template structure in the MRC format. Tomograms in IMOD format (.rec) are also okay but need to be renamed (or softlinked!) to have the correct extension (.mrc). Tomograms are ideally binned 6x or 8x to prevent excessive runtimes. The template can be an EM reconstruction (from the EMDB) or a PDB that was coverted to a density (for example via Chimera molmap).

## Template matching workflow

Using template matching in this software consists of the following steps:

1. Creating a template and mask
2. Matching the template in a tomogram
3. Extracting particles
4. Merging annotations for export to other software


## 1. Creating a template and mask

**Important**:
- The template and mask need to have the same box size.
- The template needs to have the same contrast as the tomogram (e.g. the particles 
  are black in both the tomogram and template). Contrast can be adjusted with the 
  `--invert` option.

### pytom_create_template.py

Using an EM map as a reference structure generally leads to the best results. Alternatively a structure from the PDB can be converted in Chimera(X) using the molmap command to create an MRC file that models the electrostatic potential.

```python exec="on" result="ansi" 
import argparse
code = ("""
---8<--- "entry_points.py:create_template_usage"
""")
cleaned_lines = [line.lstrip() for line in code.splitlines()
    if not any(keyword in line for keyword in ('action=', 'default=', 'type='))
]
cleaned_code = '\n'.join(cleaned_lines)
exec(cleaned_code)
print(parser.format_help())
```

### pytom_create_mask.py

The mask around the template can be quite tight to remove as much noise as possible around the particles of interest. We recommend around 10%-20% overhang relative to the particle radius. You can also generate an ellipsoidal mask for particles that do not approximate well as a sphere. Though you will probably need to reorient this mask in chimera and resample to the grid of the template. Optionally you could also create a structured mask around the template in external software (via thresholding and dilation for example). Take into account that non-spherical masks roughly double the template matching computation time.

```python exec="on" result="ansi"
import argparse
code = ("""
---8<--- "entry_points.py:create_mask_usage"
""")
cleaned_lines = [line.lstrip() for line in code.splitlines()
    if not any(keyword in line for keyword in ('action=', 'default=', 'type='))
]
cleaned_code = '\n'.join(cleaned_lines)
exec(cleaned_code)
print(parser.format_help())
```

## 2. Matching the template in a tomogram

### pytom_match_template.py

This script requires at least a tomogram, a template, a mask, the min and max tilt angles (for missing wedge constraint), an angular search, and a GPU index to run. The search can be limited along any dimension with the `--search-x`, `--search-y`, and `--search-z` parameters; for example to skip some empty regions in the z-dimension where the ice layer is not yet present, or to remove some reconstruction artifact region along the x-dimension. With the `--volume-split` option, a tomogram can be split into chunks to allow them to fit in GPU memory (useful for large tomograms). Providing multiple GPU's will allow the program to split the angular search (or the subvolume search) over multiple cards to speed up the algorithm. 

The software automatically calculates the angular search based on the available 
resolution and provided particle diameter. The required search is found from the 
Crowther criterion $\Delta \alpha = \frac{180}{\pi r_{max} d}$. For the maximal 
resolution the voxel size is used, unless a low-pass filter is specified as this 
limits the available maximal resolution. You can exploit this to reduce the angular 
search! For non-spherical particles we suggest choosing the particle diameter as the 
longest axis of the macromolecule. 

In case the template matching is run with a non-spherical mask, it is essential to set the `--non-spherical-mask` flag. It requires a slight modification of the calculation that will roughly double the computation time, so only use non-spherical masks if absolutely necessary.

#### Optimizing results: per tilt weighting with CTFs and dose accumulation

Optimal results are obtained by also incorporating information for the 3D CTF. You 
can pass the following files (and parameters):
- Tilt angles: a `.rawtlt` or `.tlt` file to the `--tilt-angles` parameter with all the 
  tilt 
  angles used to reconstruct the tomogram. You should then also set the 
  `--per-tilt-weighting` flag.
- CTF data: a `.defocus` file from IMOD or `.txt` file to `--defocus-file`. The `.txt` file 
  should specify the defocus of each tilt in **$\micro m$**. You can also give a 
  single defocus value (in $\micro m$). The CTF will also require input for 
  `--voltage`, `--amplitude-contrast`, and `--spherical-abberation`.
- Dose weighting: a `.txt` file to `--dose-accumulation` with the accumulated dose per 
  tilt (assuming the same ordering as `.tlt`). Each line contains a single float 
  specifying the accumulated dose in **e-/A2**. Dose weighting only works in 
  combination with `--per-tilt-weighting`.

_(As a side note, you can also only enable `--per-tilt-weighting` **without** dose accumulation and CTFs, or **with either** dose accumulation or CTFs.)_

When enabling the CTF model here (with the defocus file), it is important that the template is not multiplied with a CTF before passing it to this script. The template only needs to be scaled to the correct pixel size and the contrast should be adjusted to match the contrast in the tomograms.

Secondly, if the tomogram was CTF corrected, for example by using IMODs 
strip-based CTF correction or NovaCTF. Its important to add the parameter 
`--tomogram-ctf-model phase-flip` which modifies the template CTF to match the 
tomograms CTF correction.

#### Background corrections

The software contains two background correction methods that might improve results: 
`--spectral-whitening` or `--random-phase-correction` (from STOPGAP). In our 
experience the random phase correction is most reliable, while spectral whitening 
never seemed to clearly improve results.

```python exec="on" result="ansi"
import argparse
code = ("""
---8<--- "entry_points.py:match_template_usage"
""")
cleaned_lines = [line.lstrip() for line in code.splitlines()
    if not any(keyword in line for keyword in ('action=', 'default=', 'type='))
]
cleaned_code = '\n'.join(cleaned_lines)
exec(cleaned_code)
print(parser.format_help())
```

## 3. Extracting particles

Both scripts run on the job file created in `pytom_match_template.py` which contains details about correlation statistics and the output files. The job file always has the format `[TOMO_ID]_job.json`.

**IMPORTANT** For both scripts the `[-r, --radius-px]` option needs to be considered carefully. The particle extraction will mask out spheres with this radius around each peak in the score volume and prevents selecting the same macromolecule twice. It is specified as an integer **number of pixels** (not Angstrom!) and ideally it should be the radius of the particle of interest. It can be found by dividing the particle radius by the pixel size, e.g. a ribosome (r = 290A / 2) in a 15A tomogram should gets a pixel radius of 9.6. As it needs to be an integer value and ribosomes are not perfect spheres, it is best to round it down to 9 pixels. 

### pytom_extract_candidates.py

#### STAR file metadata

Resulting STAR files from extraction have three colums with extraction statistics (`LCCmax`, `CutOff`, `SearchStd`). Dividing the `LCCmax` and the `CutOff` by the `SearchStd`, will express them as a number of $\sigma$ or (3D SNR; similar to [Rickgauer et al. (2017)](https://doi.org/10.7554/eLife.25648).

STAR files written out by the template matching module will have RELION compliant column headers, i.e. `rlnCoordinateX` and `rlnAgleRot`, to simplify integration with other software. The Euler angles that are written out therefore also follow the same conventions as RELION and Warp, i.e. `rlnAngleRot`, `rlnAngleTilt`, `rlnAnglePsi` are intrinsic clockwise ZYZ Euler angles. Hence they can be directly used for subtomogram averaging in RELION. See here for more info: [https://www.ccpem.ac.uk/user_help/rotation_conventions.php](https://www.ccpem.ac.uk/user_help/rotation_conventions.php).

Please see the [For developers](Developers.md) section for more details on the 
metadata. 

#### Default true positive estimation

The particle extraction has been updated to use the formula in [Rickgauer et al. (2017)](https://doi.org/10.7554/eLife.25648) for finding the extraction threshold based on the false alarm rate. This was not yet described in our [IJMS publication](https://www.mdpi.com/1422-0067/24/17/13375) but is essentially very similar to the Gaussian fit that we used. However, it is more reliable and also specific to the standard deviation $\sigma$ of the search in each tomogram. `pytom_match_template.py` keeps track of $\sigma$ and stores it in the job file. The user can specify a number of false positives to allow per tomogram with a minimum value of 1. It can be increased to make the extraction threshold more lenient which might increase the number of true positives at the expense of more false positives. The parameter should roughly correspond to the number of false positives that end up in the extracted particle list.

Template matching has a huge search space $N_{voxels} * N_{rotations}$ which is mainly 
false positives, and has in comparison a tiny fraction of true positives. If we have 
a Gaussian for the background (with expected mean 0 and some standard deviation), 
the false alarm rate can be calculated for a certain cut-off value, as it is 
dependent on the size of the search space. For example, a false alarm rate of $(N_
{voxels} * N_{rotations})^{-1}$, indicates it would expect 1 false positive in the 
whole search. This can be calculated with the error function,

$$N^{-1} = \text{erfc}( \theta / ( \sigma \sqrt{2} ) ) / (2 n_{\text{FP}})$$

, where theta is the cut-off, sigma the standard deviation of the Gaussian, and N the search space. $n_{\text{FP}}$ represents the scaling by the user of tolerated number of false positives.

#### Tophat transform filter

This option can be used to filter the score map for sharp peaks (steep local maxima) 
which usually correspond to true positives. This will be described in a forthcoming 
publication. For now, you can check out Marten's poster at CCPEM that shows some 
preliminary results: [10.5281/zenodo.13165643](https://doi.org/10.5281/zenodo.13165643).

```python exec="on" result="ansi"
import argparse
code = ("""
---8<--- "entry_points.py:extract_candidates_usage"
""")
cleaned_lines = [line.lstrip() for line in code.splitlines()
    if not any(keyword in line for keyword in ('action=', 'default=', 'type='))
]
cleaned_code = '\n'.join(cleaned_lines)
exec(cleaned_code)
print(parser.format_help())
```

### pytom_estimate_roc.py

This script runs the Gaussian fit as described in the [IJMS publication](https://www.mdpi.com/1422-0067/24/17/13375). It requires installation with plotting dependencies as it writes out or displays a figure showing the Gaussian fit and estimated ROC curve. The benefit is that it estimates some classification statistics (such as false discovery rate and sensitivity). You can use it to esimate an extraction threshold for a representative tomogram and then supply this threshold as the `[-c, --cut-off]` parameter for `pytom_extract_candidates.py`.

```python exec="on" result="ansi"
import argparse
code = ("""
---8<--- "entry_points.py:estimate_roc_usage"
""")
cleaned_lines = [line.lstrip() for line in code.splitlines()
    if not any(keyword in line for keyword in ('action=', 'default=', 'type='))
]
cleaned_code = '\n'.join(cleaned_lines)
exec(cleaned_code)
print(parser.format_help())
```

## 4. Merging annotations for export to other software

After running template matching and candidate extraction on multiple tomograms, each tomogram will have an individual starfile with particle annotations. Each starfile will contain the `MicrographName` column which refers back to the tomogram name. Multiple starfiles can therefore be appended to results in a large list which can be used in other software (such as RELION, WarpM) to load annotations. These software will link the annotations to specific tilt-series using the `MicrographName` column.

### pytom_merge_stars.py

Without providing any parameters the script will try to merge all the starfiles in the current working directory and save them to a new file `particles.star`.

```python exec="on" result="ansi"
import argparse
code = ("""
---8<--- "entry_points.py:merge_stars_usage"
""")
cleaned_lines = [line.lstrip() for line in code.splitlines()
    if not any(keyword in line for keyword in ('action=', 'default=', 'type='))
]
cleaned_code = '\n'.join(cleaned_lines)
exec(cleaned_code)
print(parser.format_help())
```



