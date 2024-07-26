This page contains usage information for `pytom_extract_candidates.py` and `pytom_estimate_roc.py`.

Both scripts run on the job file created in `pytom_match_template.py` which contains details about correlation statistics and the output files. The job file always has the format `[TOMO_ID]_job.json`.

**IMPORTANT** For both scripts the `[-r, --radius-px]` option needs to be considered carefully. The particle extraction will mask out spheres with this radius around each peak in the score volume and prevents selecting the same macromolecule twice. It is specified as an integer **number of pixels** (not Angstrom!) and ideally it should be the radius of the particle of interest. It can be found by dividing the particle radius by the pixel size, e.g. a ribosome (r = 290A / 2) in a 15A tomogram should gets a pixel radius of 9.6. As it needs to be an integer value and ribosomes are not perfect spheres, it is best to round it down to 9 pixels. 

## Extracting particles 

Resulting STAR files from extraction have three colums with extraction statistics (`LCCmax`, `CutOff`, `SearchStd`). Dividing the `LCCmax` and the `CutOff` by the `SearchStd`, will express them as a number of $\sigma$ or (3D SNR; similar to [Rickgauer et al. (2017)](https://doi.org/10.7554/eLife.25648).

STAR files written out by the template matching module will have RELION compliant column headers, i.e. `rlnCoordinateX` and `rlnAgleRot`, to simplify integration with other software. The Euler angles that are written out therefore also follow the same conventions as RELION and Warp, i.e. `rlnAngleRot`, `rlnAngleTilt`, `rlnAnglePsi` are intrinsic clockwise ZYZ Euler angles. Hence they can be directly used for subtomogram averaging in RELION. See here for more info: [https://www.ccpem.ac.uk/user_help/rotation_conventions.php](https://www.ccpem.ac.uk/user_help/rotation_conventions.php).

The particle extraction has been updated to use the formula in [Rickgauer et al. (2017)](https://doi.org/10.7554/eLife.25648) for finding the extraction threshold based on the false alarm rate. This was not yet described in our [IJMS publication](https://www.mdpi.com/1422-0067/24/17/13375) but is essentially very similar to the Gaussian fit that we used. However, it is more reliable and also specific to the standard deviation $\sigma$ of the search in each tomogram. `pytom_match_template.py` keeps track of $\sigma$ and stores it in the job file. The user can specify a number of false positives to allow per tomogram with a minimum value of 1. It can be increased to make the extraction threshold more lenient which might increase the number of true positives at the expense of more false positives. The parameter should roughly correspond to the number of false positives that end up in the extracted particle list.

#### Mathematical background of extraction statistic

Template matching has a huge search space (N_voxels * N_rotations) which is mainly false positives, and has in comparison a tiny fraction of true positives. If we have a Gaussian for the background (with expected mean 0 and some standard deviation), the false alarm rate can be calculated for a certain cut-off value, as it is dependent on the size of the search space. For example, a false alarm rate of (N_voxels * N_rotations)^(-1), indicates it would expect 1 false positive in the whole search. This can be calculated with the error function,

$$N^{-1} = \text{erfc}( \theta / ( \sigma \sqrt{2} ) ) / (2 n_{\text{FP}})$$

, where theta is the cut-off, sigma the standard deviation of the Gaussian, and N the search space. $n_{\text{FP}}$ represents the scaling by the user of tolerated number of false positives.


### Usage


```
pytom_extract_candidates.py 
  [-h] 
  -j JOB_FILE 
  -n NUMBER_OF_PARTICLES 
  [--number-of-false-positives NUMBER_OF_FALSE_POSITIVES] 
  -r RADIUS_PX 
  [-c CUT_OFF] 
  [--log LOG]
```

### Parameters

* ` -h, --help`            
  > Show the help message and exit.
* ` -j JOB_FILE, --job-file JOB_FILE`
  > JSON file that contain all data on the template matching job, written out by pytom_match_template.py in the destination path with format `[TOMO_ID]_job.sjon`.
* ` -n NUMBER_OF_PARTICLES, --number-of-particles NUMBER_OF_PARTICLES`
  > Maximum number of particles to extract from tomogram.
* ` --number-of-false-positives NUMBER_OF_FALSE_POSITIVES`
  > Number of false positives to determine the false alarm rate. Here, the sensitivity of the particle extraction can be increased at the expense of more false positives. The default value of 1 is recommended for particles that can be distinguished well from the background (high specificity).
* ` -r RADIUS_PX, --radius-px RADIUS_PX`
  > Particle radius in pixels in the tomogram. It is used during extraction to remove areas around peaks preventing double extraction.
* ` -c CUT_OFF, --cut-off CUT_OFF`
  > Override automated extraction cutoff estimation and instead extract the number_of_particles down to this LCCmax value. Set to -1 to guarantee extracting number_of_particles. Values larger than 1 make no sense as the correlation cannot be higher than 1.
* ` --log LOG`
  > Can be switched from `info` to `debug` mode.

## Evaluate classification

This script runs the Gaussian fit as described in the [IJMS publication](https://www.mdpi.com/1422-0067/24/17/13375). It requires installation with plotting dependencies as it writes out or displays a figure showing the Gaussian fit and estimated ROC curve. The benefit is that it estimates some classification statistics (such as false discovery rate and sensitivity). You can use it to esimate an extraction threshold for a representative tomogram and then supply this threshold as the `[-c, --cut-off]` parameter for `pytom_extract_candidates.py`.

### Usage

```
pytom_estimate_roc.py 
  [-h] 
  -j JOB_FILE 
  -n NUMBER_OF_PARTICLES 
  -r RADIUS_PX 
  [--bins BINS] 
  [--gaussian-peak GAUSSIAN_PEAK] 
  [--force-peak] 
  [--crop-plot] 
  [--show-plot] 
  [--log LOG]
```

### Parameters

* ` -h, --help`
  > show this help message and exit
* `-j JOB_FILE, --job-file JOB_FILE`
  > JSON file that contain all data on the template matching job, written out by pytom_match_template.py in the destination path.
* ` -n NUMBER_OF_PARTICLES, --number-of-particles NUMBER_OF_PARTICLES`
  > The number of particles to extract and estimate the ROC on, recommended is to multiply the expected number of particles by 3.
* ` -r RADIUS_PX, --radius-px RADIUS_PX`
  > Particle radius in pixels in the tomogram. It is used during extraction to remove areas around peaks preventing double extraction.
* ` --bins BINS`
  > Number of bins for the histogram to fit Gaussians on.
* ` --gaussian-peak GAUSSIAN_PEAK`
  > Expected index of the histogram peak of the Gaussian fitted to the particle population.
* ` --force-peak`
  > Force the particle peak to the provided peak index.
* ` --crop-plot`
  > Flag to crop the plot relative to the height of the particle population.
* ` --show-plot`
  > Flag to use a pop-up window for the plot instead of writing it to the location of the job file.
