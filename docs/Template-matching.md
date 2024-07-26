This page contains usage information for `pytom_match_template.py`.

## Matching the template

This script requires at least a tomogram, a template, a mask, the min and max tilt angles (for missing wedge constraint), an angular search, and a GPU index to run. The search can be limited along any dimension with the `--search-x`, `--search-y`, and `--search-z` parameters; for example to skip some empty regions in the z-dimension where the ice layer is not yet present, or to remove some reconstruction artifact region along the x-dimension. With the `--volume-split` option, a tomogram can be split into chunks to allow them to fit in GPU memory (useful for large tomograms). Providing multiple GPU's will allow the program to split the angular search (or the subvolume search) over multiple cards to speed up the algorithm. 

Its crucial to consider the angular search for the particle of interest. The required search is found from the Crowther criterion $$\Delta \alpha = \frac{180}{\pi r_{max} d}$$ , where $r_{max}$ is the max resolution in fourier space of either template or tomogram, and d is the particle diameter. A ribosome (d=290A) in a 20A tomogram ( $r_{max}$ is $\frac{1}{2*20} A^{-1}$ ), has an optimal search of 7.9 degrees. In this case `--angular-search 7.00`  will give good separation of ribosomes from noise and artifacts.

In case the template matching is run with a non-spherical mask, it is essential to set the `--non-spherical-mask` flag. It requires a slight modification of the calculation that will roughly double the computation time, so only use non-spherical masks if absolutely necessary.

### Optimizing results: per tilt weighting with CTFs and dose accumulation

Optimal results are obtained by using the full 3D CTF dose-weighted model. You will need to pass the following files (and parameters):
- the `--per-tilt-weighting` flag needs to be set.
- a `.rawtlt` or `.tlt` file to the `--tilt-angles` parameter with all the tilt angles used to reconstruct the tomogram. Each line contains a single float value that specifies the tilt around the y-axis in degrees.
- a `.txt` file to `--dose-accumulation` with the accumulated dose per tilt (assuming the same ordering as `.tlt`). Each line contains a single float specifying the accumulated dose in **e-/A2**.
- a `.defocus` or `.txt` file to `--defocus-file` specifying the defocus of each tilt in **nm**. The `.txt` file sticks to the same format as the dose-accumulation file. The `.defocus` files are output from IMOD and can be directly passed. The CTF will also require input for `--voltage`, `--amplitude-contrast`, and `--spherical-abberation`.

_(As a side note, you can also only enable `--per-tilt-weighting` **without** dose accumulation and CTFs, or **with either** dose accumulation or CTFs.)_

When enabling the CTF model here (with the defocus file), it is important that the template is not multiplied with a CTF before passing it to this script. The template only needs to be scaled to the correct pixel size and the contrast should be adjusted to match the contrast in the tomograms.

Secondly, the tomogram should also be CTF corrected, for example by using IMODs strip-based CTF correction. However, software such as NovaCTF likely provides even better results, especially for matching high-resolution signal, as the CTF correction model is more accurate.

### Spectrum whitening

Spectrum whitening generally gives improved results by sharpening the correlation peaks. It is activated with the `--spectral-whitening` flag and does not require any further user input.

### Usage

```
pytom_match_template.py 
  [-h] 
  -t TEMPLATE 
  -m MASK 
  [--non-spherical-mask]
  -v TOMOGRAM 
  [-d DESTINATION] 
  -a TILT_ANGLES [TILT_ANGLES ...] 
  [--per-tilt-weighting] 
  --angular-search ANGULAR_SEARCH
  [--z-axis-rotational-symmetry Z_AXIS_ROTATIONAL_SYMMETRY]
  [-s VOLUME_SPLIT VOLUME_SPLIT VOLUME_SPLIT] 
  [--search-x SEARCH_X SEARCH_X] 
  [--search-y SEARCH_Y SEARCH_Y] 
  [--search-z SEARCH_Z SEARCH_Z]
  [--voxel-size-angstrom VOXEL_SIZE_ANGSTROM] 
  [--low-pass LOW_PASS] 
  [--high-pass HIGH_PASS]
  [--dose-accumulation DOSE_ACCUMULATION] 
  [--defocus-file DEFOCUS_FILE]
  [--amplitude-contrast AMPLITUDE_CONTRAST]
  [--spherical-abberation SPHERICAL_ABBERATION] 
  [--voltage VOLTAGE]
  [--spectral-whitening] 
  -g GPU_IDS [GPU_IDS ...] 
  [--log LOG]
``` 

### Parameters

* ` -h, --help`
  > show this help message and exit
* ` -t TEMPLATE, --template TEMPLATE`
  > Template; MRC file.
* ` -m MASK, --mask MASK`
  > Mask with same box size as template; MRC file.
* ` --non-spherical-mask`
  > Flag that sets the required computation for non-spherical mask inputs. Roughly doubles computation time, bus is essential when the mask is not a sphere.
* ` -v TOMOGRAM, --tomogram TOMOGRAM`
  > Tomographic volume; MRC file.
* ` -d DESTINATION, --destination DESTINATION`
  > Folder to store the files produced by template matching.
* ` -a TILT_ANGLES [TILT_ANGLES ...], --tilt-angles TILT_ANGLES [TILT_ANGLES ...]`
  > Tilt angles of the tilt-series, either the minimum and maximum values of the tilts (e.g. --tilt-angles -59.1 60.1) or a .rawtlt/.tlt file with all the angles (e.g. --tilt-angles tomo101.rawtlt). In case all the tilt angles are provided a more elaborate Fourier space constraint can be used.
* ` --per-tilt-weighting` 
  > Flag to activate per-tilt-weighting, only makes sense if a file with all tilt angles has been provided. In case not set, while a tilt angle file is provided, the minimum and maximum tilt angle are used to create a binary wedge. The base functionality of this flag creates a fanned wedge where each tilt is weighted by cos(tilt_angle). If dose accumulation and CTF parameters are provided these will all be incorporated in the tilt-weighting.
* ` --angular-search ANGULAR_SEARCH`
  > Options are: [7.00, 35.76, 19.95, 90.00, 18.00, 12.85, 38.53, 11.00, 17.86, 25.25, 50.00, 3.00] Alternatively, a .txt file can be provided with three Euler angles (in radians) per line that define the angular search. Angle format is ZXZ anti-clockwise (see: https://www.ccpem.ac.uk/user_help/rotation_conventions.php).
* ` --z-axis-rotational-symmetry Z_AXIS_ROTATIONAL_SYMMETRY`
  > Integer value indicating the rotational symmetry of the template around the z-axis. The length of the rotation search will be shortened through division by this value. Only works for template symmetry around the z-axis.
* ` -s VOLUME_SPLIT VOLUME_SPLIT VOLUME_SPLIT, --volume-split VOLUME_SPLIT VOLUME_SPLIT VOLUME_SPLIT`
  > Split the volume into smaller parts for the search, can be relevant if the volume does not fit into GPU memory. Format is x y z, e.g. --volume-split 1 2 1 will split y into 2 chunks, resulting in 2 subvolumes. --volume-split 2 2 1 will split x and y in 2 chunks, resulting in 2 x 2 = 4 subvolumes.
* ` --search-x SEARCH_X SEARCH_X`
  > Start and end indices of the search along the x-axis, e.g. --search-x 10 490
* `--search-y SEARCH_Y SEARCH_Y`
  > Start and end indices of the search along the y-axis, e.g. --search-x 10 490
* ` --search-z SEARCH_Z SEARCH_Z`
  > Start and end indices of the search along the z-axis, e.g. --search-x 30 230
* ` --voxel-size-angstrom VOXEL_SIZE_ANGSTROM`
  > Voxel spacing of tomogram/template in angstrom, if not provided will try to read from the MRC files. Argument is important for band-pass filtering!
* ` --low-pass LOW_PASS`   
  > Apply a low-pass filter to the tomogram and template. Generally desired if the template was already filtered to a certain resolution. Value is the resolution in A.
* ` --high-pass HIGH_PASS`
  > Apply a high-pass filter to the tomogram and template to reduce correlation with large low frequency variations. Value is a resolution in A, e.g. 500 could be appropriate as the CTF is often incorrectly modelled up to 50nm.
* ` --dose-accumulation DOSE_ACCUMULATION`
  > Here you can provide a file that contains the accumulated dose at each tilt angle, assuming the same ordering of tilts as the tilt angle file. Format should be a .txt file with on each line a dose value in e-/A2 .
* ` --defocus-file DEFOCUS_FILE`
  > Here you can provide an IMOD defocus file (version 2 or 3) or a text file with defocus values. This data, together with the other ctf parameters (amplitude contrast, voltage, spherical abberation), will be used to create a 3D CTF weighting function. IMPORTANT: if you provide this, the input template should not be modulated with a CTF beforehand. Format should be .defocus (IMOD) or .txt, same ordering as tilt angle list. The .txt file should contain a single defocus value (in nm) per line.
* ` --amplitude-contrast AMPLITUDE_CONTRAST`
  > Amplitude contrast fraction for CTF.
* ` --spherical-abberation SPHERICAL_ABBERATION`
  > Spherical abberation for CTF in mm.
* ` --voltage VOLTAGE`     
  > Voltage for CTF in keV.
* ` --spectral-whitening`  
  > Calculate a whitening filtering from the power spectrum of the tomogram; apply it to the tomogram patch and template. Effectively puts more weight on high resolution features and sharpens the correlation peaks.
* ` -g GPU_IDS [GPU_IDS ...], --gpu-ids GPU_IDS [GPU_IDS ...]`
  > GPU indices to run the program on.
* ` --log LOG`
  > Can be set to `info` or `debug`