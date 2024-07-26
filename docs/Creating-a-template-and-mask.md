This page contains usage information for `pytom_create_template.py` and `pytom_create_mask.py`.

**Important**:
- The template and mask need to have the same box size.
- The template needs to have the same contrast as the tomogram (e.g. the particles are black in both the tomogram and template).

## Generating the template

Using an EM map as a reference structure generally leads to the best results. Alternatively a structure from the PDB can be converted in Chimera(X) using the molmap command to create an MRC file that models the electrostatic potential.

### About CTFs

Some form of CTF **must** be applied to the template:
- In case all the neccesary parameters for CTF correction can be passed to `pytom_match_template.py`, you should only scale the template and adjust its contrast in this script.
- Otherwise the template can be multiplied with a CTF here, in which case we often cut the CTF after the first zero crossing and apply a low pass filter. This is due to defocus gradient effects leading to wrong CTF crossings and reducing the correlation.

### Usage

```
pytom_create_template.py 
  [-h] 
  -i INPUT_MAP 
  [-o OUTPUT_FILE] 
  [--input-voxel-size-angstrom INPUT_VOXEL_SIZE_ANGSTROM] 
  --output-voxel-size-angstrom OUTPUT_VOXEL_SIZE_ANGSTROM
  [--center] 
  [-c] 
  [-z DEFOCUS] 
  [-a AMPLITUDE_CONTRAST] 
  [-v VOLTAGE] 
  [--Cs CS] 
  [--cut-after-first-zero] 
  [--flip-phase] 
  [--low-pass LOW_PASS] 
  [-b BOX_SIZE]
  [--invert] 
  [-m] 
  [--display-filter] 
  [--log LOG]
```

### Parameters

* ` -h, --help`
  > show this help message and exit
* ` -i INPUT_MAP, --input-map INPUT_MAP`
  > Map to generate template from; MRC file.
* `-o OUTPUT_FILE, --output-file OUTPUT_FILE`
  > Provide path to write output, needs to end in .mrc . If not provided file is written to current directory in the following format: template_{input_map.stem}_{voxel_size}A.mrc
* ` --input-voxel-size-angstrom INPUT_VOXEL_SIZE_ANGSTROM`
  > Voxel size of input map, in Angstrom. If not provided will be read from MRC input (so make sure it is annotated correctly!).
* ` --output-voxel-size-angstrom OUTPUT_VOXEL_SIZE_ANGSTROM`
  > Output voxel size of the template, in Angstrom. Needs to be equal to the voxel size of the tomograms for template matching. Input map will be downsampled to this spacing.
* ` --center`              
  > Set this flag to automatically center the density in the volume by measuring the center of mass.
* ` -c, --ctf-correction`
  > Set this flag to multiply the input map with a CTF. The following parameters are also important to specify because the defaults might not apply to your data: --defocus, --amplitude-contrast, --voltage, --Cs.
* ` -z DEFOCUS, --defocus DEFOCUS`
  > Defocus in um (negative value is overfocus).
* ` -a AMPLITUDE_CONTRAST, --amplitude-contrast AMPLITUDE_CONTRAST`
  > Fraction of amplitude contrast in the image ctf.
* ` -v VOLTAGE, --voltage VOLTAGE`
  > Acceleration voltage of electrons in keV
* ` --Cs CS`
  > Spherical aberration in mm.
* ` --cut-after-first-zero`
  > Set this flag to cut the CTF after the first zero crossing. Generally recommended to apply as the simplistic CTF convolution will likely become inaccurate after this point due to defocus gradients.
* ` --flip-phase`
  > Set this flag to apply a phase flipped CTF. Only required if the CTF is modelled beyond the first zero crossing and if the tomograms have been CTF corrected by phase flipping.
* ` --low-pass LOW_PASS`
  > Apply a low pass filter to this resolution, in Angstrom. By default a low pass filter is applied to a resolution of (2 * output_spacing_angstrom) before downsampling the input volume.
* ` -b BOX_SIZE, --box-size BOX_SIZE`
  > Specify a desired size for the output box of the template. Only works if it is larger than the downsampled box size of the input.
* ` --invert`
  > Multiply template by -1. WARNING not needed if ctf with defocus is already applied!
* ` -m, --mirror`
  > Mirror the final template before writing to disk.
* ` --display-filter`
  > Display the combined CTF and low pass filter to the user.
* ` --log LOG`
  > Can be set to `info` or `debug.

## Generate the mask

The mask around the template can be quite tight to remove as much noise as possible around the particles of interest. We recommend around 10%-20% overhang relative to the particle radius. You can also generate an ellipsoidal mask for particles that do not approximate well as a sphere. Though you will probably need to reorient this mask in chimera and resample to the grid of the template. Optionally you could also create a structured mask around the template in external software (via thresholding and dilation for example). Take into account that non-spherical masks roughly double the template matching computation time.

### Usage

```
pytom_create_mask.py 
  [-h] 
  -b BOX_SIZE 
  [-o OUTPUT_FILE] 
  [--voxel-size VOXEL_SIZE] 
  -r RADIUS 
  [--radius-minor1 RADIUS_MINOR1] 
  [--radius-minor2 RADIUS_MINOR2] 
  [-s SIGMA]
```

### Parameters

* ` -h, --help`
  > show this help message and exit
* ` -b BOX_SIZE, --box-size BOX_SIZE`
  > Shape of square box for the mask.
* ` -o OUTPUT_FILE, --output-file OUTPUT_FILE`
  > Provide path to write output, needs to end in .mrc . If not provided file is written to current directory in the following format: ./mask_b[box_size]px_r[radius]px.mrc
* ` --voxel-size VOXEL_SIZE`
  > Provide a voxel size to annotate the MRC (currently not used for any mask calculation).
* ` -r RADIUS, --radius RADIUS`
  > Radius of the spherical mask in number of pixels. In case minor1 and minor2 are provided, this will be the radius of the ellipsoidal mask along the x-axis.
* ` --radius-minor1 RADIUS_MINOR1`
  > Radius of the ellipsoidal mask along the y-axis in number of pixels.
* ` --radius-minor2 RADIUS_MINOR2`
  > Radius of the ellipsoidal mask along the z-axis in number of pixels.
* ` -s SIGMA, --sigma SIGMA`
  > Sigma of gaussian drop-off around the mask edges in number of pixels. Values in the range from 0.5-1.0 are usually sufficient for tomograms with 20A-10A voxel sizes.