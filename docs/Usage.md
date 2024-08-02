This page describes the general usage of the software.

## Overview

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

This section contains usage information for `pytom_create_template.py` and 
`pytom_create_mask.py`.

**Important**:
- The template and mask need to have the same box size.
- The template needs to have the same contrast as the tomogram (e.g. the particles are black in both the tomogram and template).

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

#### About CTFs

Some form of CTF **should** be applied to the template:
- In case all the neccesary parameters for CTF correction can be passed to `pytom_match_template.py`, you should only scale the template and adjust its contrast in this script.
- Otherwise the template can be multiplied with a CTF here, in which case we often cut the CTF after the first zero crossing and apply a low pass filter. This is due to defocus gradient effects leading to wrong CTF crossings and reducing the correlation.


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

### pytom_extract_candidates.py

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

### pytom_merge_stars.py

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



