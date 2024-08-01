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

### pytom_create_template.py

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



