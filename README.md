![test-badge](https://github.com/SBC-Utrecht/pytom-match-pick/actions/workflows/unit-tests.yml/badge.svg?branch=main)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10728422.svg)](https://doi.org/10.5281/zenodo.10728422)


# pytom-match-pick

GPU-accelerated template matching for cryo-electron tomography, originally developed in [PyTom](https://github.com/SBC-Utrecht/PyTom), as a standalone Python package that is run from the command line. 

This software is developed by Marten L. Chaillet ([@McHaillet](https://github.com/McHaillet)) and Sander Roet ([@sroet](https://github.com/sroet)) in the group of Friedrich Förster at Utrecht University.

![cover_image](docs/images/tomo200528_100_illustration.png)

<!--
This line starts the block that is incorporated into the website via mkdocs snippets
-->
[//]: # (#--8<-- [start:docs])

## Requires

```
miniconda3
nvidia-cuda-toolkit
```

## Installation

There are 2 options for creating a conda environment. We recommend option (1) which will later allow cupy to build 
against a system installed cuda-toolkit. Compared to option (2) this can give an almost two-fold speedup:

1. **(recommended)** Create a new python 3 environment:

    ```commandline
    conda create -n pytom_tm python=3
    ```

2.  Create a new environment with a prebuild cupy version and complete CUDA-toolkit. This is more reliable, but takes more 
    disk space and has less optimal performance.

    ```commandline
    conda create -n pytom_tm -c conda-forge python=3 cupy cuda-version=11.8
    ```


Once the environment is created, activate it:

```commandline
conda activate pytom_tm
```

Then install the code with `pip` (building cupy can take a while!):

```commandline
python -m pip install pytom-match-pick[plotting]
```

The installation above also adds the optional dependencies `[matplotlib, seaborn]` which are required to run 
`pytom_estimate_roc.py`. They are not essential to the core template matching functionality, so for some systems 
(such as certain cluster environments) it might be desirable to skip them. In that case remove `[plotting]` from the pip install command:

```commandline
python -m pip install pytom-match-pick
```

### Cupy warning
Having issues running the software? If cupy is not correctly installed, 
```commandline
python -c "import pytom_tm"
```

can show a cupy warning. If this is the case, this probably means cupy is not correctly installed.
Alternatively, cupy can sometimes be installed without issue but not detect CUDA correctly. In that case, the following should raise some errors:
```commandline
python -c "import cupy as cp; a = cp.zeros((100,100))"
```

To solve cupy installation issues, please check 
[the cupy docs](https://docs.cupy.dev/en/stable/install.html#installing-cupy). It might be solved by installing a 
specific build compatible with the installed cuda toolkit.   

## Usage

The following scripts are available to run with `--help` to see parameters:

- create a template from an mrc file containing a density map: `pytom_create_template.py --help`
- create a mask for template matching: `pytom_create_mask.py --help`
- run template matching with the mask (.mrc) and template (.mrc) on a tomogram (.mrc): `pytom_match_template.py --help`
- extract candidates from a job file (.json) created in the template matching output folder: `pytom_extract_candidate.py --help`
- estimate an ROC curve from a job file (.json): `pytom_estimate_roc.py --help`
- merge multiple star files to a single starfile: `pytom_merge_stars.py --help`

Detailed usage instructions and a tutorial are available on [our site](https://SBC-Utrecht.github.io/pytom-match-pick).

## Usage questions, ideas and solutions, engagement, etc
Please use our [github discussions](https://github.com/SBC-Utrecht/pytom-match-pick/discussions) for:
 - Asking questions about bottlenecks.
 - Share ideas and solutions.
 - Engage with other community members about processing strategies.
 - etc...

## Developer install
If you want the most up-to-date version of the code you can get install it from this repository via:

```commandline
git clone https://github.com/SBC-Utrecht/pytom-match-pick.git
cd pytom-match-pick
python -m pip install '.[all]'
```

if you don't want the optional plotting dependencies use the following install command instead:
```commandline
python -m pip install '.[dev]'
```

For development, please also install pre-commit to check and autostyle the code before 
you make PRs: 

```commandline
pre-commit install
```

This uses Ruff to check and format whenever you make commits.

If you update anything in the (documentation) `docs/` folder make sure to test build the website locally:

```commandline
mkdocs serve
```

## Tests

With the developer install also comes the ability to run the unittests,
from the git repository run:

```commandline
cd tests
python -m unittest discover
```

## Contributing

Contributions to the project are very welcome! Feel free to make a pull request or suggest an implementation in the issues. For PR's we will gladly give you feedback on how to integrate the code.

## Citation

Chaillet, M. L., van der Schot, G., Gubins, I., Roet, S., Veltkamp, R. C., & Förster, F. (2023). Extensive angular sampling enables the sensitive localization of macromolecules in electron tomograms. _International Journal of Molecular Sciences_, 24(17), 13375. <https://doi.org/10.3390/ijms241713375>

<!--
This ends the block for the website
-->
[//]: # (#--8<-- [end:docs])
