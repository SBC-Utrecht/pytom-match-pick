# PyTOM GPU template matching for cryo-ET
PyTOM's GPU template matching as a single pip plugin that can only be run from the command line.

The full PyTOM repository can be found at: https://github.com/SBC-Utrecht/PyTom

## Requires

```
miniconda3
nvidia-cuda-toolkit
```

## Installation

There are 2 options for creating a conda environment:

1. **(recommended)** Create a new environment with a prebuild cupy version and complete CUDA-toolkit. This is reliable but takes more disk space.

    ```commandline
    conda create -n pytom_tm -c conda-forge python=3 cupy cuda-version=11.8
    ```

2. Create a new environment without cupy and let pip build against a system installed CUDA-toolkit.

    ```commandline
    conda create -n pytom_tm python=3
    ```

Once the environment is created, activate it:

```commandline
conda activate pytom_tm
```

Then clone the repository and install it with pip: 

```commandline
git clone https://github.com/SBC-Utrecht/pytom-template-matching-gpu.git
cd pytom-template-matching-gpu
python -m pip install '.[plotting]'
```

The installation above also adds the optional dependencies `[matplotlib, seaborn]` which are required to run `pytom_estimate_roc.py`. They are not essential to the core template matching fucntionality, so for some systems (such as certain cluster environments) it might be desirable to skip them. In that case remove `[plotting]` from the pip install command:

```commandline
python -m pip install .
```

## Tests

To run the unittests:

```commandline
cd tests
python -m unittest discover
```

If cupy is not correctly installed, this should raise some errors:

```commandline
python -c "import cupy as cp; a = cp.zeros((100,100))"
```

To solve cupy installation issues, please check 
[the cupy docs](https://docs.cupy.dev/en/stable/install.html#installing-cupy). It might be solved by installing a 
specific build compatible with the installed cuda toolkit.   

## Usage

Detailed usage instructions are available on the wiki: https://github.com/SBC-Utrecht/pytom-template-matching-gpu/wiki

Also, a tutorial can be found on the same wiki: https://github.com/SBC-Utrecht/pytom-template-matching-gpu/wiki/Tutorial
The following scripts are available to run with `--help` to see parameters:

- create a template from an mrc file containing a density map: `pytom_create_template.py --help`
- create a mask for template matching: `pytom_create_mask.py --help`
- run template matching with the mask (.mrc) and template (.mrc) on a tomogram (.mrc): `pytom_match_template.py --help`
- extract candidates from a job file (.json) created in the template matching output folder: `pytom_extract_candidate.py --help`
- estimate an ROC curve from a job file (.json): `pytom_estimate_roc.py --help`
- merge multiple star files to a single starfile: `pytom_merge_stars.py --help`

## Contributing

Contributions to the project are very welcome. Feel free to make a pull request to address a specific issue.

## Citation

For a reference on GPU accelerated template matching in tomograms please see the [IJMS publication](https://www.mdpi.com/1422-0067/24/17/13375).


```
@Article{ijms241713375,
    AUTHOR = {Chaillet, Marten L. and van der Schot, Gijs and Gubins, Ilja and Roet, Sander and Veltkamp, Remco C. and Förster, Friedrich},
    TITLE = {Extensive Angular Sampling Enables the Sensitive Localization of Macromolecules in Electron Tomograms},
    JOURNAL = {International Journal of Molecular Sciences},
    VOLUME = {24},
    YEAR = {2023},
    NUMBER = {17},
    ARTICLE-NUMBER = {13375},
    URL = {https://www.mdpi.com/1422-0067/24/17/13375},
    ISSN = {1422-0067},
    DOI = {10.3390/ijms241713375}
}
```
