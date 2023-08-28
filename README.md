# PyTOM GPU template matching for cryo-ET
PyTOM's GPU template matching as a single pip plugin that can only be run from the command line.

The full PyTOM repository can be found at: https://github.com/SBC-Utrecht/PyTom

## Citation

...

## Requires

```
miniconda3
nvidia-cuda-toolkit
```

## Installation

Create a new conda environment:

```commandline
conda create -n pytom_tm python=3
```

The clone the repository and install it with pip: 

```commandline
git clone https://github.com/SBC-Utrecht/pytom-template-matching-gpu.git
cd pytom-template-matching-gpu
pip install .
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

The following scripts are available to run with `--help` to see parameters:

- create a template from an mrc file containing a density map: `pytom_create_template.py --help`
- create a mask for template matching: `pytom_create_mask.py --help`
- run template matching with the mask (.mrc) and template (.mrc) on a tomogram (.mrc): `pytom_match_template.py --help`
- extract candidates from a job file (.json) created in the template matching output folder: `pytom_extract_candidate.py --help`
- estimate an ROC curve from a job file (.json): `pytom_estimate_roc.py --help`
- merge multiple star files to a single starfile: `pytom_merge_stars.py --help`

## Contributing

Contributions to the project are very welcome. Feel free to make a pull request to address a specific issue.
