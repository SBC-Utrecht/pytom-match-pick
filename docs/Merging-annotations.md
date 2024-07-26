This page contains usage information for `pytom_merge_stars.py`.

## Merging starfiles

After running template matching and candidate extraction on multiple tomograms, each tomogram will have an individual starfile with particle annotations. Each starfile will contain the `MicrographName` column which refers back to the tomogram name. Multiple starfiles can therefore be appended to results in a large list which can be used in other software (such as RELION, WarpM) to load annotations. These software will link the annotations to specific tilt-series using the `MicrographName` column.

### Usage

`pytom_merge_stars.py` merges multiple starfiles into a single starfile. Basic usage of the script is:

```
pytom_merge_star.py 
  [-h] 
  [-i INPUT_DIR] 
  [-o OUTPUT_FILE] 
  [--log LOG]
```

Without providing any parameters the script will try to merge all the starfiles in the current working directory and save them to a new file `particles.star`.

### Parameters

The following options are available:
* `-h, --help` 
  > Show a help message and exit.
* `-i INPUT_DIR, --input-dir INPUT_DIR` 
  > Provide a path to a directory of starfiles. The script will try to merge all files that end in '.star', it is up to the user to ensure that they are mergeable.
* `-o OUTPUT_FILE, --output-file OUTPUT_FILE` 
  > Path for writing the output starfile.
* `--log LOG` 
  > Can be switched from default `info` to `debug` mode to be more verbose.
