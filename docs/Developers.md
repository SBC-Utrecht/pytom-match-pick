Star files written out by pytom-match-pick should be easily integratable with other software as it follows RELION star file conventions. The only difference are three column headers with extraction statistics, which are important to maintain for annotations.

The exact header is:

```
# Created by the starfile Python package (version x.x.x) at xx:xx:xx on xx/xx/xxxx


data_

loop_
_rlnCoordinateX #1
_rlnCoordinateY #2
_rlnCoordinateZ #3
_rlnAngleRot #4
_rlnAngleTilt #5
_rlnAnglePsi #6
_rlnLCCmax #7
_rlnCutOff #8
_rlnSearchStd #9
_rlnDetectorPixelSize #10
_rlnMicrographName #11
```