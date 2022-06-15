# A LAGGED PARTICLE FILTER FOR STABLE FILTERING OF CERTAIN HIGH-DIMENSIONAL STATE-SPACE MODEL

## About
This repository contains the modules used in the *SJUQ* submission  **A LAGGED PARTICLE FILTER FOR STABLE FILTERING OF 
CERTAIN HIGH-DIMENSIONAL STATE-SPACE MODEL**, and consists of the application of the lagged particle filtering technique
on the non-linear shallow-water equation.

## Requirements
The python code provided in this repository requires the following packages:

- numpy
- yaml
- h5py
- scipy
- numba
- jax (in case of usage of GPU for matrix operations)

## Usage

5 executable driver scripts are provided:

- `run_enkf.py` to run the Ensemble kalman filter
- `run_etkf.py` to run the `ETKF` filter
- `run_etkf_sqrt.py` to run the `ETKF-SQRT` filter
- `run_lpf.py` to run the lagged particle filter

Prior to running any given filter, one should provide the data and eventually the predictor
statistics. This is done through executing the driver script `gnerate_pred_using_EnKF.py`.

The input for the scripts mentionned above, is specified in a yaml file format, as the one
provided here as an example. This format is self explanatory and the variables appelation is
set to match the one used in the paper. In the scripts, one may change the input filename as:
```Python
input_file = <my_input_file.yml>
```
