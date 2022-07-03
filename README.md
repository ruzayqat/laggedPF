# A LAGGED PARTICLE FILTER FOR STABLE FILTERING OF CERTAIN HIGH-DIMENSIONAL STATE-SPACE MODEL

## About
This repository contains the modules used in generating the graphs of the shallow-water model in the *SJUQ* article  **A LAGGED PARTICLE FILTER FOR STABLE FILTERING OF 
CERTAIN HIGH-DIMENSIONAL STATE-SPACE MODEL** which can be found on this link https://arxiv.org/pdf/2110.00884, and consists of the application of the lagged particle filtering technique on the non-linear shallow-water equation. For help regarding this code or the MATLAB codes that are used for the other models, please contact the first author on his email found in the article. 

## Requirements
The python code provided in this repository requires the following packages:

- numpy
- yaml
- h5py
- scipy
- numba
- jax (in case of usage of GPU for matrix operations)

## Usage

Prior to running any given filter, one should provide the data and eventually the predictor
statistics. This is done through executing the driver script `gnerate_pred_using_EnKF.py`.

4 executable driver scripts are provided:

- `run_enkf.py` to run the Ensemble kalman filter
- `run_etkf.py` to run the `ETKF` filter
- `run_etkf_sqrt.py` to run the `ETKF-SQRT` filter
- `run_lpf.py` to run the lagged particle filter



The input for the scripts mentionned above, is specified in a yaml file format, as the one
provided here as an example. This format is self explanatory and the variables appelation is
set to match the one used in the paper. In the scripts, one may change the input filename as:
```Python
input_file = <my_input_file.yml>
```

## Plotting

To plot the results simply run the MATLAB code `read_h5.m`. Currently this file generates comparisons between the Lagged PF and EnKF. You can edit the file easily to generate plots for the other supported ensemble methods, namely ETKF and ETKF-SQRT but first you have to run the files `run_etkf.py` and `run_etkf_sqrt.py`.
