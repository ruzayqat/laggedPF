"""Module container for vartious filtering methods"""
import os
import time
import datetime

import numpy as np
import h5py
import scipy.linalg as sc
from scipy.sparse import diags
from tools import (mat_mul, mvnrnd, fwd_slash, symmetric)
from solver import solve
from parameters import get_params
from data_tools import get_data

def _initialize(params):
    initial = _get_xstar(params)
    x_a = np.transpose([initial] * params["M"])
    e_loc = np.zeros((params["dimx"], params["T"] + 1))
    e_loc[:, 0] = initial
    return x_a, e_loc

def dump(filename, t_simul, e_loc):
    """Dump to file"""
    print("Dumping results to %s" %filename)
    with h5py.File(filename, "w") as fout:
        fout.create_dataset(name="E_loc", data=e_loc)
        fout.create_dataset(name="Time", data=t_simul)


def _logline(filter_name, params, step, runtime):
    filename = "%s_log_%03d.log" %(filter_name, params["isim"])
    if step == 0:
        line = "Filter         Step         Runtime     Remaining"
        with open(filename, "w") as fout:
            fout.write(line+"\n")
    remain = params["T"]*runtime/(step+1)
    remain -= runtime
    remain = str(datetime.timedelta(seconds=int(remain)))
    runtime = str(datetime.timedelta(seconds=int(runtime)))
    line = "%s %08d/%08d %s %s" %(filter_name, step+1, params["T"], runtime, remain)
    with open(filename, "a") as fout:
        fout.write(line+"\n")

def enkf(input_file, isim):
    """ Ensemble Kalman Filter

    Arguments:
    input_file {str}: Name or full path to the input yaml file
    isim {int}: Simulation identifier
    Returns: None
    """
    params = get_params(input_file)
    matrix = np.eye(params["dimo"]) * params["sig_y"]
    _filtering(params, "enkf", matrix, isim)

def etkf(input_file, isim):
    """ Ensemble Trasnform Kalman Filter

    Arguments:
    input_file {str}: Name or full path to the input yaml file
    isim {int}: Simulation identifier

    Returns:
    None
    """
    params = get_params(input_file)
    matrix = np.eye(params["dimo"]) * params["sig_y"]
    _filtering(params, "etkf", matrix, isim)

def etkf_sqrt(input_file, isim):
    """ Ensemble Trasnform Kalman SQRT Filter

    Arguments:
    input_file {str}: Name or full path to the input yaml file
    isim {int}: Simulation identifier

    Returns:
    None
    """
    params = get_params(input_file)
    matrix = np.eye(params["dimo"]) * params["sig_y"]**2
    _filtering(params, "etkf_sqrt", matrix, isim)

def _filtering(params, filter_name, matrix, isim):
    """Generung filtering function"""
    params["isim"] = isim
    filter_name = filter_name.lower().strip()
    if filter_name == "enkf":
        filter_func = enkf_run
    elif filter_name == "etkf":
        filter_func = etkf_run
    elif filter_name == "etkf_sqrt":
        filter_func = etkf_sqrt_run

    t_simul, e_loc = filter_func(matrix, params)
    directory = os.path.abspath(params["%s_dir" %filter_name])
    os.makedirs(directory, exist_ok=True)
    filename = "%s/sim_%08d.h5" %(directory, isim)
    dump(filename, t_simul, e_loc)


def enkf_run(r2_sqrt, params):
    """ One Run of Ensemble Kalman Filter """
    t_simul = 0.0
    x_a, e_loc = _initialize(params)
    if params["Y_CoefMatrix_is_eye"]:
        r2_mat = np.eye(params["dimo"]) * params["sig_y"] ** 2
    else:
        r2_mat = mat_mul(r2_sqrt, r2_sqrt)

    total_time = 0
    filename = "%s/EnKF" %params["enkf_dir"]
    for step in range(params["T"]):
        tick = time.time()
        t_simul, e_loc[:, step + 1], x_a = _enkf_step(params, r2_mat, r2_sqrt, x_a,
                                                 step)
        tack = time.time()
        total_time += tack - tick
        _logline(filename, params, step, total_time)
    return t_simul, e_loc

def _enkf_step(params, r2_mat, r2_sqrt, x_a, step):
    t_simul, x_f = solve(params['M'], x_a, params)
    if (step + 1) % params['t_freq'] == 0:
        data = get_data(params, step)
        pred_mean = np.sum(x_f, axis=1) / params['M']
        diff = x_f - pred_mean.reshape(-1, 1)
        if params['C_is_eye']:
            temp = diff.T
            temp = mat_mul(diff, temp) / params['M']
            temp = fwd_slash(params["Ido"], temp + r2_mat)
            temp = mat_mul(diff.T, temp)
            kappa = mat_mul(diff, temp) / params['M']
            noise = np.random.normal(size=(params["dimo"],params["M"]))
            if params['Y_CoefMatrix_is_eye']:
                temp = data.reshape(-1, 1) - x_f - params['sig_y'] * noise
            else:
                temp = data.reshape(-1, 1) - x_f
                temp -= mat_mul(noised_obs_mat, noise)
        else:
            temp = mat_mul(diff.T, params['C'].T)
            temp = mat_mul(diff, temp)
            temp = mat_mul(params['C'], temp) / params['M']
            temp = fwd_slash(params['Ido'], temp + r2_mat)
            temp = mat_mul(params['C'].T, temp)
            temp = mat_mul(diff.T, temp)
            kappa = mat_mul(diff, temp) / params['M']
            noise = np.random.normal(size=(params["dimo"],params["M"]))
            if params['Y_CoefMatrix_is_eye']:
                temp = data.reshape(-1, 1) - mat_mul(params['C'], x_f)
                temp -= params['sig_y'] * noise
            else:
                temp = data.reshape(-1, 1) - mat_mul(params['C'], x_f)\
                                     - mat_mul(noised_obs_mat,noise)
        x_a = x_f + mat_mul(kappa, temp)
    else:
        x_a = x_f

    return t_simul, np.sum(x_a, axis=1) / params["M"], x_a


def etkf_run(r2_sqrt_inv, params):
    """ One Run of Ensemble Transform Kalman Filter """
    t_simul = 0.0
    x_a, e_loc = _initialize(params)
    total_time = 0
    filename = "%s/ETKF" %params["etkf_dir"]
    for step in range(params["T"]):
        tick = time.time()
        t_simul, e_loc[:, step + 1], x_a = _etkf_step(params, r2_sqrt_inv, x_a, step)
        tack = time.time()
        total_time += tack - tick
        _logline(filename, params, step, total_time)
    return t_simul, e_loc


def _etkf_step(params, r2_sqrt_inv, x_a, step):
    t_simul, x_f = solve(params["M"], x_a, params)

    if (step + 1) % params['t_freq'] == 0:
        data = get_data(params, step)
        m_f = np.sum(x_f, axis=1).reshape(-1, 1) / params["M"]
        sfm = 1 / np.sqrt(params["M"] - 1) * (x_f - m_f)
        signal_hat = mat_mul(params['C'], x_f)
        mean = np.sum(signal_hat, axis=1).reshape(-1, 1) / params["M"]
        if params["Y_CoefMatrix_is_eye"]:
            phi_k = 1 / np.sqrt(params["M"] - 1) * (signal_hat - mean).T / params["sig_y"]
        else:
            phi_k = 1 / np.sqrt(params["M"] - 1) * mat_mul((signal_hat - mean).T, r2_sqrt_inv)
        eta_k = mat_mul(phi_k.T, phi_k) + params["Ido"]
        kappa = mat_mul(sfm, fwd_slash(phi_k, eta_k))
        if params["Y_CoefMatrix_is_eye"]:
            m_a = m_f + mat_mul(kappa, ((data.reshape(-1, 1) - mean) / params["sig_y"]))
        else:
            m_a = m_f + mat_mul(kappa, mat_mul(r2_sqrt_inv,
                                           (data.reshape(-1, 1) - mean)))
        unit, diag = sc.svd(mat_mul(phi_k, phi_k.T))[0:2]
        diag = diags(diag)
        sfm = mat_mul(sfm, fwd_slash(unit, np.sqrt(diag + np.identity(params["M"]))))
        x_a = np.sqrt(params["M"] - 1) * sfm + m_a
    else:
        x_a = x_f
    return t_simul, np.sum(x_a, axis=1) / params["M"], x_a


def etkf_sqrt_run(r2_mat, params):
    """ One Run of Ensemble Transform Kalman Filter with Square Root of invTTt
    (e.g. Hunt et al., 2007) see "State-of-the-art stochastic data
    assimilation methods for high-dimensional non-Gaussian problems """
    t_simul = 0.0
    x_a, e_loc = _initialize(params)
    total_time = 0
    filename = "%s/ETKF_SQRT" %params["etkf_sqrt_dir"]
    for step in range(params["T"]):
        tick = time.time()
        t_simul, e_loc[:, step + 1], x_a = _etkf_sqrt_step(params, r2_mat, x_a, step)
        tack = time.time()
        total_time += tack - tick
        _logline(filename, params, step, total_time)
    return t_simul, e_loc

def _etkf_sqrt_step(params, r2_mat, x_a, step):
    # pylint: disable-msg=too-many-locals
    t_simul, x_f = solve(params["M"], x_a, params)

    if (step + 1) % params['t_freq'] == 0:
        data = get_data(params, step)
        
        m_f = np.sum(x_f, axis=1).reshape(-1, 1) / params["M"]
        xfp = x_f - m_f
        s_mat = mat_mul(params['C'], x_f)
        mean = np.sum(s_mat, axis=1).reshape(-1, 1) / params["M"]
        s_mat = s_mat - mean
        invr2_s = sc.lstsq(r2_mat, s_mat)[0]
        inv_ttt = symmetric((params["M"] - 1) * np.identity(params["M"]) +\
                                                           mat_mul(s_mat.T, invr2_s))
        eigenvals, eigenvects = np.linalg.eig(inv_ttt)
        eigenvals = np.diag(eigenvals.real)
        eigenvects = eigenvects.real
        xap = mat_mul(eigenvects, sc.lstsq(np.sqrt(eigenvals), eigenvects.T)[0])
        xap = np.sqrt(params["M"] - 1) * mat_mul(xfp, xap)
        temp = mat_mul(invr2_s.T, data.reshape(-1, 1) - mean)
        temp = sc.lstsq(eigenvals, mat_mul(eigenvects.T, temp))[0]
        m_a = m_f + mat_mul(xfp, mat_mul(eigenvects, temp))
        x_a = xap + m_a
    else:
        x_a = x_f
    return t_simul, np.sum(x_a, axis=1) / params["M"], x_a


def _get_xstar(params):
    with h5py.File(params["data_file"], "r") as fin:
        xstar = fin["step_%08d" %0]["signal"][...]
    return xstar