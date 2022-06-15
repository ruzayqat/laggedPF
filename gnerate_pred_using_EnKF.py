"""Kalman ensemble filter"""
import os, numpy as np
from tqdm import tqdm
import h5py
from tools import mat_mul, fwd_slash, mvnrnd, nearestpd, logdet
from data_tools import sample_from_f

def ensemble_kalman_filter(noised_obs_mat, isim, x_star, params):
    """ One Run of Ensemble Kalman Filter to return X_f and P_f to be used in
        LaggedPF.
    Input:  1) noised_obs_mat: Noise coef. matrix of the observations
            2) isim: an integer.
    Output: 1) pred_mean: a 2d array that each column of it contains the mean of the
                    predictor, that is the mean of the density p(x_n|y_{0:n-1})
                    for n = 1,...,T
            2) cov_array: a 3d array that each 2d sub-array of it is the covariance
                    of the predictor density
    """
    np.random.seed(isim + 100)
    x_a = np.transpose([x_star] * params['M'])
    pred_mean = np.zeros(params['dimx'])
    cov_array = np.zeros((params['dimx'], params['dimx']))
    t_simul = 0.0
    comp = {'id_o': np.eye(params['dimo'])}
    if params['Y_CoefMatrix_is_eye']:
        comp['r2_mat'] = np.eye(params['dimo']) * params['sig_y'] ** 2
    else:
        comp['r2_mat'] = mat_mul(noised_obs_mat, noised_obs_mat)
    for step in tqdm(range(params['T'])):
        t_simul, x_f, _ = sample_from_f(params['M'], params['M'], x_a, t_simul, params)
        if (step + 1) % params['t_freq'] == 0:
            data = _load_data(params, step)
            pred_mean = np.sum(x_f, axis=1) / params['M']
            comp['diff'] = x_f - pred_mean.reshape(-1, 1)
            cov_array = mat_mul(comp['diff'], comp['diff'].T) / (params['M'] - 1)
            if params['C_is_eye']:
                temp = comp['diff'].T
                temp = mat_mul(comp['diff'], temp) / params['M']
                temp = fwd_slash(comp['id_o'], temp + comp['r2_mat'])
                temp = mat_mul(comp['diff'].T, temp)
                comp['kappa'] = mat_mul(comp['diff'], temp) / params['M']
                comp["noise"] = np.random.normal(size=(params["dimo"],params["M"]))
                if params['Y_CoefMatrix_is_eye']:
                    temp = data.reshape(-1, 1) - x_f - params['sig_y'] * comp['noise']
                else:
                    temp = data.reshape(-1, 1) - x_f
                    temp -= mat_mul(noised_obs_mat, comp['noise'])
            else:
                temp = mat_mul(comp['diff'].T, params['C'].T)
                temp = mat_mul(comp['diff'], temp)
                temp = mat_mul(params['C'], temp) / params['M']
                temp = fwd_slash(comp['id_o'], temp + comp['r2_mat'])
                temp = mat_mul(params['C'].T, temp)
                temp = mat_mul(comp['diff'].T, temp)
                comp['kappa'] = mat_mul(comp['diff'], temp) / params['M']
                comp["noise"] = np.random.normal(size=(params["dimo"],params["M"]))
                if params['Y_CoefMatrix_is_eye']:
                    temp = data.reshape(-1, 1) - mat_mul(params['C'], x_f)
                    temp -= params['sig_y'] * comp['noise']
                else:
                    temp = data.reshape(-1, 1) - mat_mul(params['C'], x_f) - mat_mul(noised_obs_mat, comp['noise'])
            x_a = x_f + mat_mul(comp['kappa'], temp)
        else:
            x_a = x_f
        cov_array = nearestpd(cov_array)
        inv_cov_array = fwd_slash(params["Idx"],cov_array)
        ldet = logdet(cov_array)
        _dump(params, step, pred_mean, inv_cov_array, ldet)

    return (pred_mean, inv_cov_array)


def _load_data(params, step):
    filename = params['data_file']
    with h5py.File(filename, 'r') as (fin):
        data = fin[('step_%08d' % step)]['data'][...]
    return data


def _dump(params, step, pred_mean, inv_cov_array, logdet_cov):
    dir_ = os.path.dirname(params['predictor_file'])
    os.makedirs(dir_, exist_ok=True)
    with h5py.File(params['predictor_file'], 'a') as (fout):
        grp = fout.create_group('step%08d' % step)
        grp.create_dataset(name='mean', data=pred_mean)
        grp.create_dataset(name='inv_cov_array', data=inv_cov_array)
        grp.create_dataset(name='logdet_cov', data=logdet_cov)
