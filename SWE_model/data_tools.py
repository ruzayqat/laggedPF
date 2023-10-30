"""Module for data functions"""
import os, numpy as np
from tqdm import tqdm
import h5py
from tools import mvnrnd, mat_mul
from solver import solve

def sample_from_f(num, num_part, noised, t_simul, params):
    """ Sample from f(x_n|x_{n-1}) """
    if (num_part != 1) & (num_part != num):
        raise ValueError('Error: M must be either equal to 1 or N')
    else:
        tstep, noised = solve(num_part, noised, params)
        t_simul += tstep
        noised = noised.real
        if num_part == 1:
            fx0 = noised
            noised = np.transpose([fx0] * num)
        else:
            fx0 = []
        if params['X_CoefMatrix_is_eye']:
            noise = np.random.normal(size=(params['dimx'], num))
            noised = noised + params['sig_x'] * noise
        else:
            raise NotImplementedError('Only X_CoefMatrix_is_eye is supported')
    return (
     t_simul, noised, fx0)


def initial_condition(params):
    """initialcondition"""
    x_mesh = np.linspace(0 - params['dx'], 2 + params['dx'], params['dim'])
    x_mesh, y_mesh = np.meshgrid(x_mesh, x_mesh)
    height = np.ones((params['dim'], params['dim']))
    cond = (x_mesh >= 0.5) & (x_mesh <= 1) & (y_mesh >= 0.5) & (y_mesh <= 1)
    height[cond] = 2.5
    x_star = np.zeros(params['dimx'])
    x_star[0:params['dim2']] = height.flatten()
    return x_star


def generate_data(params, x_star):
    """doctsring"""
    X = x_star
    signal = x_star
    dump_data(params, X, 0, "x_nonpertur")
    dump_data(params, X, 0, "signal")

    print('Generating data....')
    for step in tqdm(range(params['T'])):
        tstep, X = solve(1, X, params)
        dump_data(params, X, step+1, "x_nonpertur")
        tstep, signal = solve(1, signal, params)
        noise_x = np.random.normal(size=(params['dimx']))
        if params['X_CoefMatrix_is_eye']:
            signal = signal + params['sig_x'] * noise_x
        else:
            raise NotImplementedError('Only X_CoefMatrix_is_eye is supported')
        dump_data(params, signal, step+1, "signal")

        if (step + 1) % params["t_freq"] == 0:
            noise_o = np.random.normal(size=(params['dimo']))

            if params['Y_CoefMatrix_is_eye']:
                if params["C_is_eye"]:
                    data =  signal + params['sig_y'] * noise_o
                else:
                    data = mat_mul(params['C'] , signal) + params['sig_y'] * noise_o
            else:
                raise NotImplementedError('Only Y_CoefMatrix_is_eye is supported')
            dump_data(params, data, step, 'data')
        else:
            #There is no data at time = step..you can fill it anything.
            #I choose to fill it with zeros  or can leave it empty. 
            #But has to add to the matrix 'data' to avoid confusion with indexing later on
            dump_data(params, np.zeros(params['dimo']), step, 'data')

    return (signal, data)


def dump_data(params, array, step, name):
    """Dumping array step"""
    #print('Dumping to %s\n' %params["data_file"])
    dir_ = os.path.dirname(params["data_file"])
    os.makedirs(dir_, exist_ok=True)
    grp_name = "step_%08d" %step
    with h5py.File(params["data_file"], "a") as fout:
        if grp_name in fout:
            grp = fout[grp_name]
        else:
            grp = fout.create_group(name=grp_name)
        grp.create_dataset(name=name, data=array)


def get_predictor_stats(params, step):
    """get_predictor_stats"""
    with h5py.File(params["predictor_file"], "r") as fin:
        pred_mean = fin["step%08d"%step]["mean"][...]
        pred_inv_cov = fin["step%08d"%step]["inv_cov_array"][...]
        logdet_cov = fin["step%08d"%step]["logdet_cov"][...]
    return pred_mean, pred_inv_cov, logdet_cov

def get_data(params, step):
    """get_data"""
    with h5py.File(params["data_file"], "r") as fin:
        data = fin["step_%08d"%step]["data"][...]
    return data
