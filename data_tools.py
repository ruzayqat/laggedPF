"""Module for data functions"""
import os
import numpy as np
from tqdm import tqdm
import h5py
from tools import (mvnrnd, mat_mul)
from solver import solve

def sample_from_f(num, num_part, noised, t_simul, params):
    """ Sample from f(x_n|x_{n-1}) """
    # num_part is the number of particles
    # num only used for noise geneation
    if (num_part != 1) & (num_part != num):
        raise ValueError('Error: M must be either equal to 1 or N')
    tstep, noised = solve(num_part, noised, params)

    t_simul += tstep
    noised = noised.real
    if num_part == 1:
        fx0 = noised
        noised = np.transpose([fx0] * num)
    else:
        fx0 = []
    noise = mvnrnd(np.zeros((params["dimx"])), params["Idx"], coln=num)  # size = dx x N
    noised = noised + params["sig_x"] * noise
    return t_simul, noised, fx0


def initial_condition(params):
    """initialcondition"""
    x_mesh = np.linspace(0 - params["dx"], 2 + params["dx"], params["dim"])
    x_mesh, y_mesh = np.meshgrid(x_mesh, x_mesh)

    height = np.ones((params["dim"], params["dim"]))
    cond = (x_mesh >= 0.5) & (x_mesh <= 1) & (y_mesh >= 0.5) & (y_mesh <= 1)
    height[cond] = 2.5
    x_star = np.zeros(params["dimx"])
    x_star[0:params["dim2"]] = height.flatten()
    return x_star


def generate_data(params, x_star):
    """doctsring"""
    print("Generating signal....")
    signal = np.zeros(params["dimx"])
    signal = x_star
    dump_data(params, signal, 0, "signal")
    for step in tqdm(range(params["T"])):
        tstep, signal = solve(1, signal, params)
        dump_data(params, signal, step+1, "signal")

    data = np.zeros(params["dimo"])
    # x_pertur = np.zeros(params["dimx"])
    x_pertur = x_star
    dump_data(params, x_pertur, 0, "x_pertur")
    t_obs = 0.
    print("Generating data....")
    for step in tqdm(range(params["T"])):
        tstep, x_pertur = solve(1, x_pertur, params)
        dump_data(params, x_pertur, step+1, "x_pertur")
        t_obs += tstep
        noise_x = mvnrnd(np.zeros((params["dimx"])), params["Idx"])
        x_pertur = x_pertur + params["sig_x"] * noise_x
        noise_o = mvnrnd(np.zeros((params["dimo"])), np.eye(params["dimo"]))
        data = mat_mul(params["C"], x_pertur)
        data += params["sig_y"] * noise_o
        dump_data(params, data, step, "data")
    return signal, data

def dump_data(params, array, step, name):
    """Dumping array step"""
    # print('Dumping to %s\n' %params["data_file"])
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
        pred_cov = fin["step%08d"%step]["cov_array"][...]
    return pred_mean, pred_cov

def get_data(params, step):
    """get_data"""
    with h5py.File(params["data_file"], "r") as fin:
        data = fin["step_%08d"%step]["data"][...]
    return data
