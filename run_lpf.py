"""Main module"""
from functools import partial
import multiprocessing as mp
from parameters import get_params
from laggedpf import laggedpf
from data_tools import initial_condition, generate_data
from gnerate_pred_using_EnKF import ensemble_kalman_filter

def main():
    """Multisim"""
    
    print("Parsing input file...")
    input_file = "example_input.yml"
    params = get_params(input_file)

    simu = range(0, params["nsimu"])

    print("Generating initial state and data")
    x_star = initial_condition(params)
    #generate_data(params, x_star)

    # print("Generating the mean and cov of $\\mu$ function")
    # ensemble_kalman_filter(params["sig_y"], 0, x_star, params)

    run = partial(laggedpf, params)
    print("Performing %d simulations on %d processors" %(params["nsimu"], params["ncores"]))
    pool = mp.Pool(processes = params["ncores"])
    pool.map(run, simu)

if __name__ == '__main__':
    main()
