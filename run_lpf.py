"""Main module"""
from functools import partial
import multiprocessing as mp
from parameters import get_params
from laggedpf import laggedpf
from data_tools import initial_condition, generate_data

def main():
    """Multisim"""
    start = 0
    end_ = 30
    simu = range(start, end_)
    nsimu = len(simu)
    print("Parsing input file...")
    input_file = "example_input.yml"
    params = get_params(input_file)
    # x_star = initial_condition(params)
    # generate_data(params, x_star)
    # ensemble_kalman_filter(params["sig_y"], 0, x_star, params)

    run = partial(laggedpf, params)
    print("Performing %d simulations..." %nsimu)
    pool = mp.Pool(nsimu)
    pool.map(run, simu)

if __name__ == '__main__':
    main()
