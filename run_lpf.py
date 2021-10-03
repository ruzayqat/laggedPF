"""Main module"""
from functools import partial
import multiprocessing as mp
from parameters import get_params
from laggedpf import laggedpf


def main():
    """Multisim"""
    start = 0
    end_ = 50
    simu = range(start, end_)
    nsimu = len(simu)
    print("Parsing input file...")
    input_file = "example_input.yml"
    params = get_params(input_file)
    run = partial(laggedpf, params)
    print("Performing %d simulations..." %nsimu)
    pool = mp.Pool(nsimu)
    pool.map(run, simu)

if __name__ == '__main__':
    main()
