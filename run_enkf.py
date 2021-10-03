from filters import enkf, etkf, etkf_sqrt
from functools import partial
import multiprocessing as mp

def main(filter_name, input_file, nsimul):
    """main function"""
    if filter_name == "enkf":
        filter_func = enkf
    elif filter_name == "etkf":
        filter_func = etkf
    elif filter_name == "etkf_sqrt":
        filter_func = etkf_sqrt
    filter_func = partial(filter_func, input_file)
    simu = range(nsimul)
    pool = mp.Pool(nsimul)
    pool.map(filter_func, simu)

if __name__ == '__main__':
    nsimul = 50
    input_file = "example_input.yml"
    filter_name = "enkf"
    main(filter_name, input_file, nsimul)
