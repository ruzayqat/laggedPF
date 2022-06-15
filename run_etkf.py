from ensemble_filters import enkf, etkf, etkf_sqrt
from functools import partial
import multiprocessing as mp
from parameters import get_params

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
    input_file = "example_input.yml"
    params = get_params(input_file)
    nsimul = params["nsimu"]
    filter_name = "etkf"
    main(filter_name, input_file, nsimul)