"""Main module"""
from data_tools import (initial_condition,
                        generate_data)
from parameters import get_params
from ensemble_kalman import ensemble_kalman_filter

def main():
    """Test main function"""
    input_file = "example_input.yml"
    params = get_params(input_file)
    x_star = initial_condition(params)
    generate_data(params, x_star)
    ensemble_kalman_filter(params["sig_y"], 0, x_star, params)


if __name__ == '__main__':
    main()
