"""Doctsring here"""
import numpy as np
import h5py


def generate_grid(params):
    """Generate 2D grid given bounds and discretization
       and dumps the grid to an hdf5 file

    Parameters:
    params (dict): Dictionnary of parameters

    Returns:
    mesh (list) : List of 2D arrays for the coordinates
    """
    start = params["grid_bounds"][0]
    scale = params["grid_bounds"][1] - params["grid_bounds"][0]

    filename = "grid.h5"
    if "grid_filename" in params:
        filename = params["grid_filename"]

    if "grid_file" in params:
        filename = params["grid_file"]
    coords = np.linspace(0, 1, params["discretization"])
    coords = start + scale*coords
    mesh = np.meshgrid(coords, coords)

    with h5py.File(filename, "w") as fout:
        for i, key in enumerate(["coords_x", "coords_y"]):
            fout.create_dataset(name=key, data=mesh[i])
    return mesh
