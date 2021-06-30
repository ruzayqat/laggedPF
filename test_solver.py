import numpy as np
import matplotlib.pyplot as plt
from solver import ShallowWaterSolver
from grid import generate_grid



def generate_initial_field(bounds, discretization):
    """generate initial field"""
    grid_params = {"grid_bounds":bounds,
                   "discretization":discretization+2}
    mesh = generate_grid(grid_params)
    dim = mesh[0].shape[0]
    height = np.ones((dim, dim))
    cond = True
    for i in range(2):
        cond = np.logical_and(cond, mesh[i]>=0.5)
        cond = np.logical_and(cond, mesh[i]<=1.)
    height[cond] = 2.5

    initial = height.ravel()
    for i in range(2):
        initial = np.append(initial, np.zeros_like(height))
    return initial

params = dict()
params["discretization"] = 50
params["grid_bounds"] = [0, 2.]
params["dx"] = 2/(params["discretization"]-1)
params["solver_dim"] = params["discretization"]+2 
params["solver_cfl"] = 0.5



solver = ShallowWaterSolver(params["solver_dim"])
sol = generate_initial_field(params["grid_bounds"], params["discretization"])
nsteps = 5000
time = 0
for step in range(nsteps):
    solver.init(sol)
    sol, dt = solver.solve()
    time += dt
    print("Shallow Water Solver   Ite=%08d, Timestep=%14.8e , Time=%14.8e" %(step, dt, time))
    if step%50==0:
        solver.dump("test_solver", step+1)
# plt.imshow(solver.solution[1, :, :], cmap="jet", interpolation="bilinear")
# plt.show()