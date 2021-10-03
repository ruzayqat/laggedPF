"""Module container for Shallow Water solver"""
import os
import numpy as np
import h5py
from lxml import etree

class ShallowWaterSolver():
    """docstring for ShallowWaterSolver"""
    def __init__(self, dim, step=1, gravity=9.81, cfl=0.5):
        self.dim = dim
        dim_ = dim+2
        self.step = {"x": step, "t":0., "cfl":cfl}
        self.solution = np.zeros((3, dim_, dim_))
        self.fluxes = {"lam_u": np.zeros((dim_, dim_)),
                       "lam_v": np.zeros((dim_, dim_)),
                       "x": np.zeros((3, dim_, dim_)),
                       "y": np.zeros((3, dim_, dim_))}
        self.gravity = gravity
        self.shift = {"m1": np.roll(np.arange(dim_), -1),
                      "p1":np.roll(np.arange(dim_), 1)}


    def init(self, sol_prev):
        """Set previous solution"""
        dim = self.dim
        dim2 = dim*dim

        for i in range(3):
            self.solution[i, 1:-1, 1:-1] = sol_prev[i*dim2:(i+1)*dim2].reshape(dim, dim)
            self.solution[i, 0, :] = self.solution[i, 1, :]
            self.solution[i, :, 0] = self.solution[i, :, 1]
            self.solution[i, -1, :] = self.solution[i, -2, :]
            self.solution[i, :, -1] = self.solution[i, :, -2]

    def set_dx(self, step):
        """Set space step"""
        self.step["x"] = step

    def set_cfl(self, cfl):
        """Set CFL number"""
        self.step["cfl"] = cfl

    def _evaluate_dt(self):
        grav = self.gravity * 0.5
        for i in range(1, 3):
            self.solution[i, :, :] /= self.solution[0, :, :]

        if self.solution[0, :, :].min() < 0:
            msg = "Solver will halt. Found negative height"
            raise RuntimeError(msg)

        self.fluxes["lam_u"] = 0.5*np.abs(self.solution[1, :, :] +
                                          self.solution[1, :, self.shift["m1"]]) +\
                               np.sqrt(self.solution[0, :, :] +
                                       self.solution[0, :, self.shift["m1"]] * grav)

        self.fluxes["lam_v"] = 0.5*np.abs(self.solution[2, :, :] +
                                          self.solution[2, self.shift["m1"], :]) +\
                               np.sqrt(self.solution[0, :, :] +
                                       self.solution[0, self.shift["m1"], :] * grav)
        norm = -1e16
        for key in ["lam_u", "lam_v"]:
            norm = max(norm, self.fluxes[key].sum(axis=1).max())
        self.step["t"] = self.step["cfl"] * (self.step["x"] / norm)
        for i in range(1, 3):
            self.solution[i, :, :] *= self.solution[0, :, :]


    def _evaluate_fluxes(self):
        huv = self.solution[1, :, :]*self.solution[2, :, :]/self.solution[0, :, :]
        ghh = self.gravity*0.5* self.solution[0, :, :] ** 2

        # calculate (hu,hu^2+gh^2/2,huv)
        for i, key in enumerate(["x", "y"], 1):
            self.fluxes[key][0, :, :] = self.solution[i, :, :]
            self.fluxes[key][i, :, :] = self.solution[i, :, :]** 2
            self.fluxes[key][i, :, :] /= self.solution[0, :, :]
            self.fluxes[key][i, :, :] += ghh
            j = 1
            if key == "x":
                j = 2
            self.fluxes[key][j, :, :] = huv

        self.fluxes["x"] = self.fluxes["x"] + \
                           self.fluxes["x"][:, :, self.shift["m1"]]
        self.fluxes["x"] -= (self.solution[:, :, self.shift["m1"]] - \
                             self.solution) * self.fluxes["lam_u"]
        self.fluxes["x"] *= 0.5

        self.fluxes["y"] = self.fluxes["y"] + \
                           self.fluxes["y"][:, self.shift["m1"], :]
        self.fluxes["y"] -= (self.solution[:, self.shift["m1"], :] - \
                             self.solution) * self.fluxes["lam_v"]
        self.fluxes["y"] *= 0.5


    def _advance_sol(self):
        alpha = -self.step["t"] / self.step["x"]
        for i, key in enumerate(["x", "y"]):
            self.solution += alpha*self.fluxes[key]
            if i == 0:
                self.solution -= alpha*self.fluxes[key][:, :, self.shift["p1"]]
            else:
                self.solution -= alpha*self.fluxes[key][:, self.shift["p1"], :]

    def _apply_bc(self):
        for i in range(3):
            alpha = 1
            beta = 1
            if i == 1:
                alpha = -1
            if i == 2:
                beta = -1
            self.solution[i, :, -1] = alpha*self.solution[i, :, -2]
            self.solution[i, :, 0] = alpha*self.solution[i, :, 1]
            self.solution[i, -1, :] = beta*self.solution[i, -2, :]
            self.solution[i, 0, :] = beta*self.solution[i, 1, :]

    def solve(self):
        """Solving for next step"""
        self._evaluate_dt()
        self._evaluate_fluxes()
        self._advance_sol()
        self._apply_bc()
        sol = self.solution[:, 1:-1, 1:-1]
        return sol.reshape(-1), self.step["t"]

    def _dump_xmf(self, filename):
        #pylint: disable=I1101
        dims = dims = "%d %d" %(self.dim, self.dim)
        spacing = "%14.8e %14.8e" %(self.step["x"], self.step["x"])

        xmf_tree = dict()
        xmf_tree['root'] = etree.Element("Xdmf", Version="2.0",
                                         nsmap={"xi": "http://www.w3.org/2001/XInclude"})
        xmf_tree['dom'] = etree.SubElement(xmf_tree['root'], "Domain")
        xmf_tree['grd'] = etree.SubElement(xmf_tree['dom'], "Grid",
                                           Name="Structured Grid", GridType="Uniform")
        etree.SubElement(xmf_tree['grd'], "Topology", Name="Topo",
                         TopologyType="2DCORECTMesh", NumberOfElements=dims)

        xmf_tree['geo'] = etree.SubElement(xmf_tree['grd'], "Geometry",
                                           GeometryType="ORIGIN_DXDY")

        field = etree.SubElement(xmf_tree['geo'], "DataItem", Name="Origin",
                                 Dimensions="2", Format="XML", NumberType="Float",
                                 Precision="8")
        field.text = "\n%s%s\n%s" %(11*" ", "0 0", 8*" ")

        field = etree.SubElement(xmf_tree['geo'], "DataItem", Name="Spacing",
                                 Dimensions="2", Format="XML", NumberType="Float",
                                 Precision="8")

        field.text = "\n%s%s\n%s" %(11*" ", spacing, 8*" ")

        for var in ["H", "HU", "HV"]:
            attr = etree.SubElement(xmf_tree['grd'], "Attribute", Name=var,
                                    Center="Node", AttributeType="Scalar")

            field = etree.SubElement(attr, "DataItem", Dimensions=dims,
                                     Format="HDF", NumberType="Float", Precision="8")
            text = "%s:/%s" %(os.path.basename(filename), var)
            field.text = "\n%s%s\n%s" %(11*" ", text, 8*" ")


            xmf_ct = etree.tostring(xmf_tree['root'], pretty_print=True,
                                    xml_declaration=True,
                                    doctype='<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>')
            xmf_ct = xmf_ct.decode().replace("encoding=\'ASCII\'", "")

        xmf_file = filename.replace(".h5", ".xmf")
        with open(xmf_file, "w") as fout:
            fout.write(xmf_ct)


    def dump(self, prefix, ite):
        """Dump solution to file"""
        keys = ["H", "HU", "HV"]
        filename = "%s_%08d.h5" %(prefix, ite)
        print("\t Shalow Water Solver: Dumping solution to %s" %filename)
        with h5py.File(filename, "w") as fout:
            for i, key in enumerate(keys):
                fout.create_dataset(name=key,
                                    data=self.solution[i, 1:-1, 1:-1])
        self._dump_xmf(filename)


SOLVER = None
def _init_solver(params):
    global SOLVER
    SOLVER = ShallowWaterSolver(params["dim"])
    SOLVER.set_dx(params["dx"])
    SOLVER.set_cfl(params["c"])

def solve(num, vec_in, params):
    """solver"""
    if SOLVER is None:
        _init_solver(params)
    if num == 1:
        SOLVER.init(vec_in)
        sol, tstep = SOLVER.solve()
    else:
        sol = np.zeros_like(vec_in)
        for i in range(num):
            SOLVER.init(vec_in[:, i])
            sol[:, i], tstep = SOLVER.solve()
    return tstep, sol.real
