"""Module container for Kalman filtering  based on
   Houtekamer et al. (1998). 'Data assimilation using an
   ensemble Kalman filter technique' """
from glob import glob
import numpy as np
import h5py
from utilities import (mvnrnd,
                       fwd_slash)
from solver import ShallowWaterSolver

class EnsembleKalmanFilter():
    """docstring for EnsembleKalmanFilter"""
    def __init__(self, obs_size, ens_size, obs_dim, outfile):
        self.props = {"obs_size":obs_size,
                      "ens_size":ens_size,
                      "nite":100,
                      "prefix":outfile,
                      "predictor_stats":False,
                      "obs_all_coords":False,
                      "ite":0,
                      "time":0.,
                      'y_coefmatrix_is_eye':True,
                      "id_obs": np.eye(obs_dim),
                      "data_files":[],
                      "data_chunck":0}

        self.e_loc = np.zeros(obs_size)
        self.x_a = np.empty(())
        self.x_f = np.empty(())
        self.obs_matrix = np.eye(obs_dim)
        self.pred_stats = [None]*2
        dim = np.sqrt(obs_size//3)
        self.sampler = ShallowWaterSolver(dim)


    def set_nite(self, nite):
        """Set number of iterations"""
        self.props["nite"] = nite

    def set_obsmatrix(self, obs_matrix):
        """Set observation matrix"""
        if hasattr(obs_matrix, '__iter__'):
            self.obs_matrix = np.matmul(obs_matrix, obs_matrix)
            self.props["y_coefmatrix_is_eye"] = False
        else:
            self.obs_matrix *= obs_matrix**2

    def initialize(self, initial):
        """Set initial condition"""
        self.x_a = np.transpose([initial] * self.props["ens_size"])
        self.e_loc = initial
        self._dump("e_loc")

    def predictor_only(self):
        """Toggle only computation of predictor stats"""
        self.props["predictor_stats"] = True

    def obs_all_coords(self):
        """toggle observation of all coords"""
        self.props["obs_all_coords"] = True

    def _sample(self, sig_x):
        for i in range(self.props["ens_size"]):
            self.sampler.init(self.x_a[i, :])
            self.x_a[i, :], dt_ = self.sampler.solve()
        noise = mvnrnd(np.zeros(self.props["obs_size"]),
                       np.eye(self.props["obs_size"]),
                       coln=self.props["ens_size"])
        self.x_a = self.x_a + sig_x * noise

        self.props["time"] += dt_
        self.props["ite"] += 1

    def set_data_params(self, prefix, chunk_size=10):
        """Specify location with prefix of data files and number of data to
        be loaded at once"""
        files = glob("%s*.h5" %prefix)
        files = sorted(files)
        self.props["data_files"] = files
        self.props["data_chuck"] = chunk_size

    def _load_data(self, idx):
        dim2 = self.props['obs_size']//3
        data = np.zeros(self.props['obs_size'])
        filename = self.props["data_files"][idx]
        with h5py.File(filename, "r") as fin:
            for i, key in enumerate(["H", 'HV', "HV"]):
                data[i*dim2:(i+1)*dim2] = fin[key][...].ravel()
        return data

    def run(self, c_mat, sig_x, step, cfl=0.5):
        """Execute one run"""
        self.sampler.set_dx(step)
        self.sampler.set_cfl(cfl)
        for i in range(self.props["nite"]):
            data = self._load_data(i)
            self._sample(sig_x)
            self._advance(c_mat, data)

    def _advance(self, c_mat, data):
        """One iteration of ENKF"""
        e_size = self.props["ens_size"]

        mean = np.sum(self.x_f, axis=1)/e_size
        diff = self.x_f - mean.reshape(-1, 1)
        if self.props["predictor_stats"]:
            self.pred_stats[0] = mean
            self.pred_stats[1] = np.matmul(diff, diff.T) / (e_size - 1)
            self._dump("pred_stats")
        else:
            if self.props["obs_all_coords"]:
                temp = np.matmul(diff, diff.T) / e_size
            else:
                temp = np.matmul(diff.T, c_mat.T)
                temp = np.matmul(diff, temp)
                temp = np.matmul(c_mat, temp) / e_size


            temp = fwd_slash(self.props["id_obs"], temp + self.obs_matrix)

            if not self.props["obs_all_coords"]:
                temp = np.matmul(c_mat.T, temp)

            temp = np.matmul(diff.T, temp)
            kappa = np.matmul(diff, temp) / e_size
            noise = mvnrnd(np.zeros(self.props["obs_dim"]), self.props['id_obs'],
                           coln=e_size)
            temp = data.reshape(-1, 1)

            if self.props['y_coefmatrix_is_eye']:
                temp -= self.obs_matrix * noise
            else:
                temp -= np.matmul(self.obs_matrix, noise)

            if self.props["obs_all_coords"]:
                temp -= self.x_f
            else:
                temp -= np.matmul(c_mat, self.x_f)
            self.x_a = self.x_f + np.matmul(kappa, temp)
            self.e_loc = np.sum(self.x_a, axis=1) / e_size
            self._dump("e_loc")

    def _dump(self, kind):
        """Dump to file, spcifying kind"""
        filename = self.props["prefix"]
        if kind.lower() == "pred_stats":
            filename = "%s_pred_stats_%08d.h5"
        elif kind.lower() == "e_loc":
            filename = "%s_eloc_%08d.h5"

        with h5py.File(filename, "a") as fout:
            grpname = "%08d" %self.props["ite"]
            if grpname not in fout:
                grp = fout.create_group(grpname)
            else:
                grp = fout[grpname]

            if kind.lower() == "pred_stats":
                grp.create_dataset(name="mean", data=self.pred_stats[0])
                grp.create_dataset(name="cov", data=self.pred_stats[0])
            elif kind.lower() == "e_loc":
                grp.create_dataset(name="e_loc", data=self.e_loc)
            grp.attrs["ite"] = self.props["ite"]
            grp.attrs["time"] = self.props['time']
