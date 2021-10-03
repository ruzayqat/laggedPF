"""Module lagged particle filter"""
import os
import time
from functools import partial
import numpy as np
import h5py
from mcmc import mcmc1, mcmc2, mcmc3
from data_tools import sample_from_f, get_predictor_stats, get_data
from tools import (mat_mul, fwd_slash, bisection, logdet)
from solver import solve

def _iterline(isim, stage, step, kappa, timestart):
    elapsed = time.time()-timestart
    msg = '\t [LPF-%d] isim = %05d, p(%05d) = %05d elapsed = %14.8es\n'
    msg = msg % (stage, isim, step, kappa, elapsed)
    filename = "lpf_%08d.log" %isim
    with open(filename, "a") as fout:
        fout.write(msg)

def calc_ess(weights0, sum_weights0):
    """ Calculates the effective sample size """
    return sum_weights0 ** 2 / np.sum(weights0 ** 2)


def resample_1(ess, xvect, weights, params):
    """If ess < threshold, resample the particles according to
    their weights
    """
    if ess <= params["ESS_threshold"]:
        rand = np.random.choice(np.arange(params["N"]), params["N"], p=weights)
        xvect = xvect[:, rand]
        lw_old = np.ones(params["N"])*(-np.log(params["N"]))
        resampled = True
    else:
        resampled = False
        lw_old = np.log(weights)
    return xvect, lw_old, resampled

def resample_2(ess, nstep, path_m, weights, params):
    """Resamples the whole path from time 0 to n of all particles according to
    their weights
    """
    if ess <= params["ESS_threshold"]:
        rand = np.random.choice(np.arange(params["N"]), params["N"], p=weights)
        for step in range(nstep + 2):
            ind1 = params["N"] * step
            ind2 = params["N"] * (step + 1)
            temp = path_m[:, ind1:ind2]
            temp = temp[:, rand]
            path_m[:, ind1:ind2] = temp
        lw_old = np.ones(params["N"])*(-np.log(params["N"]))
        resampled = True
    else:
        resampled = False
        lw_old = np.log(weights)
    return path_m, lw_old, resampled


def _normalize_weights(lw_old, norm_const, temp):
    log_w = lw_old + norm_const + np.diag(temp)
    max_lw = np.max(log_w)
    weights0 = np.exp(log_w - max_lw)
    sum_weights0 = np.sum(weights0)
    weights = weights0 / sum_weights0
    return log_w, weights, weights0, sum_weights0

def weights_1(data, delta, signal, lw_old, params):
    """Calculates the weights for the SMC sampler on the log scale.
    Inputs: 1) data: Y[:,n] the data at time n.
            2) delta: The power in g(yn|xn)^delta.
            3) X: The signal of size (dimx x N).
            4) lw_old: The previous log-weights.
            5) r2_inv: The inverse of the covariance matrix of the data noise
                Note that if Y_CoefMatrix_is_eye is True, then pass None.
    Outpus: 1) lw: the log-weights
            2) We: the normalized weights ,
            3) We0: the weights minus the maximum weight
            4) sumWe0: their sum
    """
    norm_const = params["dimo"] * np.log(2.0 * np.pi) + params["ldet_R2"]
    norm_const *= -0.5 * delta
    if params["C_is_eye"]:
        vec = data - signal  # data - mu, mu is mean of g(yn|xn);
    else:
        vec = data - mat_mul(params["C"], signal)
    if params["Y_CoefMatrix_is_eye"]:
        temp = - 0.5 * delta * params["sig_y"] ** (-2) * mat_mul(vec.T, vec)
    else:
        temp = - 0.5 * delta * mat_mul(vec.T, mat_mul(params["R2_inv"], vec))

    return _normalize_weights(lw_old, norm_const, temp)

def weight_2(step, data, delta, signal, aggr, params):
    """The path is 2d array. For making it easy to understand assume the path
    is 3d array of shape (T+1 ,dimx, N), that is for each time t = 0,...,T+1:
    path[t,:,:] is all the particles at time t. However, path is a 2d
    array of shape (dimx,(T+1)*N), at time t = 0, the particles are:
    path[:,0:N], at t=1, path[:,N:2*N], etc..
    """
    path = aggr[0]
    x_f_n_ml_p1 = aggr[1]
    p_f_n_ml_p1_inv = aggr[2]
    ldet_p_f_n_ml_p1 = aggr[3]
    fxn = aggr[4]
    lw_old = aggr[5]

    if params["C_is_eye"]:
        vec1 = data - signal  # data - mu, mu is mean of g(yn|xn);
    else:
        vec1 = data - mat_mul(params["C"], signal)
    # in the following when n = L, we have mu_{X_1}(X_2) (recall:
    # path[2,:,:] = X2):
    ind1 = params["N"] * (step - params["L"] + 2)
    ind2 = params["N"] * (step - params["L"] + 3)
    temp = path[:, ind1:ind2] - x_f_n_ml_p1  # when n = L, path[2,:,:] --> X_2
    # this is from the exponential part of $log(\mu_{n-L}(x_{n-L+1})$

    vec3 = path[:, ind1:ind2] - fxn  # when n = L, have x_2 - f(
    norm_const = params["ldet_R1"] - params["dimo"] * np.log(2.0 * np.pi)
    norm_const += -params["ldet_R2"] - ldet_p_f_n_ml_p1
    norm_const = delta / 2 * norm_const

    temp = mat_mul(temp.T, mat_mul(p_f_n_ml_p1_inv, temp))

    if params["Y_CoefMatrix_is_eye"] & params["X_CoefMatrix_is_eye"]:
        # g(yn|xn)^delta
        temp += params["sig_y"] ** (-2) * mat_mul(vec1.T, vec1)
        temp -= params["sig_x"] ** (-2) * mat_mul(vec3.T, vec3)
    else:
        temp += mat_mul(vec1.T, mat_mul(params["R2_inv"], vec1))
        temp -= mat_mul(vec3.T, mat_mul(params["R1_inv"], vec3))
    temp *= -0.5 * delta

    return _normalize_weights(lw_old, norm_const, temp)

def funcofdelta_1(data, signal, delta, params):
    """needs docstring"""
    # weights_1(data, delta, signal, lw_old, params)
    if params["Y_CoefMatrix_is_eye"]:
        func_ = partial(weights_1, data=data, signal=signal,
                        lw_old=np.zeros(params["N"]), params=params)
        _, _, weights0, sum_weights0 = func_(delta=delta)
    else:
        raise NotImplementedError("Matrix R2_inv not implemented yet!")
        # need to modify to pass R2_inv:
        # lw, We, weights0, sum_weights0 = weights_1(Y1, phi[0], np.zeros(N),R2_inv)
    ess = calc_ess(weights0, sum_weights0) - params["ESS_threshold"]
    return ess


def funcofdelta_2(step, data, signal, delta, aggr, params):
    """needs docstring"""
    if params["Y_CoefMatrix_is_eye"]:
        if len(aggr) < 6:
            aggr.append(np.zeros(params["N"]))
        func_ = partial(weight_2, step=step, data=data,
                        signal=signal, params=params,
                        aggr=aggr)
        _, _, weights0, sum_weights0 = func_(delta=delta)
    else:
        raise NotImplementedError("Matrix R2_inv not implemented yet!")

    ess = calc_ess(weights0, sum_weights0) - params["ESS_threshold"]
    return ess


def _get_xstar(params):
    with h5py.File(params["data_file"], "r") as fin:
        xstar = fin["step_%08d" %0]["signal"][...]
    return xstar

def _get_conv(terminate, converged, k, phi, delta):
    terminate = False
    if (not converged) & (k == 0):
        delta = 0.00001
    if (not converged) & (k > 0):
        delta = 1.02 * (phi[k] - phi[k - 1])
    if (delta <= (1. - phi[k])) & (delta >= 0.):
        phi.append(phi[k] + delta)
    elif delta > (1. - phi[k]):
        delta = 1. - phi[k]
        phi.append(1)
        terminate = True
    if phi[k + 1] == 1.:
        terminate = True
    return terminate, converged, k, phi, delta

def laggedpf(params, isim):
    """main function of lagged particle filter"""
    lpf = LaggedPf(isim, params)
    lpf.run()

class LaggedPf():
    """Lagged particle filter main class"""
    def __init__(self, isim, params):
        print("Creating path vector for %08d...." %isim)
        self.path = np.zeros((params["dimx"], params["N"] * (params["T"] + 1)))
        self.params = params
        params["isim"] = isim
        self.signal = None
        self.t_simul = 0
        self.fx0 = None
        self.ess_saved = []
        self.lw_old = None
        self.pstep = 0
        self._setup()

    def _setup(self):
        # files = ["data_file", "predictor_file", "lagged_file"]
        files = ["lagged_file"]
        # log_file = "lpf_%08d.log" %self.params["isim"]

        # for i, file in enumerate(files):
        for _, file in enumerate(files):
            filename = os.path.basename(self.params[file])
            filename = "%s_%05d.h5" %(filename.split(".h5")[0], self.params["isim"])
            filename = os.path.join(os.path.dirname(self.params[file]), filename)
            # # if i in [0, 1]:
            #     with open(log_file, "a") as fout:
            #         fout.write("Making a local copy of %s...\n" %self.params[file])
            #     copyfile(self.params[file], filename)
            self.params[file] = filename


    def run(self):
        """Main function"""
        print("Entering main loop for %08d...." %self.params["isim"])
        self.step0()
        self._dump_restart()
        if self.params["L"] > 1:
            for self.pstep in range(1, self.params["L"]):
                self.step1()
                self._dump_restart()
        else:
            for self.pstep in range(self.params["L"], self.params["T"]):
                self.step2()
                self._dump_restart()

    def _dump_restart(self):
        if "restart_file" not in self.params:
            filename = "restarts/restart_%08d.h5"
            self.params["restart_file"] = filename %self.params["isim"]

        filename = "lpf_%08d.log" %self.params["isim"]
        with open(filename, "a") as fout:
            fout.write("Dumping restart file file %s\n" %self.params["restart_file"])

        dir_ = os.path.dirname(self.params["restart_file"])
        os.makedirs(dir_, exist_ok=True)

        if os.path.isfile(self.params["restart_file"]):
            os.remove(self.params["restart_file"])

        with h5py.File(self.params["restart_file"], "w") as fout:
            fout.create_dataset(name="t_simul", data=self.t_simul)
            fout.create_dataset(name="ess_saved", data=self.ess_saved)
            fout.create_dataset(name="path", data=self.path)
            fout.attrs["ite"] = self.pstep

    # def _dump(self):
    #     log_file = "lpf_%08d.log" %self.params["isim"]
    #     with open(log_file, "a") as fout:
    #         fout.write("Dumping results to file %s\n" %self.params["lagged_file"])
    #     dir_ = os.path.dirname(self.params["lagged_file"])
    #     os.makedirs(dir_, exist_ok=True)
    #     with h5py.File(self.params["lagged_file"], "w") as fout:
    #         fout.create_dataset(name="t_simul", data=self.t_simul)
    #         fout.create_dataset(name="ess_saved", data=self.ess_saved)
    #         fout.create_dataset(name="path", data=self.path)

    def step0(self):
        """n=0"""
        self.lw_old = -np.log(self.params["N"]) * np.ones(self.params["N"])
        phi = []
        phi.append(self.params["phi1"])
        ind1 = 0
        ind2 = self.params["N"]
        x_star = _get_xstar(self.params)
        self.path[:, ind1:ind2] = np.transpose([x_star] * self.params["N"])
        # Sample the new state X_{1} from f(x_1|X_{0})
        self.t_simul, self.signal, self.fx0 = sample_from_f(self.params["N"],
                                                            1, x_star, 0,
                                                            self.params)
        self.fx0 = self.fx0.reshape(-1, 1)
        if (self.pstep + 1) % self.params["t_freq"] == 0:
            self._step0_process(phi)
        ind1 = self.params["N"] * 1
        ind2 = self.params["N"] * 2
        self.path[:, ind1:ind2] = self.signal

    def step1(self):
        """1<n<L"""
        phi = []
        phi.append(["phi1"])
        self.t_simul, self.signal, _ = sample_from_f(self.params["N"],
                                                     self.params["N"],
                                                     self.signal,
                                                     self.t_simul,
                                                     self.params)

        if (self.pstep + 1) % self.params["t_freq"] == 0:
            self._step1_process(phi)
        ind1 = self.params["N"] * (self.pstep + 1)
        ind2 = self.params["N"] * (self.pstep + 2)
        self.path[:, ind1:ind2] = self.signal

    def step2(self):
        """step2"""
        phi = []
        phi.append(self.params["phi1"])
        self.t_simul, self.signal, _ = sample_from_f(self.params["N"],
                                                     self.params["N"],
                                                     self.signal,
                                                     self.t_simul,
                                                     self.params)

        if (self.pstep + 1) % self.params["t_freq"] == 0:
            self._step2_process(phi)
        ind1 = self.params["N"] * (self.pstep + 1)
        ind2 = self.params["N"] * (self.pstep + 2)
        self.path[:, ind1:ind2] = self.signal


    def _step2_process(self, phi):
        timestart = time.time()
        step = self.pstep - self.params["L"] + 1
        pred_mean_n_ml_p1, pred_conv_n_ml_p1 = get_predictor_stats(self.params,
                                                                   step)
        pred_mean_n_ml_p1 = pred_mean_n_ml_p1.reshape(-1, 1)
        pred_cov_n_ml_p1_inv = fwd_slash(self.params["Idx"],
                                         pred_conv_n_ml_p1)
        ldet_pred_cov_n_ml_p1 = logdet(pred_conv_n_ml_p1)

        step = self.pstep - self.params["L"]
        pred_mean_n_ml, pred_conv_n_ml = get_predictor_stats(self.params,
                                                             step)
        pred_mean_n_ml = pred_mean_n_ml.reshape(-1, 1)
        pred_cov_n_ml_inv = fwd_slash(self.params["Idx"], pred_conv_n_ml)

        #------
        ind1 = self.params["N"] * (self.pstep - self.params["L"] + 1)
        ind2 = self.params["N"] * (self.pstep - self.params["L"] + 2)
        _, fxn = solve(self.params["N"], self.path[:, ind1:ind2], self.params)
        data_n = get_data(self.params, self.pstep).reshape(-1, 1)
        if self.params["X_CoefMatrix_is_eye"] & self.params["Y_CoefMatrix_is_eye"]:
            _, weights, weights0, sum_weights0 = weight_2(self.pstep, data_n, phi[0],
                                                          self.signal,
                                                          [self.path, pred_mean_n_ml_p1,
                                                           pred_cov_n_ml_p1_inv,
                                                           ldet_pred_cov_n_ml_p1,
                                                           fxn, self.lw_old],
                                                          self.params)
        else:
            raise NotImplementedError("only Y_CoefMatrix_is_eye is supported")
        ess = calc_ess(weights0, sum_weights0)
        self.ess_saved.append(ess)
        self.signal, self.lw_old = resample_1(ess, self.signal, weights, self.params)[0:2]
        terminate = False
        k = 0

        while not terminate:
            # (ESS - ESS_threshold) as a function of delta:
            aggr = [self.path, pred_mean_n_ml_p1, pred_cov_n_ml_p1_inv,
                    ldet_pred_cov_n_ml_p1, fxn]

            func_ = lambda delta: funcofdelta_2(self.pstep, data_n,
                                                self.signal, delta,
                                                aggr, self.params)
            delta, converged = bisection(func_, self.params)
            terminate, converged, k, phi, delta = _get_conv(terminate, converged, k, phi, delta)
            if self.params["X_CoefMatrix_is_eye"] & self.params["Y_CoefMatrix_is_eye"]:
                _, weights, weights0, sum_weights0 = weight_2(self.pstep,
                                                              data_n, delta,
                                                              self.signal,
                                                              [self.path, pred_mean_n_ml_p1,
                                                               pred_cov_n_ml_p1_inv,
                                                               ldet_pred_cov_n_ml_p1,
                                                               fxn, self.lw_old],
                                                              self.params)

            else:
                raise NotImplementedError("only Y_CoefMatrix_is_eye is supported")
            # resample if necessary:
            ess = calc_ess(weights0, sum_weights0)
            self.ess_saved.append(ess)
            self.signal, self.lw_old, resampled = resample_1(ess, self.signal,
                                                             weights, self.params)

            if self.params["X_CoefMatrix_is_eye"] & self.params["Y_CoefMatrix_is_eye"]:
                path2update = mcmc3(self.params, self.pstep, self.params["isim"],
                                    self.path, self.signal, phi, k, self.fx0,
                                    pred_mean_n_ml_p1, pred_cov_n_ml_p1_inv,
                                    pred_mean_n_ml, pred_cov_n_ml_inv)
            else:
                raise NotImplementedError("only Y_CoefMatrix_is_eye is supported")
                # Need to modify here to include R1_inv and R2_inv:
                # X = mcmc2(self.params, n,h,phi,k,self.path,X,Yn, R1_inv, R2_inv)
            ind1 = self.params["N"] * (self.pstep - self.params["L"] + 1)
            ind2 = self.params["N"] * (self.pstep + 2)
            self.path[:, ind1:ind2] = path2update
            ind1 = self.params["N"] * self.params["L"]
            ind2 = self.params["N"] * (self.params["L"] + 1)
            self.signal = path2update[:, ind1:ind2]
            k += 1
            if not terminate:
                ind1 = self.params["N"] * (self.pstep - self.params["L"] + 1)
                ind2 = self.params["N"] * (self.pstep - self.params["L"] + 2)
                _, fxn = solve(self.params["N"], self.path[:, ind1:ind2], self.params)

        _iterline(self.params["isim"], 2, self.pstep, k, timestart)
        if not resampled:
            self.path, self.lw_old = resample_2(ess, self.pstep, self.path,
                                                weights, self.params)[0:2]
        ind1 = self.params["N"] * (self.pstep + 1)
        ind2 = self.params["N"] * (self.pstep + 2)
        self.signal = self.path[:, ind1:ind2]



    def _step1_process(self, phi):
        timestart = time.time()
        data = get_data(self.params, self.pstep).reshape(-1, 1)
        if self.params["Y_CoefMatrix_is_eye"]:
            log_w, weights, weights0, sum_weights0 = weights_1(data, phi[0], self.signal,
                                                               self.lw_old, self.params)
        else:
            raise NotImplementedError("only Y_CoefMatrix_is_eye is supported")
            # need to modify to include R2_inv:
            # lw, We, We0, sumWe0 = weights_1(Y1, phi[0],X,np.zeros(N),R2_inv)
        # resample if necessary:
        ess = calc_ess(weights0, sum_weights0)
        self.ess_saved.append(ess)
        self.signal, self.lw_old = resample_1(ess, self.signal, log_w, weights)[0:2]
        terminate = False
        k = 0

        while not terminate:
            # (ESS - ESS_threshold) as a function of delta:
            func_ = lambda delta: funcofdelta_1(data, self.signal, delta, self.params)
            delta, converged = bisection(func_, self.params)
            terminate, converged, k, phi, delta = _get_conv(terminate, converged, k, phi, delta)
            if self.params["Y_CoefMatrix_is_eye"]:
                log_w, weights, weights0, sum_weights0 = weights_1(data, delta,
                                                                   self.signal,
                                                                   self.lw_old,
                                                                   self.params)
            else:
                raise NotImplementedError("only Y_CoefMatrix_is_eye is supported")
                # need to modify to pass R2_inv:
                # lw,We,We0,sumWe0 = weights_1(Y1,delta,X,np.zeros(N),R2_inv)
            # resample if necessary:
            ess = calc_ess(weights0, sum_weights0)
            self.ess_saved.append(ess)
            self.signal, self.lw_old, resampled = resample_1(ess, self.signal, log_w, weights)
            ##### mcmc step - random walk ########
            if self.params["X_CoefMatrix_is_eye"] & self.params["Y_CoefMatrix_is_eye"]:
                path2update = mcmc2(self.params, self.pstep,
                                    self.params["isim"], phi, k,
                                    self.path, self.signal, data)
            else:
                raise NotImplementedError("only Y_CoefMatrix_is_eye is supported")
                # Need to modify here to include R1_inv and R2_inv:
                # path2update =mcmc2(self.params, n,h,phi,k,path,X,Yn,R1_inv,R2_inv)
            self.path[:, 0:self.params["N"] * (self.pstep + 2)] = path2update
            # at this laggedpf step, self.pstep is in [1,L-1]
            ind1 = self.params["N"] * (self.pstep + 1)
            ind2 = self.params["N"] * (self.pstep + 2)
            self.signal = path2update[:, ind1:ind2]
            k += 1
        # end while
        _iterline(self.params["isim"], 1, self.pstep, k, timestart)
        if not resampled:
            self.path, self.lw_old = resample_2(ess, self.pstep, self.path,
                                                weights, self.params)[0:2]
        ind1 = self.params["N"] * (self.pstep + 1)
        ind2 = self.params["N"] * (self.pstep + 2)
        self.signal = self.path[:, ind1:ind2]

    def _step0_process(self, phi):
        timestart = time.time()
        data = get_data(self.params, 0).reshape(-1, 1)
        if self.params["Y_CoefMatrix_is_eye"]:
            _, weights, weights0, sum_weights0 = weights_1(data, phi[0], self.signal,
                                                           np.zeros(self.params["N"]),
                                                           self.params)
        else:
            raise NotImplementedError("only Y_CoefMatrix_is_eye is supported")
            # need to modify to include R2_inv:
            # lw, We, We0, sum_weights0 = weights_1(Y1, phi[0],X,np.zeros(N),R2_inv)
        # resample if necessary:
        ess = calc_ess(weights0, sum_weights0)
        self.ess_saved.append(ess)
        self.signal, lw_old = resample_1(ess, self.signal, weights, self.params)[0:2]
        terminate = False
        k = 0
        while not terminate:
            func_ = lambda delta: funcofdelta_1(data, self.signal, delta, self.params)
            delta, converged = bisection(func_, self.params)
            terminate, converged, k, phi, delta = _get_conv(terminate, converged, k, phi, delta)

            if self.params["Y_CoefMatrix_is_eye"]:
                _, weights, weights0, sum_weights0 = weights_1(data, delta,
                                                               self.signal, lw_old,
                                                               self.params)
            else:
                raise NotImplementedError("only Y_CoefMatrix_is_eye is supported")
                # lw,We,We0,sumWe0 = weights_1(Y1,delta,X,np.zeros(N),R2_inv)
            # resample if necessary:
            ess = calc_ess(weights0, sum_weights0)
            self.ess_saved.append(ess)
            self.signal, lw_old, resampled = resample_1(ess, self.signal,
                                                        weights, self.params)

            if self.params["X_CoefMatrix_is_eye"] & self.params["Y_CoefMatrix_is_eye"]:
                self.signal = mcmc1(self.params, 0, self.params["isim"],
                                    phi, k, self.signal, data, self.fx0)
            else:
                raise NotImplementedError("only Y_CoefMatrix_is_eye is supported")
                # Need to modify here to include R1_inv and R2_inv:
                # X = mcmc1(self.params, n,h,phi,k,X,Yn,fx0, R1_inv, R2_inv)
            k += 1
        _iterline(self.params["isim"], 0, 1, k, timestart)
        if not resampled:
            self.signal, lw_old = resample_1(ess, self.signal,
                                             weights, self.params)[0:2]
