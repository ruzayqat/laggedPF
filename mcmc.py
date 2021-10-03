"""Module container for MCMC"""
import time
import numpy as np
from tools import (mvnrnd, mat_mul)
from solver import solve
from data_tools import get_data

def _iterline(stage, isim, meta):
    pstep, mcmciter, aar, phi, timestart = meta
    elapsed = time.time() - timestart
    msg = 'isim = %05d, step = %05d, steps = %05d,'
    msg = msg +' rate = %9.3e, phi = %9.3e, elapsed = %9.3es'
    msg = "\t\t[MCMC-%d]:" %stage + msg
    msg = msg % (isim, pstep, mcmciter, aar, phi, elapsed)
    log_file = "mcmc_%08d.log" %isim
    with open(log_file, "a") as fout:
        fout.write(msg+"\n")


def mcmc1(params, pstep, isim, phi, k, signal, data_n, fx0):
    accept = np.zeros(params["N"])
    aar = 0
    mcmciter = 0
    terminatemcmc = False
    timestart = time.time()
    while not terminatemcmc:
        mcmciter += 1
        cond = (mcmciter > params["MCMCnmin"]) &\
               (aar >= params["aar_min"]) &\
               (aar <= params["aar_max"])
        if cond:
            terminatemcmc = True
        if mcmciter >= params["MCMCnmax"]:
            terminatemcmc = True
        covm = params["sig"]
        if aar < params["aar_min"]:
            covm = params["sig"] * (phi[k] + 2) / (phi[k] + 1) / (mcmciter ** 5 + 10)
        if aar > params["aar_max"]:
            covm = params["sig"] * (phi[k] + 2) / (phi[k] + 1)
        x_per = mvnrnd(signal, covm, coln=params["N"])
        if params["C_is_eye"]:
            vg_p = data_n - x_per  # p stands for proposed
            vgn = data_n - signal
        else:
            vg_p = data_n - mat_mul(params["C"], x_per)
            vgn = data_n - mat_mul(params["C"], signal)
        vf_p = x_per - fx0
        vfn = signal - fx0
        if params["Y_CoefMatrix_is_eye"]:
            temp1 = -0.5 * phi[k + 1] *params["sig_y"] ** (-2) * mat_mul(vg_p.T, vg_p) \
                    + 0.5 * phi[k + 1] * params["sig_y"] ** (-2) * mat_mul(vgn.T, vgn)
        else:
            temp1 = -0.5 * phi[k + 1] * mat_mul(vg_p.T, mat_mul(params["R2_inv"], vg_p))
            temp1 += 0.5 * phi[k + 1] * mat_mul(vgn.T, mat_mul(params["R2_inv"], vgn))

        if params["X_CoefMatrix_is_eye"]:
            temp2 = -0.5 * params["sig_x"] ** (-2) * mat_mul(vf_p.T, vf_p) \
                    + 0.5 * params["sig_x"] ** (-2) * mat_mul(vfn.T, vfn)
        else:
            temp2 = -0.5 * mat_mul(vf_p.T, mat_mul(params["R1_inv"], vf_p)) \
                    + 0.5 * mat_mul(vfn.T, mat_mul(params["R1_inv"], vfn))

        log_accep = np.minimum(np.zeros(params["N"]), np.diag(temp1 + temp2))
        log_u = np.log(np.random.uniform(size=params["N"]))
        for j in range(params["N"]):
            if log_u[j] < log_accep[j]:
                signal[:, j] = x_per[:, j]
                accept[j] += 1
        accep_rate = accept / mcmciter
        aar = np.mean(accep_rate)
    _iterline(1, isim, [pstep+1, mcmciter, aar, phi[k+1], timestart])
    return signal


###########################
def mcmc2(params, pstep, isim, phi, k, path, signal, data):
    timestart = time.time()
    accept = np.zeros(params["N"])
    aar = 0 #average acceptance rate
    mcmciter = 0
    terminatemcmc = False

    path2update = np.zeros((params["dimx"], params["N"] * (pstep + 2)))
    path2update[:, 0:params["N"] * (pstep + 1)] = path[:, 0:params["N"] * (pstep + 1)]
    ind1 = params["N"] * (pstep + 1)
    ind2 = params["N"] * (pstep + 2)
    path2update[:, ind1:ind2] = signal

    lf_p = np.zeros((params["N"], pstep + 1))  # p stands for proposed
    lg_p = np.zeros((params["N"], pstep + 1))
    lo_f = np.zeros((params["N"], pstep + 1))
    lo_g = np.zeros((params["N"], pstep + 1))
    while not terminatemcmc:
        mcmciter += 1
        cond = (mcmciter > params["MCMCnmin"]) &\
               (aar >= params["aar_min"]) &\
               (aar <= params["aar_max"])
        if cond:
            terminatemcmc = True
        if mcmciter >= params["MCMCnmax"]:
            terminatemcmc = True

        covm = params["sig"]
        if aar < params["aar_min"]:
            covm = params["sig"] * (phi[k] + 2) / (phi[k] + 1) / (mcmciter ** 6 + 15)
        elif aar > params["aar_max"]:
            covm = params["sig"] * (phi[k] + 2) / (phi[k] + 1) / np.log(mcmciter + 1)
        else:
            covm = params["sig"] * (phi[k] + 2) / (phi[k] + 1) / 2
        path_p = mvnrnd(path2update, covm, coln=params["N"] * (pstep + 2))

        for istep in range(pstep):
            ind1 = params["N"] * istep
            ind2 = params["N"] * (istep + 1)
            jnd1 = params["N"] * (istep + 1)
            jnd2 = params["N"] * (istep + 2)

            _, sol = solve(params["N"], path_p[:, ind1:ind2], params)
            vf_p = path_p[:, jnd1:jnd2] - sol
            _, sol = solve(params["N"], path2update[:, ind1:ind2], params)
            vfn = path2update[:, jnd1:jnd2] - sol
            if params["C_is_eye"]:
                vg_p = data.reshape(-1, 1) - path_p[:, jnd1:jnd2]
                vgn = data.reshape(-1, 1) - path2update[:, jnd1:jnd2]
            else:
                vg_p = data.reshape(-1, 1) - mat_mul(params["C"], path_p[:, jnd1:jnd2])
                vgn = data.reshape(-1, 1) - mat_mul(params["C"],
                                                    path2update[:, jnd1:jnd2])

            if params["X_CoefMatrix_is_eye"]:
                lf_p[:, istep] = np.diag(-0.5 * params["sig_x"] ** (-2) * \
                                        mat_mul(vf_p.T, vf_p))
                lo_f[:, istep] = np.diag(-0.5 * params["sig_x"] ** (-2) * \
                                      mat_mul(vfn.T, vfn))
            else:
                lf_p[:, istep] = np.diag(-0.5 * mat_mul(vf_p.T,
                                                        mat_mul(params["R1_inv"],
                                                                vf_p)))
                lo_f[:, istep] = np.diag(-0.5 * mat_mul(vfn.T,
                                                        mat_mul(params["R1_inv"],
                                                                vfn)))
            if istep < pstep - 1:
                if params["Y_CoefMatrix_is_eye"]:
                    lg_p[:, istep] = np.diag(-0.5 * params["sig_y"] ** (-2) \
                                            * mat_mul(vg_p.T, vg_p))
                    lo_g[:, istep] = np.diag(-0.5 * params["sig_y"] ** (-2) \
                                          * mat_mul(vgn.T, vgn))
                else:
                    lg_p[:, istep] = np.diag(-0.5 * mat_mul(vg_p.T,
                                                            mat_mul(params["R2_inv"],
                                                                    vg_p)))
                    lo_g[:, istep] = np.diag(-0.5 * mat_mul(vgn.T,
                                                            mat_mul(params["R2_inv"],
                                                                    vgn)))
            else:
                if params["Y_CoefMatrix_is_eye"]:
                    lg_p[:, istep] = np.diag(-0.5 * phi[k + 1] * params["sig_y"] ** (-2) \
                                            * mat_mul(vg_p.T, vg_p))
                    lo_g[:, istep] = np.diag(-0.5 * phi[k + 1] * params["sig_y"] ** (-2) \
                                          * mat_mul(vgn.T, vgn))
                else:
                    lg_p[:, istep] = np.diag(-0.5 * phi[k + 1] \
                                            * mat_mul(vg_p.T,
                                                      mat_mul(params["R2_inv"],
                                                              vg_p)))
                    lo_g[:, istep] = np.diag(-0.5 * phi[k + 1] \
                                          * mat_mul(vgn.T,
                                                    mat_mul(params["R2_inv"],
                                                            vgn)))

        log_accep = np.minimum(np.zeros(params["N"]), np.sum(lf_p, axis=1) \
                               + np.sum(lg_p, axis=1) - np.sum(lo_f, axis=1) \
                               - np.sum(lo_g, axis=1))
        log_u = np.log(np.random.uniform(size=params["N"]))
        for j in range(params["N"]):
            if log_u[j] < log_accep[j]:
                for istep in range(pstep):
                    ind1 = params["N"] * istep + j
                    path2update[:, ind1] = path_p[:, ind1]
                accept[j] += 1
        accep_rate = accept / mcmciter
        aar = np.mean(accep_rate)

    _iterline(2, isim, [pstep+1, mcmciter, aar, phi[k+1], timestart])

    return path2update

def mcmc3(params, pstep, isim, path, signal, phi, k, fx0, x_f_n_ml_p1, p_f_n_ml_p1_inv,
          x_f_n_ml, p_f_n_ml_inv):

    timestart = time.time()
    accept = np.zeros(params["N"])
    aar = 0 #average acceptance rate
    mcmciter = 0
    terminatemcmc = False
    path2update = np.zeros((params["dimx"], params["N"] * (params["L"] + 1)))
    ind1 = params["N"] * (pstep + 1 - params["L"])
    ind2 = params["N"] * (pstep + 1)
    path2update[:, 0:params["N"] * params["L"]] = path[:, ind1:ind2]
    path2update[:, params["N"] * params["L"]: params["N"] * (params["L"] + 1)] = signal
    lf_p = np.zeros((params["N"], params["L"]))
    lg_p = np.zeros((params["N"], params["L"]))
    lo_f = np.zeros((params["N"], params["L"]))
    lo_g = np.zeros((params["N"], params["L"]))
    while not terminatemcmc:
        mcmciter += 1
        cond = (mcmciter > params["MCMCnmin"]) &\
               (aar >= params["aar_min"]) &\
               (aar <= params["aar_max"])
        if cond:
            terminatemcmc = True
        if mcmciter >= params["MCMCnmax"]:
            terminatemcmc = True

        covm = params["sig"]
        if aar < params["aar_min"]:
            covm = params["sig"] * (phi[k] + 2) / (phi[k] + 1) / (mcmciter**6 + 10)
        elif aar > params["aar_max"]:
            covm = params["sig"] * (phi[k] + 2) / (phi[k] + 1) / (mcmciter**2 + 1)
        else:
            covm = params["sig"] * (phi[k] + 2) / (phi[k] + 1) / 2
        path_p = mvnrnd(path2update, covm, coln=params["N"] * (params["L"] + 1))
        if pstep == params["L"]:
            vf_1_0_p = path_p[:, 0:params["N"]] - fx0
            vf_1_0 = path2update[:, 0:params["N"]] - fx0
            if params["X_CoefMatrix_is_eye"]:
                lf_1_0p = np.diag(-0.5 * params["sig_x"] ** (-2) * \
                                     mat_mul(vf_1_0_p.T, vf_1_0_p))
                lf_1_0 = np.diag(-0.5 * params["sig_x"] ** (-2) * \
                                    mat_mul(vf_1_0.T, vf_1_0))
            else:
                lf_1_0p = np.diag(-0.5 * mat_mul(vf_1_0_p.T,
                                                 mat_mul(params["R1_inv"],
                                                         vf_1_0_p)))
                lf_1_0 = np.diag(-0.5 * params["sig_x"] ** (-2) * \
                                 mat_mul(vf_1_0.T, mat_mul(params["R1_inv"],
                                                           vf_1_0)))
            for istep in range(params["L"]):
                ind1 = params["N"] * istep
                ind2 = params["N"] * (istep + 1)
                _, fix = solve(params["N"], path_p[:, ind1:ind2], params)
                jnd1 = params["N"] * (istep + 1)
                jnd2 = params["N"] * (istep + 2)
                vf_p = path_p[:, jnd1:jnd2] - fix
                data = get_data(params, istep)
                if params["C_is_eye"]:
                    vg_p = data.reshape(-1, 1) - path_p[:, ind1:ind2]
                else:
                    vg_p = data.reshape(-1, 1) - mat_mul(params["C"],
                                                                   path_p[:, ind1:ind2])
                _, fix = solve(params["N"], path2update[:, ind1:ind2], params)

                vfn = path2update[:, jnd1:jnd2] - fix
                if params["C_is_eye"]:
                    vgn = data.reshape(-1, 1) - path2update[:, ind1:ind2]
                else:
                    vgn = data.reshape(-1, 1) - mat_mul(params["C"],
                                                                  path2update[:, ind1:ind2])
                if params["X_CoefMatrix_is_eye"]:
                    lf_p[:, istep] = np.diag(-0.5 * params["sig_x"] ** (-2) * \
                                            mat_mul(vf_p.T, vf_p))
                    lo_f[:, istep] = np.diag(-0.5 * params["sig_x"] ** (-2) \
                                          * mat_mul(vfn.T, vfn))
                else:
                    lf_p[:, istep] = np.diag(-0.5 * mat_mul(vf_p.T,
                                                            mat_mul(params["R1_inv"],
                                                                    vf_p)))
                    lo_f[:, istep] = np.diag(-0.5 * mat_mul(vfn.T,
                                                            mat_mul(params["R1_inv"],
                                                                    vfn)))
                if params["Y_CoefMatrix_is_eye"]:
                    lg_p[:, istep] = np.diag(-0.5 * params["sig_y"] ** (-2) * \
                                            mat_mul(vg_p.T, vg_p))
                    lo_g[:, istep] = np.diag(-0.5 * params["sig_y"] ** (-2) * \
                                          mat_mul(vgn.T, vgn))
                else:
                    lg_p[:, istep] = np.diag(-0.5 * mat_mul(vg_p.T,
                                                            mat_mul(params["R2_inv"],
                                                                    vg_p)))
                    lo_g[:, istep] = np.diag(-0.5 * mat_mul(vgn.T,
                                                            mat_mul(params["R2_inv"],
                                                                    vgn)))
            # calculate log(the ratio to the power phi[k+1]):
            ind1 = params["N"]
            ind2 = 2 * params["N"]
            jnd1 = params["N"] * params["L"]
            jnd2 = params["N"] * (params["L"] + 1)
            data = get_data(params, pstep)
            if params["C_is_eye"]:
                vg_pr = data.reshape(-1, 1) - path_p[:, jnd1:jnd2]
                vg_r = data.reshape(-1, 1) - path2update[:, jnd1:jnd2]
            else:
                vg_pr = data.reshape(-1, 1) - mat_mul(params["C"],
                                                                path_p[:, jnd1:jnd2])
                vg_r = data.reshape(-1, 1) - mat_mul(params["C"],
                                                               path2update[:, jnd1:jnd2])

            vec_mu_p = path_p[:, ind1:ind2] - x_f_n_ml_p1
            vec_mu = path2update[:, ind1:ind2] - x_f_n_ml_p1
            if params["Y_CoefMatrix_is_eye"]:
                lg_pr = np.diag(-0.5 * params["sig_y"] ** (-2) * mat_mul(vg_pr.T,
                                                                         vg_pr))
                lg_r = np.diag(-0.5 * params["sig_y"] ** (-2) * mat_mul(vg_r.T,
                                                                        vg_r))
            else:
                lg_pr = np.diag(-0.5 * mat_mul(vg_pr.T,
                                               mat_mul(params["R2_inv"], vg_pr)))
                lg_r = np.diag(-0.5 * mat_mul(vg_r.T,
                                              mat_mul(params["R2_inv"], vg_r)))
            log_mu_p = np.diag(-0.5 * mat_mul(vec_mu_p.T,
                                              mat_mul(p_f_n_ml_p1_inv,
                                                      vec_mu_p)))
            log_mu = np.diag(-0.5 * mat_mul(vec_mu.T,
                                            mat_mul(p_f_n_ml_p1_inv,
                                                    vec_mu)))
            log_ratio = phi[k + 1] * (lg_pr + log_mu_p + lo_f[:, 0]
                                      - lg_r - log_mu - lf_p[:, 0])
            log_accep = np.minimum(np.zeros(params["N"]), log_ratio \
                                   + np.sum(lf_p, axis=1) + lf_1_0p \
                                   + np.sum(lg_p, axis=1) \
                                   - np.sum(lo_f, axis=1) - lf_1_0 \
                                   - np.sum(lo_g, axis=1))
        else:
            for istep in range(params["L"]):
                ind1 = params["N"] * istep
                ind2 = params["N"] * (istep + 1)
                _, fix = solve(params["N"], path_p[:, ind1:ind2], params)
                jnd1 = params["N"] * (istep + 1)
                jnd2 = params["N"] * (istep + 2)
                vf_p = path_p[:, jnd1:jnd2] - fix
                data = get_data(params, istep + pstep - params["L"])
                if params["C_is_eye"]:
                    vg_p = data.reshape(-1, 1)- path_p[:, ind1:ind2]
                else:
                    vg_p = data.reshape(-1, 1) \
                              - mat_mul(params["C"], path_p[:, ind1:ind2])
                _, fix = solve(params["N"], path2update[:, ind1:ind2], params)

                vfn = path2update[:, jnd1:jnd2] - fix
                if params["C_is_eye"]:
                    vgn = data.reshape(-1, 1) - path2update[:, ind1:ind2]
                else:
                    vgn = data.reshape(-1, 1) - mat_mul(params["C"],
                                                        path2update[:, ind1:ind2])

                if params["X_CoefMatrix_is_eye"]:
                    lf_p[:, istep] = np.diag(-0.5 * params["sig_x"] ** (-2) * \
                                            mat_mul(vf_p.T, vf_p))
                    lo_f[:, istep] = np.diag(-0.5 * params["sig_x"] ** (-2) \
                                          * mat_mul(vfn.T, vfn))
                else:
                    lf_p[:, istep] = np.diag(-0.5 * mat_mul(vf_p.T,
                                                            mat_mul(params["R1_inv"],
                                                                    vf_p)))
                    lo_f[:, istep] = np.diag(-0.5 * mat_mul(vfn.T,
                                                            mat_mul(params["R1_inv"],
                                                                    vfn)))
                if params["Y_CoefMatrix_is_eye"]:
                    lg_p[:, istep] = np.diag(-0.5 * params["sig_y"] ** (-2) * \
                                            mat_mul(vg_p.T, vg_p))
                    lo_g[:, istep] = np.diag(-0.5 * params["sig_y"] ** (-2) \
                                          * mat_mul(vgn.T, vgn))
                else:
                    lg_p[:, istep] = np.diag(-0.5 * mat_mul(vg_p.T,
                                                            mat_mul(params["R2_inv"],
                                                                    vg_p)))
                    lo_g[:, istep] = np.diag(-0.5 * mat_mul(vgn.T,
                                                            mat_mul(params["R2_inv"],
                                                                    vgn)))
            # calculate log(the ratio to the power phi[k+1]):
            ind1 = params["N"]
            ind2 = 2 * params["N"]
            jnd1 = params["N"] * params["L"]
            jnd2 = params["N"] * (params["L"] + 1)
            data = get_data(params, pstep)
            if params["C_is_eye"]:
                vg_pr = data.reshape(-1, 1) - path_p[:, jnd1:jnd2]
                vg_r = data.reshape(-1, 1) - path2update[:, jnd1:jnd2]
            else:
                vg_pr = data.reshape(-1, 1) - mat_mul(params["C"],
                                                      path_p[:, jnd1:jnd2])
                vg_r = data.reshape(-1, 1) - mat_mul(params["C"],
                                                     path2update[:, jnd1:jnd2])

            vec_mu_p = path_p[:, ind1:ind2] - x_f_n_ml_p1
            vec_mu = path2update[:, ind1:ind2] - x_f_n_ml_p1
            if params["Y_CoefMatrix_is_eye"]:
                lg_pr = np.diag(-0.5 * params["sig_y"] ** (-2) * mat_mul(vg_pr.T,
                                                                         vg_pr))
                lg_r = np.diag(-0.5 * params["sig_y"] ** (-2) * mat_mul(vg_r.T,
                                                                        vg_r))
            else:
                lg_pr = np.diag(-0.5 * mat_mul(vg_pr.T,
                                               mat_mul(params["R2_inv"], vg_pr)))
                lg_r = np.diag(-0.5 * mat_mul(vg_r.T,
                                              mat_mul(params["R2_inv"], vg_r)))
            log_mu_p = np.diag(-0.5 * mat_mul(vec_mu_p.T,
                                              mat_mul(p_f_n_ml_p1_inv,
                                                      vec_mu_p)))
            log_mu = np.diag(-0.5 * mat_mul(vec_mu.T,
                                            mat_mul(p_f_n_ml_p1_inv,
                                                    vec_mu)))
            log_ratio = phi[k + 1] * (lg_pr + log_mu_p + lo_f[:, 0]
                                      - lg_r - log_mu - lf_p[:, 0])

            vec_mu_p = path_p[:, 0:params["N"]] - x_f_n_ml
            vec_mu = path2update[:, 0:params["N"]] - x_f_n_ml
            log_mu_p = np.diag(-0.5 * mat_mul(vec_mu_p.T,
                                              mat_mul(p_f_n_ml_inv,
                                                      vec_mu_p)))
            log_mu = np.diag(-0.5 * mat_mul(vec_mu.T,
                                            mat_mul(p_f_n_ml_inv,
                                                    vec_mu)))

            log_accep = np.minimum(np.zeros(params["N"]), log_ratio + log_mu_p \
                                   + np.sum(lf_p, axis=1) \
                                   + np.sum(lg_p, axis=1) - log_mu \
                                   - np.sum(lo_f, axis=1) \
                                   - np.sum(lo_g, axis=1))
            # end if
        log_u = np.log(np.random.uniform(size=params["N"]))
        for j in range(params["N"]):
            if log_u[j] < log_accep[j]:
                for istep in range(params["L"]+1):
                    ind1 = params["N"] * istep + j
                    path2update[:, ind1] = path_p[:, ind1]
                accept[j] += 1
        accep_rate = accept / mcmciter
        aar = np.mean(accep_rate)
    _iterline(3, isim, [pstep+1, mcmciter, aar, phi[k+1], timestart])
    return path2update
