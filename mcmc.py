"""Module container for MCMC"""
import time, numpy as np
from tools import mvnrnd, mat_mul
from solver import solve
from data_tools import get_data

def _iterline(params, stage, isim, meta):
    pstep, mcmciter, aar, phi, timestart, k = meta
    elapsed = time.time() - timestart
    msg = 'k = %05d, isim = %05d, step = %05d, mcmciter = %05d,'
    msg = msg +' rate = %9.2e, phi = %9.2e, elapsed = %9.2es'
    msg = "\t\t[MCMC-%d]:" %stage + msg
    msg = msg % (k, isim, pstep, mcmciter, aar, phi, elapsed)
    if (k%8 == 0):
        print(msg)
    log_file = "%s/mcmc_%08d.log" % (params["mcmc_dir"],isim)
    with open(log_file, "a") as fout:
       fout.write(msg+"\n")


def mcmc1(params, pstep, isim, phi, k, signal, data_n, fx0):
    accept = np.zeros(params['N'])
    aar = 0
    mcmciter = 0
    terminatemcmc = False
    timestart = time.time()
    if params['C_is_eye']:
        vgn = data_n - signal
    else:
        vgn = data_n - mat_mul(params['C'], signal)
    vfn = signal - fx0


    if params["Y_CoefMatrix_is_eye"]:
        log_g = -0.5 * phi[k + 1] * params["sig_y"] ** (-2) * np.sum(vgn * vgn, axis = 0)
    else:
        log_g = np.diag(-0.5 * phi[k + 1] * mat_mul(vgn.T, mat_mul(params["R2_inv"], vgn)))

    if params["X_CoefMatrix_is_eye"]:
        log_f = -0.5 * params["sig_x"] ** (-2) * np.sum(vfn * vfn, axis=0)
    else:
        log_f = np.diag(-0.5 * mat_mul(vfn.T, mat_mul(params["R1_inv"], vfn)))

    while not terminatemcmc:
        mcmciter += 1
        cond = (mcmciter > params["MCMCnmin"]) &\
               (aar >= params["aar_min"]) &\
               (aar <= params["aar_max"])
        if cond:
            terminatemcmc = True
        if mcmciter >= params['MCMCnmax']:
            terminatemcmc = True
        covm = params['sig']
        if aar < params['aar_min']:
            covm = params['sig'] * (phi[k] + 2) / (phi[k] + 1) / (mcmciter ** 5 + 10)
        if aar > params['aar_max']:
            covm = params['sig'] * (phi[k] + 2) / (phi[k] + 1)
        noise_prop = np.random.normal(size=(params['dimx'],params['N']))
        x_per = signal + np.sqrt(covm) * noise_prop
        if params['C_is_eye']:
            vg_p = data_n - x_per
        else:
            vg_p = data_n - mat_mul(params['C'], x_per)
        vf_p = x_per - fx0
        if params['Y_CoefMatrix_is_eye']:
            log_gp = -0.5 * phi[k + 1] * params['sig_y'] ** (-2) * np.sum(vg_p * vg_p, axis=0)
        else:
            log_gp = np.diag(-0.5 * phi[k + 1] * mat_mul(vg_p.T, mat_mul(params["R2_inv"], vg_p)))
        if params['X_CoefMatrix_is_eye']:
            log_fp = -0.5 * params['sig_x'] ** (-2) * np.sum(vf_p * vf_p, axis=0)
        else:
            log_fp = np.diag(-0.5 * mat_mul(vf_p.T, mat_mul(params["R1_inv"], vf_p)))
        log_accep = log_fp + log_gp - log_f - log_g
        log_u = np.log(np.random.uniform(size=(params["N"])))
        for j in range(params["N"]):
            if log_u[j] < log_accep[j]:
                signal[:, j] = x_per[:, j]
                log_f[j] = log_fp[j]
                log_g[j] = log_gp[j]
                accept[j] += 1
        accep_rate = accept / mcmciter
        aar = np.mean(accep_rate)
    _iterline(params, 1, isim, [pstep+1, mcmciter, aar, phi[k+1], timestart, k+1])
    return signal


def mcmc2(params, pstep, isim, phi, k, path, signal):
    timestart = time.time()
    accept = np.zeros(params['N'])
    aar = 0 #average acceptance rate
    mcmciter = 0
    terminatemcmc = False

    path2update = np.zeros((params['dimx'], params['N'] * (pstep + 2)))
    path2update[:, 0:params['N'] * (pstep + 1)] = path[:, 0:params['N'] * (pstep + 1)]
    ind1 = params['N'] * (pstep + 1)
    ind2 = params['N'] * (pstep + 2)
    path2update[:, ind1:ind2] = signal

    lf_p = np.zeros((params['N'], pstep + 1))   # p stands for proposed
    lo_f = np.zeros((params['N'], pstep + 1))
    if pstep  >= params["t_freq"]:
        leng = np.sum((np.arange(1, pstep + 1, dtype=int) % params["t_freq"])==0)
        #np.arange starts from 1 but ends at pstep not pstep + 1
        lg_p = np.zeros((params['N'], leng))
        lo_g = np.zeros((params['N'], leng))


    for istep in range(pstep):
        ind1 = params['N'] * istep
        ind2 = params['N'] * (istep + 1)
        jnd1 = params['N'] * (istep + 1)
        jnd2 = params['N'] * (istep + 2)

        _, sol = solve(params['N'], path2update[:, ind1:ind2], params)
        vfn = path2update[:, jnd1:jnd2] - sol

        if params['X_CoefMatrix_is_eye']:
            lo_f[:, istep] = -0.5 * params['sig_x'] ** (-2) * np.sum(vfn * vfn, axis=0)
        else:
            lo_f[:, istep] = np.diag(-0.5 * mat_mul(vfn.T, mat_mul(params['R1_inv'], vfn)))


        if (istep + 1) % params["t_freq"] == 0:
            data = get_data(params, istep)

            if params['C_is_eye']:
                vgn = data.reshape(-1, 1) - path2update[:, jnd1:jnd2]
            else:
                vgn = data.reshape(-1, 1) - mat_mul(params['C'], path2update[:, jnd1:jnd2])

            ind = int((istep + 1)/params["t_freq"] - 1)
            if istep < pstep - 1:
                if params['Y_CoefMatrix_is_eye']:
                    lo_g[:, ind] = -0.5 * params['sig_y'] ** (-2) * np.sum(vgn * vgn, axis=0)
                else:
                    lo_g[:, ind] = np.diag(-0.5 * mat_mul(vgn.T, mat_mul(params['R2_inv'], vgn)))
            else:
                if params['Y_CoefMatrix_is_eye']:
                    lo_g[:, ind] = -0.5 * phi[(k + 1)] * params['sig_y'] ** (-2) * np.sum(vgn * vgn, axis=0)
                else:
                    lo_g[:, ind] = np.diag(-0.5 * phi[(k + 1)] * mat_mul(vgn.T, mat_mul(params['R2_inv'], vgn)))


    while not terminatemcmc:
        mcmciter += 1
        cond = (mcmciter > params["MCMCnmin"]) &\
               (aar >= params["aar_min"]) &\
               (aar <= params["aar_max"])
        if cond:
            terminatemcmc = True
        if mcmciter >= params["MCMCnmax"]:
            terminatemcmc = True

        covm = params['sig']
        if aar < params["aar_min"]:
            covm = params["sig"] * (phi[k] + 2) / (phi[k] + 1) / (mcmciter ** 6 + 15)
        elif aar > params["aar_max"]:
            covm = params["sig"] * (phi[k] + 2) / (phi[k] + 1) / np.log(mcmciter + 1)
        else:
            covm = params["sig"] * (phi[k] + 2) / (phi[k] + 1) / 2
        path_p = path2update + np.sqrt(covm) * \
                 np.random.normal(size=(params['dimx'], 
                    params['N'] * (pstep + 2)))

        for istep in range(pstep):
            ind1 = params['N'] * istep
            ind2 = params['N'] * (istep + 1)
            jnd1 = params['N'] * (istep + 1)
            jnd2 = params['N'] * (istep + 2)

            _, sol = solve(params['N'], path_p[:, ind1:ind2], params)
            vf_p = path_p[:, jnd1:jnd2] - sol
            if params['X_CoefMatrix_is_eye']:
                lf_p[:, istep] = -0.5 * params['sig_x'] ** (-2) * np.sum(vf_p * vf_p, axis=0)
            else:
                lf_p[:, istep] = np.diag(-0.5 * mat_mul(vf_p.T, mat_mul(params['R1_inv'], vf_p)))

            if (istep + 1) % params["t_freq"] == 0:
                data = get_data(params, istep)
                if params['C_is_eye']:
                    vg_p = data.reshape(-1, 1) - path_p[:, jnd1:jnd2]
                else:
                    vg_p = data.reshape(-1, 1) - mat_mul(params['C'], path_p[:, jnd1:jnd2])
                
                ind = int((istep + 1)/params["t_freq"] - 1)
                if istep < pstep - 1:
                    if params['Y_CoefMatrix_is_eye']:
                        lg_p[:, ind] = -0.5 * params['sig_y'] ** (-2) * np.sum(vg_p * vg_p, axis=0)
                    else:
                        lg_p[:, ind] = np.diag(-0.5 * mat_mul(vg_p.T, mat_mul(params['R2_inv'], vg_p)))
                else:
                    if params['Y_CoefMatrix_is_eye']:
                        lg_p[:, ind] = -0.5 * phi[(k + 1)] * params['sig_y'] ** (-2) * np.sum(vg_p * vg_p, axis=0)
                    else:
                        lg_p[:, ind] = np.diag(-0.5 * phi[(k + 1)] * mat_mul(vg_p.T, mat_mul(params['R2_inv'], vg_p)))

        log_accep = np.sum(lf_p, axis=1) + np.sum(lg_p, axis=1) - np.sum(lo_f, axis=1) - np.sum(lo_g, axis=1)
        log_u = np.log(np.random.uniform(size=(params['N'])))
        for j in range(params['N']):
            if log_u[j] < log_accep[j]:
                lo_g[j, :] = lg_p[j, :]
                lo_f[j, :] = lf_p[j, :]
                for istep in range(pstep):
                    ind1 = params['N'] * istep + j
                    path2update[:, ind1] = path_p[:, ind1]
                accept[j] += 1
        accep_rate = accept / mcmciter
        aar = np.mean(accep_rate)

    _iterline(params, 2, isim, [pstep+1, mcmciter, aar, phi[k+1], timestart, k+1])

    return path2update


def mcmc3(params, pstep, isim, path, signal, phi, k, fx0, x_f_n_ml_p1,
             p_f_n_ml_p1_inv, x_f_n_ml, p_f_n_ml_inv):
    timestart = time.time()
    accept = np.zeros(params['N'])
    aar = 0
    mcmciter = 0
    terminatemcmc = False

    path2update = np.zeros((params['dimx'], params['N'] * (params['L'] + 1)))
    ind1 = params['N'] * (pstep + 1 - params['L'])
    ind2 = params['N'] * (pstep + 1)
    path2update[:, 0:params['N'] * params['L']] = path[:, ind1:ind2]
    path2update[:, params['N'] * params['L']: params['N'] * (params['L'] + 1)] = signal

    lf_p = np.zeros((params['N'], params['L']))
    lo_f = np.zeros((params['N'], params['L']))
    leng = np.sum((np.arange(pstep + 1 - params["L"], pstep + 1, dtype=int) % params["t_freq"])==0)
    if leng >= 1:
        lg_p = np.zeros((params['N'], leng))
        lo_g = np.zeros((params['N'], leng))

    while not terminatemcmc:
        mcmciter += 1
        cond = (mcmciter > params["MCMCnmin"]) &\
               (aar >= params["aar_min"]) &\
               (aar <= params["aar_max"])
        if cond:
            terminatemcmc = True
        if mcmciter >= params["MCMCnmax"]:
            terminatemcmc = True

        covm = params["sig"] # sig is a number not a matrix
        if aar < params["aar_min"]:
            covm = params["sig"] * (phi[k] + 2) / (phi[k] + 1) / (mcmciter**6 + 10)
        elif aar > params["aar_max"]:
            covm = params["sig"] * (phi[k] + 2) / (phi[k] + 1) / (mcmciter**2 + 1)
        else:
            covm = params["sig"] * (phi[k] + 2) / (phi[k] + 1) / 2

        #path_p = mvnrnd(path2update, covm, coln=params["N"] * (params["L"] + 1))
        #assuming covm = number * identity
        path_p = path2update + np.sqrt(covm) * \
                 np.random.normal(size=(params['dimx'], params['N'] * (params['L'] + 1)))

        for istep in range(params['L']):
            ind1 = params['N'] * istep
            ind2 = params['N'] * (istep + 1)
            _, fix = solve(params['N'], path_p[:, ind1:ind2], params)
            jnd1 = params['N'] * (istep + 1)
            jnd2 = params['N'] * (istep + 2)
            vf_p = path_p[:, jnd1:jnd2] - fix
            if mcmciter == 1:
                _, fix = solve(params['N'], path2update[:, ind1:ind2], params)
                vfn = path2update[:, jnd1:jnd2] - fix

            if params['X_CoefMatrix_is_eye']:
                lf_p[:, istep] = -0.5 * params['sig_x'] ** (-2) * np.sum(vf_p * vf_p, axis=0)

                if mcmciter == 1:
                    lo_f[:, istep] = -0.5 * params['sig_x'] ** (-2) * np.sum(vfn * vfn, axis=0)

            else:
                lf_p[:, istep] = np.diag(-0.5 * mat_mul(vf_p.T, mat_mul(params['R1_inv'], vf_p)))

                if mcmciter == 1:
                    lo_f[:, istep] = np.diag(-0.5 * mat_mul(vfn.T, mat_mul(params['R1_inv'], vfn)))

            stepr = istep + pstep - params['L'] 
            ind = 0
            if (stepr + 1) % params["t_freq"] == 0:       
                data = get_data(params, stepr)
                if params['C_is_eye']:
                    vg_p = data.reshape(-1, 1) - path_p[:, ind1:ind2]
                else:
                    vg_p = data.reshape(-1, 1) - mat_mul(params['C'], path_p[:, ind1:ind2])

                if mcmciter == 1:
                    if params['C_is_eye']:
                        vgn = data.reshape(-1, 1) - path2update[:, ind1:ind2]
                    else:
                        vgn = data.reshape(-1, 1) - mat_mul(params['C'], path2update[:, ind1:ind2])

                if params['Y_CoefMatrix_is_eye']:
                    lg_p[:, ind] = -0.5 * params['sig_y'] ** (-2) * np.sum(vg_p * vg_p, axis=0)
                    if mcmciter == 1:
                        lo_g[:, ind] = -0.5 * params['sig_y'] ** (-2) * np.sum(vgn * vgn, axis=0)
                else:
                    lg_p[:, ind] = np.diag(-0.5 * mat_mul(vg_p.T, mat_mul(params['R2_inv'], vg_p)))
                    if mcmciter == 1:
                        lo_g[:, ind] = np.diag(-0.5 * mat_mul(vgn.T, mat_mul(params['R2_inv'], vgn)))
                ind += 1

        # calculate log(the ratio to the power phi[k+1]):
        ind1 = params['N']
        ind2 = 2 * params['N']
        jnd1 = params['N'] * params['L']
        jnd2 = params['N'] * (params['L'] + 1)

        if (pstep + 1) % params["t_freq"] == 0:
            data = get_data(params, pstep)
            if params['C_is_eye']:
                vg_pr = data.reshape(-1, 1) - path_p[:, jnd1:jnd2]
                if mcmciter == 1:
                    vg_r = data.reshape(-1, 1) - path2update[:, jnd1:jnd2]
            else:
                vg_pr = data.reshape(-1, 1) - mat_mul(params['C'], path_p[:, jnd1:jnd2])
                if mcmciter == 1:
                    vg_r = data.reshape(-1, 1) - mat_mul(params['C'], path2update[:, jnd1:jnd2])

            if params['Y_CoefMatrix_is_eye']:
                lg_pr = -0.5 * params['sig_y'] ** (-2) * np.sum(vg_pr * vg_pr, axis=0)
                if mcmciter == 1:
                    lg_r = -0.5 * params['sig_y'] ** (-2) * np.sum(vg_r * vg_r, axis=0)
            else:
                lg_pr = np.diag(-0.5 * mat_mul(vg_pr.T, mat_mul(params['R2_inv'], vg_pr)))
                if mcmciter == 1:
                    lg_r = np.diag(-0.5 * mat_mul(vg_r.T, mat_mul(params['R2_inv'], vg_r)))

        if pstep == params['L']:
            vec_mu1_p = path_p[:, 0:params['N']] - fx0
            if mcmciter == 1:
                vec_mu1 = path2update[:, 0:params['N']] - fx0
            if params['X_CoefMatrix_is_eye']:
                log_mu1_p = -0.5 * params['sig_x'] ** (-2) * np.sum(vec_mu1_p * vec_mu1_p, axis=0)
                if mcmciter == 1:
                    log_mu1 = -0.5 * params['sig_x'] ** (-2) * np.sum(vec_mu1 * vec_mu1, axis=0)
            else:
                log_mu1_p = np.diag(-0.5 * mat_mul(vec_mu1_p.T, mat_mul(params['R1_inv'], vec_mu1_p)))
                if mcmciter == 1:
                    log_mu1 = np.diag(-0.5 * params['sig_x'] ** (-2) * mat_mul(vec_mu1.T,
                                                             mat_mul(params['R1_inv'], vec_mu1)))
        else:
            vec_mu1_p = path_p[:, 0:params['N']] - x_f_n_ml
            log_mu1_p = np.diag(-0.5 * mat_mul(vec_mu1_p.T, mat_mul(p_f_n_ml_inv, vec_mu1_p)))

            if mcmciter == 1:
                vec_mu1 = path2update[:, 0:params['N']] - x_f_n_ml
                log_mu1 = np.diag(-0.5 * mat_mul(vec_mu1.T, mat_mul(p_f_n_ml_inv, vec_mu1))).copy()

        vec_mu2_p = path_p[:, ind1:ind2] - x_f_n_ml_p1
        log_mu2_p = np.diag(-0.5 * mat_mul(vec_mu2_p.T, mat_mul(p_f_n_ml_p1_inv,
                                                                 vec_mu2_p)))
        if mcmciter == 1:
            vec_mu2 = path2update[:, ind1:ind2] - x_f_n_ml_p1
            log_mu2 = np.diag(-0.5 * mat_mul(vec_mu2.T, mat_mul(p_f_n_ml_p1_inv, vec_mu2))).copy()

        log_ratio = phi[(k + 1)] * (lg_pr + log_mu2_p + lo_f[:, 0] \
                                     - lg_r - log_mu2 - lf_p[:, 0])
        log_accep = log_ratio + log_mu1_p \
                            + np.sum(lf_p, axis=1) \
                            + np.sum(lg_p, axis=1) - log_mu1 \
                            - np.sum(lo_f, axis=1) \
                            - np.sum(lo_g, axis=1)

        log_u = np.log(np.random.uniform(size=params["N"]))
        for j in range(params["N"]):
            if log_u[j] < log_accep[j]:
                lg_r[j] = lg_pr[j]
                log_mu1[j] = log_mu1_p[j]
                log_mu2[j] = log_mu2_p[j]
                lo_f[j, :] = lf_p[j, :]
                lo_g[j, :] = lg_p[j, :]
                for istep in range(params["L"]+1):
                    ind1 = params["N"] * istep + j
                    path2update[:, ind1] = path_p[:, ind1]
                accept[j] += 1
        accep_rate = accept / mcmciter
        aar = np.mean(accep_rate)
    _iterline(params, 3, isim, [pstep+1, mcmciter, aar, phi[k+1], timestart, k+1])
    return path2update
