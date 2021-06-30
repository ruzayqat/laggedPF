#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 17:39:12 2021

@author: Ruzayqat, Hamza
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from scipy import linalg as sLA
# from scipy.sparse import identity as Identity
from scipy.sparse import diags
import multiprocessing as MP
import datetime
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import gc
import time
from functools import partial

# from scipy import optimize

np.seterr(divide='ignore', invalid='ignore')


# %%###########################################################################
# functions
def mvnrnd(mean, cov, coln=1):
    """ multivariate_normal (faster than the numpy built-in function:
        np.random.multivariate_normal) """
    mndim = mean.ndim
    if (mndim == 1) & (coln == 1):
        result = mean + np.matmul(sLA.cholesky(cov),
                                  np.random.standard_normal(mean.size))
    elif (mndim == 1) & (coln > 1):
        result = mean.reshape(-1, 1) + np.matmul(sLA.cholesky(cov),
                                                 np.random.standard_normal(size=(mean.size, coln)))
    elif mndim > 1:
        result = mean + np.matmul(sLA.cholesky(cov),
                                  np.random.standard_normal(size=mean.shape))
    return result


#########################
def fwd_slash(A, B):
    """ equivalent to A/B in MATLAB. That is solve for x in: x * B = A
        This is equivalent to: B.T * x.T = A.T """
    return LA.solve(B.T, A.T).T  # accepts dense arrays only
    # return ssLA.spsolve(B.T, A.T).T #accepts sparse A,B in CSC or CSR format
    # return sLA.lstsq(B.T, A.T)[0].T #B & A should be dense arrays
    # return sLA.solve(B.T,A.T).T #accepts dense arrays only


#########################
def symmetric(A):
    return np.triu(A) + np.triu(A, 1).T


#########################
def swe_onestep(N, Uold):
    """ One step in time for the Solver of Shallow Water 
    Equations (Conservative form) using Finite Volume Scheme 
    Input:  1) Uold: 3d or 4d array. If it is 3d, its shape is (3,dim,dim)
              where U[0,:,:] = height, U[1,:,:] = height * velocity in x-axis
                    U[3,:,:] = height * velocity in y-axis
              If it is 4d array, its shape is (N,3,dim,dim), where N is the 
              number of particles.
            2) N: number of particles.
    Output: 1) U1: SWE solution using FV in one time step
            2) dt: timestep size
    """
    # Uold is a 3d or 4d array
    U1 = np.zeros(Uold.shape)
    for i in range(N):
        if N == 1:
            Uold0 = Uold
        else:
            Uold0 = Uold[i, :, :, :]
            # u = hu./h;
        uold = Uold0[1, :, :] / Uold0[0, :, :]
        # v = hv./h;
        vold = Uold0[2, :, :] / Uold0[0, :, :]
        # calculate lambda = |u| + sqrt(gh) used for finding flux
        lambdau = 0.5 * np.absolute(uold + uold[:, shiftm1]) \
                  + np.sqrt(g * 0.5 * (Uold0[0, :, :] + Uold0[0, :, shiftm1]))
        lambdav = 0.5 * np.absolute(vold + vold[shiftm1, :]) \
                  + np.sqrt(g * 0.5 * (Uold0[0, :, :] + Uold0[0, shiftm1, :]))
        lambdamax = LA.norm(np.concatenate((lambdau, lambdav)), np.inf)
        # calculate time stepsize
        dt = c * (dx / lambdamax)
        # calculate h*u*v
        huv = Uold0[1, :, :] * Uold0[2, :, :] / Uold0[0, :, :]
        ghh = half_g * Uold0[0, :, :] ** 2
        # calculate (hu,hu^2+gh^2/2,huv)
        ffu = np.stack((Uold0[1, :, :],
                        Uold0[1, :, :] ** 2 / Uold0[0, :, :] + ghh, huv))
        # calculate (hv,huv,hv^2+gh^2/2)
        ffv = np.stack((Uold0[2, :, :], huv,
                        Uold0[2, :, :] ** 2 / Uold0[0, :, :] + ghh))
        # calculate fluxes in x and y directions 
        fluxx = 0.5 * (ffu + ffu[:, :, shiftm1]) - \
                0.5 * (Uold0[:, :, shiftm1] - Uold0) * lambdau
        fluxy = 0.5 * (ffv + ffv[:, shiftm1, :]) - \
                0.5 * (Uold0[:, shiftm1, :] - Uold0) * lambdav
        # time step
        U = Uold0 - (dt / dx) * (fluxx - fluxx[:, :, shiftp1]) \
            - (dt / dy) * (fluxy - fluxy[:, shiftp1, :])
        # impose boundary conditions on h
        U[0, :, -1] = U[0, :, -2]
        U[0, :, 0] = U[0, :, 1]
        U[0, -1, :] = U[0, -2, :]
        U[0, 0, :] = U[0, 1, :]
        # impose boundary conditions on hu
        U[1, :, -1] = - U[1, :, -2]
        U[1, :, 0] = - U[1, :, 1]
        U[1, -1, :] = U[1, -2, :]
        U[1, 0, :] = U[1, 1, :]
        # impose boundary conditions on hv
        U[2, :, -1] = U[2, :, -2]
        U[2, :, 0] = U[2, :, 1]
        U[2, -1, :] = - U[2, -2, :]
        U[2, 0, :] = - U[2, 1, :]
        # set U1 = U
        if N == 1:
            U1 = U
        else:
            U1[i, :, :, :] = U
    return U1, dt


#########################
def OneOr2D_to_3dOr4d(N, v):
    """ converts 1d or 2d arrays to 3d or 4d array.
        Input: 1) v: 1d or 2d array. If it is 1d, its shape is (3*dim2). 
                    If it is 2d, its shape is (3*dim2, N)
               2) N: number of columns in v
        Output: U: If N = 1, then it is a 3d array of shape (3,dim,dim)
                   If N != 1, then it is a 4d array of shape (N,3,dim,dim)  
    """
    if N == 1:
        h = v[0:dim2].reshape(dim, dim)
        hu = v[dim2:2 * dim2].reshape(dim, dim)
        hv = v[2 * dim2:3 * dim2].reshape(dim, dim)
        U = np.stack((h, hu, hv))  # 3d array of shape = 3 x dim x dim
    else:
        U = np.zeros((N, 3, dim, dim))
        for i in range(N):
            h = v[0:dim2, i].reshape(dim, dim)
            hu = v[dim2:2 * dim2, i].reshape(dim, dim)
            hv = v[2 * dim2:3 * dim2, i].reshape(dim, dim)
            U[i, :, :, :] = np.stack((h, hu, hv))  # 4d array shape N x 3 x dim x dim
    return U


#########################
def ThreeOr4D_to_1dOr2d(N, U):
    """ converts 3d or 4d arrays to 1d or 2d arrays.
        Inputs: 1) U: a 3d or 4d array. If it is a 3d then its shape is 
                   (3,dim,dim). If it is 4d, its shape is (N,3,dim,dim)
                2) N: number of particles.
        Output: v: a 1d (if N=1) or 2d array (otherwise) of shape (3*dim2,N) 
    """
    if N == 1:
        v = np.zeros(3 * dim2)
        h = U[0, :, :]
        hu = U[1, :, :]
        hv = U[2, :, :]
        v[0:dim2] = h.reshape(dim2)
        v[dim2:2 * dim2] = hu.reshape(dim2)
        v[2 * dim2:3 * dim2] = hv.reshape(dim2)
    else:
        v = np.zeros((3 * dim2, N))
        for i in range(N):
            h = U[i, 0, :, :]
            hu = U[i, 1, :, :]
            hv = U[i, 2, :, :]
            v[0:dim2, i] = h.reshape(dim2)
            v[dim2:2 * dim2, i] = hu.reshape(dim2)
            v[2 * dim2:3 * dim2, i] = hv.reshape(dim2)
    return v


#########################
def EnKF(R2_sqrt, h):
    """ One Run of Ensemble Kalman Filter """
    seed = h + 100
    np.random.seed(seed)
    x_a = np.transpose([x_star] * M)  # dx x M
    E_loc = np.zeros((dimx, T + 1))
    E_loc[:, 0] = x_star
    t_simul = 0.0
    Ido = np.eye(dimo)
    if Y_CoefMatrix_is_eye:
        R2 = np.eye(dimo) * sig_y ** 2
    else:
        R2 = np.matmul(R2_sqrt, R2_sqrt)
    for n in range(T):
        t_simul, x_f = sampleFromf(M, M, x_a, t_simul)[0:2]
        m = np.sum(x_f, axis=1).reshape(-1, 1) / M  # size =dx x 1
        diff = x_f - m
        if C_is_eye:
            temp1 = diff.T  # (M * dx) * (dx * dy) = M * dy
            temp2 = np.matmul(diff, temp1) / M  # (dx * M) * (M* dy) = dx * dy
            temp3 = fwd_slash(Ido, temp2 + R2)  # this is faster than sLA.inv
            temp4 = np.matmul(diff.T, temp3)  # (M * dx) * (dx * dy) = M * dy
            K = np.matmul(diff, temp4) / M  # (dx * M) * (M* dy) = dx * dy
            dV = mvnrnd(np.zeros(dimo), Ido, coln=M)  # size = dy x M
            if Y_CoefMatrix_is_eye:
                temp7 = Y[:, n].reshape(-1, 1) - x_f - sig_y * dV  # dy * M
            else:
                temp7 = Y[:, n].reshape(-1, 1) - x_f - np.matmul(R2_sqrt, dV)  # dy * M
        else:
            temp1 = np.matmul(diff.T, C.T)  # (M * dx) * (dx * dy) = M * dy
            temp2 = np.matmul(diff, temp1)  # (dx * M) * (M* dy) = dx * dy
            temp3 = np.matmul(C, temp2) / M  # (dy * dx) * (dx * dy) = dy * dy
            temp4 = fwd_slash(Ido, temp3 + R2)  # this is faster than sLA.inv
            temp5 = np.matmul(C.T, temp4)  # dx * dy
            temp6 = np.matmul(diff.T, temp5)  # (M * dx) * (dx * dy) = M * dy
            K = np.matmul(diff, temp6) / M  # (dx * M) * (M* dy) = dx * dy
            dV = mvnrnd(np.zeros(dimo), Ido, coln=M)  # size = dy x M
            if Y_CoefMatrix_is_eye:
                temp7 = Y[:, n].reshape(-1, 1) - np.matmul(C, x_f) - sig_y * dV  # dy * M
            else:
                temp7 = Y[:, n].reshape(-1, 1) - np.matmul(C, x_f) \
                        - np.matmul(R2_sqrt, dV)  # dy * M
        x_a = x_f + np.matmul(K, temp7)  # dx * M
        E_loc[:, n + 1] = np.sum(x_a, axis=1) / M
    return t_simul, E_loc


#########################
def EnKF1(R2_sqrt, h):
    """ One Run of Ensemble Kalman Filter to return X_f and P_f to be used in
        LaggedPF.
    Input:  1) R2_sqrt: Noise coef. matrix of the observations
            2) h: an integer.
    Output: 1) X_f: a 2d array that each column of it contains the mean of the
                    predictor, that is the mean of the density p(x_n|y_{0:n-1})
                    for n = 1,...,T
            2) P_f: a 3d array that each 2d sub-array of it is the covariance 
                    of the predictor density
    """
    seed = h + 100
    np.random.seed(seed)
    x_a = np.transpose([x_star] * M)  # dx x M
    E_loc = np.zeros((dimx, T + 1))
    E_loc[:, 0] = x_star
    X_f = np.zeros((dimx, T))  # Will save the mean of the predictor in it
    # X_f[:,0] = is the mean of the predictor p(x_{1}|y_{0})
    # X_f[:,1] = is the mean of the predictor p(x_{2}|y_{0:1})
    P_f = np.zeros((T, dimx, dimx))  # will save the cov of the predictor in it
    Ido = np.eye(dimo)
    t_simul = 0.0
    if Y_CoefMatrix_is_eye:
        R2 = np.eye(dimo) * sig_y ** 2
    else:
        R2 = np.matmul(R2_sqrt, R2_sqrt)
    for n in range(T):
        t_simul, x_f = sampleFromf(M, M, x_a, t_simul)[0:2]
        X_f[:, n] = np.sum(x_f, axis=1) / M  # size =dx x 1
        diff = x_f - X_f[:, n].reshape(-1, 1)
        P_f[n, :, :] = np.matmul(diff, diff.T) / (M - 1)
        if C_is_eye:
            temp1 = diff.T  # (M * dx) * (dx * dy) = M * dy
            temp2 = np.matmul(diff, temp1) / M  # (dx * M) * (M* dy) = dx * dy
            temp3 = fwd_slash(Ido, temp2 + R2)  # this is faster than sLA.inv
            temp4 = np.matmul(diff.T, temp3)  # (M * dx) * (dx * dy) = M * dy
            K = np.matmul(diff, temp4) / M  # (dx * M) * (M* dy) = dx * dy
            dV = mvnrnd(np.zeros((dimo)), Ido, coln=M)  # size = dy x M
            if Y_CoefMatrix_is_eye:
                temp7 = Y[:, n].reshape(-1, 1) - x_f - sig_y * dV  # dy * M
            else:
                temp7 = Y[:, n].reshape(-1, 1) - x_f - np.matmul(R2_sqrt, dV)  # dy * M
        else:
            temp1 = np.matmul(diff.T, C.T)  # (M * dx) * (dx * dy) = M * dy
            temp2 = np.matmul(diff, temp1)  # (dx * M) * (M* dy) = dx * dy
            temp3 = np.matmul(C, temp2) / M  # (dy * dx) * (dx * dy) = dy * dy
            temp4 = fwd_slash(Ido, temp3 + R2)  # this is faster than sLA.inv
            temp5 = np.matmul(C.T, temp4)  # dx * dy
            temp6 = np.matmul(diff.T, temp5)  # (M * dx) * (dx * dy) = M * dy
            K = np.matmul(diff, temp6) / M  # (dx * M) * (M* dy) = dx * dy
            dV = mvnrnd(np.zeros((dimo)), Ido, coln=M)  # size = dy x M
            if Y_CoefMatrix_is_eye:
                temp7 = Y[:, n].reshape(-1, 1) - np.matmul(C, x_f) - sig_y * dV  # dy * M
            else:
                temp7 = Y[:, n].reshape(-1, 1) - np.matmul(C, x_f) \
                        - np.matmul(R2_sqrt, dV)  # dy * M
        x_a = x_f + np.matmul(K, temp7)  # dx * M
    return X_f, P_f


#########################
def EtKF(R2_sqrt_inv, h):
    """ One Run of Ensemble Transform Kalman Filter """
    seed = h + 100
    np.random.seed(seed)
    x_a = np.transpose([x_star] * M)  # dx x M
    E_loc = np.zeros((dimx, T + 1))
    E_loc[:, 0] = x_star
    t_simul = 0.0
    Ido = np.eye(dimo)
    for n in range(T):
        t_simul, x_f = sampleFromf(M, M, x_a, t_simul)[0:2]
        m_f = np.sum(x_f, axis=1).reshape(-1, 1) / M  # size =dx x 1
        S_f = 1 / np.sqrt(M - 1) * (x_f - m_f)  # dx * M
        if C_is_eye:
            Y_hat = x_f
        else:
            Y_hat = np.matmul(C, x_f)  # dy * M
        my = np.sum(Y_hat, axis=1).reshape(-1, 1) / M  # dy * 1
        if Y_CoefMatrix_is_eye:
            Fk = 1 / np.sqrt(M - 1) * (Y_hat - my).T / sig_y  # M * dy
        else:
            Fk = 1 / np.sqrt(M - 1) * np.matmul((Y_hat - my).T, R2_sqrt_inv)
        Tk = np.matmul(Fk.T, Fk) + Ido  # dy * dy
        K = np.matmul(S_f, fwd_slash(Fk, Tk))  # (dx * M) * (M * dy)
        if Y_CoefMatrix_is_eye:
            m_a = m_f + np.matmul(K, ((Y[:, n].reshape(-1, 1) - my) / sig_y))
        else:
            m_a = m_f + np.matmul(K, np.matmul(R2_sqrt_inv,
                                               (Y[:, n].reshape(-1, 1) - my)))
            # the above step was corrected it is wrong in the manual:
        # "Data assimilation toolbox for Matlab"
        Un, Dn = sLA.svd(np.matmul(Fk, Fk.T))[0:2]  # return the Un and Dn only -
        # no need for Vn
        Dn = diags(Dn)  # convert Dn from 1d array to sparse M x M array
        S_a = np.matmul(S_f, fwd_slash(Un, np.sqrt(Dn + np.identity(M))))  # dx * M
        x_a = np.sqrt(M - 1) * S_a + m_a

        E_loc[:, n + 1] = np.sum(x_a, axis=1) / M
    return t_simul, E_loc


#########################
def EtKF_sqrt(R2, h):
    """ One Run of Ensemble Transform Kalman Filter with Square Root of invTTt
    (e.g. Hunt et al., 2007) see "State-of-the-art stochastic data 
    assimilation methods for high-dimensional non-Gaussian problems """
    seed = h + 100
    np.random.seed(seed)
    x_a = np.transpose([x_star] * M)  # dimx x M
    E_loc = np.zeros((dimx, T + 1))
    E_loc[:, 0] = x_star
    t_simul = 0.0
    for n in range(T):
        t_simul, x_f = sampleFromf(M, M, x_a, t_simul)[0:2]
        m_f = np.sum(x_f, axis=1).reshape(-1, 1) / M  # size =dimx x 1
        Xfp = x_f - m_f  # dimx x M
        if C_is_eye:
            Cxf = x_f
        else:
            Cxf = np.matmul(C, x_f)  # dimo * M
        my = np.sum(Cxf, axis=1).reshape(-1, 1) / M  # dimo x 1
        S = Cxf - my  # dimo * M
        invR2_S = sLA.lstsq(R2, S)[0]  # dimo * M  = R2\S in MATLAB
        invTTt = symmetric((M - 1) * np.identity(M) + np.matmul(S.T, invR2_S))
        # M * M ...see symetric function
        Sigma, Vt = LA.eig(invTTt)  # returns eigenvalues and eignevectors
        Sigma = np.diag(Sigma.real)  # M x M, it might return complex numbers
        Vt = Vt.real  # it might return complex numbers with the imiginary part
        # being very small = e-10 or e-20
        Tm = np.matmul(Vt, sLA.lstsq(np.sqrt(Sigma), Vt.T)[0])  # M x 1
        Xap = np.sqrt(M - 1) * np.matmul(Xfp, Tm)  # dimx x 1
        temp = np.matmul(invR2_S.T, Y[:, n].reshape(-1, 1) - my)
        temp = sLA.lstsq(Sigma, np.matmul(Vt.T, temp))[0]
        m_a = m_f + np.matmul(Xfp, np.matmul(Vt, temp))  # dimx x 1
        x_a = Xap + m_a
        E_loc[:, n + 1] = np.sum(x_a, axis=1) / M
    return t_simul, E_loc


#########################
def LaggedPF(X_f, P_f, h):
    """ One Run of Lagged Particle Filter """
    seed = h + 100
    np.random.seed(seed)
    ESS_saved = []
    E = np.zeros((dimx, T + 1))
    path = np.zeros((dimx, N * (T + 1)))
    t_simul = 0.0
    for n in range(T):
        if n == 0:
            """ ----------------------- n = 0 ------------------------ """
            phi = []
            phi.append(phi1)
            i1 = 0
            i2 = N
            path[:, i1:i2] = np.transpose([x_star] * N)
            E[:, 0] = x_star
            # Sample the new state X_{1} from f(x_1|X_{0})
            t_simul, X, fx0 = sampleFromf(N, 1, x_star, t_simul)
            fx0 = fx0.reshape(-1, 1)
            ##### initial step in SMC sampler ####
            # calculate the weight w_1
            Yn = Y[:, 0].reshape(-1, 1)
            if Y_CoefMatrix_is_eye:
                lw, We, We0, sumWe0 = weight1(Yn, phi[0], X, np.zeros(N))
            else:
                pass  # need to modify to include R2_inv:
                # lw, We, We0, sumWe0 = weight1(Y1, phi[0],X,np.zeros(N),R2_inv)
            # resample if necessary:
            ESS = calcESS(We0, sumWe0)
            ESS_saved.append(ESS)
            X, lw_old, We = resample1(ESS, X, lw, We)[0:3]
            terminate = False
            k = 0
            ##### Later steps in SMC Sampler ####
            while not terminate:
                # (ESS - ESS_threshold) as a function of delta:
                funcOfDelta = lambda delta: FuncOfDelta1(Yn, X, delta)
                """ Note: Sometimes this function "funcOfDelta" is always 
                positive on [0,1], i.e. it does not have a zero """
                delta, converged = bisection(funcOfDelta, 0., 1., 1e-5, 1e-1)
                #print("delta = ", delta)
                if (not converged) & (k == 0):
                    delta = 0.0001
                if (not converged) & (k > 0):
                    delta = 1.2 * (phi[k] - phi[k - 1])
                if (delta <= (1. - phi[k])) & (delta >= 0.):
                    phi.append(phi[k] + delta)
                elif delta > (1. - phi[k]):
                    # print('delta = %.5f\n', delta)
                    delta = 1. - phi[k]
                    phi.append(1)
                    terminate = True
                if phi[k + 1] == 1:
                    terminate = True
                if Y_CoefMatrix_is_eye:
                    lw, We, We0, sumWe0 = weight1(Yn, delta, X, lw_old)
                else:
                    pass  # need to modify to pass R2_inv:
                    # lw,We,We0,sumWe0 = weight1(Y1,delta,X,np.zeros(N),R2_inv)
                # resample if necessary:
                ESS = calcESS(We0, sumWe0)
                ESS_saved.append(ESS)
                X, lw_old, We, resampled = resample1(ESS, X, lw, We)
                ##### MCMC step - random walk ########
                if X_CoefMatrix_is_eye & Y_CoefMatrix_is_eye:
                    X = MCMC1(n,h, phi, k, X, Yn, fx0)
                else:
                    pass  # Need to modify here to include R1_inv and R2_inv:
                    # X = MCMC1(n,h,phi,k,X,Yn,fx0, R1_inv, R2_inv)
                k += 1
            print('h = %d, p(%d) = %d \n' % (h,n+1, k))
            if not resampled:
                ESS = N  # to force the resample function to do resampling
                X = resample1(ESS, X, lw, We)[0]
            i1 = N * (n + 1)
            i2 = N * (n + 2)
            path[:, i1:i2] = X
        elif (n > 0) & (n < L):
            """ ------------------ n >= 1 and < L -------------------"""
            phi = []
            phi.append(phi1)
            # sample from f(x_n|X_{n-1})
            t_simul, X = sampleFromf(N, N, X, t_simul)[0:2]
            ##### initial step in SMC sampler ####
            # calculate the weight w_1
            Yn = Y[:, n].reshape(-1, 1)
            if Y_CoefMatrix_is_eye:
                lw, We, We0, sumWe0 = weight1(Yn, phi[0], X, np.zeros(N))
            else:
                pass  # need to modify to include R2_inv:
                # lw, We, We0, sumWe0 = weight1(Y1, phi[0],X,np.zeros(N),R2_inv)
            # resample if necessary:
            ESS = calcESS(We0, sumWe0)
            ESS_saved.append(ESS)
            X, lw_old, We = resample1(ESS, X, lw, We)[0:3]
            terminate = False
            k = 0
            ##### Later steps in SMC Sampler ####
            while not terminate:
                # (ESS - ESS_threshold) as a function of delta:
                funcOfDelta = lambda delta: FuncOfDelta1(Yn, X, delta)
                delta, converged = bisection(funcOfDelta, 0., 1., 1e-4, 1e-1)
                """ Note: Sometimes this function "funcOfDelta" is always 
                positive on [0,1], i.e. it does not have a zero """
                # sol = optimize.root_scalar(funcOfDelta, bracket=[0, 1],
                #                             method='brentq')
                # delta = sol.root
                # converged = sol.converged
                #print("delta = ", delta)
                if (not converged) & (k == 0):
                    delta = 0.0001
                if (not converged) & (k > 0):
                    delta = 1.2 * (phi[k] - phi[k - 1])
                if (delta <= (1 - phi[k])) & (delta >= 0):
                    phi.append(phi[k] + delta)
                elif delta > (1 - phi[k]):
                    # print('delta = %.5f\n', delta)
                    delta = 1 - phi[k]
                    phi.append(1)
                    terminate = True
                if phi[k + 1] == 1.:
                    terminate = True
                if Y_CoefMatrix_is_eye:
                    lw, We, We0, sumWe0 = weight1(Yn, delta, X, lw_old)
                else:
                    pass  # need to modify to pass R2_inv:
                    # lw,We,We0,sumWe0 = weight1(Y1,delta,X,np.zeros(N),R2_inv)
                # resample if necessary:
                ESS = calcESS(We0, sumWe0)
                ESS_saved.append(ESS)
                X, lw_old, We, resampled = resample1(ESS, X, lw, We)
                ##### MCMC step - random walk ########
                if X_CoefMatrix_is_eye & Y_CoefMatrix_is_eye:
                    pathToUpdate = MCMC2(n, h, phi, k, path, X, Y)
                else:
                    pass  # Need to modify here to include R1_inv and R2_inv:
                    # pathToUpdate =MCMC2(n,h,phi,k,path,X,Yn,R1_inv,R2_inv)
                path[:, 0:N * (n + 2)] = pathToUpdate
                i1 = N * (n + 1)
                i2 = N * (n + 2)
                X = pathToUpdate[:, i1:i2]
                k += 1
            # end while
            print('h = %d, p(%d) = %d \n' % (h, n + 1, k))
            if not resampled:
                path = resample2(n, path, We)
            i1 = N * (n + 1)
            i2 = N * (n + 2)
            X = path[:, i1:i2]
        else:
            """ ---------------------- n >= L -----------------------"""
            phi = []
            phi.append(phi1)
            # sample from f(x_n|X_{n-1})
            t_simul, X = sampleFromf(N, N, X, t_simul)[0:2]
            # In the following, X_f, P_f are calculated by calling the function
            # EnKF1 outside this function.
            X_f_n_mL_p1 = X_f[:, n - L + 1].reshape(-1, 1)# mean of the predictor distrib.
            # P(X_{n-L+1}|y_{0:n-L}) when n = L, we have
            # X_f[:,1] mean of the predictor p(x_{2}|y_{0:1})
            # transiting from x1 to x2.
            P_f_n_mL_p1_inv = fwd_slash(Idx, P_f[n - L + 1, :, :])
            # when n = L, have inv(P_f[1,:,:]
            ldet_P_f_n_mL_p1 = logdet(P_f[n - L + 1, :, :])
            X_f_n_mL = X_f[:, n - L].reshape(-1, 1)
            P_f_n_mL_inv = fwd_slash(Idx, P_f[n - L, :, :])
            # when n = L, have inv(P_f[0,:,:])
            ##### initial step in SMC sampler ####
            i1 = N * (n - L + 1)
            i2 = N * (n - L + 2)
            U = OneOr2D_to_3dOr4d(N, path[:, i1:i2])
            U, dt = swe_onestep(N, U)
            fXN = ThreeOr4D_to_1dOr2d(N, U).real
            # calculate the weight w_1
            Yn = Y[:, n].reshape(-1, 1)
            if X_CoefMatrix_is_eye & Y_CoefMatrix_is_eye:
                lw, We, We0, sumWe0 = weight2(n, Yn, phi[0], path, X,
                                             X_f_n_mL_p1, P_f_n_mL_p1_inv,
                                        ldet_P_f_n_mL_p1, np.zeros(N), fXN)
            else:
                pass  # need to modify here to include R1_inv, R2_inv:
                # lw, We, We0, sumWe0 = weight2(n,Yn, phi[0], path, X,
                #          X_f_n_mL_p1, P_f_n_mL_p1_inv,ldet_P_f_n_mL_p1,
                #          np.zeros(N),fXN, R1_inv, R2_inv)
            ESS = calcESS(We0, sumWe0)
            ESS_saved.append(ESS)
            X, lw_old, We = resample1(ESS, X, lw, We)[0:3]
            terminate = False
            k = 0
            ##### Later steps in SMC Sampler ####
            while not terminate:
                # (ESS - ESS_threshold) as a function of delta:
                funcOfDelta = lambda delta: FuncOfDelta2(n, Yn, X, path,
                                                X_f_n_mL_p1, P_f_n_mL_p1_inv,
                                                ldet_P_f_n_mL_p1, fXN, delta)
                delta, converged = bisection(funcOfDelta, 0., 1., 1e-4, 1e-1)
                """ Note: Sometimes this "funcOfDelta" is always 
                    positive on [0,1], i.e. it does not have a zero """
                #print("delta = ", delta)
                if (not converged) & (k == 0):
                    delta = 0.0001
                if (not converged) & (k > 0):
                    delta = 1.2 * (phi[k] - phi[k - 1])
                if (delta <= (1. - phi[k])) & (delta >= 0.):
                    phi.append(phi[k] + delta)
                elif delta > (1. - phi[k]):
                    # print('delta = %.5f\n', delta)
                    delta = 1. - phi[k]
                    phi.append(1)
                    terminate = True
                if phi[k + 1] == 1.:
                    terminate = True
                if X_CoefMatrix_is_eye & Y_CoefMatrix_is_eye:
                    lw, We, We0, sumWe0 = weight2(n, Yn, delta, path, X,
                                                  X_f_n_mL_p1, P_f_n_mL_p1_inv,
                                                  ldet_P_f_n_mL_p1, lw_old, fXN)
                else:
                    pass  # need to modify here to include R1_inv, R2_inv:
                    # lw, We, We0, sumWe0, fXN = weight2(n,Yn, delta, path, X,
                    #                X_f_n_mL_p1, P_f_n_mL_p1_inv,
                    #                ldet_P_f_n_mL_p1, lw_old, R1_inv, R2_inv)
                # resample if necessary:
                ESS = calcESS(We0, sumWe0)
                ESS_saved.append(ESS)
                X, lw_old, We, resampled = resample1(ESS, X, lw, We)
                ##### MCMC step - random walk ########
                if X_CoefMatrix_is_eye & Y_CoefMatrix_is_eye:
                    pathToUpdate = MCMC3(n, h, path, X, phi, k, fx0, Y,
                                         X_f_n_mL_p1, P_f_n_mL_p1_inv,
                                         X_f_n_mL, P_f_n_mL_inv)
                else:
                    pass  # Need to modify here to include R1_inv and R2_inv:
                    # X = MCMC2(n,h,phi,k,path,X,Yn, R1_inv, R2_inv)
                i1 = N * (n - L + 1)
                i2 = N * (n + 2)
                path[:, i1:i2] = pathToUpdate
                i1 = N * L
                i2 = N * (L + 1)
                X = pathToUpdate[:, i1:i2]
                k += 1
                if not terminate:
                    i1 = N * (n - L + 1)
                    i2 = N * (n - L + 2)
                    U = OneOr2D_to_3dOr4d(N, path[:, i1:i2])
                    U, dt = swe_onestep(N, U)
                    fXN = ThreeOr4D_to_1dOr2d(N, U).real
            # end while
            print('h = %d, p(%d) = %d \n' % (h, n + 1, k))
            if not resampled:
                path = resample2(n, path, We)
            i1 = N * (n + 1)
            i2 = N * (n + 2)
            X = path[:, i1:i2]
        # end if n == 0
    return t_simul, ESS_saved, path


def logdet(A, chol=True):
    """ LOGDET Computation of logarithm of determinant of a matrix

      v = logdet(A, chol = False)
          computes the logarithm of determinant of A. 

      v = logdet(A)
          If A is positive definite, you can tell the function 
          to use Cholesky factorization to accomplish the task, which is 
          substantially more efficient for positive definite matrix. 
   
      Copyright 2008, Dahua Lin, MIT
      Email: dhlin@mit.edu
      Edited by Hamza Ruzayqat (From MATLAB to PYTHON)
      This file can be freely modified or distributed for any kind of 
      purposes.
      """
    if chol:
        v = 2 * np.sum(np.log(np.diag(sLA.cholesky(A))))
    else:
        P, L, U = sLA.lu(A)
        du = np.diag(U)
        c = LA.det(P) * np.prod(np.sign(du))
        v = np.log(c) + np.sum(np.log(np.abs(du)))
    return v


###########################
def sampleFromf(N, M, X, t_simul):
    """ Sample from f(x_n|x_{n-1}) """
    if (M != 1) & (M != N):
        print('Error: M must be either equal to 1 or N')
        return ()
    U = OneOr2D_to_3dOr4d(M, X)
    U, dt = swe_onestep(M, U)
    t_simul += dt
    X = ThreeOr4D_to_1dOr2d(M, U)
    X = X.real
    if M == 1:
        fx0 = X
        X = np.transpose([fx0] * N)
    else:
        fx0 = []
    dW = mvnrnd(np.zeros((dimx)), Idx, coln=N)  # size = dx x N
    X = X + sig_x * dW
    return t_simul, X, fx0


###########################    
def MCMC1(n,h, phi, k, X, Yn, fx0, R1_inv=None, R2_inv=None):
    ac = np.zeros(N)
    aar = 0 #average acceptance rate
    MCMCiter = 0
    terminateMCMC = False
    while not terminateMCMC:
        MCMCiter += 1
        if (MCMCiter > MCMCnmin) & (aar >= aar_min) & (aar <= aar_max):
            terminateMCMC = True
        if MCMCiter >= MCMCnmax:
            terminateMCMC = True
        covm = sig
        if aar < aar_min:
            covm = sig * (phi[k] + 2) / (phi[k] + 1) / (MCMCiter ** 5 + 10)
        if aar > aar_max:
            covm = sig * (phi[k] + 2) / (phi[k] + 1)
        Xp = mvnrnd(X, covm, coln=N)
        if C_is_eye:
            vg_p = Yn - Xp  # p stands for proposed
            vg = Yn - X
        else:
            vg_p = Yn - np.matmul(C, Xp)
            vg = Yn - np.matmul(C, X)
        vf_p = Xp - fx0
        vf = X - fx0
        if Y_CoefMatrix_is_eye:
            temp1 = -0.5 * phi[k + 1] *sig_y ** (-2) * np.matmul(vg_p.T, vg_p) \
                    + 0.5 * phi[k + 1] * sig_y ** (-2) * np.matmul(vg.T, vg)
        else:
            temp1 = -0.5 * phi[k + 1] * np.matmul(vg_p.T,
                        np.matmul(R2_inv, vg_p)) + 0.5 * phi[k + 1] \
                    * np.matmul(vg.T, np.matmul(R2_inv, vg))

        if X_CoefMatrix_is_eye:
            temp2 = -0.5 * sig_x ** (-2) * np.matmul(vf_p.T, vf_p) \
                    + 0.5 * sig_x ** (-2) * np.matmul(vf.T, vf)
        else:
            temp2 = -0.5 * np.matmul(vf_p.T, np.matmul(R1_inv, vf_p)) \
                    + 0.5 * np.matmul(vf.T, np.matmul(R1_inv, vf))

        log_accep = np.minimum(np.zeros(N), np.diag(temp1 + temp2))
        log_U = np.log(np.random.uniform(size=N))
        for j in range(N):
            if log_U[j] < log_accep[j]:
                X[:, j] = Xp[:, j]
                ac[j] += 1
        accep_rate = ac / MCMCiter
        aar = np.mean(accep_rate)
    print('h = %d, time_step n = %d, MCMCsteps = %d,'
          ' accep_rate_avg = %.4f, phi = %.5f\n' %
          (h,n+1, MCMCiter, aar, phi[k + 1]))
    return X


###########################
def MCMC2(n, h, phi, k, path, X, Y, R1_inv=None, R2_inv=None):
    ac = np.zeros(N)
    aar = 0 #average acceptance rate
    MCMCiter = 0
    terminateMCMC = False

    pathToUpdate = np.zeros((dimx, N * (n + 2)))
    pathToUpdate[:, 0:N * (n + 1)] = path[:, 0:N * (n + 1)]
    i1 = N * (n + 1)
    i2 = N * (n + 2)
    pathToUpdate[:, i1:i2] = X

    lf_p = np.zeros((N, n + 1))  # p stands for proposed
    lg_p = np.zeros((N, n + 1))
    lf = np.zeros((N, n + 1))
    lg = np.zeros((N, n + 1))
    while not terminateMCMC:
        MCMCiter += 1
        if (MCMCiter > MCMCnmin) & (aar >= aar_min) & (aar <= aar_max):
            terminateMCMC = True
        if MCMCiter >= MCMCnmax:
            terminateMCMC = True

        covm = sig
        if aar < aar_min:
            covm = sig * (phi[k] + 2) / (phi[k] + 1) / (MCMCiter ** 6 + 15)
        elif aar > aar_max:
            covm = sig * (phi[k] + 2) / (phi[k] + 1) / np.log(MCMCiter + 1)
        else:
            covm = sig * (phi[k] + 2) / (phi[k] + 1) / 2
        path_p = mvnrnd(pathToUpdate, covm, coln=N * (n + 2))

        for v in range(n):
            i1 = N * v
            i2 = N * (v + 1)
            j1 = N * (v + 1)
            j2 = N * (v + 2)
            U = OneOr2D_to_3dOr4d(N, path_p[:, i1:i2])
            U, dt = swe_onestep(N, U)
            f = ThreeOr4D_to_1dOr2d(N, U).real
            vf_p = path_p[:, j1:j2] - f
            U = OneOr2D_to_3dOr4d(N, pathToUpdate[:, i1:i2])
            U, dt = swe_onestep(N, U)
            f = ThreeOr4D_to_1dOr2d(N, U).real
            vf = pathToUpdate[:, j1:j2] - f
            if C_is_eye:
                vg_p = Y[:, v].reshape(-1, 1) - path_p[:, j1:j2]
                vg = Y[:, v].reshape(-1, 1) - pathToUpdate[:, j1:j2]
            else:
                vg_p = Y[:, v].reshape(-1, 1) - np.matmul(C, path_p[:, j1:j2])
                vg = Y[:, v].reshape(-1, 1) - np.matmul(C,
                                                        pathToUpdate[:, j1:j2])
            if X_CoefMatrix_is_eye:
                lf_p[:, v] = np.diag(-0.5 * sig_x ** (-2) * \
                                        np.matmul(vf_p.T, vf_p))
                lf[:, v] = np.diag(-0.5 * sig_x ** (-2) * \
                                      np.matmul(vf.T, vf))
            else:
                lf_p[:, v] = np.diag(-0.5 * np.matmul(vf_p.T,
                                                np.matmul(R1_inv, vf_p)))
                lf[:, v] = np.diag(-0.5 * np.matmul(vf.T,
                                                np.matmul(R1_inv, vf)))
            if v < n - 1:
                if Y_CoefMatrix_is_eye:
                    lg_p[:, v] = np.diag(-0.5 * sig_y ** (-2) \
                                            * np.matmul(vg_p.T, vg_p))
                    lg[:, v] = np.diag(-0.5 * sig_y ** (-2) \
                                          * np.matmul(vg.T, vg))
                else:
                    lg_p[:, v] = np.diag(-0.5 * np.matmul(vg_p.T,
                                                np.matmul(R2_inv, vg_p)))
                    lg[:, v] = np.diag(-0.5 * np.matmul(vg.T,
                                                np.matmul(R2_inv, vg)))
            else:
                if Y_CoefMatrix_is_eye:
                    lg_p[:, v] = np.diag(-0.5 * phi[k + 1] * sig_y ** (-2) \
                                            * np.matmul(vg_p.T, vg_p))
                    lg[:, v] = np.diag(-0.5 * phi[k + 1] * sig_y ** (-2) \
                                          * np.matmul(vg.T, vg))
                else:
                    lg_p[:, v] = np.diag(-0.5 * phi[k + 1] \
                                            * np.matmul(vg_p.T,
                                                    np.matmul(R2_inv, vg_p)))
                    lg[:, v] = np.diag(-0.5 * phi[k + 1] \
                                          * np.matmul(vg.T,
                                                      np.matmul(R2_inv, vg)))

        log_accep = np.minimum(np.zeros(N), np.sum(lf_p, axis=1) \
                               + np.sum(lg_p, axis=1) - np.sum(lf, axis=1) \
                               - np.sum(lg, axis=1))
        log_U = np.log(np.random.uniform(size=N))
        for j in range(N):
            if log_U[j] < log_accep[j]:
                for v in range(n):
                    i1 = N * v + j
                    pathToUpdate[:, i1] = path_p[:, i1]
                ac[j] += 1
        accep_rate = ac / MCMCiter
        aar = np.mean(accep_rate)
    print('h = %d, time_step n = %d, MCMCsteps = %d,'
          ' accep_rate_avg = %.4f, phi = %.5f\n' %
          (h,n+1, MCMCiter, aar, phi[k + 1]))
    return pathToUpdate


##########################
def MCMC3(n, h, path, X, phi, k, fx0, Y, X_f_n_mL_p1, P_f_n_mL_p1_inv,
          X_f_n_mL, P_f_n_mL_inv, R1_inv=None, R2_inv=None):
    ac = np.zeros(N)
    aar = 0 #average acceptance rate
    MCMCiter = 0
    terminateMCMC = False

    pathToUpdate = np.zeros((dimx, N * (L + 1)))
    i1 = N * (n + 1 - L)
    i2 = N * (n + 1)
    pathToUpdate[:, 0:N * L] = path[:, i1:i2]  # From X_{n+1-L} to X_{n}
    pathToUpdate[:, N * L: N * (L + 1)] = X
    # = X_{n+1},so pathToUpdate is from X_{n+1-L} to X_{n+1}
    # When n = L, pathToUpdate[:,0:N] = path[:,N:2*N] = X_1
    #   and pathToUpdate[:,N*(L-1):N*L] = path[:,N*L:N*(L+1)] = X_{L+1}
    # When n = L+2, pathToUpdate[:,0:N] = X_2
    #   and pathToUpdate[:,N*(L-1): N*L] = X_{L+2}
    lf_p = np.zeros((N, L))  # p stands for proposed
    lg_p = np.zeros((N, L))
    lf = np.zeros((N, L))
    lg = np.zeros((N, L))
    while not terminateMCMC:
        MCMCiter += 1
        if (MCMCiter > MCMCnmin) & (aar >= aar_min) & (aar <= aar_max):
            terminateMCMC = True
        if MCMCiter >= MCMCnmax:
            terminateMCMC = True

        covm = sig
        if aar < aar_min:
            covm = sig * (phi[k] + 2) / (phi[k] + 1) / (MCMCiter**6 + 10)
        elif aar > aar_max:
            covm = sig * (phi[k] + 2) / (phi[k] + 1) / (MCMCiter**2 + 1)
        else:
            covm = sig * (phi[k] + 2) / (phi[k] + 1) / 2
        path_p = mvnrnd(pathToUpdate, covm, coln=N * (L + 1))
        if n == L:
            # if n = L, path_p[:,0:N] or pathToUpdate[:,0:N]corresponds to X_1
            # calc log[f(x_1^{'j}|x_0)] and log[f(x_1^{j}|x_0)]
            vf_1_0_p = path_p[:, 0:N] - fx0
            vf_1_0 = pathToUpdate[:, 0:N] - fx0
            if X_CoefMatrix_is_eye:
                lf_1_0p = np.diag(-0.5 * sig_x ** (-2) * \
                                     np.matmul(vf_1_0_p.T, vf_1_0_p))
                lf_1_0 = np.diag(-0.5 * sig_x ** (-2) * \
                                    np.matmul(vf_1_0.T, vf_1_0))
            else:
                lf_1_0p = np.diag(-0.5 * \
                                     np.matmul(vf_1_0_p.T,
                                               np.matmul(R1_inv, vf_1_0_p)))
                lf_1_0 = np.diag(-0.5 * sig_x ** (-2) * \
                                    np.matmul(vf_1_0.T,
                                              np.matmul(R1_inv, vf_1_0)))
            for v in range(L):
                i1 = N * v
                i2 = N * (v + 1)
                U = OneOr2D_to_3dOr4d(N, path_p[:, i1:i2])
                U, dt = swe_onestep(N, U)
                fX = ThreeOr4D_to_1dOr2d(N, U).real
                j1 = N * (v + 1)
                j2 = N * (v + 2)
                vf_p = path_p[:, j1:j2] - fX
                if C_is_eye:
                    vg_p = Y[:, v].reshape(-1, 1) - path_p[:, i1:i2]
                else:
                    vg_p = Y[:, v].reshape(-1, 1) - np.matmul(C,
                                                              path_p[:, i1:i2])
                U = OneOr2D_to_3dOr4d(N, pathToUpdate[:, i1:i2])
                U, dt = swe_onestep(N, U)
                fX = ThreeOr4D_to_1dOr2d(N, U).real
                vf = pathToUpdate[:, j1:j2] - fX
                if C_is_eye:
                    vg = Y[:, v].reshape(-1, 1) - pathToUpdate[:, i1:i2]
                else:
                    vg = Y[:, v].reshape(-1, 1) - np.matmul(C,
                                                    pathToUpdate[:, i1:i2])
                if X_CoefMatrix_is_eye:
                    lf_p[:, v] = np.diag(-0.5 * sig_x ** (-2) * \
                                            np.matmul(vf_p.T, vf_p))
                    lf[:, v] = np.diag(-0.5 * sig_x ** (-2) \
                                          * np.matmul(vf.T, vf))
                else:
                    lf_p[:, v] = np.diag(-0.5 * np.matmul(vf_p.T,
                                                np.matmul(R1_inv, vf_p)))
                    lf[:, v] = np.diag(-0.5 * np.matmul(vf.T,
                                                np.matmul(R1_inv, vf)))
                if Y_CoefMatrix_is_eye:
                    lg_p[:, v] = np.diag(-0.5 * sig_y ** (-2) * \
                                            np.matmul(vg_p.T, vg_p))
                    lg[:, v] = np.diag(-0.5 * sig_y ** (-2) * \
                                          np.matmul(vg.T, vg))
                else:
                    lg_p[:, v] = np.diag(-0.5 * np.matmul(vg_p.T,
                                                    np.matmul(R2_inv, vg_p)))
                    lg[:, v] = np.diag(-0.5 * np.matmul(vg.T,
                                                    np.matmul(R2_inv, vg)))
            # calculate log(the ratio to the power phi[k+1]):
            i1 = N
            i2 = 2 * N
            j1 = N * L
            j2 = N * (L + 1)
            if C_is_eye:
                vg_pr = Y[:, n].reshape(-1, 1) - path_p[:, j1:j2]
                vg_r = Y[:, n].reshape(-1, 1) - pathToUpdate[:, j1:j2]
            else:
                vg_pr = Y[:, n].reshape(-1, 1) - np.matmul(C, path_p[:, j1:j2])
                vg_r = Y[:, n].reshape(-1, 1) - np.matmul(C,
                                                         pathToUpdate[:,j1:j2])

            vec_mu_p = path_p[:, i1:i2] - X_f_n_mL_p1
            vec_mu = pathToUpdate[:, i1:i2] - X_f_n_mL_p1
            if Y_CoefMatrix_is_eye:
                lg_pr = np.diag(-0.5 * sig_y ** (-2) * np.matmul(vg_pr.T,
                                                                    vg_pr))
                lg_r = np.diag(-0.5 * sig_y ** (-2) * np.matmul(vg_r.T,
                                                                   vg_r))
            else:
                lg_pr = np.diag(-0.5 * np.matmul(vg_pr.T,
                                                    np.matmul(R2_inv, vg_pr)))
                lg_r = np.diag(-0.5 * np.matmul(vg_r.T,
                                                   np.matmul(R2_inv, vg_r)))
            log_mu_p = np.diag(-0.5 * np.matmul(vec_mu_p.T,
                                         np.matmul(P_f_n_mL_p1_inv,
                                                   vec_mu_p)))
            log_mu = np.diag(-0.5 * np.matmul(vec_mu.T,
                                              np.matmul(P_f_n_mL_p1_inv,
                                                        vec_mu)))
            log_ratio = phi[k + 1] * (lg_pr + log_mu_p + lf[:, 0]
                                      - lg_r - log_mu - lf_p[:, 0])
            log_accep = np.minimum(np.zeros(N), log_ratio \
                                   + np.sum(lf_p, axis=1) + lf_1_0p \
                                   + np.sum(lg_p, axis=1) \
                                   - np.sum(lf, axis=1) - lf_1_0 \
                                   - np.sum(lg, axis=1))
        else:  # n = L+1, L+2, ...
            for v in range(L):
                i1 = N * v
                i2 = N * (v + 1)
                U = OneOr2D_to_3dOr4d(N, path_p[:, i1:i2])
                U, dt = swe_onestep(N, U)
                fX = ThreeOr4D_to_1dOr2d(N, U).real
                j1 = N * (v + 1)
                j2 = N * (v + 2)
                vf_p = path_p[:, j1:j2] - fX
                if C_is_eye:
                    vg_p = Y[:, v + n - L].reshape(-1, 1) - path_p[:, i1:i2]
                    # when v=L-1 --> Y[:,n-1] - C @ path_p[:,N*(L-1): N*L]
                    # when n = L+1 --> Y[:,L] - C @ path_p[:,N*(L-1): N*L]
                else:
                    vg_p = Y[:, v + n - L].reshape(-1, 1) \
                              - np.matmul(C, path_p[:, i1:i2])

                U = OneOr2D_to_3dOr4d(N, pathToUpdate[:, i1:i2])
                U, dt = swe_onestep(N, U)
                fX = ThreeOr4D_to_1dOr2d(N, U).real
                vf = pathToUpdate[:, j1:j2] - fX
                if C_is_eye:
                    vg = Y[:, v + n - L].reshape(-1, 1) - pathToUpdate[:,i1:i2]
                else:
                    vg = Y[:, v + n - L].reshape(-1, 1) \
                            - np.matmul(C, pathToUpdate[:, i1:i2])
                if X_CoefMatrix_is_eye:
                    lf_p[:, v] = np.diag(-0.5 * sig_x ** (-2) * \
                                            np.matmul(vf_p.T, vf_p))
                    lf[:, v] = np.diag(-0.5 * sig_x ** (-2) \
                                          * np.matmul(vf.T, vf))
                else:
                    lf_p[:, v] = np.diag(-0.5 * np.matmul(vf_p.T,
                                                np.matmul(R1_inv, vf_p)))
                    lf[:, v] = np.diag(-0.5 * np.matmul(vf.T,
                                                    np.matmul(R1_inv, vf)))
                if Y_CoefMatrix_is_eye:
                    lg_p[:, v] = np.diag(-0.5 * sig_y ** (-2) * \
                                            np.matmul(vg_p.T, vg_p))
                    lg[:, v] = np.diag(-0.5 * sig_y ** (-2) \
                                          * np.matmul(vg.T, vg))
                else:
                    lg_p[:, v] = np.diag(-0.5 * np.matmul(vg_p.T,
                                                np.matmul(R2_inv, vg_p)))
                    lg[:, v] = np.diag(-0.5 * np.matmul(vg.T,
                                                    np.matmul(R2_inv, vg)))
            # calculate log(the ratio to the power phi[k+1]):
            i1 = N
            i2 = 2 * N
            j1 = N * L
            j2 = N * (L + 1)
            if C_is_eye:
                vg_pr = Y[:, n].reshape(-1, 1) - path_p[:, j1:j2]
                vg_r = Y[:, n].reshape(-1, 1) - pathToUpdate[:, j1:j2]
            else:
                vg_pr = Y[:, n].reshape(-1, 1) - np.matmul(C, path_p[:, j1:j2])
                vg_r = Y[:, n].reshape(-1, 1) \
                          - np.matmul(C, pathToUpdate[:, j1:j2])

            vec_mu_p = path_p[:, i1:i2] - X_f_n_mL_p1
            vec_mu = pathToUpdate[:, i1:i2] - X_f_n_mL_p1
            if Y_CoefMatrix_is_eye:
                lg_pr = np.diag(-0.5 * sig_y ** (-2) * np.matmul(vg_pr.T,
                                                                    vg_pr))
                lg_r = np.diag(-0.5 * sig_y ** (-2) * np.matmul(vg_r.T,
                                                                   vg_r))
            else:
                lg_pr = np.diag(-0.5 * np.matmul(vg_pr.T,
                                                    np.matmul(R2_inv, vg_pr)))
                lg_r = np.diag(-0.5 * np.matmul(vg_r.T,
                                                   np.matmul(R2_inv, vg_r)))
            log_mu_p = np.diag(-0.5 * np.matmul(vec_mu_p.T,
                                                np.matmul(P_f_n_mL_p1_inv, 
                                                          vec_mu_p)))
            log_mu = np.diag(-0.5 * np.matmul(vec_mu.T,
                                              np.matmul(P_f_n_mL_p1_inv,
                                                        vec_mu)))
            log_ratio = phi[k + 1] * (lg_pr + log_mu_p + lf[:, 0]
                                      - lg_r - log_mu - lf_p[:, 0])
            # calculate %\mu_{n-L-1}(x_{n-L})
            vec_mu_p = path_p[:, 0:N] - X_f_n_mL
            vec_mu = pathToUpdate[:, 0:N] - X_f_n_mL
            log_mu_p = np.diag(-0.5 * np.matmul(vec_mu_p.T,
                                                np.matmul(P_f_n_mL_inv,
                                                          vec_mu_p)))
            log_mu = np.diag(-0.5 * np.matmul(vec_mu.T,
                                              np.matmul(P_f_n_mL_inv, 
                                                        vec_mu)))

            log_accep = np.minimum(np.zeros(N), log_ratio + log_mu_p \
                                   + np.sum(lf_p, axis=1) \
                                   + np.sum(lg_p, axis=1) - log_mu \
                                   - np.sum(lf, axis=1) \
                                   - np.sum(lg, axis=1))
            # end if
        log_U = np.log(np.random.uniform(size=N))
        for j in range(N):
            if log_U[j] < log_accep[j]:
                for v in range(L+1):
                    i1 = N * v + j
                    pathToUpdate[:, i1] = path_p[:, i1]
                ac[j] += 1
        accep_rate = ac / MCMCiter
        aar = np.mean(accep_rate)
    # end while
    print('h = %d, time_step n = %d MCMCsteps = %d,'
          ' accep_rate_avg = %.4f, phi = %.5f\n' %
          (h,n+1, MCMCiter, aar, phi[k + 1]))
    return pathToUpdate


##########################
def weight1(Yn, delta, X, lw_old, R2_inv=None):
    """ 
    Calculates the weights for the SMC sampler on the log scale.
    Inputs: 1) Yn: Y[:,n] the data at time n.
            2) delta: The power in g(yn|xn)^delta.
            3) X: The signal of size (dimx x N).
            4) lw_old: The previous log-weights.
            5) R2_inv: The inverse of the covariance matrix of the data noise
                Note that if Y_CoefMatrix_is_eye is True, then pass None.
    Outpus: 1) lw: the log-weights
            2) We: the normalized weights , 
            3) We0: the weights minus the maximum weight
            4) sumWe0: their sum 
    """
    norm_const = -0.5 * delta * (dimo * log2pi + ldet_R2)
    if C_is_eye:
        vec = Yn - X  # Yn - mu, mu is mean of g(yn|xn);
    else:
        vec = Yn - np.matmul(C, X)
    if Y_CoefMatrix_is_eye:
        # g(yn|xn)^delta
        tempMatrix = - 0.5 * delta * sig_y ** (-2) * np.matmul(vec.T, vec)
    else:
        tempMatrix = - 0.5 * delta * np.matmul(vec.T, np.matmul(R2_inv, vec))
    lw = lw_old + norm_const + np.diag(tempMatrix)
    # normalize the weight
    max_lw = np.max(lw)
    We0 = np.exp(lw - max_lw)
    sumWe0 = np.sum(We0)
    We = We0 / sumWe0
    return lw, We, We0, sumWe0


###########################
def weight2(n, Yn, delta, path, X, X_f_n_mL_p1, P_f_n_mL_p1_inv,
            ldet_P_f_n_mL_p1, lw_old, fXN, R1_inv=None, R2_inv=None):
    # The path is 2d array. For make it easy to understand assume the path
    # is 3d array of shape (T+1 ,dimx, N), that is for each time t = 0,...,T:
    # path[t,:,:] is all the particles at time t. However, path is a 2d
    # array of shape (dimx,(T+1)*N), at time t = 0, the particles are:
    # path[:,0:N], at t=1, path[:,N:2*N], etc..
    if C_is_eye:
        vec1 = Yn - X  # Yn - mu, mu is mean of g(yn|xn);
    else:
        vec1 = Yn - np.matmul(C, X)
    # in the following when n = L, we have mu_{X_1}(X_2) (recall:
    # path[2,:,:] = X2):
    i1 = N * (n - L + 2)
    i2 = N * (n - L + 3)
    vec2 = path[:, i1:i2] - X_f_n_mL_p1  # when n = L, path[2,:,:] --> X_2
    # this is from the exponential part of $log(\mu_{n-L}(x_{n-L+1})$
    vec3 = path[:, i1:i2] - fXN  # when n = L, have x_2 - f(
    norm_const = delta / 2 * (ldet_R1 - dimo * log2pi - ldet_R2 - ldet_P_f_n_mL_p1)
    if Y_CoefMatrix_is_eye & X_CoefMatrix_is_eye:
        # g(yn|xn)^delta
        tempMatrix = -0.5 * delta * (sig_y ** (-2) * np.matmul(vec1.T, vec1)
                                     + np.matmul(vec2.T, np.matmul(P_f_n_mL_p1_inv, vec2))
                                     - sig_x ** (-2) * np.matmul(vec3.T, vec3))
    else:
        tempMatrix = -0.5 * delta * (np.matmul(vec1.T, np.matmul(R2_inv, vec1))
                                     + np.matmul(vec2.T, np.matmul(P_f_n_mL_p1_inv, vec2))
                                     - np.matmul(vec3.T, np.matmul(R1_inv, vec3)))
    lw = lw_old + norm_const + np.diag(tempMatrix)
    # normalize the weight
    max_lw = np.max(lw)
    We0 = np.exp(lw - max_lw)
    sumWe0 = np.sum(We0)
    We = We0 / sumWe0
    return lw, We, We0, sumWe0


###########################
def calcESS(We0, sumWe0):
    """ Calculates the effective sample size """
    return sumWe0 ** 2 / np.sum(We0 ** 2)


##########################
def resample1(ESS, X, lw, We):
    """ 
    If ESS < threshold, resample the particles according to 
    their weights 
    """
    if ESS <= ESS_threshold:
        An = np.random.choice(np.arange(N), N, p=We)
        X = X[:, An]
        lw_old = np.zeros(N)
        We = np.ones(N) / N
        resampled = True
    else:
        resampled = False
        lw_old = lw
    return X, lw_old, We, resampled


##########################
def resample2(n, path, We):
    """ 
    Resamples the whole path from time 0 to n of all particles according to 
    their weights 
    """
    An = np.random.choice(np.arange(N), N, p=We)
    for v in range(n + 2):
        i1 = N * v
        i2 = N * (v + 1)
        Xtemp = path[:, i1:i2]
        Xtemp = Xtemp[:, An]
        path[:, i1:i2] = Xtemp
    return path


##########################
def FuncOfDelta1(Yn, X, delta):
    if Y_CoefMatrix_is_eye:
        funcOfDelta = partial(weight1, Yn=Yn, X=X, lw_old=np.zeros(N))
        We0, sumWe0 = funcOfDelta(delta=delta)[2:4]
    else:
        pass  # need to modify to pass R2_inv:
        # lw, We, We0, sumWe0 = weight1(Y1, phi[0], np.zeros(N),R2_inv)
    ESS = calcESS(We0, sumWe0)
    ESS -= ESS_threshold
    return ESS


##########################
def FuncOfDelta2(n, Yn, X, path, X_f_n_mL_p1, P_f_n_mL_p1_inv,
                 ldet_P_f_n_mL_p1, fXN, delta):
    if Y_CoefMatrix_is_eye:
        funcOfDelta = partial(weight2, n=n, Yn=Yn, path=path, X=X,
                              X_f_n_mL_p1=X_f_n_mL_p1,
                              P_f_n_mL_p1_inv=P_f_n_mL_p1_inv,
                              ldet_P_f_n_mL_p1=ldet_P_f_n_mL_p1,
                              lw_old=np.zeros(N), fXN=fXN)
        We0, sumWe0 = funcOfDelta(delta=delta)[2:4]
    else:
        pass  # need to modify to pass R1_inv & R2_inv:
        # funcOfDelta = partial(weight2, n = n,Yn =Yn, path = path, X = X,
        #                       X_f_n_mL_p1 = X_f_n_mL_p1,
        #                       P_f_n_mL_p1_inv = P_f_n_mL_p1_inv,
        #                       ldet_P_f_n_mL_p1 = ldet_P_f_n_mL_p1,
        #                       lw_old=np.zeros(N),
        #                       R1_inv = R1_inv, R2_inv = R2_inv)

    ESS = calcESS(We0, sumWe0)
    ESS -= ESS_threshold
    return ESS


#########################
def bisection(func, a, b, diff, error):
    fa = func(a)
    fb = func(b)
    if fa * fb >= 0:
        print("f(a) and f(b) must have different signs\n")
        print(" f(a) = %.4f, f(b) = %.4f\n" % (fa, fb))
        print('Try different values of b\n')
        iters = 0
        b = b/20
        fb = func(b)
        while (fa * fb >= 0) & (iters < 100):
            iters += 1
            b = b - 0.0005
            fb = func(b)
        if (fa * fb >= 0):
            print('Failed to find b so that f(a)*f(b) >= 0, f(b)= %.3f \n' % fb)
            c = a
            converged = False
            return c, converged
    iters = 0
    converged = False
    while ((b - a) >= diff) & (iters <= bisection_nmax):
        iters += 1
        # Find middle point
        c = (a + b) / 2
        # Check if middle point is root
        if np.abs(func(c)) <= error:
            converged = True
            break
        # Decide the side to repeat the steps
        if func(c) * func(a) < 0.:
            b = c
        else:
            a = c
    if iters > bisection_nmax:
        c = a
        converged = False
    return c, converged


################### Plotting ################################# 

def setFigure():
    fig = plt.figure()
    fig.set_size_inches(12, 8)
    return fig


def initialize_subplot(fig, E):
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xx, yy, E, cmap=cm.coolwarm, linewidth=0.2,
                           rstride=1, cstride=1)
    fig.colorbar(surf, shrink=0.5)
    ax.view_init(10, 35)
    ax.set_zlim(0, htop)
    ax.set_xlim(a, b)
    ax.set_ylim(a, b)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return ax, surf, fig


##################
def animate_3dArray(n, V, fig, surf, ax):
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(10, 35)
    surf = ax.plot_surface(xx, yy, V[n, :, :], cmap=cm.coolwarm,
                           linewidth=0.2, rstride=1, cstride=1)
    fig.colorbar(surf, shrink=0.5)
    ax.set_zlim(0, htop)
    ax.set_xlim(a, b)
    ax.set_ylim(a, b)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return ax, surf, fig


#################
def animate_2dArray(n, X, fig, surf, ax):
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(10, 35)
    surf = ax.plot_surface(xx, yy, X[0:dim2, n].reshape(dim, dim),
                           cmap=cm.coolwarm, linewidth=0.2, rstride=1, cstride=1)
    fig.colorbar(surf, shrink=0.5)
    ax.set_zlim(0, htop)
    ax.set_xlim(a, b)
    ax.set_ylim(a, b)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return ax, surf, fig


#########################

##############################################################################
##############################################################################
########################## M A I N ###########################################
##############################################################################
# %%###########################################################################
"""
In this program, we filter the following discrete nonlinear sysfx0:
    X_t = f(X_{t-1}) + R1_sqrt * dWt
    Y_t = C @ X_t + R2_sqrt * dVt
where dWt, dVt are independent random variables sampled from 
N(np.zeros(dimx),np.eye(dimx)) and N(np.zeros(dimo),np.eye(dimo)) respectively
f(X_t) is the solution of Shallow-Water Equations one step in time.
X_t is a random vector in R^dimx, where X_t[0:dim2] is the hight of the water,
X_t[dim2:2*dim2] is the velocity of the fluid in x-axis and X_t[2*dim2:3*dim2]
is the velocity in y-axis. Y_t is random vector in R^dimo, is the observation
of some or all coordinates of X_t with some noise. R1_sqrt and R2_sqrt are the 
noise coeficient matrices for X_t and Y_t, respectively.

The filtering methods we use are:
    1) EnKF : Ensemble Kalman Filter.
        (see e.g. Houtekamer et al. (1998). "Data assimilation using an 
         ensemble Kalman filter technique")
    2) EtKF : Ensemble transform Kalman Filter.
       (see e.g. Adaptive Sampling with the Ensemble Transform Kalman Filter. 
        Part I: Theoretical Aspects by Craig H. Bishop et al.)
    3) EtKF_sqrt: Ensemble transform Kalman Filter with Square-root.
        (e.g. Hunt et al., 2007) see "State-of-the-art stochastic data 
        assimilation methods for high-dimensional non-Gaussian problems"
    4) Our proposed method for particle filter: Lagged Particle Filter.
"""
# Set some parameters
""" All the following are global variables """
np.random.seed(1234)
nsimul = 8
d = 5  # grid points in direction x and direction y
dim = d + 2  # these two extra points are for ghost cells
dim2 = dim ** 2
dimx = 3 * dim2  # x is in R^2 of size (d x d) x 3, thus we have 3*d^2 elements
# x contains height, velocity in x and velocity in y directions
h_freq = 1  # spatial frequency (in both directions).. observe height every
# "h_freq" steps
v_freq = 3  # velocity frequency ...observe velocity every "v_freq" steps in x
# and y directions
# if freq = 3, the observed elements are 0,3,6,9,12,...
# if freq = 2, the observed elements are 0,2,4,6,8,...
# if freq = 1, the observed elements are 0,1,2,3,4,...
no_h = int(np.ceil(dim / h_freq) ** 2)  # number of height observations ( in x and y)
no_v = int(2 * np.ceil(dim / v_freq) ** 2)  # number of velocity observations. The 2
# because we have two components of velocity u & v
dimo = no_h + no_v  # dimension of observations
# The grid is a square [a,b] x [a,b]
a = 0.
b = 2.
dx = (b - a) / (d - 1)
dy = dx
htop = 2.7  # used in plotting (a little larger than the maximum initial height)
N = 100  # number of particles
M = 1000  # size of the ensemble for EnKF, ETKF, ETKF_sqrt
L = 3  # Lag ... for Lagged Particle Filter
MCMCnmax = 100
MCMCnmin = 10
aar_min = 0.2222 #minimum average acceptance rate for MCMC
aar_max = 0.3111 #maximum average acceptance rate for MCMC
bisection_nmax = 1000
ESS_threshold = 0.5 * N  # effective sample size threshold for resampling
phi1 = 0.  # inital annealing parameter
T = 100  # number of iterations
# we will calculate the following filters : p(x_t|y_{0:t}) for t =1,...,T
c = 0.5  # Courant number threshold for FV solver
g = 9.81  # gravity constant
half_g = 0.5 * g
sig = 2.38 ** 2 / dimx * np.eye(dimx)  # Cov - used in MCMC for the random walk
log2pi = np.log(2.0 * np.pi)

### FLAGS:
save_to_files = False
plot_flag = False
X_CoefMatrix_is_eye = True  # is R1_sqrt the identity matrix?
Y_CoefMatrix_is_eye = True  # is R2_sqrt the identity matrix?
C_is_eye = False  # if it is True, then dimx = dimo observing all coords
# see h_freq and v_freq..if it is True change to 1
# get the date and time right now:
start_time = time.time()
date = datetime.datetime.now()
date_time = np.array([date.year, date.month, date.day, date.hour, date.minute])
if save_to_files:
    string = 'DateOfSimul-%d-%d-%d-%d-%d.txt' % (date.year, date.month, date.day,
                                                 date.hour, date.minute)
    np.savetxt(string, date_time, fmt='%d')
# set up the grid
x = np.linspace(a - dx, b + dx, num=dim)
y = x.copy()

if save_to_files:
    # Save x and y to files:
    string = 'x_%d-%d-%d-%d-%d-dx%d_dy%d_L%d_N%d_T%d.txt' \
             % (date.year, date.month, date.day, date.hour,
                date.minute, dimx, dimo, L, N, T)
    np.savetxt(string, x, fmt='%f')
    string = 'y_%d-%d-%d-%d-%d-dx%d_dy%d_L%d_N%d_T%d.txt' \
             % (date.year, date.month, date.day,
                date.hour, date.minute, dimx, dimo, L, N, T)
    np.savetxt(string, y, fmt='%f')

xx, yy = np.meshgrid(x, y)  # needed for plotting (global)
#----------
del x
del y
gc.collect()
# Initial conditions: 
# initial height
h0 = np.ones((dim, dim))
h0[(xx >= 0.5) & (xx <= 1) & (yy >= 0.5) & (yy <= 1)] = 2.5
x_star = np.zeros(dimx)  # initial velocities zero
x_star[0:dim2] = h0.flatten()  # Initial state
# vectors for circular shift:
shiftp1 = np.roll(np.arange(dim), 1)
shiftm1 = np.roll(np.arange(dim), -1)
# Choose the matrices R1_sqrt, R2_sqrt, and C
if not C_is_eye:
    C = np.zeros((dimo, dimx))
    C[0, 0] = 1.  # always observe the first element
    for i in range(dimo):
        for j in range(dimx):
            if (i <= no_h) & (j % h_freq == 0):
                C[i, j] = 1.
            if (i > no_h) & (j % v_freq == 0):
                C[i, j] = 1.
Idx = np.eye(dimx)
if X_CoefMatrix_is_eye:
    sig_x = np.sqrt(0.0001)
    ldet_R1 = 2 * dimx * np.log(sig_x)  # det(R1) = sig_x^(2*dimu)
else:
    pass  # need to provide R1_sqrt & R1 = np.matmul(R1_sqrt, R1_sqrt) & ldet_R1

if Y_CoefMatrix_is_eye:
    sig_y = np.sqrt(0.0001)
    ldet_R2 = 2 * dimo * np.log(sig_y)  # det(R2) = sig_y^(2*dimo)
else:
    pass  # need to provide R2_sqrt & R2 = np.matmul(R2_sqrt, R2_sqrt) & ldet_R2

# R2_sqrt = sig_y * Ido # sig_y * np.identity(dimo)
# R2_sqrt_inv  = Ido / sig_y #np.identity(dimo)/sig_y
# R2 = sig_y**2  * Ido #covariance of observation noise
log_det_R2 = 2 * dimo * np.log(sig_y)  # det(R2) = sig_y^(2*dimo);
R1_sqrt = sig_x * Idx
R1 = sig_x ** 2 * Idx  # covariance of state noise
log_det_R1 = 2 * dimx * np.log(sig_x)  # det(R1) = sig_x^(2*dimu);
# Generate the data 
Y = np.zeros((dimo, T))
X_pertur = np.zeros((dimx, T + 1))
X_pertur[:, 0] = x_star
t_obs = 0.
U1 = np.zeros((T, dim, dim))
for n in range(T):
    U = OneOr2D_to_3dOr4d(1, X_pertur[:, n])
    U, dt = swe_onestep(1, U)
    t_obs += dt
    U1[n, :, :] = U[0, :, :]
    X_pertur[:, n + 1] = ThreeOr4D_to_1dOr2d(1, U)
    dW = mvnrnd(np.zeros((dimx)), Idx)
    X_pertur[:, n + 1] = X_pertur[:, n + 1] + sig_x * dW
    dV = mvnrnd(np.zeros((dimo)), np.eye(dimo))
    Y[:, n] = np.matmul(C, X_pertur[:, n + 1]) + sig_y * dV

print('finished generating the observations\n')
if save_to_files:
    string = 'X_pertur_%d-%d-%d-%d-%d-dx%d_dy%d_L%d_N%d_T%d.txt' \
             % (date.year, date.month, date.day, date.hour,
                date.minute, dimx, dimo, L, N, T)
    np.savetxt(string, X_pertur, fmt='%f')
    string = 'Y_%d-%d-%d-%d-%d-dx%d_dy%d_L%d_N%d_T%d.txt' \
             % (date.year, date.month, date.day, date.hour,
                date.minute, dimx, dimo, L, N, T)
    np.savetxt(string, Y, fmt='%f')

X_nonpertur = np.zeros((dimx, T + 1))
X_nonpertur[:, 1] = x_star
for n in range(T):
    U = OneOr2D_to_3dOr4d(1, X_nonpertur[:, n]);
    U, dt = swe_onestep(1, U)
    X_nonpertur[:, n + 1] = ThreeOr4D_to_1dOr2d(1, U)
if save_to_files:
    string = 'X_nonpertur_%d-%d-%d-%d-%d-dx%d_dy%d_L%d_N%d_T%d.txt' \
             % (date.year, date.month, date.day, date.hour,
                date.minute, dimx, dimo, L, N, T)
    np.savetxt(string, X_nonpertur, fmt='%f')
# %%####
if plot_flag:
    fig0 = setFigure()
    ax0, surf0, fig0 = initialize_subplot(fig0, U1[0, :, :])
    ani0 = FuncAnimation(fig0, animate_3dArray, frames=T,
                         fargs=(U1, fig0, surf0, ax0), interval=10, repeat=True)
# %%####
del U
del X_pertur
del X_nonpertur
gc.collect()
# %%###########################################################################
# EnKF
if __name__ == "__main__":
    # call EnKF1 to return the mean and covariance of the predictor to be used
    # later
    X_f, P_f = EnKF1(sig_y, 0)  # to use them in LaggedPF
    # creating a pool object
    p = MP.Pool(processes=8)
    # map list to target function
    if Y_CoefMatrix_is_eye:
        func0 = partial(EnKF, sig_y)  # can pass any thing for the second input
    else:
        pass  # need to modify this if R2_sqrt is available to :
        # func0 = partial(EnKF,R2_sqrt)
    results0 = p.map(func0, range(nsimul))
    print('Ensemble Kalman Filter ...Done')
    if Y_CoefMatrix_is_eye:
        func1 = partial(EtKF, sig_y)  # can pass any thing for the second input
    else:
        pass  # need to modify this if R2_sqrt_inv is available to :
        # func1 = partial(EtKF,R2_sqrt_inv)
    results1 = p.map(func1, range(nsimul))
    print('Ensemble Transform Kalman Filter ...Done')
    # if Y_CoefMatrix_is_eye:
    #    func2 = partial(EtKF_sqrt,sig_y**2 * np.eye(dimo)) 
    # else:
    #    pass #need to modify this if R2 is available to :
    #    #func2 = partial(EtKF_sqrt,R2)   
    #results2 = p.map(func2, range(nsimul))
    #print('Ensemble Transform Kalman Filter with SQRT...Done')

    func3 = partial(LaggedPF, X_f, P_f)
    results3 = p.map(func3, range(nsimul))
    print('Lagged Particle Filter ...Done')
    p.close()
    p.join()

    t_simul_EnKF = sum([atuple[0] for atuple in results0]) / nsimul
    E_EnKF = sum([atuple[1] for atuple in results0]) / nsimul

    t_simul_EtKF = sum([atuple[0] for atuple in results1]) / nsimul
    E_EtKF = sum([atuple[1] for atuple in results1]) / nsimul

    # t_simul_EtKF_sqrt = sum([atuple[0] for atuple in results2])/nsimul
    # E_EtKF_sqrt = sum([atuple[1] for atuple in results2])/nsimul

    t_simul_LaggedPF = sum([atuple[0] for atuple in results3]) / nsimul
    # just get the first ESS_saved
    ESS_saved = results3[0][1]
    path = sum([atuple[2] for atuple in results3]) / nsimul
    E_LaggedPF = np.zeros((dimx, T + 1))
    for i in range(T + 1):
        i1 = N * i
        i2 = N * (i + 1)
        E_LaggedPF[:, i] = np.sum(path[:, i1:i2], axis=1) / N

    if plot_flag:
        fig1 = setFigure()
        ax1, surf1, fig1 = initialize_subplot(fig1,
                                              E_EnKF[0:dim2, 0].reshape(dim, dim))
        ani1 = FuncAnimation(fig1, animate_2dArray, frames=T,
                             fargs=(E_EnKF, fig1, surf1, ax1), interval=10, repeat=True)

        fig2 = setFigure()
        ax2, surf2, fig2 = initialize_subplot(fig2,
                                              E_EtKF[0:dim2, 0].reshape(dim, dim))
        ani2 = FuncAnimation(fig2, animate_2dArray, frames=T,
                             fargs=(E_EtKF, fig2, surf2, ax2), interval=10, repeat=True)

        # fig3 = setFigure()
        # ax3, surf3, fig3 = initialize_subplot(fig3,
        #                            E_EtKF_sqrt[0:dim2,0].reshape(dim,dim))
        # ani3 = FuncAnimation(fig3, animate_2dArray, frames = T,
        #        fargs=(E_EtKF_sqrt,fig3,surf3, ax3),interval=10, repeat=True)

        fig4 = setFigure()
        ax4, surf4, fig4 = initialize_subplot(fig4,
                                              E_LaggedPF[0:dim2, 0].reshape(dim, dim))
        ani4 = FuncAnimation(fig4, animate_2dArray, frames=T,
                             fargs=(E_LaggedPF, fig4, surf4, ax4), interval=10, repeat=True)
    if save_to_files:
        string = 'E_EnKF_%d-%d-%d-%d-%d-dx%d_dy%d_L%d_N%d_T%d.txt' \
                 % (date.year, date.month, date.day, date.hour,
                    date.minute, dimx, dimo, L, N, T)
        np.savetxt(string, E_EnKF, fmt='%f')

        string = 'E_EtKF_%d-%d-%d-%d-%d-dx%d_dy%d_L%d_N%d_T%d.txt' \
                 % (date.year, date.month, date.day, date.hour,
                    date.minute, dimx, dimo, L, N, T)
        np.savetxt(string, E_EtKF, fmt='%f')

        # string = 'E_EtKF_sqrt_%d-%d-%d-%d-%d-dx%d_dy%d_L%d_N%d_T%d.txt'\
        #    % (date.year,date.month,date.day,date.hour,\
        #        date.minute,dimx,dimo,L,N,T)
        # np.savetxt(string, E_EtKF_sqrt, fmt='%f')

        string = 'E_LaggedPF_%d-%d-%d-%d-%d-dx%d_dy%d_L%d_N%d_T%d.txt' \
                 % (date.year, date.month, date.day, date.hour,
                    date.minute, dimx, dimo, L, N, T)
        np.savetxt(string, E_LaggedPF, fmt='%f')
