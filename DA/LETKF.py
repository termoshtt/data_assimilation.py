# -*- coding: utf-8 -*-

"""
Local Ensemble Transform Kalman Filter (LETKF)

Notations
----------
N : int
    length of the global state vector
n : int
    length of local state vectors
L : int
    length of the global observation vector
l : int
    length of local observation vectors
P : int
    Number of cites. :code:`N=n*P`, :code:`L=l*P`
p : int
    index of cite
K : int
    Number of ensembles

H : scipy.sparse.linalg.LinearOperator, (N) -> (L)
    Observation operator. **Assume Linear**.

"""

import numpy as np
from .linalg import symmetric_square_root


def observation(yG, YbG, RG, p):
    def each(n):
        y = np.roll(yG, p-n)[:2*p+1]
        Yb = np.roll(YbG, p-n, axis=0)[:2*p+1, :]
        R = np.roll(np.roll(RG, p-n, axis=1), p-n, axis=0)[:2*p+1, :2*p+1]
        return y, Yb, R
    return each


def analysis(H, RG, p, rho=1.0):
    def update(xbG, XbG, yOG):
        N, k = XbG.shape
        _, L = H.shape
        YbG = H(XbG)
        yG = yOG - H(xbG)
        obs = observation(yG, YbG, RG, p)
        for n in range(N):
            y, Yb, R = obs(n)
            YR = np.dot(Yb.T, np.linalg.inv(R))
            Pa = np.linalg.inv(np.dot(YR, Yb) + ((k-1)/rho)*np.identity(k))
            wa = np.dot(Pa, np.dot(YR, y))
            Wa = symmetric_square_root((k-1)*Pa)
            xbG[n] += np.dot(XbG[n, :], wa)
            XbG[n, :] = np.dot(XbG[n, :], Wa)
        return xbG, XbG
    return update
