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


def analysis(H, RG, P, rho=1.0):
    def update(xbG, XbG, yOG):
        N, k = XbG.shape
        _, L = H.shape
        if N % P != 0:
            raise RuntimeError("cites cannot be divided equally")
        if L % P != 0:
            raise RuntimeError("observations cannot be divided equally")
        n = N // P
        l = L // P
        ybG = H(xbG)
        YbG = H(XbG)
        for p in range(P):
            sl = slice(p*l, (p+1)*l)
            yb = ybG[sl]
            yO = yOG[sl]
            Yb = YbG[sl, :]
            R = RG[sl, sl]
            YR = np.dot(Yb.T, np.linalg.inv(R))
            Pa = np.linalg.inv(np.dot(YR, Yb) + ((k-1)/rho)*np.identity(k))
            wa = np.dot(Pa, np.dot(YR, yO - yb))
            Wa = symmetric_square_root((k-1)*Pa)
            sn = slice(p*n, (p+1)*n)
            xbG[sn] += np.dot(XbG[sn, :], wa)
            XbG[sn, :] = np.dot(XbG[sn, :], Wa)
        return xbG, XbG
    return update
