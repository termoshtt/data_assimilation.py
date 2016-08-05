# -*- coding: utf-8 -*-

"""
Local Ensemble Transform Kalman Filter (LETKF)

Notations
----------
N : int
    length of the global state vector
p : int
    Number of observations used in a local analysis step is :code:`2*p+1`
K : int
    Number of ensembles
"""

import numpy as np
from . import ensemble
from .linalg import symmetric_square_root


def observation(yG, YbG, RG, p):
    def each(n):
        y = np.roll(yG, p-n)[:2*p+1]
        Yb = np.roll(YbG, p-n, axis=0)[:2*p+1, :]
        R = np.roll(np.roll(RG, p-n, axis=1), p-n, axis=0)[:2*p+1, :2*p+1]
        return y, Yb, R
    return each


def analysis(H, RG, p, rho=1.0):
    def update(xs, yOG):
        xbG, XbG = ensemble.deviations(xs)
        XbG = XbG.T
        N, k = XbG.shape
        YbG = np.dot(H, XbG)
        yG = yOG - np.dot(H, xbG)
        obs = observation(yG, YbG, RG, p)
        for n in range(N):
            y, Yb, R = obs(n)
            YR = np.dot(Yb.T, np.linalg.inv(R))
            Pa = np.linalg.inv(np.dot(YR, Yb) + ((k-1)/rho)*np.identity(k))
            wa = np.dot(Pa, np.dot(YR, y))
            Wa = symmetric_square_root((k-1)*Pa)
            xbG[n] += np.dot(XbG[n, :], wa)
            XbG[n, :] = np.dot(XbG[n, :], Wa)
        return ensemble.reconstruct(xbG, XbG.T)
    return update
