# -*- coding: utf-8 -*-

"""
Equivalent-weight Particle Filter (EWPF)

Notations
----------
N : int
    length of the global state vector
p : int
    length of observation vector
K : int
    Number of ensembles
"""

import numpy as np
from numpy.linalg import inv
from numpy.random import normal
from . import linalg, ensemble


def analysis(H, Q, R, M, Nth, gm=0, n=3):
    """
    Analysis step of Equivalent-weight Particle Filter

    Parameters
    -----------
    H : np.array (Nxp matrix)
        Observation
    Q : np.array (NxN matrix)
        Covariant matrix of EoM
    R : np.array (pxp matrix)
        Covariant matrix of observation
    M : int
        parameter for equivalent weight
    gm : float
        Intensity of noise added in the development of the proposed density
    Nth : int
        Threshold of resampling
    n : int, optional(default=3)
        Parameter for :py:func:`ensemble.merge_resampling`
    """
    V_inv = linalg.dot3(H, Q, H.T) + R
    K = linalg.dot3(Q, H.T, V_inv)
    M_inv = inv(Q) + linalg.dot3(H.T, inv(R), H)
    KMK = linalg.dot3(K.T, M_inv, K)
    Qs = linalg.symmetric_square_root(Q)

    def update(xs, cs, yO):
        Cs = []
        for x, c in zip(xs, cs):
            r = yO - np.dot(H, x)
            Cmin = c + 0.5*linalg.norm(r, V_inv)
            Cs.append(Cmin)
        C = sorted(Cs)[M]
        for i, (x, c) in enumerate(zip(xs, cs)):
            r = yO - np.dot(H, x)
            Cmin = c + 0.5*linalg.norm(r, V_inv)
            if Cmin < C:
                A = 0.5*linalg.norm(r, KMK)
                a = 1 - np.sqrt((C-Cmin)/A)
                cs[i] = C
            else:
                a = 1
                cs[i] = Cmin
            xi = normal(size=x.shape)
            cs[i] += np.linalg.norm(xi) / 2
            xs[i] += a*np.dot(K, r) + gm * np.dot(Qs, xi)
        cs -= np.min(cs)
        cs[cs > 30] = 30  # avoid overflow
        ws = ensemble.weight(cs)
        if ensemble.Neff(ws) < Nth:
            xs = ensemble.merge_resampling(ws, xs, n)
            cs = np.zeros_like(cs)
        return xs, cs
    return update
