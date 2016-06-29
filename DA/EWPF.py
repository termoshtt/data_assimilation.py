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
from . import linalg, ensemble


def analysis(H, Q, R, Nth):
    V_inv = linalg.dot3(H, Q, H.T) + R
    K = linalg.dot3(Q, H.T, V_inv)
    M_inv = inv(Q) + linalg.dot3(H, inv(R), H.T)
    KMK = linalg.dot3(K, M_inv, K.T)

    def update(xs, cs, yO):
        Cs = []
        for x, c in zip(xs, cs):
            r = yO - np.dot(H, x)
            Cmin = c + 0.5*linalg.norm(r, V_inv)
            Cs.append(Cmin)
        C = sorted(Cs)[len(Cs) // 5]
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
            xs[i] += a*np.dot(K, r)
        cs -= np.max(cs)
        cs[cs < -30] = -30
        ws = ensemble.weight(cs)
        print(ensemble.Neff(ws))
        if ensemble.Neff(ws) < Nth:
            xs = ensemble.merge_resampling(ws, xs, n=10)
            cs = np.zeros_like(cs)
        return xs, cs
    return update
