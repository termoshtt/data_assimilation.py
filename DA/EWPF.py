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


def _norm(r, A):
    return np.dot(r, np.dot(A, r))


def _dot3(A, B, C):
    return np.dot(A, np.dot(B, C))


def analysis(H, Q, R):
    V_inv = _dot3(H, Q, H.T) + R
    K = _dot3(Q, H.T, V_inv)
    M_inv = inv(Q) + _dot3(H, inv(R), H.T)
    KMK = _dot3(K, M_inv, K.T)

    def update(xs, ws, yO):
        Cs = []
        for x, w in zip(xs, ws):
            r = yO - np.dot(H, x)
            Cmin = -np.log(w) + 0.5*_norm(r, V_inv)
            Cs.append(Cmin)
        C = sorted(Cs)[-len(Cs) // 5]
        for i, (x, w) in enumerate(zip(xs, ws)):
            r = yO - np.dot(H, x)
            Cmin = -np.log(w) + 0.5*_norm(r, V_inv)
            if Cmin < C:
                A = 0.5*_norm(r, KMK)
                a = 1 - np.sqrt((C-Cmin)/A)
                ws[i] = np.exp(-C)
            else:
                a = 1
                ws[i] = np.exp(-Cmin)
            xs[i] += a*np.dot(K, r)
        ws /= np.sum(ws)
        return xs, ws
    return update
