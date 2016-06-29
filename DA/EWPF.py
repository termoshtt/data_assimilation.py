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


def analysis(H, Q, R, Nth):
    V_inv = _dot3(H, Q, H.T) + R
    K = _dot3(Q, H.T, V_inv)
    M_inv = inv(Q) + _dot3(H, inv(R), H.T)
    KMK = _dot3(K, M_inv, K.T)

    def update(xs, cs, yO):
        Cs = []
        for x, c in zip(xs, cs):
            r = yO - np.dot(H, x)
            Cmin = c + 0.5*_norm(r, V_inv)
            Cs.append(Cmin)
        C = sorted(Cs)[len(Cs) // 5]
        for i, (x, c) in enumerate(zip(xs, cs)):
            r = yO - np.dot(H, x)
            Cmin = c + 0.5*_norm(r, V_inv)
            if Cmin < C:
                A = 0.5*_norm(r, KMK)
                a = 1 - np.sqrt((C-Cmin)/A)
                cs[i] = C
            else:
                a = 1
                cs[i] = Cmin
            xs[i] += a*np.dot(K, r)
        cs -= np.max(cs)
        cs[cs < -30] = -30
        ws = weight(cs)
        if Neff(ws) < Nth:
            cws = np.cumsum(ws)
            xs = np.array([xs[np.searchsorted(cws, np.random.random())]
                           for _ in range(len(xs))])
            cs = np.zeros_like(cs)
        return xs, cs
    return update


def weight(cs):
    ws = np.exp(-cs)
    return ws / np.sum(ws)


def Neff(ws):
    return 1. / np.sum(ws**2)
