# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import inv
from . import linalg


def V_inv(H, Q, R):
    return inv(linalg.dot3(H, Q, H.T) + R)


def M_inv(H, Q, R):
    return inv(Q) + linalg.dot3(H.T, inv(R), H)


def gain_matrix(H, Q, V_inv):
    return linalg.dot3(Q, H.T, V_inv)


def analysis(H, Q, R, x, y):
    Vi = V_inv(H, Q, R)
    K = gain_matrix(H, Q, Vi)
    Q = Q - linalg.dot3(K, H, Q)
    x = x + np.dot(K, y-np.dot(H, x))
    return x, Q
