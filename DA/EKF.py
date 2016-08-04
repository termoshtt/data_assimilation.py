# -*- coding: utf-8 -*-

"""
Extended Kalman Filter
"""

import numpy as np
from .lyapunov import Jacobi
from .linalg import dot3
from numpy.linalg import inv


def forcast(U):
    def f(x, P):
        J = Jacobi(U, x)
        A = J(P).T
        P = J(A).T
        return U(x), P
    return f


def analysis(H, R):
    def f(x, P, y):
        V_inv = inv(dot3(H, P, H.T) + R)
        K = dot3(P, H.T, V_inv)
        P -= dot3(K, H, P)
        x += np.dot(K, y-np.dot(H, x))
        return x, P
    return f
