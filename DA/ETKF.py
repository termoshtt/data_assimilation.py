# -*- coding: utf-8 -*-

"""
Ensamble Trasnform Kalman Filter

Notations
----------
N : int
    length of state vector
p : int
    length of observation vector
K : int
    Number of ensembles

H : scipy.sparse.linalg.LinearOperator, (N) -> (p)
    Obvservation operator. **Assume Linear**.

"""

import numpy as np
from .linalg import symmetric_square_root
from scipy.sparse.linalg import LinearOperator


def H_trivial(N):
    """ Observe all as it is. """
    return LinearOperator((N, N), matvec=lambda x: x)


def H_head(N, p):
    """ Observe first p data """
    return LinearOperator((p, N), matvec=lambda x: x[:p])


def analysis(H, R_inv):
    def update(xb, Xb, yO):
        _, k = Xb.shape
        yb = H(xb)
        Yb = H(Xb)
        YR = np.dot(Yb.T, R_inv)
        Pa = np.linalg.inv(np.dot(YR, Yb) + (k-1)*np.identity(k))
        wa = np.dot(Pa, np.dot(YR, yO - yb))
        Wa = symmetric_square_root((k-1)*Pa)
        return xb + np.dot(Xb, wa), np.dot(Xb, Wa)
    return update
