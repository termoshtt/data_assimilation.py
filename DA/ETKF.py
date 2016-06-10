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


def make_ensemble(N, K, noise_intensity):
    """ Create ensemble with zero mean

    Examples
    ---------
    >>> Xa = make_ensemble(10, 5, 1)
    >>> Xa.shape
    (10, 5)
    >>> np.allclose(np.average(Xa, axis=1), np.zeros(10))
    True
    """
    xs = np.random.normal(size=(K, N))
    return (xs - np.average(xs, axis=0)).T


def forcast(teo):
    def update(xa, Xa):
        xs = np.array([xa + dxa for dxa in Xa.T])
        for i, x in enumerate(xs):
            xs[i] = teo(x)
        xb = np.average(xs, axis=0)
        Xb = np.array([x - xb for x in xs]).T
        return xb, Xb
    return update


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
