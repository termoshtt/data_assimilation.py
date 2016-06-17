# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse.linalg import LinearOperator


def trivial(N):
    """ Observe all as it is. """
    return LinearOperator((N, N), matvec=lambda x: x)


def head(N, p):
    """ Observe first p data """
    return LinearOperator((p, N), matvec=lambda x: x[:p])


def add_noise(H, intensity):
    _, N = H.shape
    return lambda x: H(x) + intensity*np.random.normal(size=N)
