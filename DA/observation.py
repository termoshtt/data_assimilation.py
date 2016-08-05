# -*- coding: utf-8 -*-

import numpy as np


def trivial(N):
    """ Observe all as it is. """
    return np.identity(N)


def head(N, p):
    """ Observe first p data """
    return np.eye(N, p).T


def add_noise(H, intensity):
    _, N = H.shape
    return lambda x: np.dot(H, x) + intensity*np.random.normal(size=N)
