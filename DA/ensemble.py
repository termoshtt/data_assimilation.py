# -*- coding: utf-8 -*-

import numpy as np


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
    xs = noise_intensity*np.random.normal(size=(K, N))
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
