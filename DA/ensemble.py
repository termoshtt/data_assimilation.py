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


def forcast_ensemble(teo):
    def update(xs):
        for i, x in enumerate(xs):
            xs[i] = teo(x)
        return xs
    return update


def deviations(xs):
    xb = np.average(xs, axis=0)
    Xb = np.array([x - xb for x in xs]).T
    return xb, Xb


def reconstruct(xb, Xb):
    return np.array([xb + dxb for dxb in Xb.T])


def forcast_deviations(teo):
    U = forcast_ensemble(teo)

    def update(xb, Xb):
        xs = reconstruct(xb, Xb)
        xs = U(xs)
        return deviations(xs)
    return update


def resampling(ws, xs):
    cws = np.cumsum(ws)
    return np.array([xs[np.searchsorted(cws, np.random.random())]
                     for _ in range(len(xs))])


def merge_resampling(ws, xs, n=3):
    cws = np.cumsum(ws)
    return np.array([np.average([xs[np.searchsorted(cws, np.random.random())]
                                 for _ in range(n)], axis=0)
                     for _ in range(len(xs))])


def Neff(ws):
    return 1. / np.sum(ws**2)


def weight(cs):
    ws = np.exp(-cs)
    return ws / np.sum(ws)
