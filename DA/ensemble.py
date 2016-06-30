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


def sampling(cws, xs):
    return xs[np.searchsorted(cws, np.random.random())]


def resampling(ws, xs):
    cws = np.cumsum(ws)
    return np.array([sampling(cws, xs) for _ in range(len(xs))])


def _gen_weight(n):
    """
    Examples
    ---------
    >>> a = _gen_weight(5)
    >>> np.allclose(np.sum(a), 1.0)
    True
    >>> np.allclose(np.sum(a**2), 1.0)
    True
    """
    a = np.random.random(n)
    M1 = np.sum(a[1:])
    M2 = np.sum(a[1:]**2)
    A = 2*M1/(M1**2 + M2)
    a *= A
    a[0] = (M2-M1**2) / (M2+M1**2)
    return a


def merge_resampling(ws, xs, n=3):
    cws = np.cumsum(ws)
    a = _gen_weight(n)
    return np.array([np.dot(a, [sampling(cws, xs) for _ in range(n)])
                     for _ in range(len(xs))])


def Neff(ws):
    return 1. / np.sum(ws**2)


def weight(cs):
    ws = np.exp(-cs)
    return ws / np.sum(ws)
