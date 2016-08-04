# -*- coding: utf-8 -*-

import numpy as np


def new(N, K, center=None, noise=1.0):
    """ Create ensemble with zero mean

    Examples
    ---------
    >>> xs = new(10, 5)
    >>> xs.shape
    (5, 10)
    >>> np.allclose(average(xs), np.zeros(10))
    True

    >>> x = np.random.random(10)
    >>> xs = new(10, 5, center=x)
    >>> xs.shape
    (5, 10)
    >>> xs[0].shape
    (10,)
    >>> np.allclose(average(xs), x)
    True
    """
    xs = np.random.normal(size=(K, N))
    xm = np.average(xs, axis=0)
    if center is None:
        return noise*(xs - xm)
    if len(center) != N:
        raise RuntimeError("Size of center mismatches")
    return center + noise*(xs - xm)


def average(xs):
    return np.average(xs, axis=0)


def deviations(xs):
    """ Get deviation vectors

    Examples
    ---------
    >>> xs = new(10, 5)
    >>> x, X = deviations(xs)
    >>> x.shape
    (10,)
    >>> np.allclose(x, np.zeros(10))
    True
    >>> X.shape
    (5, 10)
    >>> xs2 = reconstruct(x, X)
    >>> np.allclose(xs, xs2)
    True
    """
    xb = average(xs)
    return xb, xs - xb


def reconstruct(xb, Xb):
    return xb + Xb


def forcast(teo):
    def update(xs):
        for i, x in enumerate(xs):
            xs[i] = teo(x)
        return xs
    return update


def sampling(cws, xs):
    """ Get sample from `xs` by with accumulated weight `cws` """
    return xs[np.searchsorted(cws, cws[-1]*np.random.random())]


def resampling(ws, xs):
    cws = np.cumsum(ws)
    return np.array([sampling(cws, xs) for _ in range(len(xs))])


def _gen_weight(n):
    """
    Generate weight for merge-resampling

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
    if n < 2:
        raise RuntimeError("Too small n for merge resampling: n={}".format(n))
    cws = np.cumsum(ws)
    a = _gen_weight(n)
    return np.array([np.dot(a, [sampling(cws, xs) for _ in range(n)])
                     for _ in range(len(xs))])


def Neff(ws):
    """ effective number of ensembles """
    return 1. / np.sum(ws**2)


def weight(cs):
    ws = np.exp(-cs)
    return ws / np.sum(ws)
