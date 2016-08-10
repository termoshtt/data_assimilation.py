# -*- coding: utf-8 -*-

import numpy as np
import itertools
from . import ensemble, linalg


def random_covar(N):
    """
    Generate covariance matrix

    Examples
    ---------
    >>> P = random_covar(5)
    >>> np.testing.assert_allclose(P.trace(), 1.0)  # trace is normalized
    >>> np.testing.assert_allclose(P, P.T)
    """
    P = np.random.normal(size=(N, N))
    P = np.dot(P.T, P)
    return P / P.trace()


def pdf(mu, P_inv):
    """
    Examples
    ---------
    >>> N = 5
    >>> P = random_covar(N)
    >>> xp = np.random.normal(size=N)
    >>> p = pdf(xp, P)
    >>> px = p(xp+np.random.normal(size=N))
    >>> isinstance(px, float)
    True
    >>> px > 0
    True
    """
    k = len(mu)
    N = np.sqrt(np.linalg.det(P_inv) / (2*np.pi)**(-k))

    def p(x):
        return N*np.exp(-linalg.quad(x-mu, P_inv)/2)
    return p


def generator(mu, P):
    while True:
        yield np.random.multivariate_normal(mu, P)


def MonteCarlo(mu, P):
    def eval_mc(f, M):
        rands = itertools.islice(generator(mu, P), M)
        return sum(f(x) for x in rands) / M
    return eval_mc


def KL_div(P_pre_inv, P_post, dx):
    """
    KL-divergence between two multi-Gaussians

    Parameters
    -----------
    P_pre_inv : np.array(2d)
        Inverse of covariance matrix of prior distribution
    P_post : np.array(2d)
        Covariance matrix of posterior distribution
    dx : np.array(1d)
        Difference of two centers
    """
    N = dx.shape
    QP = np.dot(P_pre_inv, P_post)
    return (
        -np.log(np.linalg.det(QP)) - N
        + QP.trace() + linalg.quad(dx, P_pre_inv)
    ) / 2


def KL_div_approx(xs_pre, xs_post):
    """
    KL-divergence between two ensembles under Gaussian approximation
    """
    xm_pre = ensemble.average(xs_pre)
    P_pre_inv = np.linalg.inv(ensemble.covar(xs_pre))
    xm_post = ensemble.average(xs_post)
    P_post = ensemble.covar(xs_post)
    return KL_div(P_pre_inv, P_post, xm_post - xm_pre)
