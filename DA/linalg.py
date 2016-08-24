# -*- coding: utf-8 -*-

import numpy as np


def quad(r, A):
    """
    :math:`\| r \|_A = (r, Ar)`
    """
    return np.dot(r, np.dot(A, r))


def dot3(A, B, C):
    return np.einsum("ij,jk,kl->il", A, B, C)


def bracket(A, B):
    """

    Examples
    ---------
    >>> N = 5
    >>> A = np.random.random((N, N))
    >>> B = np.random.random((N, N))
    >>> C = np.random.random((N, N))
    >>> np.allclose(bracket(A+B, C), bracket(A, C) + bracket(B, C))
    True
    >>> np.allclose(bracket(bracket(A, B), C), bracket(A, np.dot(B, C)))
    True
    """
    return dot3(B.T, A, B)


def bracket_diag(A, D):
    return np.einsum("i,ij,j->ij", D, A, D)


def symmetric_square_root(A):
    """
    calc symmetric square root matrix of
    symmetric positive definite matrix using SVD

    Examples
    ---------
    >>> A = np.random.random((10, 10))
    >>> A = np.dot(A, A.T)  # symmetric positive def.
    >>> Q = symmetric_square_root(A)
    >>> np.allclose(Q, Q.T)
    True
    >>> np.allclose(np.dot(Q, Q), A)
    True
    """
    U, S, _ = np.linalg.svd(A)
    return np.dot(U*np.sqrt(S), U.T)


def curvature(x_pre, x_now, x_next):
    """
    Examples
    ---------
    On a line
    >>> x = np.array([0., 0.])
    >>> y = np.array([1., 1.])
    >>> z = np.array([2., 2.])
    >>> np.testing.assert_allclose(curvature(x, y, z), 0.)

    Unit cycle
    >>> x = np.array([np.cos(0),   np.sin(0)])
    >>> y = np.array([np.cos(0.1), np.sin(0.1)])
    >>> z = np.array([np.cos(0.2), np.sin(0.2)])
    >>> np.testing.assert_allclose(curvature(x, y, z), 1., rtol=1e-2)
    """
    p_n = x_next - x_now
    p_p = x_now - x_pre
    pp = p_n - p_p
    p_mean = (p_n + p_p) / 2
    p_norm = np.linalg.norm(p_mean)
    cross = np.sqrt((np.linalg.norm(pp)*p_norm)**2 - np.dot(pp, p_mean)**2)
    return cross / (p_norm**3)
