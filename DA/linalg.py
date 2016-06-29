# -*- coding: utf-8 -*-

import numpy as np


def norm(r, A):
    """
    :math:`\| r \|_A = (r, Ar)`
    """
    return np.dot(r, np.dot(A, r))


def dot3(A, B, C):
    return np.dot(A, np.dot(B, C))


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
