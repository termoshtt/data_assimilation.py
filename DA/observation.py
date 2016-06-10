# -*- coding: utf-8 -*-

from scipy.sparse.linalg import LinearOperator


def trivial(N):
    """ Observe all as it is. """
    return LinearOperator((N, N), matvec=lambda x: x)


def head(N, p):
    """ Observe first p data """
    return LinearOperator((p, N), matvec=lambda x: x[:p])
