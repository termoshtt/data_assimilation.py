# -*- coding: utf-8 -*-

"""
Extended Kalman Filter
"""

from . import lyapunov, Kalman


def forcast(U):
    def f(x, P):
        J = lyapunov.Jacobi(U, x)
        A = J(P).T
        P = J(A).T
        return U(x), P
    return f


def analysis(H, R):
    def f(x, P, y):
        return Kalman.analysis(H, P, R, x, y)
    return f
