# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import inv
from unittest import TestCase

from . import EKF, normal
from .linalg import bracket


class TestEKF(TestCase):

    def test_forcast(self):
        N = 10
        a = np.identity(N) + 0.001*np.random.normal(size=(N, N))

        def U(x): return np.dot(a, x)  # linear dynamics
        F = EKF.forcast(U)
        P = normal.random_covar(N)
        x = np.random.random(N)
        _, Pn = F(x, P)
        np.testing.assert_allclose(Pn, bracket(P, a.T), rtol=5e-6)

    def test_riccati(self):
        N = 5
        R = normal.random_covar(N)
        H = np.random.normal(size=(N, N))
        Omg = bracket(inv(R), H)
        a = np.identity(N) + 0.001*np.random.normal(size=(N, N))

        def U(x): return np.dot(a, x)  # linear dynamics
        F = EKF.forcast(U)
        A = EKF.analysis(H, R)

        x = np.random.random(N)
        P = normal.random_covar(N)
        J = inv(P)
        Jr = bracket(J+Omg, inv(a))

        x, P = A(x, P, np.dot(H, x))
        x, P = F(x, P)
        Jn = inv(P)

        np.testing.assert_allclose(Jn, Jr, 1e-4)
