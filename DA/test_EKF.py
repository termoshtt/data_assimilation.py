# -*- coding: utf-8 -*-

import numpy as np
from unittest import TestCase

from . import EKF
from .linalg import dot3


class TestEKF(TestCase):

    def test_forcast(self):
        N = 10
        A = np.random.random((N, N))
        U = lambda x: np.dot(A, x)  # linear dynamics
        F = EKF.forcast(U)
        P = np.random.random((N, N))
        P = np.dot(P.T, P)
        x = np.random.random(N)
        _, Pn = F(x, P)
        np.testing.assert_allclose(Pn, dot3(A, P, A.T))

    def assimilation(self, A, obs):
        x = self.init.copy()
        xm = self.init.copy()
        P = np.identity(self.N)
        T_transit = self.T // 10
        F = EKF.forcast(self.U)
        for t in range(self.T + T_transit):
            x = self.U(x)
            xm, P = F(xm, P)
            xm, P = A(xm, P)
