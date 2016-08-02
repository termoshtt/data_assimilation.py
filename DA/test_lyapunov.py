# -*- coding: utf-8 -*-

from . import lyapunov, model

import numpy as np
from unittest import TestCase


class TestLyapunov(TestCase):

    def setUp(self):
        self.U = model.RK4(model.Lorenz63(p=10, r=28, b=8./3.), 0.01)
        x = np.random.random(3)
        for _ in range(5000):
            x = self.U(x)
        self.x = x

    def test_clv_forward(self):
        T = 1000
        tl = lyapunov._clv_forward(self.U, self.x, T)
        for t in range(T-1):
            now = tl[t]
            nex = tl[t+1]
            x = now["x"]
            Q = now["Q"]
            Qn = nex["Q"]
            Rn = nex["R"]
            J = lyapunov.Jacobi(self.U, x)
            np.testing.assert_allclose(J(Q), np.dot(Qn, Rn))

    def test_clv_backward_C(self):
        """ :math:`R_nC_{n-1}D_n = C_n` """
        T = 1000
        tl = lyapunov._clv_forward(self.U, self.x, T)
        tl = lyapunov._clv_backward(tl)
        for t in range(T-1):
            now = tl[t]
            nex = tl[t+1]
            C = np.dot(now["Q"].T, now["V"])
            Cn = np.dot(nex["Q"].T, nex["V"])
            Rn = nex["R"]
            D = nex["D"]
            RCD = lyapunov.rescaled(np.dot(Rn, C), D)
            np.testing.assert_almost_equal(RCD, Cn)

    def test_clv_backward_V(self):
        T = 10
        tl = lyapunov._clv_forward(self.U, self.x, T)
        tl = lyapunov._clv_backward(tl)
        for t in range(T-1):
            now = tl[t]
            nex = tl[t+1]
            x = now["x"]
            J = lyapunov.Jacobi(self.U, x)
            V = now["V"]
            Vn = nex["V"]
            D = nex["D"]
            JVD = lyapunov.rescaled(J(V), D)
            np.testing.assert_allclose(JVD, Vn)
