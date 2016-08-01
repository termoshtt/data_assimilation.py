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

    def test_scaled(self):
        A = np.random.random((5, 5))
        B, d = lyapunov.scaled(A.copy())
        for a, b, n in zip(A.T, B.T, d):
            np.testing.assert_allclose(np.linalg.norm(b), 1)
            np.testing.assert_allclose(b * n, a)

    def test_clv_forward(self):
        T = 1000
        tl = lyapunov._clv_forward(self.U, self.x, T)
        for t in range(T-1):
            now = tl[t]
            nex = tl[t+1]
            x = now["x"]
            Q = now["Q"]
            D = lyapunov.Jacobi(self.U, x)
            np.testing.assert_allclose(D(Q), np.dot(nex["Q"], nex["R"]))

    def test_clv_backward(self):
        T = 10
        tl = lyapunov._clv_forward(self.U, self.x, T)
        tl = lyapunov._clv_backward(tl)
        for t in range(T-1):
            now = tl[t]
            nex = tl[t+1]
            C = np.dot(now["Q"].T, now["V"])
            Cn = np.dot(nex["Q"].T, nex["V"])
            RC, _ = lyapunov.scaled(np.dot(nex["R"], C))
            np.testing.assert_almost_equal(RC, Cn)

            print("C", C)
            print("V", now["V"])
            print("Q", now["Q"])

            x = now["x"]
            J = lyapunov.Jacobi(self.U, x)
            JV, _ = lyapunov.scaled(J(now["V"]))
            Vn, _ = lyapunov.scaled(nex["V"])
            np.testing.assert_allclose(JV, Vn)
        raise RuntimeError()
