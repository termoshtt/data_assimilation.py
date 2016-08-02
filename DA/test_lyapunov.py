# -*- coding: utf-8 -*-

from . import lyapunov, model

import numpy as np
from numpy.testing import assert_allclose, assert_equal
from unittest import TestCase


class TestLyapunov(TestCase):

    def setUp(self):
        self.U = model.RK4(model.Lorenz63(p=10, r=28, b=8./3.), 0.01)
        x = np.random.random(3)
        for _ in range(5000):
            x = self.U(x)
        self.x = x

    def test_jacobi_linear(self):
        a = np.random.random(3)
        b = np.random.random(3)
        J = lyapunov.Jacobi(self.U, self.x)
        assert_allclose(J(a+b), J(a)+J(b), atol=1e-6)

    def test_jacobi_nonsquare_matrix(self):
        A = np.random.random((3, 2))
        J = lyapunov.Jacobi(self.U, self.x)
        assert_equal(J(A).shape, (3, 2))

    def test_jacobi_matrix_assoc(self):
        A = np.random.random((3, 3))
        B = np.random.random((3, 3))
        J = lyapunov.Jacobi(self.U, self.x)
        assert_allclose(np.dot(J(A), B), J(np.dot(A, B)), atol=1e-7)

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
            assert_allclose(J(Q), np.dot(Qn, Rn), atol=1e-7)

    def test_clv_backward_QR(self):
        T = 1000
        tl = lyapunov._clv_forward(self.U, self.x, T)
        tl = lyapunov._clv_backward(tl)
        for t in range(T-1):
            now = tl[t]
            nex = tl[t+1]
            C = np.dot(now["Q"].T, now["V"])
            Cn = np.dot(nex["Q"].T, nex["V"])
            Q = now["Q"]
            Qn = nex["Q"]
            Rn = nex["R"]
            V = now["V"]
            Vn = nex["V"]
            Dn = nex["D"]
            x = now["x"]
            J = lyapunov.Jacobi(self.U, x)
            assert_allclose(np.dot(J(Q), C), J(np.dot(Q, C)), atol=1e-6)
            assert_allclose(np.dot(np.dot(Qn, Rn), C), J(V), atol=1e-6)
            assert_allclose(lyapunov.rescaled(np.dot(Rn, C), Dn),
                            Cn, atol=1e-8)
            assert_allclose(lyapunov.rescaled(J(V), Dn), Vn, atol=1e-6)
