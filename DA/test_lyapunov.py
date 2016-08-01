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
            D = lyapunov.Jacobi(self.U, x)
            np.testing.assert_allclose(D(Q), np.dot(nex["Q"], nex["R"]))
