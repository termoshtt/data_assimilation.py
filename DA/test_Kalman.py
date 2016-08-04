# -*- coding: utf-8 -*-

import numpy as np
from . import Kalman, linalg
from numpy.random import normal
from numpy.linalg import inv
from unittest import TestCase


class TestKalman(TestCase):

    def setUp(self):
        self.N = 10
        self.p = 5
        Q = normal(size=(self.N, self.N))
        self.Q = np.dot(Q, Q.T)
        R = normal(size=(self.p, self.p))
        self.R = np.dot(R, R.T)
        self.H = np.eye(self.p, self.N)

    def test_analysis(self):
        x = normal(size=self.N)   # truth
        xm = normal(size=self.N)  # pre-estimate
        y = normal(size=self.p)   # observation

        Q_inv = inv(self.Q)
        R_inv = inv(self.R)
        M_inv = Kalman.M_inv(self.H, self.Q, self.R)
        V_inv = Kalman.V_inv(self.H, self.Q, self.R)

        a, Qn = Kalman.analysis(self.H, self.Q.copy(), self.R, xm, y)

        np.testing.assert_allclose(inv(Qn), M_inv)

        np.testing.assert_almost_equal(
            linalg.quad(x-xm, Q_inv) + linalg.quad(y-np.dot(self.H, x), R_inv),
            linalg.quad(x-a, M_inv) + linalg.quad(y-np.dot(self.H, xm), V_inv),
        )
