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

    def test_identity(self):
        x = normal(size=self.N)
        xm = normal(size=self.N)
        y = normal(size=self.p)

        Q_inv = inv(self.Q)
        R_inv = inv(self.R)
        M_inv = Kalman.M_inv(self.H, self.Q, self.R)
        V_inv = Kalman.V_inv(self.H, self.Q, self.R)
        K = Kalman.gain_matrix(self.H, self.Q, V_inv)

        a = xm + np.dot(K, y-np.dot(self.H, xm))

        np.testing.assert_almost_equal(
            linalg.quad(x-xm, Q_inv) + linalg.quad(y-np.dot(self.H, x), R_inv),
            linalg.quad(x-a, M_inv) + linalg.quad(y-np.dot(self.H, xm), V_inv),
        )
