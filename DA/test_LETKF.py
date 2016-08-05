# -*- coding: utf-8 -*-

import numpy as np
from . import LETKF, observation, misc, ensemble
from unittest import TestCase


class TestLETKF(TestCase):

    def test_observation(self):
        N = 10
        K = 5
        p = 1
        yG = np.arange(N)
        YbG = (ensemble.replica(yG, K, 1) - yG).T
        RG = np.diag(1+yG)
        f = LETKF.observation(yG, YbG, RG, p)
        y, Yb, R = f(5)
        np.testing.assert_equal(y, np.array([4, 5, 6]))
        np.testing.assert_equal(Yb, YbG[4:7])
        np.testing.assert_equal(R, np.diag([5, 6, 7]))

    def test_dimension(self):
        N = 40
        p = 3
        L = 30
        K = 5
        H = observation.head(N, L)
        xb = np.random.normal(size=N)
        xs = ensemble.replica(xb, K, 1)
        A = LETKF.analysis(H, np.identity(L), p)
        xs = A(xs, np.dot(H, xb))


class TestLETKF2(misc.TestLorenz96):

    def setUp(self):
        super().setUp(F=8, dt=0.01, N=40, T=1000, K=8)

    def test_assimilation(self):
        H = observation.trivial(self.N)
        obs = observation.add_noise(H, 1)
        R = np.identity(self.N)
        A = LETKF.analysis(H, R, p=6, rho=1.1)
        rms = self.eval_rms(A, obs)
        self.assertLess(rms, 0.2)
