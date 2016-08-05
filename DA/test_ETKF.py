# -*- coding: utf-8 -*-

import numpy as np
from . import ETKF, observation, misc, ensemble
from unittest import TestCase


class TestETKF(TestCase):

    def test_dimension(self):
        N = 40
        p = 20
        K = 10
        H = observation.head(N, p)
        xb = np.random.normal(size=N)
        xs = ensemble.replica(xb, K, noise=1.0)
        A = ETKF.analysis(H, np.identity(p))
        xs = A(xs, np.dot(H, xb))


class TestETKF2(misc.TestLorenz96):

    def setUp(self):
        super().setUp(F=8, dt=0.01, N=40, T=1000, K=40)

    def test_assimilation(self):
        H = np.identity(self.N)
        R = np.identity(self.N)
        A = ETKF.analysis(H, R)
        obs = observation.add_noise(H, 1)
        rms = self.eval_rms(A, obs)
        self.assertLess(rms, 0.2)
