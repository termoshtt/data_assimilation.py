# -*- coding: utf-8 -*-

import numpy as np
from . import ETKF, observation, misc
from unittest import TestCase


class TestETKF(TestCase):

    def test_dimension(self):
        N = 40
        p = 20
        K = 10
        H = observation.head(N, p)
        xb = np.random.normal(size=N)
        Xb = np.array([xb + np.random.normal(size=N) for _ in range(K)]).T
        A = ETKF.analysis(H, np.identity(p))
        xa, Xa = A(xb, Xb, H(xb))

    def test_assimilation(self):
        N = 40
        K = 40
        H = observation.trivial(N)
        obs = observation.add_noise(H, 1)
        R = np.identity(N)
        A = ETKF.analysis(H, R)
        rms = misc.evaluate_rms(N, 8, 0.01, A, obs, K, 1000)
        self.assertLess(rms, 0.2)
