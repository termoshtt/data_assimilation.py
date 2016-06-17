# -*- coding: utf-8 -*-

import numpy as np
from . import LETKF, observation
from .ensemble import make_ensemble
from unittest import TestCase


class TestLETKF(TestCase):

    def test_observation(self):
        N = 10
        K = 5
        p = 1
        yG = np.arange(N)
        YbG = make_ensemble(N, K, 1)
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
        Xb = make_ensemble(N, K, 1)
        A = LETKF.analysis(H, np.identity(L), p)
        xa, Xa = A(xb, Xb, H(xb))
