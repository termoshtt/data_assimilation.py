# -*- coding: utf-8 -*-

import numpy as np
from . import LETKF, observation
from unittest import TestCase


class TestLETKF(TestCase):

    def test_dimension(self):
        N = 40
        P = 10
        L = 30
        K = 5
        H = observation.head(N, L)
        xb = np.random.normal(size=N)
        Xb = np.array([xb + np.random.normal(size=N) for _ in range(K)]).T
        A = LETKF.analysis(H, np.identity(L), P)
        xa, Xa = A(xb, Xb, H(xb))
