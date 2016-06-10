# -*- coding: utf-8 -*-

import numpy as np
from . import ETKF
from unittest import TestCase


class TestETKF(TestCase):

    def test_dimension(self):
        N = 40
        p = 20
        K = 10
        H = ETKF.H_head(N, p)
        xb = np.random.normal(size=N)
        Xb = np.array([xb + np.random.normal(size=N) for _ in range(K)]).T
        A = ETKF.analysis(H, np.identity(p))
        xa, Xa = A(xb, Xb, H(xb))
