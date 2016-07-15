# -*- coding: utf-8 -*-

import numpy as np
from . import MPF, misc


class TestMPF(misc.TestLorenz63):

    def setUp(self):
        super().setUp(p=10, r=28, b=8./3., dt=0.01, T0=10, T=100, K=500)

    def test_small_n(self):
        H = np.identity(3)
        R = np.identity(3)
        with self.assertRaises(RuntimeError):
            MPF.analysis(H, R, n=1)

    def test_assimilation(self):
        H = np.identity(3)
        R = np.identity(3)

        def obs(x):
            return np.dot(H, x) + np.random.normal(size=3)

        A = MPF.analysis(H, R)
        rms = self.eval_rms(A, obs)
        self.assertLess(rms, 0.5)
