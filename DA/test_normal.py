# -*- coding: utf-8 -*-

import numpy as np
from unittest import TestCase
from . import normal


class TestNormal(TestCase):

    def test_mc_mean(self):
        N = 2
        M = 10000
        P = normal.random_covar(N)
        xp = np.random.normal(size=N)
        mc = normal.MonteCarlo(xp, P)
        xm = mc(lambda x: x, M)
        np.testing.assert_allclose(xm, xp, atol=10/np.sqrt(M))

    def test_kl_div(self):
        N = 2
        M = 10000
        P = normal.random_covar(N)
        Q = normal.random_covar(N)
        xp = np.zeros(N)
        xq = np.ones(N)
        p = normal.pdf(xp, P)
        q = normal.pdf(xq, Q)

        mc = normal.MonteCarlo(xp, P)
        kl_mc = mc(lambda x: np.log(p(x)/q(x)), M)
        Qinv = np.linalg.inv(Q)
        kl = normal.KL_div(Qinv, P, xp-xq)

        np.testing.assert_allclose(kl_mc, kl)
