# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import inv
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

    def test_mc_covar(self):
        N = 2
        M = 10000
        P = normal.random_covar(N)
        xp = np.zeros(N)
        mc = normal.MonteCarlo(xp, P)
        Pm = (M / (M-1)) * mc(lambda x: np.outer(x, x), M)
        np.testing.assert_allclose(Pm, P, atol=10/np.sqrt(M))

    def test_kl_div_zero(self):
        N = 11
        P = normal.random_covar(N)
        Qinv = np.linalg.inv(P)
        dx = np.zeros(N)
        kl = normal.KL_div(Qinv, P, dx)
        np.testing.assert_allclose(kl, 0.0, atol=1e-13)

    def test_kl_div_exact(self):
        N = 3
        xp = np.zeros(N)
        P = np.identity(N)
        xq = np.ones(N)
        Q = np.identity(N)
        D = normal.KL_div(inv(Q), P, xp-xq)
        np.testing.assert_allclose(D, 0.5*N)

    def test_kl_div_approx(self):
        N = 2
        xp = np.random.normal(size=N)
        P = normal.random_covar(N)
        xq = np.random.normal(size=N)
        Q = normal.random_covar(N)
        D = normal.KL_div(inv(Q), P, xp-xq)

        M = 10000
        xsp = normal.gen_ensemble(xp, P, M)
        xsq = normal.gen_ensemble(xq, Q, M)
        D_approx = normal.KL_div_approx(xsq, xsp)

        np.testing.assert_allclose(D_approx, D, rtol=10/np.sqrt(M))
