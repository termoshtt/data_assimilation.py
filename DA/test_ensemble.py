# -*- coding: utf-8 -*-

import numpy as np
from . import ensemble, normal
from unittest import TestCase


class TestResampling(TestCase):

    def same_gaussian(self, method):
        N = 2
        K = 10000
        x = np.random.normal(size=N)
        xs = ensemble.replica(x, K)
        ws = np.ones(K) / K  # reproduce same pdf
        xs = method(ws, xs)
        np.testing.assert_allclose(ensemble.average(xs), x, atol=10/np.sqrt(K))
        np.testing.assert_allclose(
                ensemble.covar(xs), np.identity(N), atol=10/np.sqrt(K))

    def gaussian(self, method):
        N = 2
        K = 10000
        x = np.array([1, 0])
        xs = ensemble.replica(x, K)
        ws = np.array([np.exp(-np.dot(x, x)/2) for x in xs])
        ws /= ws.sum()
        xs = method(ws, xs)
        np.testing.assert_allclose(
                ensemble.average(xs), x/2, atol=10/np.sqrt(K))
        np.testing.assert_allclose(
                ensemble.covar(xs), np.identity(N)/2, atol=10/np.sqrt(K))

    def test_importance_sampling_same(self):
        self.same_gaussian(ensemble.importance_sampling)
        self.gaussian(ensemble.importance_sampling)

    def test_merge_resampling_same(self):
        self.same_gaussian(ensemble.merge_resampling)
        self.gaussian(ensemble.merge_resampling)

    def test_kl_div_hist_exact(self):
        N = 2
        K = 10000
        xp = np.zeros(N)
        P = np.identity(N)
        xsp = normal.gen_ensemble(xp, P, K)
        xq = np.ones(N)
        Q = np.identity(N)
        xsq = normal.gen_ensemble(xq, Q, K)
        D_hist = ensemble.KL_div_hist(xsp, xsq)
        np.testing.assert_allclose(D_hist, 0.5*N, rtol=0.5)

    def test_kl_div_hist_random(self):
        N = 2
        K = 100000
        xp = np.random.normal(size=N)
        P = normal.random_covar(N)
        xsp = normal.gen_ensemble(xp, P, K)
        xq = np.random.normal(size=N)
        Q = normal.random_covar(N)
        xsq = normal.gen_ensemble(xq, Q, K)
        D_hist = ensemble.KL_div_hist(xsp, xsq)
        D = normal.KL_div(P, Q, xp-xq)
        np.testing.assert_allclose(D_hist, D, rtol=0.5)

    def test_gaussian_non_gaussianity(self):
        N = 2
        K = 10000
        mu = np.random.normal(size=N)
        P = normal.random_covar(N)
        xs = normal.gen_ensemble(mu, P, K)
        D = ensemble.non_gaussianity(xs)
        np.testing.assert_allclose(D, 0.0, atol=K**-0.5)
