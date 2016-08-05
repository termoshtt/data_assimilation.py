# -*- coding: utf-8 -*-

import numpy as np
from . import ensemble
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
        np.testing.assert_allclose(ensemble.average(xs), x/2, atol=10/np.sqrt(K))
        np.testing.assert_allclose(
                ensemble.covar(xs), np.identity(N)/2, atol=10/np.sqrt(K))

    def test_importance_sampling_same(self):
        self.same_gaussian(ensemble.importance_sampling)
        self.gaussian(ensemble.importance_sampling)

    def test_merge_resampling_same(self):
        self.same_gaussian(ensemble.merge_resampling)
        self.gaussian(ensemble.merge_resampling)
