# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from .model import RK4, Lorenz96, Lorenz63
from . import ensemble
from unittest import TestCase


def generate_l63_timeseries(dt, T, x0):
    U = RK4(Lorenz63(p=10., r=28., b=8./3.), dt=dt)

    def iterate(x, T):
        for t in range(T):
            yield dt*t, x
            x = U(x)

    return pd.DataFrame(
        [(t, x[0], x[1], x[2]) for (t, x) in
         iterate(np.array([1.0, 0, 0]), 10000)],
        columns=["time", "X", "Y", "Z"]
    ).set_index("time")


class _TestEnsembleDA(TestCase):

    def setUp(self, U, N, T, K, init_noise=1):
        self.N = N
        self.K = K
        self.T = T
        self.init_noise = init_noise

        self.U = U
        self.F = ensemble.forcast(self.U)

        x = np.sin(np.arange(0, np.pi, np.pi/N))
        for t in range(T):
            x = self.U(x)
        self.init = x

    def assimilation(self, A, obs):
        x = self.init.copy()
        xa = self.init.copy()
        xs = ensemble.replica(xa, self.K, noise=self.init_noise)
        for t in range(self.T + self.T // 10):
            x = self.U(x)
            xs = self.F(xs)
            xs = A(xs, obs(x))
            yield x, xs

    def eval_rms(self, A, obs):
        x = self.init.copy()
        da = self.assimilation(A, obs)
        rms_sum = 0
        for _ in zip(range(self.T // 10), da):
            pass  # remove initial transit
        for x, xs in da:
            xa = np.average(xs, axis=0)
            rms_sum += np.linalg.norm(x-xa) / np.sqrt(len(x))
        return rms_sum / self.T


class TestLorenz96(_TestEnsembleDA):

    def setUp(self, F, dt, N, T, K, init_noise=1):
        super().setUp(RK4(Lorenz96(F), dt), N, T, K, init_noise)


class TestLorenz63(_TestEnsembleDA):

    def setUp(self, p, r, b, dt, T0, T, K, init_noise=1):
        super().setUp(RK4(Lorenz63(p, r, b), dt, T0), 3, T, K, init_noise)
