# -*- coding: utf-8 -*-

import numpy as np
from .model import RK4, Lorenz96
from . import ensemble


def make_init(N, F, dt, T):
    U = RK4(Lorenz96(F), dt)
    x = np.sin(np.arange(0, np.pi, np.pi/N))
    for t in range(T):
        x = U(x)
    return x


def assimilation(F, dt, A, obs, K, T, init_noise=1):
    U = RK4(Lorenz96(F), dt)
    F = ensemble.forcast_deviations(U)

    def da(x):
        xa = x.copy()
        Xa = ensemble.make_ensemble(len(x), K, init_noise)
        for t in range(T):
            x = U(x)
            xb, Xb = F(xa, Xa)
            xa, Xa = A(xb, Xb, obs(x))
            yield x, xa, Xa
    return da


def evaluate_rms(N, F, dt, A, obs, K, T, init_noise=1):
    x = make_init(N, F, dt, T)
    da = assimilation(F, dt, A, obs, K, T + T // 10, init_noise)(x)
    rms_sum = 0
    for _ in zip(range(T // 10), da):
        pass  # remove initial transit
    for x, xa, _ in da:
        rms_sum += np.linalg.norm(x-xa) / np.sqrt(len(x))
    return rms_sum / T
