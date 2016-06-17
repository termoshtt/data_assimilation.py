# -*- coding: utf-8 -*-


import numpy as np
from . import ensemble


def Lorenz96(F):
    def f(x):
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
    return f


def RK4(f, dt):
    def teo(x):
        k1 = dt*f(x)
        k2 = dt*f(x+k1/2)
        k3 = dt*f(x+k2/2)
        k4 = dt*f(x+k3)
        return x + (k1+2*k2+2*k3+k4)/6
    return teo


def Lorenz96_RK4(F, dt):
    return RK4(Lorenz96(F), dt)


def forcast(F, dt):
    return ensemble.forcast(Lorenz96_RK4(F, dt))


def make_init(N, F, dt, T):
    teo = Lorenz96_RK4(F, dt)
    x = np.sin(np.arange(0, np.pi, np.pi/N))
    for t in range(T):
        x = teo(x)
    return x


def assimilation(F, dt, A, obs, K, T, init_noise=1):
    U = Lorenz96_RK4(F, dt)
    F = forcast(F, dt)

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
    da = assimilation(F, dt, A, obs, K, T, init_noise)
    rms_sum = 0
    for x, xa, _ in da(x):
        rms_sum += np.linalg.norm(x-xa) / np.sqrt(len(x))
    return rms_sum / T
