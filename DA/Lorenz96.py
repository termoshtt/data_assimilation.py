# -*- coding: utf-8 -*-


import numpy as np


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


def teo(F, T, n_step):
    dt = T / n_step
    U = Lorenz96_RK4(F, dt)

    def f(x):
        for _ in range(n_step):
            x = U(x)
        return x
    return f
