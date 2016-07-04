# -*- coding: utf-8 -*-

import numpy as np


def Lorenz96(F):
    def f(x):
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
    return f


def Lorenz63(p, r, b):
    def f(x):
        return np.array([p*(x[1]-x[0]), x[0]*(r-x[2])-x[1], x[0]*x[1]-b*x[2]])
    return f


def RK4(f, dt, T=1):
    def teo(x):
        k1 = dt*f(x)
        k2 = dt*f(x+k1/2)
        k3 = dt*f(x+k2/2)
        k4 = dt*f(x+k3)
        return x + (k1+2*k2+2*k3+k4)/6

    if T == 1:
        return teo

    def repeat(x):
        for t in range(T):
            x = teo(x)
        return x
    return repeat
