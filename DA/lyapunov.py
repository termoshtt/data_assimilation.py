# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import solve_triangular


def Jacobi(F, x, alpha=1e-7):
    def f(dx):
        s = np.linalg.norm(dx) / alpha
        if s > 0.0:
            return (F(x+dx/s) - F(x))*s
        else:
            return np.zeros_like(dx)

    def D(V):
        if len(V.shape) == 1:
            return f(V)
        if len(V.shape) == 2:
            return np.array([f(v) for v in V.T])
        raise RuntimeError("Higher order (>=3) tensor does not support")

    return D


def scale(C):
    norms = np.array([np.linalg.norm(c) for c in C])
    for c, n in zip(C, norms):
        c /= n
    return n


def _clv_forward(U, x, T):
    tl = []
    N = len(x)
    Q = np.identity(N)
    R = np.identity(N)  # dummy
    for t in range(T):
        x = U(x)
        D = Jacobi(U, x)
        tl.append({
            "x": x,
            "Q": Q,
            "R": R,
        })
        Q, R = np.linalg.qr(D(Q))
    return tl


def _clv_backward(tl, T, T_pre, T_post):
    N = len(tl[0]["x"])
    C = np.random.random((N, N))
    D = np.zeros(N)
    count = 0
    for t, info in reversed(enumerate(tl)):
        R = info["R"]
        Q = info["Q"]
        C = solve_triangular(R, C)
        info["V"] = np.dot(Q, C)
        if T_pre < t and t < T - T_post:
            D += np.log(scale(C))
            count += 1
    return tl, np.exp(D/count)


def CLV(U, x0, T, T_pre=None, T_post=None):
    if T_pre is None:
        T_pre = T // 2
    if T_post is None:
        T_post = T // 2
    tl = _clv_forward(U, x0.copy(), T_pre + T + T_post)
    return _clv_backward(tl, T, T_pre, T_post)
