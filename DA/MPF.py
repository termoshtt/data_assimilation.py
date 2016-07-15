# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import inv
from . import ensemble, linalg


def analysis(H, R, n=3):
    if n < 2:
        raise RuntimeError("Too small n for merge resampling: n={}".format(n))
    R_inv = inv(R)

    def update(xs, yO):
        cs = np.array([linalg.quad(yO-np.dot(H, x), R_inv)/2 for x in xs])
        ws = ensemble.weight(cs)
        return ensemble.merge_resampling(ws, xs, n)
    return update
