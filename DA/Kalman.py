# -*- coding: utf-8 -*-

from numpy.linalg import inv
from . import linalg


def V_inv(H, Q, R):
    return inv(linalg.dot3(H, Q, H.T) + R)


def M_inv(H, Q, R):
    return inv(Q) + linalg.dot3(H.T, inv(R), H)


def gain_matrix(H, Q, V_inv):
    return linalg.dot3(Q, H.T, V_inv)
