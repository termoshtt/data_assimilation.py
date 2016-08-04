# -*- coding: utf-8 -*-

import numpy as np
from . import EKF, observation, misc, ensemble


class TestEKF(misc.TestLorenz96):

    def assimilation(self, A, obs):
        x = self.init.copy()
        xm = self.init.copy()
        P = np.identity(self.N)
        T_transit = self.T // 10
        F = EKF.forcast(self.U)
        for t in range(self.T + T_transit):
            x = self.U(x)
            xm, P = F(xm, P)
            xm, P = A(xm, P)
