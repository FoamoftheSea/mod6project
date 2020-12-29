# Code in this file is copied and adapted from
# https://github.com/openai/evolution-strategies-starter.

# This code has been modified by Nate Cibik to include functionality for
# learning rate decay parameter

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

# OPTIMIZERS FOR MINIMIZING OBJECTIVES
class Optimizer(object):
    def __init__(self, w_policy):
        self.w_policy = w_policy.flatten()
        self.dim = w_policy.size
        self.t = 0

    def update(self, globalg):
        self.t += 1
        step = self._compute_step(globalg)
        ratio = np.linalg.norm(step) / (np.linalg.norm(self.w_policy) + 1e-5)
        return self.w_policy + step, ratio

    def _compute_step(self, globalg):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, pi, learning_rate, lr_decay):
        Optimizer.__init__(self, pi)
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay

    def _compute_step(self, globalg):
        step = -self.learning_rate * globalg
        self.learning_rate *= (1 - self.lr_decay)
        if self.lr_decay != 0:
            print('New learning rate:', self.learning_rate)
        return step

