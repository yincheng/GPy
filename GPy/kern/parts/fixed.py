# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from kernpart import Kernpart
import numpy as np
import hashlib

class Fixed(Kernpart):
    def __init__(self,D,X,K,variance=1.):
        """
        :param D: the number of input dimensions
        :type D: int
        :param variance: the variance of the kernel
        :type variance: float
        """
        self.D = D
        self.fixed_K = K
        self.num_params = 1
        self.name = 'fixed'
        self._set_params(np.array([variance]).flatten())
        self._X = X
        assert self._X.shape[0] is self.fixed_K.shape[0]

    def _get_params(self):
        return self.variance

    def _set_params(self,x):
        assert x.shape==(1,)
        self.variance = x

    def _get_param_names(self):
        return ['variance']

    def K(self,X,X2,target):
        if np.all(X == self._X) and X2 is None:
            target += self.variance * self.fixed_K

    def Kdiag(self,X,target):
        if np.all(X == self._X):
            target += self.variance * np.diag(self.fixed_K)

    def dK_dtheta(self,partial,X,X2,target):
        if np.all(X == self._X):# and X2 is None:
            target += (partial * self.fixed_K).sum()

    def dK_dX(self, partial,X, X2, target):
        pass

    def dKdiag_dX(self,partial,X,target):
        pass
