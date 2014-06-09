# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from kernpart import Kernpart
import numpy as np
from ...util.linalg import tdot
from ...util.misc import fast_array_equal
from ...util.config import *
from ..cython import kernels as c_kernels

class RBF(Kernpart):
    """
    Radial Basis Function kernel, aka squared-exponential, exponentiated quadratic or Gaussian kernel:

    .. math::

       k(r) = \sigma^2 \exp \\bigg(- \\frac{1}{2} r^2 \\bigg) \ \ \ \ \  \\text{ where  } r^2 = \sum_{i=1}^d \\frac{ (x_i-x^\prime_i)^2}{\ell_i^2}

    where \ell_i is the lengthscale, \sigma^2 the variance and d the dimensionality of the input.

    :param input_dim: the number of input dimensions
    :type input_dim: int
    :param variance: the variance of the kernel
    :type variance: float
    :param lengthscale: the vector of lengthscale of the kernel
    :type lengthscale: array or list of the appropriate size (or float if there is only one lengthscale parameter)
    :param ARD: Auto Relevance Determination. If equal to "False", the kernel is isotropic (ie. one single lengthscale parameter \ell), otherwise there is one lengthscale parameter per dimension.
    :type ARD: Boolean
    :rtype: kernel object

    .. Note: this object implements both the ARD and 'spherical' version of the function
    """

    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False):
        self.input_dim = input_dim
        self.name = 'rbf'
        self.ARD = ARD
        if not ARD:
            self.num_params = 2
            if lengthscale is not None:
                lengthscale = np.asarray(lengthscale)
                assert lengthscale.size == 1, "Only one lengthscale needed for non-ARD kernel"
            else:
                lengthscale = np.ones(1)
        else:
            self.num_params = self.input_dim + 1
            if lengthscale is not None:
                lengthscale = np.asarray(lengthscale)
                assert lengthscale.size == self.input_dim, "bad number of lengthscales"
            else:
                lengthscale = np.ones(self.input_dim)

        self._set_params(np.hstack((variance, lengthscale.flatten())))

        # initialize cache
        self._Z, self._mu, self._S = np.empty(shape=(3, 1))
        self._X, self._X2, self._params = np.empty(shape=(3, 1))


    def _get_params(self):
        return np.hstack((self.variance, self.lengthscale))

    def _set_params(self, x):
        assert x.size == (self.num_params)
        self.variance = x[0]
        self.lengthscale = x[1:]
        self.lengthscale2 = np.square(self.lengthscale)
        # reset cached results
        self._X, self._X2, self._params = np.empty(shape=(3, 1))
        self._Z, self._mu, self._S = np.empty(shape=(3, 1)) # cached versions of Z,mu,S

    def _get_param_names(self):
        if self.num_params == 2:
            return ['variance', 'lengthscale']
        else:
            return ['variance'] + ['lengthscale_%i' % i for i in range(self.lengthscale.size)]

    def K(self, X, X2, target):
        self._K_computations(X, X2)
        target += self.variance * self._K_dvar

    def Kdiag(self, X, target):
        np.add(target, self.variance, target)

    def dK_dtheta(self, dL_dK, X, X2, target):
        self._K_computations(X, X2)
        target[0] += np.sum(self._K_dvar * dL_dK)
        if self.ARD:
            dvardLdK = self._K_dvar * dL_dK
            var_len3 = self.variance / np.power(self.lengthscale, 3)
            num_data, input_dim = int(X.shape[0]), int(self.input_dim)
            if X2 is not None:
                num_inducing = int(X2.shape[0])
            else:
                # save computation for the symmetrical case
                dvardLdK = dvardLdK + dvardLdK.T
                num_inducing = 0

            # [np.add(target[1+q:2+q],var_len3[q]*np.sum(dvardLdK*np.square(X[:,q][:,None]-X2[:,q][None,:])),target[1+q:2+q]) for q in range(self.input_dim)]
            c_kernels.rbf_dK_dtheta(num_data, num_inducing, input_dim, X, X2, target, dvardLdK, var_len3)
        else:
            target[1] += (self.variance / self.lengthscale) * np.sum(self._K_dvar * self._K_dist2 * dL_dK)

    def dKdiag_dtheta(self, dL_dKdiag, X, target):
        # NB: derivative of diagonal elements wrt lengthscale is 0
        target[0] += np.sum(dL_dKdiag)

    def dK_dX(self, dL_dK, X, X2, target):
        self._K_computations(X, X2)
        if X2 is None:
            _K_dist = 2*(X[:, None, :] - X[None, :, :])
        else:
            _K_dist = X[:, None, :] - X2[None, :, :] # don't cache this in _K_computations because it is high memory. If this function is being called, chances are we're not in the high memory arena.
        dK_dX = (-self.variance / self.lengthscale2) * np.transpose(self._K_dvar[:, :, np.newaxis] * _K_dist, (1, 0, 2))
        target += np.sum(dK_dX * dL_dK.T[:, :, None], 0)

    def dKdiag_dX(self, dL_dKdiag, X, target):
        pass


    #---------------------------------------#
    #             PSI statistics            #
    #---------------------------------------#

    def psi0(self, Z, mu, S, target):
        target += self.variance

    def dpsi0_dtheta(self, dL_dpsi0, Z, mu, S, target):
        target[0] += np.sum(dL_dpsi0)

    def dpsi0_dmuS(self, dL_dpsi0, Z, mu, S, target_mu, target_S):
        pass

    def psi1(self, Z, mu, S, target):
        self._psi_computations(Z, mu, S)
        target += self._psi1

    def dpsi1_dtheta(self, dL_dpsi1, Z, mu, S, target):
        self._psi_computations(Z, mu, S)
        target[0] += np.sum(dL_dpsi1 * self._psi1 / self.variance)
        d_length = self._psi1[:,:,None] * ((self._psi1_dist_sq - 1.)/(self.lengthscale*self._psi1_denom) +1./self.lengthscale)
        dpsi1_dlength = d_length * np.atleast_3d(dL_dpsi1)
        if not self.ARD:
            target[1] += dpsi1_dlength.sum()
        else:
            target[1:] += dpsi1_dlength.sum(0).sum(0)

    def dpsi1_dZ(self, dL_dpsi1, Z, mu, S, target):
        self._psi_computations(Z, mu, S)
        denominator = (self.lengthscale2 * (self._psi1_denom))
        dpsi1_dZ = -self._psi1[:, :, None] * ((self._psi1_dist / denominator))
        target += np.sum(dL_dpsi1[:, :, None] * dpsi1_dZ, 0)

    def dpsi1_dmuS(self, dL_dpsi1, Z, mu, S, target_mu, target_S):
        self._psi_computations(Z, mu, S)
        tmp = self._psi1[:, :, None] / self.lengthscale2 / self._psi1_denom
        target_mu += np.sum(dL_dpsi1[:, :, None] * tmp * self._psi1_dist, 1)
        target_S += np.sum(dL_dpsi1[:, :, None] * 0.5 * tmp * (self._psi1_dist_sq - 1), 1)

    def psi2(self, Z, mu, S, target):
        self._psi_computations(Z, mu, S)
        target += self._psi2

    def _crossterm_mu_S(self, Z, mu, S):
        # compute the crossterm expectation for K as the other kernel:
        Sigma = 1./self.lengthscale2[None,None,:] + 1./S[:,None,:] # is independent across M,
        Sigma_tilde = (self.lengthscale2[None, :] + S)
        M = (S*mu/Sigma_tilde)[:, None, :] + (self.lengthscale2[None,:]*Z)[None, :, :]/Sigma_tilde[:, None, :]
        # make sure return is [N x M x Q]
        return M, Sigma.repeat(Z.shape[0],1)

    def dpsi2_dtheta(self, dL_dpsi2, Z, mu, S, target):
        """Shape N,num_inducing,num_inducing,Ntheta"""
        self._psi_computations(Z, mu, S)
        d_var = 2.*self._psi2 / self.variance
        d_length = 2.*self._psi2[:, :, :, None] * (self._psi2_Zdist_sq * self._psi2_denom + self._psi2_mudist_sq + S[:, None, None, :] / self.lengthscale2) / (self.lengthscale * self._psi2_denom)
        target[0] += np.sum(dL_dpsi2 * d_var)
        dpsi2_dlength = d_length * dL_dpsi2[:, :, :, None]
        if not self.ARD:
            target[1] += dpsi2_dlength.sum()
        else:
            target[1:] += dpsi2_dlength.sum(0).sum(0).sum(0)

    def dpsi2_dZ(self, dL_dpsi2, Z, mu, S, target):
        self._psi_computations(Z, mu, S)
        term1 = self._psi2_Zdist / self.lengthscale2 # num_inducing, num_inducing, input_dim
        term2 = self._psi2_mudist / self._psi2_denom / self.lengthscale2 # N, num_inducing, num_inducing, input_dim
        dZ = self._psi2[:, :, :, None] * (term1[None] + term2)
        target += (dL_dpsi2[:, :, :, None] * dZ).sum(0).sum(0)

    def dpsi2_dmuS(self, dL_dpsi2, Z, mu, S, target_mu, target_S):
        """Think N,num_inducing,num_inducing,input_dim """
        self._psi_computations(Z, mu, S)
        tmp = self._psi2[:, :, :, None] / self.lengthscale2 / self._psi2_denom
        target_mu += -2.*(dL_dpsi2[:, :, :, None] * tmp * self._psi2_mudist).sum(1).sum(1)
        target_S += (dL_dpsi2[:, :, :, None] * tmp * (2.*self._psi2_mudist_sq - 1)).sum(1).sum(1)

    #---------------------------------------#
    #            Precomputations            #
    #---------------------------------------#

    def _K_computations(self, X, X2):
        params = self._get_params()
        if not (fast_array_equal(X, self._X) and fast_array_equal(X2, self._X2) and fast_array_equal(self._params , params)):
            self._X = X.copy()
            self._params = params.copy()
            if X2 is None:
                self._X2 = None
                X = X / self.lengthscale
                Xsquare = np.sum(np.square(X), 1)
                self._K_dist2 = -2.*tdot(X) + (Xsquare[:, None] + Xsquare[None, :])
            else:
                self._X2 = X2.copy()
                X = X / self.lengthscale
                X2 = X2 / self.lengthscale
                self._K_dist2 = -2.*np.dot(X, X2.T) + (np.sum(np.square(X), 1)[:, None] + np.sum(np.square(X2), 1)[None, :])
            self._K_dvar = np.exp(-0.5 * self._K_dist2)

    def _psi_computations(self, Z, mu, S):
        # here are the "statistics" for psi1 and psi2
        Z_changed = not fast_array_equal(Z, self._Z)
        if Z_changed:
            # Z has changed, compute Z specific stuff
            self._psi2_Zhat = 0.5 * (Z[:, None, :] + Z[None, :, :]) # M,M,Q
            self._psi2_Zdist = 0.5 * (Z[:, None, :] - Z[None, :, :]) # M,M,Q
            self._psi2_Zdist_sq = np.square(self._psi2_Zdist / self.lengthscale) # M,M,Q

        if Z_changed or not fast_array_equal(mu, self._mu) or not fast_array_equal(S, self._S):
            # something's changed. recompute EVERYTHING

            # psi1
            self._psi1_denom = S[:, None, :] / self.lengthscale2 + 1.
            self._psi1_dist = Z[None, :, :] - mu[:, None, :]
            self._psi1_dist_sq = np.square(self._psi1_dist) / self.lengthscale2 / self._psi1_denom
            self._psi1_exponent = -0.5 * np.sum(self._psi1_dist_sq + np.log(self._psi1_denom), -1)
            self._psi1 = self.variance * np.exp(self._psi1_exponent)

            # psi2
            self._psi2_denom = 2.*S[:, None, None, :] / self.lengthscale2 + 1. # N,M,M,Q
            self._psi2_mudist, self._psi2_mudist_sq, self._psi2_exponent, _ = self.weave_psi2(mu, self._psi2_Zhat)
            # self._psi2_mudist = mu[:,None,None,:]-self._psi2_Zhat #N,M,M,Q
            # self._psi2_mudist_sq = np.square(self._psi2_mudist)/(self.lengthscale2*self._psi2_denom)
            # self._psi2_exponent = np.sum(-self._psi2_Zdist_sq -self._psi2_mudist_sq -0.5*np.log(self._psi2_denom),-1) #N,M,M,Q
            self._psi2 = np.square(self.variance) * np.exp(self._psi2_exponent) # N,M,M,Q

            # store matrices for caching
            self._Z, self._mu, self._S = Z, mu, S

    def weave_psi2(self, mu, Zhat):
        N, input_dim = mu.shape
        num_inducing = Zhat.shape[0]

        mudist = np.empty((N, num_inducing, num_inducing, input_dim))
        mudist_sq = np.empty((N, num_inducing, num_inducing, input_dim))
        psi2_exponent = np.zeros((N, num_inducing, num_inducing))
        psi2 = np.empty((N, num_inducing, num_inducing))

        psi2_Zdist_sq = self._psi2_Zdist_sq
        _psi2_denom = self._psi2_denom.squeeze().reshape(-1, input_dim)
        half_log_psi2_denom = 0.5 * np.log(self._psi2_denom).squeeze().reshape(-1, input_dim)
        # variance_sq = float(np.square(self.variance))
        if self.ARD:
            lengthscale2 = self.lengthscale2
        else:
            lengthscale2 = np.ones(input_dim) * self.lengthscale2

        N, num_inducing, input_dim = int(N), int(num_inducing), int(input_dim)
        c_kernels.rbf_psi2(N, num_inducing, input_dim, mu, Zhat, mudist_sq,
                           mudist, lengthscale2, _psi2_denom, psi2_Zdist_sq,
                           psi2_exponent, half_log_psi2_denom, psi2)

        return mudist, mudist_sq, psi2_exponent, psi2
