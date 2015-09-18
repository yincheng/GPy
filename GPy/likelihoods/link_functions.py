# Copyright (c) 2012-2014 The GPy authors (see AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from IPython.core.debugger import Tracer

import numpy as np
from scipy import stats
from scipy.misc import logsumexp
import scipy as sp
from scipy.stats import beta
from scipy.stats import laplace
from GPy.util.univariate_Gaussian import std_norm_pdf,std_norm_cdf,inv_std_norm_cdf
from scipy.optimize import brentq

_exp_lim_val = np.finfo(np.float64).max
_lim_val = np.log(_exp_lim_val)

class GPTransformation(object):
    """
    Link function class for doing non-Gaussian likelihoods approximation

    :param Y: observed output (Nx1 numpy.darray)

    .. note:: Y values allowed depend on the likelihood_function used

    """
    def __init__(self):
        pass

    def transf(self,f):
        """
        Gaussian process tranformation function, latent space -> output space
        """
        raise NotImplementedError

    def dtransf_df(self,f):
        """
        derivative of transf(f) w.r.t. f
        """
        raise NotImplementedError

    def d2transf_df2(self,f):
        """
        second derivative of transf(f) w.r.t. f
        """
        raise NotImplementedError

    def d3transf_df3(self,f):
        """
        third derivative of transf(f) w.r.t. f
        """
        raise NotImplementedError

class MarginalDistribution(GPTransformation):
    def __init__(self, params):
        raise NotImplementedError, "This function is not implemented!"
    def get_param_vec(self):
        raise NotImplementedError, "This function is not implemented!"
    def set_param_vec(self, params):
        raise NotImplementedError, "This function is not implemented!"
    def pdf(self, f):
        raise NotImplementedError, "This function is not implemented!"
    def cdf(self, f):
        raise NotImplementedError, "This function is not implemented!"
    def quantile(self, f):
        raise NotImplementedError, "This function is not implemented!"
    def dcdf_dtheta(self, f):
        raise NotImplementedError, "This function is not implemented!"

    def dquantile_dcdf(self, f):
        output = 1./self.pdf(self.quantile(f))
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output

    def dquantile_dtheta(self, f):
        output = self.dquantile_dcdf(f) * self.dcdf_dtheta(self.quantile(f))
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output

#Begin: Laplace
class LaplaceDistribution(MarginalDistribution):
    def __init__(self, params):
        self.mu = 0.
        self.b = 0.
        self.laplace_obj = None
        self.param_vec = np.array([self.mu, self.b])
        self.set_param_vec(params)

    def get_param_vec(self):
        return self.param_vec

    def set_param_vec(self, params):
        assert len(params) == 2, "Laplace Marginal Distribution requires exactly 2 parameters: np.array([mu, b])"
        self.mu = params[0]
        self.b = params[1]
        self.param_vec[0] = self.mu
        self.param_vec[1] = self.b
        assert self.b>0., "b cannot be <=0."
        self.laplace_obj = laplace(self.mu, self.b)
        return True

    def pdf(self, f):
        output = self.laplace_obj.pdf(f)
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output

    def cdf(self, f):
        output = self.laplace_obj.cdf(f)
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output

    def quantile(self, f):
        assert not(np.any(f>1.)), 'Input to quantile needs to be in <=1.'
        assert not(np.any(f<0.)), 'Input to quantile needs to be in >=0.'
        f = np.clip(f, 1.e-323, 1.-1.e-16)
        output = self.laplace_obj.ppf(f)
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output

    def dcdf_dtheta(self, f):
        raise NotImplementedError, "This function is not implemented!"
        assert not(np.isnan(output).any().any()), 'function returns NaN'
        assert not(np.isinf(output).any().any()), 'function returns inf'
        return output

    def dquantile_dtheta(self, f):
        raise NotImplementedError, "This function is not implemented!"
        assert not(np.isnan(output).any().any()), 'function returns NaN'
        assert not(np.isinf(output).any().any()), 'function returns inf'
        return output
#End: Laplace

class BetaDistribution(MarginalDistribution):
    def __init__(self, params):
        self.alpha = 0.
        self.beta = 0.
        self.beta_obj = None
        self.param_vec = np.array([self.alpha, self.beta])
        self.set_param_vec(params)

    def get_param_vec(self):
        return self.param_vec

    def set_param_vec(self, params):
        assert len(params) == 2, "Beta Marginal Distribution requires exactly 2 parameters: np.array([alpha, beta])"
        self.alpha = params[0]
        self.beta = params[1]
        self.param_vec[0] = self.alpha
        self.param_vec[1] = self.beta
        assert self.alpha>0., "Alpha cannot be <=0."
        assert self.beta>0., "Beta cannot be <=0."
        self.beta_obj = beta(self.alpha, self.beta)
        return True

    def pdf(self, f):
        output = self.beta_obj.pdf(f)
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output

    def cdf(self, f):
        output = self.beta_obj.cdf(f)
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output

    def quantile(self, f):
        assert not(np.any(f>1.)), 'Input to quantile needs to be in <=1.'
        assert not(np.any(f<0.)), 'Input to quantile needs to be in >=0.'
        output = self.beta_obj.ppf(f)
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output

    def dcdf_dtheta(self, f):
        raise NotImplementedError, "This function is not implemented!"
        assert not(np.isnan(output).any().any()), 'function returns NaN'
        assert not(np.isinf(output).any().any()), 'function returns inf'
        return output

    def dquantile_dtheta(self, f):
        raise NotImplementedError, "This function is not implemented!"
        assert not(np.isnan(output).any().any()), 'function returns NaN'
        assert not(np.isinf(output).any().any()), 'function returns inf'
        return output


class MoG(MarginalDistribution):
    def __init__(self, mu_vec, k_vec = None, var_vec = None):
        self.mu_vec = np.reshape(mu_vec, (1, -1))[0].copy()

        if k_vec is None:
            self.k_vec = (1./len(self.mu_vec)) * np.ones(len(self.mu_vec))
        else:
            self.k_vec = np.reshape(k_vec, (1, -1))[0].copy()
        assert len(self.mu_vec) == len(self.k_vec), ('Dim. mismatch: mu_vec and k_vec ' 
                + str(len(self.mu_vec)) + ', ' + str(len(self.k_vec)))

        if var_vec is None:
            self.var_vec = np.ones(len(self.mu_vec))
        else:
            self.var_vec = np.reshape(var_vec, (1, -1))[0].copy()
        assert len(self.mu_vec) == len(self.var_vec), ('Dim. mismatch: mu_vec and var_vec ' 
                + str(len(self.mu_vec)) + ', ' + str(len(self.var_vec)))

        self.sigma_vec = np.sqrt(self.var_vec)
        self.param_vec = np.reshape(np.array([self.k_vec, self.mu_vec, self.var_vec]), (1, -1))[0]
        self.quantile_map = {}

    def get_param_vec(self):
        #HACK
        return self.mu_vec

    def set_param_vec(self, type, params):
        params = np.reshape(params, (1, -1))[0]
        if type == 'k':
            assert len(params) == len(self.k_vec), 'Invalid input param dimension'
            assert np.sum(params) == 1., 'Weight param does not sum to 1.'
            assert all(params>=0.), 'Some mixture weight appears to be <0.\n' + str(params)
            self.k_vec = params.copy()
        elif type == 'mu':
            assert len(params) == len(self.mu_vec), 'Invalid input param dimension'
            self.mu_vec = params.copy()
        elif type == 'var':
            assert len(params) == len(self.var_vec), 'Invalid input param dimension'
            assert all(params>0.), 'Some component variance appears to be <=0.\n' + str(params)
            self.var_vec = params.copy()
            self.sigma_vec = np.sqrt(self.var_vec)
        elif type == 'all':
            assert len(params) == 3 * len(self.mu_vec), 'Param. vector length mis-match!'
            m_dim = len(self.mu_vec)
            self.set_param_vec('k', params[0:m_dim])
            self.set_param_vec('mu', params[m_dim: 2*m_dim])
            self.set_param_vec('var', params[2*m_dim:])
            self.param_vec = np.reshape(np.array([self.k_vec, self.mu_vec, self.var_vec]), (1, -1))[0]
        else:
            assert False, 'Invalid param. type'
        self.quantile_map = {}
        return True

    def pdf(self, f):
        input = (f - self.mu_vec)/(self.sigma_vec)
        output = np.dot(self.k_vec, np.reshape(std_norm_pdf(input)/self.sigma_vec, (1, -1))[0])
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output

    def cdf(self, f):
        input = (f - self.mu_vec)/(self.sigma_vec)
        output = np.dot(self.k_vec, np.reshape(std_norm_cdf(input), (1, -1))[0])
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output

    def quantile_find_bracket(self, q, startpoint = 0.0, inc = 1.):
        if(np.log(inc)>=50):
            print 'Stuck in quantile_find_bracket' + str(q)
            raw_input()
        qstart = self.cdf(startpoint)
        if(qstart<=q):
            qend = self.cdf(startpoint+inc)
            return (startpoint, startpoint+inc) if (qend>=q) else self.quantile_find_bracket(q, startpoint = startpoint+inc, inc = 10. * inc)
        else:
            qend = self.cdf(startpoint-inc)
            return (startpoint-inc, startpoint) if (qend<=q) else self.quantile_find_bracket(q, startpoint = startpoint-inc, inc = 10. * inc)

    def mog_inv_cdf(self, q):
        eps = 1e-13
        itr = 0 
        maxiter = 1e6

        (lo, hi) = self.quantile_find_bracket(q)
        mid = (lo + hi)/2.0
        q_mid = self.cdf(mid)

        while(abs(lo-hi)>eps):
            if(itr>=maxiter):
                print 'WARNING: mog_inv_cdf exceeded maxiter!', abs(lo-hi), lo, hi, q, q_mid
                break # TODO: Remove this later on
            lo = mid if q>q_mid else lo
            hi = mid if q<=q_mid else hi
            mid = (lo + hi)/2.0
            q_mid = self.cdf(mid)
            itr += 1
        assert np.abs(q - self.cdf(mid))<=1e-6, 'Disagreement in computed quantile value!'
        return np.round(mid, 20)
    
    def quantile(self, f):
        # FIXME: only allow 1 input at a time
        assert np.all(f>=0.), 'Input to quantile function needs to be in [0., 1.]'
        assert np.all(f<=1.), 'Input to quantile function needs to be in [0., 1.]'
        # Bisection method
        try:
            output = self.quantile_map[f]
        except:
            try:
                output = brentq(lambda x: self.cdf(x) - f, -100., 100.)
            except:
                output = brentq(lambda x: self.cdf(x) - f, -10000., 10000.)
            self.quantile_map[f] = output
        assert np.abs(f - self.cdf(output))<=1.e-6, 'Disagreement in computed quantile value' 
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output

    def dcdf_dtheta(self, f):
        # FIXME: Limit to only mu_vec at the moment.
        input = (f - self.mu_vec)/(self.sigma_vec)
        output = -1. * std_norm_pdf(input)/(self.sigma_vec * self.k_vec)
        output = np.reshape(output, (1, -1))[0]
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output

class NormalDistribution(MarginalDistribution):
    def __init__(self, params):
        self.mu = 0.
        self.var = 0.
        self.sigma = np.sqrt(self.var)
        self.param_vec = np.array([self.mu, self.var])
        self.set_param_vec(params)

    def get_param_vec(self):
        return self.param_vec

    def set_param_vec(self, params):
        assert len(params) == 2, "Normal Marginal Distribution requires exactly 2 parameters: np.array([mean, variance])"
        self.mu = params[0]
        self.var = params[1]
        self.sigma = np.sqrt(self.var)
        self.param_vec[0] = self.mu
        self.param_vec[1] = self.var
        assert self.var>=0., "Variance cannot be <0."
        return True

    def pdf(self, f):
        output = std_norm_pdf((f - self.mu)/self.sigma)/self.sigma
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output

    def cdf(self, f):
        output = std_norm_cdf((f - self.mu)/self.sigma)
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output

    def quantile(self, f):
        qt_bound = inv_std_norm_cdf(1e-16)
        output = self.sigma * np.clip(inv_std_norm_cdf(f), qt_bound, -qt_bound) + self.mu
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output

    def dcdf_dtheta(self, f):
        dmu = - self.pdf(f)
        dvar = - .5 * self.pdf(f) * (f - self.mu)/(self.sigma ** 2)
        output = np.array([dmu, dvar])
        assert not(np.isnan(output).any().any()), 'function returns NaN'
        assert not(np.isinf(output).any().any()), 'function returns inf'
        return output

    #FIXME: HACK
    def dquantile_dtheta(self, f):
        qt_bound = inv_std_norm_cdf(1e-16)
        output = np.array([1., 0.5 * np.clip(inv_std_norm_cdf(f), qt_bound, -qt_bound)/self.sigma])
        assert not(np.isnan(output).any().any()), 'function returns NaN'
        assert not(np.isinf(output).any().any()), 'function returns inf'
        return output

class Identity(GPTransformation):
    """
    .. math::

        g(f) = f

    """
    def transf(self,f):
        return f

    def dtransf_df(self,f):
        return np.ones_like(f)

    def d2transf_df2(self,f):
        return np.zeros_like(f)

    def d3transf_df3(self,f):
        return np.zeros_like(f)

class SVCopula(GPTransformation):
    def __init__(self, w):
        self.a = w[0]
        self.b = w[1]
        self.c = w[2]
    
    def get_params(self):
        return np.array([self.a, self.b, self.c])
 
    def transf(self, f):
        if type(f) == np.ndarray:
            f_shape = np.shape(f)
            f_n = f_shape[0] if len(f_shape) == 1 else f_shape[0] * f_shape[1]
            f_vec = np.reshape(f, (f_n,))
        else:
            f_vec = np.array([f])
            f_n = len(f_vec)
        et = self.b * (f_vec + self.c)
        et = np.c_[et, np.zeros(f_n)]
        output = self.a * logsumexp(et, 1)
        output = np.reshape(output, f_shape) if f_n > 1 else output[0]
        return output + 1.e-10

    def inv_transf(self, f):
        if type(f) == np.ndarray:
            f_shape = np.shape(f)
            f_n = f_shape[0] if len(f_shape) == 1 else f_shape[0] * f_shape[1]
            f_vec = np.reshape(f, (f_n,))
        else:
            f_vec = np.array([f])
            f_n = len(f_vec)
        et = (f_vec - 1.e-10)/self.a
        et = np.c_[et, np.zeros(f_n)]
        output = (logsumexp(et, 1, np.c_[np.ones(f_n), -1. * np.ones(f_n)]) / self.b) - self.c
        output = np.reshape(output, f_shape) if f_n > 1 else output[0]
        return output

class IdentityCopula(GPTransformation):
    def __init__(self, k, marginal):
        assert k>0., 'k param in copula needs to be >0.'
        # assert isinstance(marginal, MarginalDistribution), "Please define marginal distribution!"
        self.k = k
        self.marginal = marginal

    def copula(self, z):
        output = self.marginal.quantile(std_norm_cdf(z))
        output = np.clip(output, -1.e50, 1.e50)
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output
    
    def dcopula_dz(self, z):
        output = self.marginal.dquantile_dcdf(std_norm_cdf(z)) * std_norm_pdf(z)
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output
    
    def dcopula_dtheta(self, z):
        output = self.marginal.dquantile_dtheta(std_norm_cdf(z))
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output

    def dcopula_dk(self, z):
        # Deprecated: k == 1. for copula, so this should be 0.
        #output = self.marginal.dquantile_dcdf(std_norm_cdf(z)) * std_norm_pdf(z) * -0.5 * z * (self.k ** (-1.5))
        raise NotImplementedError, "This function is not implemented!"
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output

    def transf(self, z):
        output = self.copula(z)
        '''
        if type(z) == type(np.ones(1)):
            z_shape = np.shape(z)
            z = np.reshape(z, (1, -1))[0]
            output = np.array(map(lambda z_in: std_norm_cdf(self.copula(z_in)) , z))
            output = np.reshape(output, z_shape)
        else:
            output = std_norm_cdf(self.copula(z))
        '''
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output
    
    def dtransf_df(self, z):
        output = 1. if type(z) != type(np.array([])) else np.ones_like(z)
        '''
        if type(z) == type(np.ones(1)):
            z_shape = np.shape(z)
            z = np.reshape(z, (1, -1))[0]
            output = np.array(map(lambda z_in: std_norm_pdf(self.copula(z_in)) * self.dcopula_dz(z_in), z))
            output = np.reshape(output, z_shape)
        else:
            output = std_norm_pdf(self.copula(z)) * self.dcopula_dz(z)
        '''
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output

    def dtransf_dtheta(self, z):
        ''' f being z'''
        output = self.dcopula_dtheta(z)
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output

    def dtransf_dk(self, z):
        ''' f being z'''
        # Deprecated: k == 1. for copula, so this should be 0.
        #output = std_norm_pdf(self.copula(z)) * self.dcopula_dk(z)
        raise NotImplementedError, "This function is not implemented!"
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output

    def d2transf_df2(self,f):
        raise NotImplementedError, "This function is not implemented!"

    def d3transf_df3(self,f):
        raise NotImplementedError, "This function is not implemented!"

class RegressionCopula(IdentityCopula):
    def __init__(self, k, marginal):
        assert k>0., 'k param in copula needs to be >0.'
        # assert isinstance(marginal, MarginalDistribution), "Please define marginal distribution!"
        self.k = k
        self.marginal = marginal

class Probit(GPTransformation):
    """
    .. math::

        g(f) = \\Phi^{-1} (mu)
    
    """
    def transf(self,f):
        return std_norm_cdf(f)

    def dtransf_df(self,f):
        return std_norm_pdf(f)

    def d2transf_df2(self,f):
        #FIXME
        return -f * std_norm_pdf(f)

    def d3transf_df3(self,f):
        #FIXME
        f2 = f**2
        return -(1/(np.sqrt(2*np.pi)))*np.exp(-0.5*(f2))*(1-f2)

# Link function implementation for Gaussian Copula Process Classification (GCPC)
class ProbitCopula(GPTransformation):
    """
    .. math::

        g(f) = \\Phi^{-1} (mu)
    
    """

    def __init__(self, k, marginal):
        assert k>0., 'k param in copula needs to be >0.'
        # assert isinstance(marginal, MarginalDistribution), "Please define marginal distribution!"

        self.k = k
        self.marginal = marginal

    def copula(self, z):
        if type(z) == type(np.ones(1)):
            z_shape = np.shape(z)
            z = np.reshape(z, (1, -1))[0]
            output = np.array(map(lambda z_in: self.marginal.quantile(std_norm_cdf(z_in/np.sqrt(self.k))), z))
            output = np.reshape(output, z_shape)
        else:
            output = self.marginal.quantile(std_norm_cdf(z/np.sqrt(self.k)))
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output
    
    def dcopula_dz(self, z):
        output = self.marginal.dquantile_dcdf(std_norm_cdf(z/np.sqrt(self.k))) * (std_norm_pdf(z/np.sqrt(self.k))) / np.sqrt(self.k)
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output
    
    def dcopula_dtheta(self, z):
        output = self.marginal.dquantile_dtheta(std_norm_cdf(z/np.sqrt(self.k)))
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output

    def dcopula_dk(self, z):
        output = self.marginal.dquantile_dcdf(std_norm_cdf(z/np.sqrt(self.k))) * (std_norm_pdf(z/np.sqrt(self.k))) * -0.5 * z * (self.k ** (-1.5))
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output

    def transf(self, z):
        if type(z) == type(np.ones(1)):
            z_shape = np.shape(z)
            z = np.reshape(z, (1, -1))[0]
            output = np.array(map(lambda z_in: std_norm_cdf(self.copula(z_in)) , z))
            output = np.reshape(output, z_shape)
        else:
            output = std_norm_cdf(self.copula(z))
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output
    
    def dtransf_df(self, z):
        if type(z) == type(np.ones(1)):
            z_shape = np.shape(z)
            z = np.reshape(z, (1, -1))[0]
            output = np.array(map(lambda z_in: std_norm_pdf(self.copula(z_in)) * self.dcopula_dz(z_in), z))
            output = np.reshape(output, z_shape)
        else:
            output = std_norm_pdf(self.copula(z)) * self.dcopula_dz(z)
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output

    def dtransf_dtheta(self, z):
        ''' f being z'''
        output = std_norm_pdf(self.copula(z)) * self.dcopula_dtheta(z)
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output

    def dtransf_dk(self, z):
        ''' f being z'''
        output = std_norm_pdf(self.copula(z)) * self.dcopula_dk(z)
        assert not(np.isnan(output).any()), 'function returns NaN'
        assert not(np.isinf(output).any()), 'function returns inf'
        return output

    def d2transf_df2(self,f):
        raise NotImplementedError, "This function is not implemented!"

    def d3transf_df3(self,f):
        raise NotImplementedError, "This function is not implemented!"


class Cloglog(GPTransformation):
    """
    Complementary log-log link
    .. math::

        p(f) = 1 - e^{-e^f}

        or

        f = \log (-\log(1-p))
    
    """
    def transf(self,f):
        return 1-np.exp(-np.exp(f))

    def dtransf_df(self,f):
        return np.exp(f-np.exp(f))

    def d2transf_df2(self,f):
        ef = np.exp(f)
        return -np.exp(f-ef)*(ef-1.)

    def d3transf_df3(self,f):
        ef = np.exp(f)
        return np.exp(f-ef)*(1.-3*ef + ef**2)


class Log(GPTransformation):
    """
    .. math::

        g(f) = \\log(\\mu)

    """
    def transf(self,f):
        return np.exp(np.clip(f, -_lim_val, _lim_val))

    def dtransf_df(self,f):
        return np.exp(np.clip(f, -_lim_val, _lim_val))

    def d2transf_df2(self,f):
        return np.exp(np.clip(f, -_lim_val, _lim_val))

    def d3transf_df3(self,f):
        return np.exp(np.clip(f, -_lim_val, _lim_val))

class Log_ex_1(GPTransformation):
    """
    .. math::

        g(f) = \\log(\\exp(\\mu) - 1)

    """
    def transf(self,f):
        return np.log(1.+np.exp(f))

    def dtransf_df(self,f):
        return np.exp(f)/(1.+np.exp(f))

    def d2transf_df2(self,f):
        aux = np.exp(f)/(1.+np.exp(f))
        return aux*(1.-aux)

    def d3transf_df3(self,f):
        aux = np.exp(f)/(1.+np.exp(f))
        daux_df = aux*(1.-aux)
        return daux_df - (2.*aux*daux_df)

class Reciprocal(GPTransformation):
    def transf(self,f):
        return 1./f

    def dtransf_df(self,f):
        return -1./(f**2)

    def d2transf_df2(self,f):
        return 2./(f**3)

    def d3transf_df3(self,f):
        return -6./(f**4)

class Heaviside(GPTransformation):
    """

    .. math::

        g(f) = I_{x \\in A}

    """
    def transf(self,f):
        #transformation goes here
        return np.where(f>0, 1, 0)

    def dtransf_df(self,f):
        raise NotImplementedError, "This function is not differentiable!"

    def d2transf_df2(self,f):
        raise NotImplementedError, "This function is not differentiable!"
