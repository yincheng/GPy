# Copyright (c) 2012, 2013 Ricardo Andrade
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import cython.gaussian as c_gauss

def std_norm_pdf(x):
    """Standard Gaussian density function"""
    return 1./np.sqrt(2.*np.pi)*np.exp(-.5*x**2)

def std_norm_cdf(x):
    """
    Cumulative standard Gaussian distribution
    Based on Abramowitz, M. and Stegun, I. (1970)
    """
    #Generalize for many x
    x = np.asarray(x).copy()
    cdf_x = np.zeros_like(x)
    N = x.size
    c_gauss.norm_cdf(x, cdf_x, N)

    return cdf_x

def inv_std_norm_cdf(x):
    """
    Inverse cumulative standard Gaussian distribution
    Based on Winitzki, S. (2008)
    """
    z = 2*x -1
    ln1z2 = np.log(1-z**2)
    a = 8*(np.pi -3)/(3*np.pi*(4-np.pi))
    b = 2/(np.pi * a) + ln1z2/2
    inv_erf = np.sign(z) * np.sqrt( np.sqrt(b**2 - ln1z2/a) - b )
    return np.sqrt(2) * inv_erf

