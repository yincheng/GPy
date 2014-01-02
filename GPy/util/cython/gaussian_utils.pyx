cimport numpy as np
import numpy as np
from cpython cimport bool
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def norm_cdf(np.ndarray[double, ndim=1] x, np.ndarray[double, ndim=1] cdf_x, int N):
    cdef double sign, t, erf
    cdef int i

    for i in xrange(N):
        sign = 1.0
        if x[i] < 0.0:
            sign = -1.0
            x[i] = -x[i]
        x[i] = x[i]/np.sqrt(2.0)
        t = 1.0/(1.0 +  0.3275911*x[i])

        erf = 1. - np.exp(-x[i]*x[i])*t*(0.254829592 + t*(-0.284496736 + t*(1.421413741 + t*(-1.453152027 + t*(1.061405429)))))
        cdf_x[i] = 0.5*(1.0 + sign*erf)
