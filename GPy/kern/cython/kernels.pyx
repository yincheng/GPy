import cython
cimport numpy as np
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def linear_dpsi2_dmuS(int N, int num_inducing, int input_dim,
                      np.ndarray[double, ndim=2] mu, np.ndarray[double, ndim=4] AZZA,
                      np.ndarray[double, ndim=2] target_mu,
                      np.ndarray[double, ndim=2] target_S, np.ndarray[double, ndim=3] dL_dpsi2):

    cdef int n
    cdef int m
    cdef int mm
    cdef int q
    cdef int qq
    cdef double factor
    cdef double tmp
    # TODO: #pragma omp parallel for private(m,mm,q,qq,factor,tmp)
    for n in xrange(N):
        for m in xrange(num_inducing):
            for mm in xrange(m+1):
                # add in a factor of 2 for the off-diagonal terms (and then count them only once)
                if m == mm:
                    factor = dL_dpsi2[n, m, mm]
                else:
                    factor = 2.0*dL_dpsi2[n, m, mm]

                for q in xrange(input_dim):
                    # take the dot product of mu[n,:] and AZZA[:,m,mm,q]
                    tmp = np.dot(mu[n, :], AZZA[:, m, mm, q])
                    target_mu[n, q] += factor*tmp
                    target_S[n, q] += factor*AZZA[q, m, mm, q]/2.

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def linear_dpsi2_dZ(int N, int num_inducing, int input_dim,
                    np.ndarray[double, ndim=3] AZA, np.ndarray[double, ndim=2] target,
                    np.ndarray[double, ndim=3] dL_dpsi2):
    cdef int n
    cdef int m
    cdef int mm
    cdef int q
    for m in xrange(num_inducing):
        for q in xrange(input_dim):
            for mm in xrange(num_inducing):
                for n in xrange(N):
                    target[m,q] += 2*dL_dpsi2[n,m,mm]*AZA[n,mm,q]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def rbf_psi2_lengthscale_grads(int N, int M, int Q, 
                               np.ndarray[double, ndim=2] S,
                               np.ndarray[double, ndim=3] Zdist_sq,
                               np.ndarray[double, ndim=4] mudist_sq,
                               np.ndarray[double, ndim=3] dL_dpsi2,
                               np.ndarray[double, ndim=3] psi2,
                               np.ndarray[double, ndim=1] l2,
                               np.ndarray[double, ndim=1] result):
    cdef double tmp
    for q in xrange(Q):
        tmp = 0.0
        for n in xrange(N):
            for m in xrange(M):
                tmp += dL_dpsi2[n,m,m] * psi2[n,m,m] * (Zdist_sq[m,m,q] * (2.0*S[n,q]/l2[q] + 1.0) + mudist_sq[n,m,m,q] + S[n,q]/l2[q]) / (2.0*S[n,q] + l2[q]) ;
                for mm in xrange(m):
                    tmp += 2.0 * dL_dpsi2[n,m,mm] * psi2[n,m,mm] * (Zdist_sq[m,mm,q] * (2.0*S[n,q]/l2[q] + 1.0) + mudist_sq[n,m,mm,q] + S[n,q]/l2[q]) / (2.0*S[n,q] + l2[q]) ;

        result[q] = tmp    
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def rbf_dK_dtheta(int num_data, int num_inducing, int input_dim,
                  np.ndarray[double, ndim=2] X,
                  np.ndarray[double, ndim=2] X2,
                  np.ndarray[double, ndim=1] target,
                  np.ndarray[double, ndim=2] dvardLdK,
                  np.ndarray[double, ndim=1] var_len3):
    cdef int q,i,j
    cdef double tmp
    if X2 is None:
        for q in xrange(input_dim):
            tmp = 0.0
            for i in xrange(num_data):
                for j in xrange(i):
                    tmp += (X[i,q]-X[j,q])*(X[i,q]-X[j,q])*dvardLdK[i,j]
            target[q+1] += var_len3[q]*tmp
    else:
        for q in xrange(input_dim):
            tmp = 0.0
            for i in xrange(num_data):
                for j in xrange(num_inducing):
                    tmp += (X[i,q]-X2[j,q])*(X[i,q]-X2[j,q])*dvardLdK[i,j]

            target[q+1] += var_len3[q]*tmp;


#from libc.stdlib cimport abort, malloc, free

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def rbf_psi2(int N, int num_inducing, int input_dim,
             double variance_sq,
             np.ndarray[double, ndim=2] mu,
             np.ndarray[double, ndim=3] Zhat,
             np.ndarray[double, ndim=3] psi2_Zdist_sq,
             np.ndarray[double, ndim=4] mudist,
             np.ndarray[double, ndim=4] mudist_sq,
             np.ndarray[double, ndim=2] denom_l2,
             np.ndarray[double, ndim=2] half_log_psi2_denom,
             np.ndarray[double, ndim=3] psi2):
    
    from cython.parallel import prange, parallel
    from cython import nogil

    cdef double tmp
    cdef double tmp2 
    cdef double exponent_tmp
    cdef int n, m, mm, q

    cdef extern from "math.h":
        double exp(double x) nogil    

    with nogil, parallel():
        for n in prange(N):
            for m in xrange(num_inducing):
                for mm in xrange(m+1):
                    exponent_tmp = <double> 0.0#malloc(sizeof(double))
                    for q in xrange(input_dim):
                        #compute mudist
                        tmp = <double> 0.0#malloc(sizeof(double))
                        tmp = mu[n,q] - Zhat[m,mm,q]
                        mudist[n,m,mm,q] = tmp
                        mudist[n,mm,m,q] = tmp
    
                        # now mudist_sq
                        tmp = tmp*tmp/denom_l2[n,q]
                        mudist_sq[n,m,mm,q] = tmp
                        mudist_sq[n,mm,m,q] = tmp
    
                        # now psi2_exponent
                        tmp = -psi2_Zdist_sq[m,mm,q] - tmp - half_log_psi2_denom[n,q]
                        exponent_tmp = exponent_tmp+tmp
                        
                        # psi2 would be computed like this, but np is faster
                    tmp2 = variance_sq * exp(exponent_tmp)
                    psi2[n,m,mm] = tmp2
                    psi2[n,mm,m] = tmp2
