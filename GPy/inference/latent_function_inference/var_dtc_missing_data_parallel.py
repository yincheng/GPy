'''
Created on Jul 25, 2014

@author: t-mazwie
'''
from GPy.inference.latent_function_inference import LatentFunctionInference
import numpy as np
import itertools, logging
from GPy.core.parameterization.variational import VariationalPosterior
from GPy.util import diag
from GPy.util.linalg import jitchol, dtrtri, dtrtrs, tdot, backsub_both_sides,\
    dpotri, symmetrify, mdot
from GPy.inference.latent_function_inference.posterior import Posterior

from multiprocessing import Queue, Process, cpu_count
from Queue import Empty

logger = logging.getLogger(__name__)

KILL_SWITCH = "KILL_WORKERS"

class Worker(Process):
    def __init__(self, inq, outq, keep_alive=False):
        super(Worker, self).__init__()
        self.inq = inq
        self.outq = outq
        self.keep_alive=keep_alive
        self.daemon = True

    def run(self):
        # Multiprocessing stuff!!
        try:
            arg = self.inq.get(timeout=1)
            while arg is not KILL_SWITCH:
                self.outq.put(self.worker_partial(*arg))
                arg = self.inq.get(timeout=1)
        except Empty:
            if self.keep_alive:
                self.run()

    def worker_partial(self, index, uncertain_inputs, het_noise, num_inducing, Lm, LmInv, trYYT, beta, VVT_factor, output_dim, psi0, psi1, psi2):
        num_data = psi1.shape[0]
        if uncertain_inputs:
            if het_noise:
                psi2_beta = psi2 * (beta.flatten().reshape(num_data, 1, 1)).sum(0)
            else:
                psi2_beta = psi2.sum(0) * beta
            A = LmInv.dot(psi2_beta.dot(LmInv.T))
        else:
            if het_noise:
                tmp = psi1 * (np.sqrt(beta.reshape(num_data, 1)))
            else:
                tmp = psi1 * (np.sqrt(beta))
            tmp, _ = dtrtrs(Lm, tmp.T, lower=1)
            A = tdot(tmp) #print A.sum()
    # factor B
        B = np.eye(num_inducing) + A
        LB = jitchol(B)
        psi1Vf = psi1.T.dot(VVT_factor)
        tmp, _ = dtrtrs(Lm, psi1Vf, lower=1, trans=0)
        _LBi_Lmi_psi1Vf, _ = dtrtrs(LB, tmp, lower=1, trans=0)
        tmp, _ = dtrtrs(LB, _LBi_Lmi_psi1Vf, lower=1, trans=1)
        Cpsi1Vf, _ = dtrtrs(Lm, tmp, lower=1, trans=1)
    # data fit and derivative of L w.r.t. Kmm
        delit = tdot(_LBi_Lmi_psi1Vf)
        data_fit = np.trace(delit)
        DBi_plus_BiPBi = backsub_both_sides(LB, output_dim * np.eye(num_inducing) + delit)
        delit = -0.5 * DBi_plus_BiPBi
        delit += -0.5 * B * output_dim
        delit += output_dim * np.eye(num_inducing)
        dL_dKmm_ = backsub_both_sides(Lm, delit)
    # derivatives of L w.r.t. psi
        dL_dpsi0, dL_dpsi1, dL_dpsi2 = _compute_dL_dpsi(num_inducing, num_data, output_dim, beta, Lm,
            VVT_factor, Cpsi1Vf, DBi_plus_BiPBi,
            psi1, het_noise, uncertain_inputs)
    # log marginal likelihood
        log_marginal_ = _compute_log_marginal_likelihood(num_data, output_dim, beta, het_noise,
            psi0, A, LB, trYYT, data_fit, VVT_factor)
    #put the gradients in the right places
        dL_dR_ = _compute_dL_dR(
            het_noise, uncertain_inputs, LB,
            _LBi_Lmi_psi1Vf, DBi_plus_BiPBi, Lm, A,
            psi0, psi1, beta,
            data_fit, num_data, output_dim, trYYT, VVT_factor)
        Bi, _ = dpotri(LB, lower=1)
        symmetrify(Bi)
        Bi = -dpotri(LB, lower=1)[0]
        diag.add(Bi, 1)
        return index, log_marginal_, dL_dKmm_, dL_dpsi0, dL_dpsi1, dL_dpsi2, dL_dR_, Bi, Cpsi1Vf


class VarDTCMissingDataParallel(LatentFunctionInference):
    const_jitter = 1e-10
    def __init__(self, limit=1, inan=None, keep_workers_alive=False):
        from ...util.caching import Cacher
        self._Y = Cacher(self._subarray_computations, limit)
        if inan is not None: self._inan = ~inan
        else: self._inan = None
        self.keep_alive = keep_workers_alive
        self.cpu_count = cpu_count()
        self.restart_workers()

    def restart_workers(self):
        self.inq = Queue()
        self.outq = Queue()
        self.workers = [Worker(self.outq, self.inq, self.keep_alive) for _ in xrange(self.cpu_count)]
        for p in self.workers:
            p.start()

    def set_limit(self, limit):
        self._Y.limit = limit

    def __getstate__(self):
        # has to be overridden, as Cacher objects cannot be pickled.
        return self._Y.limit, self._inan

    def __setstate__(self, state):
        # has to be overridden, as Cacher objects cannot be pickled.
        from ...util.caching import Cacher
        self.limit = state[0]
        self._inan = state[1]
        self._Y = Cacher(self._subarray_computations, self.limit)

    def _subarray_computations(self, Y):
        if self._inan is None:
            inan = np.isnan(Y)
            has_none = inan.any()
            self._inan = ~inan
        else:
            inan = self._inan
            has_none = True
        if has_none:
            #print "caching missing data slices, this can take several minutes depending on the number of unique dimensions of the data..."
            #csa = common_subarrays(inan, 1)
            size = Y.shape[1]
            #logger.info('preparing subarrays {:3.3%}'.format((i+1.)/size))
            Ys = []
            next_ten = [0.]
            count = itertools.count()
            for v, y in itertools.izip(inan.T, Y.T[:,:,None]):
                i = count.next()
                if ((i+1.)/size) >= next_ten[0]:
                    logger.info('preparing subarrays {:>6.1%}'.format((i+1.)/size))
                    next_ten[0] += .1
                Ys.append(y[v,:])

            next_ten = [0.]
            count = itertools.count()
            def trace(y):
                i = count.next()
                if ((i+1.)/size) >= next_ten[0]:
                    logger.info('preparing traces {:>6.1%}'.format((i+1.)/size))
                    next_ten[0] += .1
                y = y[inan[:,i],i:i+1]
                return np.einsum('ij,ij->', y,y)
            traces = [trace(Y) for _ in xrange(size)]
            return Ys, traces
        else:
            self._subarray_indices = [[slice(None),slice(None)]]
            return [Y], [(Y**2).sum()]

    def terminate_workers(self):
        for _ in self.workers:
            self.outq.put(KILL_SWITCH)

    def on_optimization_start(self):
        LatentFunctionInference.on_optimization_start(self)

    def on_optimization_end(self):
        LatentFunctionInference.on_optimization_end(self)

    def inference(self, kern, X, Z, likelihood, Y, Y_metadata=None):
        if isinstance(X, VariationalPosterior):
            uncertain_inputs = True
            psi0_all = kern.psi0(Z, X)
            psi1_all = kern.psi1(Z, X)
            psi2_all = kern.psi2(Z, X)
        else:
            uncertain_inputs = False
            psi0_all = kern.Kdiag(X)
            psi1_all = kern.K(X, Z)
            psi2_all = None

        Ys, traces = self._Y(Y)
        beta_all = 1./np.fmax(likelihood.gaussian_variance(Y_metadata), 1e-6)
        het_noise = beta_all.size != 1

        num_inducing = Z.shape[0]

        dL_dpsi0_all = np.zeros(Y.shape[0])
        dL_dpsi1_all = np.zeros((Y.shape[0], num_inducing))
        if uncertain_inputs:
            dL_dpsi2_all = np.zeros((Y.shape[0], num_inducing, num_inducing))

        dL_dR = 0
        woodbury_vector = np.zeros((num_inducing, Y.shape[1]))
        woodbury_inv_all = np.zeros((num_inducing, num_inducing, Y.shape[1]))
        dL_dKmm = 0
        log_marginal = 0

        Kmm = kern.K(Z).copy()
        diag.add(Kmm, self.const_jitter)
        #factor Kmm
        Lm = jitchol(Kmm)
        if uncertain_inputs: LmInv = dtrtri(Lm)

        size = Y.shape[1]
        next_ten = 0

        # set up multiprocessing
        num_received = 0
        count = itertools.count()
        data_it = itertools.izip(Ys, self._inan.T, traces)
        old_num = num_received
        num_sent = 0

        if any([not w.is_alive() for w in self.workers]):
            self.restart_workers()

        stop = False

        while num_received != len(Ys) and not stop:
            if old_num != num_received:
                print num_received
                old_num = num_received

            if num_sent != len(Ys) and (num_sent - num_received) <= self.cpu_count:
                i = count.next()
                [y, v, trYYT] = data_it.next()

                if ((i+1.)/size) >= next_ten:
                    logger.info('inference {:> 6.1%}'.format((i+1.)/size))
                    next_ten += .1
                if het_noise: beta = beta_all[i]
                else: beta = beta_all

                VVT_factor = (y*beta)
                output_dim = 1#len(ind)

                psi0 = psi0_all[v]
                psi1 = psi1_all[v, :]
                if uncertain_inputs: psi2 = psi2_all[v, :]
                else: psi2 = None

                self.outq.put((i, uncertain_inputs, het_noise, num_inducing, Lm, LmInv, trYYT, beta, VVT_factor, output_dim, psi0, psi1, psi2))
                num_sent += 1
            else:
                try:
                    arg = self.inq.get_nowait()
                    i, log_marginal_, dL_dKmm_, dL_dpsi0, dL_dpsi1, dL_dpsi2, dL_dR_, Bi, Cpsi1Vf = arg

                    # collect worker stuff:
                    log_marginal += log_marginal_

                    dL_dKmm += dL_dKmm_
                    dL_dpsi0_all[v] += dL_dpsi0
                    dL_dpsi1_all[v, :] += dL_dpsi1
                    if uncertain_inputs:
                        dL_dpsi2_all[v, :] += dL_dpsi2

                    dL_dR += dL_dR_

                    woodbury_inv_all[:, :, i:i+1] = backsub_both_sides(Lm, Bi)[:,:,None]
                    woodbury_vector[:, i:i+1] = Cpsi1Vf

                    num_received += 1
                except Empty:
                    if not stop:
                        continue
                    else:
                        break

        dL_dthetaL = likelihood.exact_inference_gradients(dL_dR)

        # gradients:
        if uncertain_inputs:
            grad_dict = {'dL_dKmm': dL_dKmm,
                         'dL_dpsi0':dL_dpsi0_all,
                         'dL_dpsi1':dL_dpsi1_all,
                         'dL_dpsi2':dL_dpsi2_all,
                         'dL_dthetaL':dL_dthetaL}
        else:
            grad_dict = {'dL_dKmm': dL_dKmm,
                         'dL_dKdiag':dL_dpsi0_all,
                         'dL_dKnm':dL_dpsi1_all,
                         'dL_dthetaL':dL_dthetaL}

        post = Posterior(woodbury_inv=woodbury_inv_all, woodbury_vector=woodbury_vector, K=Kmm, mean=None, cov=None, K_chol=Lm)

        return post, log_marginal, grad_dict

def _compute_dL_dpsi(num_inducing, num_data, output_dim, beta, Lm, VVT_factor, Cpsi1Vf, DBi_plus_BiPBi, psi1, het_noise, uncertain_inputs):
    dL_dpsi0 = -0.5 * output_dim * (beta[:,None] * np.ones([num_data, 1])).flatten()
    dL_dpsi1 = np.dot(VVT_factor, Cpsi1Vf.T)
    dL_dpsi2_beta = 0.5 * backsub_both_sides(Lm, output_dim * np.eye(num_inducing) - DBi_plus_BiPBi)
    if het_noise:
        if uncertain_inputs:
            dL_dpsi2 = beta[:, None, None] * dL_dpsi2_beta[None, :, :]
        else:
            dL_dpsi1 += 2.*np.dot(dL_dpsi2_beta, (psi1 * beta.reshape(num_data, 1)).T).T
            dL_dpsi2 = None
    else:
        dL_dpsi2 = beta * dL_dpsi2_beta
        if uncertain_inputs:
            # repeat for each of the N psi_2 matrices
            dL_dpsi2 = np.repeat(dL_dpsi2[None, :, :], num_data, axis=0)
        else:
            # subsume back into psi1 (==Kmn)
            dL_dpsi1 += 2.*np.dot(psi1, dL_dpsi2)
            dL_dpsi2 = None

    return dL_dpsi0, dL_dpsi1, dL_dpsi2


def _compute_dL_dR(het_noise, uncertain_inputs, LB, _LBi_Lmi_psi1Vf, DBi_plus_BiPBi, Lm, A, psi0, psi1, beta, data_fit, num_data, output_dim, trYYT, VVT_factor):
    # the partial derivative vector for the likelihood
    if het_noise:
        if uncertain_inputs:
            raise NotImplementedError, "heteroscedatic derivates with uncertain inputs not implemented"
        else:
            #from ...util.linalg import chol_inv
            #LBi = chol_inv(LB)
            LBi, _ = dtrtrs(LB,np.eye(LB.shape[0]))

            Lmi_psi1, _ = dtrtrs(Lm, psi1.T, lower=1, trans=0)
            _LBi_Lmi_psi1, _ = dtrtrs(LB, Lmi_psi1, lower=1, trans=0)

            dL_dR = -0.5 * beta + 0.5 * (VVT_factor)**2
            dL_dR += 0.5 * output_dim * (psi0 - np.sum(Lmi_psi1**2,0))[:,None] * beta**2

            dL_dR += 0.5*np.sum(mdot(LBi.T,LBi,Lmi_psi1)*Lmi_psi1,0)[:,None]*beta**2

            dL_dR += -np.dot(_LBi_Lmi_psi1Vf.T,_LBi_Lmi_psi1).T * VVT_factor * beta**2
            dL_dR += 0.5*np.dot(_LBi_Lmi_psi1Vf.T,_LBi_Lmi_psi1).T**2 * beta**2
    else:
        # likelihood is not heteroscedatic
        dL_dR = -0.5 * num_data * output_dim * beta + 0.5 * trYYT * beta ** 2
        dL_dR += 0.5 * output_dim * (psi0.sum() * beta ** 2 - np.trace(A) * beta)
        dL_dR += beta * (0.5 * np.sum(A * DBi_plus_BiPBi) - data_fit)
    return dL_dR

def _compute_log_marginal_likelihood(num_data, output_dim, beta, het_noise, psi0, A, LB, trYYT, data_fit,Y):
    #compute log marginal likelihood
    if het_noise:
        lik_1 = -0.5 * num_data * output_dim * np.log(2. * np.pi) + 0.5 * np.sum(np.log(beta)) - 0.5 * np.sum(beta * np.square(Y).sum(axis=-1))
        lik_2 = -0.5 * output_dim * (np.sum(beta.flatten() * psi0) - np.trace(A))
    else:
        lik_1 = -0.5 * num_data * output_dim * (np.log(2. * np.pi) - np.log(beta)) - 0.5 * beta * trYYT
        lik_2 = -0.5 * output_dim * (np.sum(beta * psi0) - np.trace(A))
    lik_3 = -output_dim * (np.sum(np.log(np.diag(LB))))
    lik_4 = 0.5 * data_fit
    log_marginal = lik_1 + lik_2 + lik_3 + lik_4
    return log_marginal
