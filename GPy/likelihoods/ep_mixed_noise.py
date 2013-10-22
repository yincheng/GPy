import numpy as np
from scipy import stats
from ..util.linalg import pdinv,mdot,jitchol,chol_inv,DSYR,tdot,dtrtrs
from likelihood import likelihood
from gaussian import Gaussian
from ep import EP
from collections import Counter

class EPMixedNoise(likelihood):
    def __init__(self,data_list,noise_models_list):
        """
        Expectation Propagation for mixed noise models (coregionalized outputs)

        Arguments
        ---------
        :param data_list: list of outputs
        :param noise_models_list: a list of noise models
        """

        assert len(data_list) == len(noise_models_list)

        self.data = np.vstack(data_list)
        _transf_data = [] #NOTE this might not be needed.

        self.num_data, self.output_dim = self.data.shape #FIXME this is not the original output_dim!!!

        self._offset = np.zeros((1, self.output_dim)) #TODO handle normalization
        self._scale = np.ones((1, self.output_dim)) #TODO handle normalization

        self.outputs = []
        self.noise_models = []
        self.not_ep = []
        self.slices = []
        self.Nparams = 0
        self.Nparams_list = []

        start = 0
        for dj,mj,zj in zip(data_list,noise_models_list,range(len(data_list))):
            end = start + dj.shape[0]
            self.slices.append( slice(start,end) )
            start = end

            if isinstance(mj,Gaussian): #TODO Gaussian should be a noise model also
                self.not_ep.append(zj) #TODO: Is this useful?
                self.noise_models.append(mj.__class__.__name__)

                self.outputs.append(mj) #TODO handle variance and normalization of dj
                _transf_data.append(self.outputs[-1].Y) #NOTE this might not be needed

            else:
                self.outputs.append(EP(dj, mj))
                self.noise_models.append(mj.__class__.__name__)
                _transf_data.append(self.outputs[-1].noise_model._preprocess_values(dj))

            self.Nparams += self.outputs[-1]._get_params().size
            self.Nparams_list.append(self.outputs[-1]._get_params().size)

        self.is_heteroscedastic = True


        self._transf_data = np.vstack(_transf_data)
        #self._set_params(np.asarray(noise_params)) #TODO

        #Initial values - Likelihood approximation parameters:
        #p(y|f) = t(f|tau_tilde,v_tilde)
        self.tau_tilde = np.zeros(self.num_data)
        self.v_tilde = np.zeros(self.num_data)

        #initial values for the GP variables
        self.Y = np.zeros((self.num_data,1))
        self.precision = np.ones(self.num_data)[:,None]

        #Overwrite values in case there is a Gaussian model in the mixture
        for mj in self.not_ep: #TODO check if this is needed
            #p(y|f) = t(f|tau_tilde,v_tilde)
            self.tau_tilde[self.slices[mj]] = 1./self.outputs[mj]._get_params()
            self.v_tilde[self.slices[mj]] = self.Y[self.slices[mj],:].flatten() * self.tau_tilde[self.slices[mj]]

            #initial values for the GP variables
            self.Y[self.slices[mj],:] = self.outputs[mj].Y
            self.precision[self.slices[mj]] = 1./self.outputs[mj]._get_params()

        self.covariance_matrix = np.diag(1./self.precision.flatten())
        self.Z = 0
        self.YYT = None
        self.V = self.precision * self.Y
        self.VVT_factor = self.V
        self.trYYT = 0.

        super(EPMixedNoise, self).__init__()

    def restart(self):
        for sj,nj in zip(self.slices,range(len(self.slices))):
            if nj not in self.not_ep:
                self.tau_tilde[sj] = 0
                self.v_tilde[sj] = 0
                self.Y[sj,:] = 0
                self.precision[sj,:] = 1.
        self.Z = 0
        self.YYT = None
        self.V = self.precision * self.Y
        self.VVT_factor = self.V
        self.trYYT = 0.
        self.covariance_matrix = np.diag(1./self.precision.flatten())

    def _get_params(self):
        params = []
        for lik in self.outputs:
            if isinstance(lik,Gaussian): #TODO Gaussian should be a noise model also
                params.append(lik._get_params().flatten())
            else:
                params.append(lik.noise_model._get_params().flatten())
        return np.hstack(params)

    def _get_param_names(self):
        counter = Counter(self.noise_models)
        all_names = []

        for lik,noise in zip(self.outputs[::-1],self.noise_models[::-1]):
            string = noise + '_'

            noise_params = []
            if isinstance(lik,Gaussian): #TODO Gaussian should be a noise model also
                for p in lik._get_param_names()[::-1]:
                    noise_params.append(string + p + '_%s' %(counter[noise] - 1))
            else:
                for p in lik.noise_model._get_param_names()[::-1]:
                    noise_params.append(string + p + '_%s' %(counter[noise] - 1))
            counter[noise] -= 1
            all_names += noise_params
        return all_names[::-1]

    def _set_params(self,p):
        if self.Nparams: #FIXME
            cs_params = np.cumsum([0]+self.Nparams_list)
            for i in range(len(self.Nparams_list)):
                self.outputs[i]._set_params(p[cs_params[i]:cs_params[i+1]])
        else:
            pass

        for nj in self.not_ep:
            self.precision[self.slices[nj]] = 1./self.outputs[nj]._get_params()

        self.V = self.precision * self.Y
        self.VVT_factor = self.V
        self.covariance_matrix = np.diag(1./self.precision.flatten())


    def predictive_values(self,mu,var,full_cov,noise_model):
        """
        Predicts the output given the GP

        :param mu: GP's mean
        :param var: GP's variance
        :param full_cov: whether to return the full covariance matrix, or just the diagonal
        :type full_cov: False|True
        :param noise_model: noise model to use
        :type noise_model: integer
        """
        if full_cov:
            raise NotImplementedError, "Cannot make correlated predictions with an EP likelihood"
        return self.outputs[noise_model].predictive_values(mu,var,full_cov)

    def _gradients(self,partial):
        gradients = []
        for lik,sj in zip(self.outputs,self.slices):
            if isinstance(lik,Gaussian): #TODO Gaussian should be a noise model also
                gradients += [lik._gradients(partial[sj])]
            else:
                gradients += [lik.noise_model._gradients(partial[sj])]
        return np.hstack(gradients)

    def _compute_GP_variables(self):
        #Variables to be called from GP
        self.Z = 0
        ep_index = set(range(len(self.slices))) - set(self.not_ep)
        #for sj in self.slices:
        for sj in ep_index:
            mu_tilde = self.v_tilde[self.slices[sj]]/self.tau_tilde[self.slices[sj]]
            sigma_sum = 1./self.tau_[self.slices[sj]] + 1./self.tau_tilde[self.slices[sj]]
            mu_diff_2 = (self.v_[self.slices[sj]]/self.tau_[self.slices[sj]] - mu_tilde)**2
            self.Z += np.sum(np.log(self.Z_hat[self.slices[sj]])) + 0.5*np.sum(np.log(sigma_sum)) + 0.5*np.sum(mu_diff_2/sigma_sum) #Normalization constant, aka Z_ep
            self.Y[self.slices[sj],:] =  mu_tilde[:,None]
            self.precision[self.slices[sj],:] = self.tau_tilde[self.slices[sj]][:,None]

        self.YYT = np.dot(self.Y,self.Y.T)
        self.covariance_matrix = np.diag(1./self.precision.flatten())
        self.V = self.precision * self.Y
        self.VVT_factor = self.V
        self.trYYT = np.trace(self.YYT)

    def fit_full(self, K, epsilon=1e-3,power_ep=[1.,1.]):
        """
        The expectation-propagation algorithm.
        For nomenclature see Rasmussen & Williams 2006.

        :param epsilon: Convergence criterion, maximum squared difference allowed between mean updates to stop iterations (float)
        :type epsilon: float
        :param power_ep: Power EP parameters
        :type power_ep: list of floats

        """
        self.epsilon = epsilon
        self.eta, self.delta = power_ep

        #Initial values - Posterior distribution parameters: q(f|X,Y) = N(f|mu,Sigma)
        mu = np.zeros(self.num_data)
        Sigma = K.copy()

        """
        Initial values - Cavity distribution parameters:
        q_(f|mu_,sigma2_) = Product{q_i(f|mu_i,sigma2_i)}
        sigma_ = 1./tau_
        mu_ = v_/tau_
        """
        self.tau_ = np.empty(self.num_data,dtype=float)
        self.v_ = np.empty(self.num_data,dtype=float)

        #Initial values - Marginal moments
        z = np.empty(self.num_data,dtype=float)
        self.Z_hat = np.empty(self.num_data,dtype=float)
        phi = np.empty(self.num_data,dtype=float)
        mu_hat = np.empty(self.num_data,dtype=float)
        sigma2_hat = np.empty(self.num_data,dtype=float)

        #Approximation
        epsilon_np1 = self.epsilon + 1.
        epsilon_np2 = self.epsilon + 1.
       	self.iterations = 0
        self.np1 = [self.tau_tilde.copy()]
        self.np2 = [self.v_tilde.copy()]

        data_index = []
        model_index = []
        for sj,nj in zip(self.slices,range(len(self.slices))):
            if nj not in self.not_ep:
                _range = range(self.num_data)[sj]
                data_index += _range
                model_index += [nj]*len(_range)

        data_index = np.array(data_index)
        model_index = np.array(model_index)
        num_ep_data = len(data_index)

        while num_ep_data and (epsilon_np1 > self.epsilon or epsilon_np2 > self.epsilon):
            if len(self.not_ep) == len(self.outputs):
                break
            #update_order = np.random.permutation(self.num_data)
            #for i in update_order:
            update_order = np.random.permutation(num_ep_data)
            for i,mj in zip(data_index[update_order],model_index[update_order]):
                #Cavity distribution parameters
                self.tau_[i] = 1./Sigma[i,i] - self.eta*self.tau_tilde[i]
                self.v_[i] = mu[i]/Sigma[i,i] - self.eta*self.v_tilde[i]
                #Marginal moments
                #FIXME _transf_data needs to be removed
                self.Z_hat[i], mu_hat[i], sigma2_hat[i] = self.outputs[mj].noise_model.moments_match(self._transf_data[i],self.tau_[i],self.v_[i])
                #Site parameters update
                Delta_tau = self.delta/self.eta*(1./sigma2_hat[i] - 1./Sigma[i,i])
                Delta_v = self.delta/self.eta*(mu_hat[i]/sigma2_hat[i] - mu[i]/Sigma[i,i])
                self.tau_tilde[i] += Delta_tau
                self.v_tilde[i] += Delta_v
                #Posterior distribution parameters update
                DSYR(Sigma,Sigma[:,i].copy(), -float(Delta_tau/(1.+ Delta_tau*Sigma[i,i])))
                mu = np.dot(Sigma,self.v_tilde)
                self.iterations += 1
            #Sigma recomptutation with Cholesky decompositon
            Sroot_tilde_K = np.sqrt(self.tau_tilde)[:,None]*K
            B = np.eye(self.num_data) + np.sqrt(self.tau_tilde)[None,:]*Sroot_tilde_K
            L = jitchol(B)
            V,info = dtrtrs(L,Sroot_tilde_K,lower=1)
            Sigma = K - np.dot(V.T,V)
            mu = np.dot(Sigma,self.v_tilde)
            epsilon_np1 = sum((self.tau_tilde-self.np1[-1])**2)/self.num_data
            epsilon_np2 = sum((self.v_tilde-self.np2[-1])**2)/self.num_data
            self.np1.append(self.tau_tilde.copy())
            self.np2.append(self.v_tilde.copy())

        return self._compute_GP_variables()

    def fit_DTC(self, Kmm, Kmn, epsilon=1e-3,power_ep=[1.,1.]):
        """
        The expectation-propagation algorithm with sparse pseudo-input.
        For nomenclature see ... 2013.

        :param epsilon: Convergence criterion, maximum squared difference allowed between mean updates to stop iterations (float)
        :type epsilon: float
        :param power_ep: Power EP parameters
        :type power_ep: list of floats

        """
        self.epsilon = epsilon
        self.eta, self.delta = power_ep

        num_inducing = Kmm.shape[0]

        #TODO: this doesn't work with uncertain inputs!

        """
        Prior approximation parameters:
        q(f|X) = int_{df}{N(f|KfuKuu_invu,diag(Kff-Qff)*N(u|0,Kuu)} = N(f|0,Sigma0)
        Sigma0 = Qnn = Knm*Kmmi*Kmn
        """
        KmnKnm = np.dot(Kmn,Kmn.T)
        Lm = jitchol(Kmm)
        Lmi = chol_inv(Lm)
        Kmmi = np.dot(Lmi.T,Lmi)
        KmmiKmn = np.dot(Kmmi,Kmn)
        Qnn_diag = np.sum(Kmn*KmmiKmn,-2)
        LLT0 = Kmm.copy()

        #Kmmi, Lm, Lmi, Kmm_logdet = pdinv(Kmm)
        #KmnKnm = np.dot(Kmn, Kmn.T)
        #KmmiKmn = np.dot(Kmmi,Kmn)
        #Qnn_diag = np.sum(Kmn*KmmiKmn,-2)
        #LLT0 = Kmm.copy()

        """
        Posterior approximation: q(f|y) = N(f| mu, Sigma)
        Sigma = Diag + P*R.T*R*P.T + K
        mu = w + P*Gamma
        """
        mu = np.zeros(self.num_data)
        LLT = Kmm.copy()
        Sigma_diag = Qnn_diag.copy()

        """
        Initial values - Cavity distribution parameters:
        q_(g|mu_,sigma2_) = Product{q_i(g|mu_i,sigma2_i)}
        sigma_ = 1./tau_
        mu_ = v_/tau_
        """
        self.tau_ = np.empty(self.num_data,dtype=float)
        self.v_ = np.empty(self.num_data,dtype=float)

        #Initial values - Marginal moments
        z = np.empty(self.num_data,dtype=float)
        self.Z_hat = np.empty(self.num_data,dtype=float)
        phi = np.empty(self.num_data,dtype=float)
        mu_hat = np.empty(self.num_data,dtype=float)
        sigma2_hat = np.empty(self.num_data,dtype=float)

        #Approximation
        epsilon_np1 = 1
        epsilon_np2 = 1
       	self.iterations = 0
        np1 = [self.tau_tilde.copy()]
        np2 = [self.v_tilde.copy()]
        while epsilon_np1 > self.epsilon or epsilon_np2 > self.epsilon:
            update_order = np.random.permutation(self.num_data)
            for i in update_order:
                #Cavity distribution parameters
                self.tau_[i] = 1./Sigma_diag[i] - self.eta*self.tau_tilde[i]
                self.v_[i] = mu[i]/Sigma_diag[i] - self.eta*self.v_tilde[i]
                #Marginal moments
                self.Z_hat[i], mu_hat[i], sigma2_hat[i] = self.noise_model.moments_match(self._transf_data[i],self.tau_[i],self.v_[i])
                #Site parameters update
                Delta_tau = self.delta/self.eta*(1./sigma2_hat[i] - 1./Sigma_diag[i])
                Delta_v = self.delta/self.eta*(mu_hat[i]/sigma2_hat[i] - mu[i]/Sigma_diag[i])
                self.tau_tilde[i] += Delta_tau
                self.v_tilde[i] += Delta_v
                #Posterior distribution parameters update
                DSYR(LLT,Kmn[:,i].copy(),Delta_tau) #LLT = LLT + np.outer(Kmn[:,i],Kmn[:,i])*Delta_tau
                L = jitchol(LLT)
                #cholUpdate(L,Kmn[:,i]*np.sqrt(Delta_tau))
                V,info = dtrtrs(L,Kmn,lower=1)
                Sigma_diag = np.sum(V*V,-2)
                si = np.sum(V.T*V[:,i],-1)
                mu += (Delta_v-Delta_tau*mu[i])*si
                self.iterations += 1
            #Sigma recomputation with Cholesky decompositon
            LLT = LLT0 + np.dot(Kmn*self.tau_tilde[None,:],Kmn.T)
            L = jitchol(LLT)
            V,info = dtrtrs(L,Kmn,lower=1)
            V2,info = dtrtrs(L.T,V,lower=0)
            Sigma_diag = np.sum(V*V,-2)
            Knmv_tilde = np.dot(Kmn,self.v_tilde)
            mu = np.dot(V2.T,Knmv_tilde)
            epsilon_np1 = sum((self.tau_tilde-np1[-1])**2)/self.num_data
            epsilon_np2 = sum((self.v_tilde-np2[-1])**2)/self.num_data
            np1.append(self.tau_tilde.copy())
            np2.append(self.v_tilde.copy())

        self._compute_GP_variables()

    def fit_FITC(self, Kmm, Kmn, Knn_diag, epsilon=1e-3,power_ep=[1.,1.]):
        """
        The expectation-propagation algorithm with sparse pseudo-input.
        For nomenclature see Naish-Guzman and Holden, 2008.

        :param epsilon: Convergence criterion, maximum squared difference allowed between mean updates to stop iterations (float)
        :type epsilon: float
        :param power_ep: Power EP parameters
        :type power_ep: list of floats
        """
        self.epsilon = epsilon
        self.eta, self.delta = power_ep

        num_inducing = Kmm.shape[0]

        """
        Prior approximation parameters:
        q(f|X) = int_{df}{N(f|KfuKuu_invu,diag(Kff-Qff)*N(u|0,Kuu)} = N(f|0,Sigma0)
        Sigma0 = diag(Knn-Qnn) + Qnn, Qnn = Knm*Kmmi*Kmn
        """
        Lm = jitchol(Kmm)
        Lmi = chol_inv(Lm)
        Kmmi = np.dot(Lmi.T,Lmi)
        P0 = Kmn.T
        KmnKnm = np.dot(P0.T, P0)
        KmmiKmn = np.dot(Kmmi,P0.T)
        Qnn_diag = np.sum(P0.T*KmmiKmn,-2)
        Diag0 = Knn_diag - Qnn_diag
        R0 = jitchol(Kmmi).T

        """
        Posterior approximation: q(f|y) = N(f| mu, Sigma)
        Sigma = Diag + P*R.T*R*P.T + K
        mu = w + P*Gamma
        """
        self.w = np.zeros(self.num_data)
        self.Gamma = np.zeros(num_inducing)
        mu = np.zeros(self.num_data)
        P = P0.copy()
        R = R0.copy()
        Diag = Diag0.copy()
        Sigma_diag = Knn_diag
        RPT0 = np.dot(R0,P0.T)

        """
        Initial values - Cavity distribution parameters:
        q_(g|mu_,sigma2_) = Product{q_i(g|mu_i,sigma2_i)}
        sigma_ = 1./tau_
        mu_ = v_/tau_
        """
        self.tau_ = np.empty(self.num_data,dtype=float)
        self.v_ = np.empty(self.num_data,dtype=float)

        #Initial values - Marginal moments
        z = np.empty(self.num_data,dtype=float)
        self.Z_hat = np.empty(self.num_data,dtype=float)
        phi = np.empty(self.num_data,dtype=float)
        mu_hat = np.empty(self.num_data,dtype=float)
        sigma2_hat = np.empty(self.num_data,dtype=float)

        #Approximation
        epsilon_np1 = 1
        epsilon_np2 = 1
       	self.iterations = 0
        self.np1 = [self.tau_tilde.copy()]
        self.np2 = [self.v_tilde.copy()]
        while epsilon_np1 > self.epsilon or epsilon_np2 > self.epsilon:
            update_order = np.random.permutation(self.num_data)
            for i in update_order:
                #Cavity distribution parameters
                self.tau_[i] = 1./Sigma_diag[i] - self.eta*self.tau_tilde[i]
                self.v_[i] = mu[i]/Sigma_diag[i] - self.eta*self.v_tilde[i]
                #Marginal moments
                self.Z_hat[i], mu_hat[i], sigma2_hat[i] = self.noise_model.moments_match(self._transf_data[i],self.tau_[i],self.v_[i])
                #Site parameters update
                Delta_tau = self.delta/self.eta*(1./sigma2_hat[i] - 1./Sigma_diag[i])
                Delta_v = self.delta/self.eta*(mu_hat[i]/sigma2_hat[i] - mu[i]/Sigma_diag[i])
                self.tau_tilde[i] += Delta_tau
                self.v_tilde[i] += Delta_v
                #Posterior distribution parameters update
                dtd1 = Delta_tau*Diag[i] + 1.
                dii = Diag[i]
                Diag[i] = dii - (Delta_tau * dii**2.)/dtd1
                pi_ = P[i,:].reshape(1,num_inducing)
                P[i,:] = pi_ - (Delta_tau*dii)/dtd1 * pi_
                Rp_i = np.dot(R,pi_.T)
                RTR = np.dot(R.T,np.dot(np.eye(num_inducing) - Delta_tau/(1.+Delta_tau*Sigma_diag[i]) * np.dot(Rp_i,Rp_i.T),R))
                R = jitchol(RTR).T
                self.w[i] += (Delta_v - Delta_tau*self.w[i])*dii/dtd1
                self.Gamma += (Delta_v - Delta_tau*mu[i])*np.dot(RTR,P[i,:].T)
                RPT = np.dot(R,P.T)
                Sigma_diag = Diag + np.sum(RPT.T*RPT.T,-1)
                mu = self.w + np.dot(P,self.Gamma)
                self.iterations += 1
            #Sigma recomptutation with Cholesky decompositon
            Iplus_Dprod_i = 1./(1.+ Diag0 * self.tau_tilde)
            Diag = Diag0 * Iplus_Dprod_i
            P = Iplus_Dprod_i[:,None] * P0
            safe_diag = np.where(Diag0 < self.tau_tilde, self.tau_tilde/(1.+Diag0*self.tau_tilde), (1. - Iplus_Dprod_i)/Diag0)
            L = jitchol(np.eye(num_inducing) + np.dot(RPT0,safe_diag[:,None]*RPT0.T))
            R,info = dtrtrs(L,R0,lower=1)
            RPT = np.dot(R,P.T)
            Sigma_diag = Diag + np.sum(RPT.T*RPT.T,-1)
            self.w = Diag * self.v_tilde
            self.Gamma = np.dot(R.T, np.dot(RPT,self.v_tilde))
            mu = self.w + np.dot(P,self.Gamma)
            epsilon_np1 = sum((self.tau_tilde-self.np1[-1])**2)/self.num_data
            epsilon_np2 = sum((self.v_tilde-self.np2[-1])**2)/self.num_data
            self.np1.append(self.tau_tilde.copy())
            self.np2.append(self.v_tilde.copy())

        return self._compute_GP_variables()

    def update_additional_params(self,gp_mean):
        self.noise_model.update_additional_params(gp_mean,self.data)
