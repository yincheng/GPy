"""
Modified By: Yin Cheng Ng
This is Sohl-Dickstein's implementation of ADAGrad, modified to do doubly stochastic
variational inference, instead of mini-batch type sgd

Adapted from:
Author: Ben Poole, Jascha Sohl-Dickstein (2013)
This software is made available under the Creative Commons
Attribution-Noncommercial License.
( http://creativecommons.org/licenses/by-nc/3.0/ )

This is an implementation of the ADAGrad algorithm: 
    Duchi, J., Hazan, E., & Singer, Y. (2010).
    Adaptive subgradient methods for online learning and stochastic optimization.
    Journal of Machine Learning Research
    http://www.eecs.berkeley.edu/Pubs/TechRpts/2010/EECS-2010-24.pdf
"""

#from numpy import *
import numpy as np

class ADAGrad(object):

    def __init__(self, f_df, theta, n_samples, max_iter = 100, reps=1, learning_rate=0.1, args=(), kwargs={}):

        self.reps = reps
        self.learning_rate = learning_rate

        #self.N = len(subfunction_references)
        #self.sub_ref = subfunction_references
        self.f_df = f_df
        self.n_samples = n_samples
        self.args = args
        self.kwargs = kwargs

        self.num_steps = 0.
        self.theta = theta.copy().reshape((-1,1))
        self.grad_history = np.zeros_like(self.theta)
        self.M = self.theta.shape[0]
        
        self.theta_del_norm_hist = np.array([])
        self.f_hist = np.array([])
        self.theta_history = np.array([np.transpose(self.theta)[0]])

        #self.f = np.ones((self.N))*np.nan


    def optimize(self, num_passes = 10, num_steps = 100):
#        if num_steps==None:
#            num_steps = num_passes*self.N
        for i in np.arange(0, num_steps):
            if not self.optimization_step():
                break
            if np.mod(i, 10) == 0.:
                print 'iter. '+str(i)+': '+str(self.f_hist[-1])
        #print 'L ', self.L
        return self.theta

    def optimization_step(self):
        #idx = np.random.randint(self.N)
        gradii = np.zeros_like(self.theta)
        lossii = 0.
        for i in np.arange(0, self.reps):
            lossi, gradi = self.f_df(self.theta, self.n_samples, *self.args, **self.kwargs)
            lossii += lossi / self.reps
            gradii += gradi.reshape(gradii.shape) / self.reps

        self.num_steps += 1.
        learning_rates = self.learning_rate / (np.sqrt(1./self.num_steps + self.grad_history))
        learning_rates[np.isinf(learning_rates)] = self.learning_rate
        self.theta -= learning_rates * gradii
        self.grad_history += gradii**2
        #self.f[idx] = lossii
        
        self.theta_history = np.r_[self.theta_history, np.transpose(self.theta)]
        self.theta_del_norm_hist = np.append(self.theta_del_norm_hist, np.linalg.norm(gradi))
        self.f_hist = np.append(self.f_hist, lossi)

        if not np.isfinite(lossii):
            print("Non-finite subfunction.  Ending run.")
            return False
        return True

