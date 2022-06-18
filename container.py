import numpy as np
import pdb
from ssll_kinetic.probability import*


log_marginal_functions = log_marginal

class EMData:
    def __init__( self, spikes, Q, ID, mu ):
        # def run(spikes,FSUM,state_cov,sigma_o,theta_o,max_iter=100, mstep=True, show_llk=True):

        self.spikes,  self.Q, self.ID, self.mu = spikes, Q, ID, mu

        self.T= spikes.shape[0]-1
        self.R = spikes.shape[1]
        self.N = spikes.shape[2]


        self.e_step_time = 0

        self.filtering_time = 0
        self.smoothing_time = 0


        self.m_step_time = 0
        self.llk_time = 0


        self.FSUM = np.zeros((self.T, self.N, self.N+1))
        for l in range(self.R):
            for t in range(1, self.T+1):
                for n in range(self.N):
                    self.FSUM[t-1, n] += np.append( self.spikes[t, l, n],  self.spikes[t, l, n]* self.spikes[t-1,l])

        


        self.theta_f = np.zeros((self.T,self.N,  self.N+1))
        self.theta_s = np.zeros((self.T, self.N, self.N+1))
        self.theta_o = np.zeros((self.T, self.N, self.N+1))


        self.sigma_f = np.zeros((self.T,self.N,self.N+1,self.N+1))
        self.sigma_o = np.zeros((self.T,self.N,self.N+1,self.N+1))

        for t in range(self.T):
            for i in range(self.N):
                 #Q[i]=0.1*np.identity(N+1)
                self.sigma_f[t,i]=np.identity(self.N+1)
                self.sigma_o[t,i]=np.identity(self.N+1)



        self.sigma_f_i = np.linalg.inv(self.sigma_f)
        self.sigma_o_i = np.linalg.inv(self.sigma_o)

        self.sigma_s = np.ones((self.T,self.N,self.N+1,self.N+1))
        self.A = np.zeros((self.T-1, self.N, self.N+1, self.N+1))
        self.lag_one_covariance = np.zeros((self.T, self.N, self.N+1, self.N+1))


        self.marg_llk = log_marginal_functions
        self.mllk = np.inf
        self.mllk_list = []
        self.iterations_list = []
        self.iterations = 0
        self.CONVERGED = 1e-4
        self.convergence = np.inf
