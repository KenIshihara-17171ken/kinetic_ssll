import numpy as np
import pdb
# import probability
from ssll.probability import*


#importのところが怪しい#
log_marginal_functions = log_marginal

class EMData:
    def __init__( self, spikes, FSUM, Q, ID, mu ):
        self.spikes, self.FSUM, self.Q, self.ID, self.mu = spikes, FSUM, Q, ID, mu

        self.T= spikes.shape[0]-1
        self.R = spikes.shape[1]
        self.N = spikes.shape[2]





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
        self.iterations = 0
        self.CONVERGED = 1e-4
        self.convergence = np.inf
