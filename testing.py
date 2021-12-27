import unittest

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
import math
import matplotlib.pyplot as plt
import random
import time

import  synthsis
import __init__



class TestEstimator(unittest.TestCase):


    def test_1_fo_varying(self):
        T = 10
        R = 10
        N = 2
        I= []

        THETA= synthsis.get_THETA_function(T,R,N)
        spikes=synthsis.get_S_function(T,R,N,THETA)
        FSUM=synthsis.get_FSUM(spikes,T,R,N)


        Q=np.zeros((N,N+1,N+1))
        for i in range(N):
             #Q[i]=0.1*np.identity(N+1)
            Q[i]=0.05*np.identity(N+1)

        ID = np.zeros((N,N+1,N+1))
        mu = np.zeros((N,N+1))

        print("Test First-Order Time-Varying Interactions.")


        emd = __init__.run(I,spikes,FSUM,Q,ID,mu,max_iter=100, mstep=True)

        # Check the consistency with the expected result.
        expected_mllk =  -127.730249
        print('Log marginal likelihood = %.6f (expected)' % expected_mllk)
        self.assertFalse(np.absolute(emd.mllk-expected_mllk) > 1e-6)


if __name__ == "__main__":
    unittest.main()
