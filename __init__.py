import numpy as np
import pdb

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import container
import exp_max

import matplotlib.pyplot as plt
from IPython.display import display, clear_output

# import time
import timeit



# def run(spikes,FSUM,state_cov,sigma_o,theta_o,max_iter=100, mstep=True):
# def run(spikes,state_cov,sigma_o,theta_o,max_iter=100, mstep=True):
def run(spikes,state_cov,init_cov,init_theta,max_iter=100, mstep=True):

    # emd = container.EMData(spikes,FSUM,state_cov,sigma_o,theta_o)
    # emd = container.EMData(spikes,state_cov,sigma_o,theta_o)
    emd = container.EMData(spikes,state_cov,init_cov,init_theta)
    lmc = emd.marg_llk(emd)

    # Iterate the EM algorithm until convergence or failure
    while (emd.iterations < max_iter) and (emd.convergence > emd.CONVERGED):
        # print('EM Iteration: %d - Convergence %.6f > %.6f' % (emd.iterations,
        #                                                         emd.convergence,
        #                                                         emd.CONVERGED))

        # print('EM Iteration: %d - Convergence %.6f > %.6f' % (emd.iterations,\
        # emd.convergence,emd.CONVERGED))

        # Perform EM

        # e_step_time_start = time.time()
        # exp_max.e_step(emd)
        # emd.e_step_time = time.time() - e_step_time_start

        
        exp_max.e_step(emd)
        loop = 1
        result = timeit.timeit(lambda: exp_max.e_step(emd), number=loop)
        emd.e_step_time = result / loop


        if mstep == True:

            # m_step_time_start = time.time()
            # exp_max.m_step(emd)
            # emd.m_step_time = time.time() - m_step_time_start

            exp_max.m_step(emd)
            loop = 1
            result = timeit.timeit(lambda: exp_max.m_step(emd), number=loop)
            emd.m_step_time = result / loop
           




        # Update previous and current log marginal values

        lmp = lmc

        # llk_time_start = time.time()
        # lmc = emd.marg_llk(emd)
        # emd.llk_time = time.time() - llk_time_start

        lmc = emd.marg_llk(emd)
        loop = 1
        result = timeit.timeit(lambda: emd.marg_llk(emd), number=loop)
        emd.llk_time = result / loop


        emd.mllk_list.append(lmc)
        emd.mllk = lmc

        emd.iterations_list.append(emd.iterations)

        # Update EM algorithm metadata
        emd.iterations += 1
        #
        emd.convergence = (lmp - lmc) / lmp

        print('Log marginal likelihood = %.6f' % (emd.mllk))


    print('Log Likelihood: ',emd.mllk, 'iter: ',emd.iterations)

    return emd


