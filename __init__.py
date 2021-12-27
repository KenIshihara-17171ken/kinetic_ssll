import numpy as np
import pdb
from ssll_kinetic.container import*
from ssll_kinetic.exp_max import*

#Use it when test
import container


def run(I,spikes,FSUM,Q,ID,mu,max_iter=100, mstep=True):


    emd = container.EMData(spikes,FSUM,Q,ID,mu)
    lmc = emd.marg_llk(emd)

    # Iterate the EM algorithm until convergence or failure
    while (emd.iterations < max_iter) and (emd.convergence > emd.CONVERGED):
        print('EM Iteration: %d - Convergence %.6f > %.6f' % (emd.iterations,\
        emd.convergence,emd.CONVERGED))

        # Perform EM
        e_step(emd)
        if mstep == True:
            m_step(emd)
        # Update previous and current log marginal values

        lmp = lmc
        lmc = emd.marg_llk(emd)

        emd.mllk_list.append(lmc)
        emd.mllk = lmc

        I.append(emd.iterations)

        # Update EM algorithm metadata
        emd.iterations += 1
        emd.convergence = (lmp - lmc) / lmp

    print('Log marginal likelihood = %.6f' % (emd.mllk))
    return emd
