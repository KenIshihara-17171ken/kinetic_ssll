import numpy as np

def log_marginal(emd):
    # Initialization of the log  marginal likelihood
    log_p = 0
    # Calculating the log marginal likelihood
    for t in range(emd.T):
        for i in range(emd.N):
            # Logarithm of the determinant
            # sign, logdet_sigma_f = np.linalg.slogdet(emd.sigma_f[t,i])
            # sign, logdet_sigma_o = np.linalg.slogdet(emd.sigma_o[t,i])

            # Computing the integrand at the map estimate
            a = emd.theta_f[t,i] - emd.theta_o[t,i]
            b = .5*np.dot(np.dot(a.T,emd.sigma_o_i[t,i]),a)
            # Calculation of psi summed over trials
            PSI = 0
            for l in range(emd.R):

                s = np.append(1, emd.spikes[t,l])
                PSI += np.log(1+np.exp(np.dot(emd.theta_f[t,i],s)))

            q = (np.dot(emd.theta_f[t,i], emd.FSUM[t,i]) - PSI) - b
            # log_p += .5*logdet_sigma_f - .5*logdet_sigma_o + q
            log_p += .5*np.log(np.linalg.det(emd.sigma_f[t,i])) \
            - .5*np.log(np.linalg.det(emd.sigma_o[t,i])) + q


    return log_p
