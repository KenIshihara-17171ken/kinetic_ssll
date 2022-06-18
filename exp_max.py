import numpy as np
import timeit

MAX_GA_ITERATIONS = 5000
GA_CONVERGENCE = 1e-4


def e_step(emd):


    loop = 1
    result = timeit.timeit(lambda: filter_function(emd), number=loop)
    emd.filtering_time = result / loop

    loop = 1
    result = timeit.timeit(lambda: smoothing_function(emd), number=loop)
    emd.smoothing_time = result / loop

    emd.e_step_time = emd.filtering_time + emd.smoothing_time
    
    


def m_step(emd):

    get_Q(emd)


def get_Q(emd):

    # Initialization
    emd.Q = np.zeros((emd.N, emd.N+1, emd.N+1))
    for i in range(emd.N):
        tmp = np.zeros((emd.N+1, emd.N+1))

        for t in range(1,emd.T):
            tmp += np.outer(emd.theta_s[t,i],emd.theta_s[t,i]) + emd.sigma_s[t,i] \
            - np.outer(emd.theta_s[t-1,i],emd.theta_s[t,i]) - emd.lag_one_covariance[t-1,i] \
            - np.outer(emd.theta_s[t,i],emd.theta_s[t-1,i]) - emd.lag_one_covariance[t-1,i].T \
            + np.outer(emd.theta_s[t-1,i],emd.theta_s[t-1,i]) + emd.sigma_s[t-1,i]
        emd.Q[i]=tmp/(emd.T-1)
    return emd.Q


def log_marginal_likelihood(emd):

    # Initialization of the log  marginal likelihood
    mllk = 0
    # Calculating the log marginal likelihood
    for t in range(emd.T):
        for i in range(emd.N):
            # Logarithm of the determinant
            sign, logdet_sigma_f = np.linalg.slogdet(emd.sigma_f[t,i])
            sign, logdet_sigma_o = np.linalg.slogdet(emd.sigma_o[t,i])
            # Computing the integrand at the map estimate
            a = emd.theta_f[t,i] - emd.theta_o[t,i]
            b = .5*np.dot(np.dot(a.T,emd.sigma_o_i[t,i]),a)
            # Calculation of psi summed over trials
            PSI = 0
            for l in range(emd.R):

                s = np.append(1, emd.spikes[t,l])
                PSI += np.log(1+np.exp(np.dot(emd.theta_f[t,i],s)))

            q = (np.dot(emd.theta_f[t,i], emd.FSUM[t,i]) - PSI) - b
            mllk += .5*logdet_sigma_f - .5*logdet_sigma_o + q

    return mllk


# Derivation of expectation parameters and Fisher information matrix
def cal_eta_G(emd,theta, t,R, N, spikes ):


    # Initialization
    # Expectation parameters

    eta = np.zeros(emd.N + 1)
    # Fisher information
    G = np.zeros((emd.N + 1, emd.N + 1 ))

    # Calculate the sum of the expectation parameters
    # and Fisher information matrix derived in each trial
    for l in range(emd.R):
        # print(len(spikes))
        # print("l",l)
        # print("t",t)

        F1 = np.append(1, spikes[t][l])
        r = 1/(1 + np.exp( -np.dot(theta, F1)) )
        eta += r * F1
        G += r*(1-r) * np.outer(F1, F1)

    return  eta, G

# def filter_function(R, T, N, FSUM, Q, spikes):
def filter_function(emd):
    emd.state_cov = np.zeros((emd.N,emd.N+1,emd.N+1))
    emd.theta_o = np.zeros((emd.N,emd.N+1))



    for i in range(emd.N):
        emd.state_cov[i] = 0.1*np.identity(emd.N+1)

    # Filtering at the initial time bin
    for i in range(emd.N):

        emd.state_cov_inv = np.linalg.inv(emd.state_cov[i])

        # Estimation of theta by Newton Rapson method
        max_dlpo = np.inf
        iterations = 0
        while max_dlpo > GA_CONVERGENCE:
            # Compute the first derivative of negative of the posterior prob. w.r.t. theta_max
            eta, G = cal_eta_G(emd,emd.theta_f[0,i],0,emd.R,emd.N,emd.spikes)
            dlpo = -(emd.FSUM[0,i] - eta) + np.dot(emd.state_cov_inv,emd.theta_f[0][i]- emd.theta_o[i])
            # Compute the second derivative of negative of the posterior prob. w.r.t. theta_max
            ddlpo = G +emd.state_cov_inv
            ddlpo_i = np.linalg.inv(ddlpo)
            # Update theta
            emd.theta_f[0,i] = emd.theta_f[0,i] - np.dot(ddlpo_i, dlpo)

            # Update the look guard
            max_dlpo = np.amax(np.absolute(dlpo)) / emd.R
            # Count iterations
            iterations += 1

            # Check for check for overrun
            if iterations == MAX_GA_ITERATIONS:
                raise Exception('The maximum-a-posterior gradient-ascent '+\
                    'algorithm did not converge before reaching the maximum '+\
                    'number iterations.')

        emd.sigma_f[0,i] = ddlpo_i
        emd.sigma_f_i[0,i] = ddlpo


        emd.sigma_o[0,i] =emd.state_cov[i]
        emd.sigma_o_i[0,i] = np.linalg.inv(emd.sigma_o[0,i])

    # Filtering for time bins t > 1
    for t in range(1,emd.T):

        for i in range(emd.N):

            # print("emd.Q[i]",emd.Q[i])
            # print("emd.sigma_f[t-1,i]",emd.sigma_f[t-1,i])
            # print("emd.sigma_o[t,i]",emd.sigma_o[t,i])
            # Compute one-step prediction density
            emd.sigma_o[t,i] = emd.sigma_f[t-1,i]+emd.Q[i]
           

            emd.sigma_o_i[t,i] = np.linalg.inv(emd.sigma_o[t,i])

            #Estimation of theta by Newton Rapson method
            max_dlpo = np.inf
            iterations = 0
            while max_dlpo > GA_CONVERGENCE:
                # Compute the first derivative of negative of the posterior prob. w.r.t. theta_max
                eta, G = cal_eta_G(emd,emd.theta_f[t,i],t, emd.R, emd.N, emd.spikes)
                tmp = emd.theta_f[t,i]-emd.theta_f[t-1,i]
                dlpo = -(emd.FSUM[t,i]- eta) + np.dot(emd.sigma_o_i[t,i], tmp)
                # Compute the second derivative of negative of the posterior prob. w.r.t. theta_max
                ddlpo = G + emd.sigma_o_i[t,i]
                ddlpo_i = np.linalg.inv(ddlpo)

                emd.theta_f[t,i] = emd.theta_f[t,i] - np.dot(ddlpo_i, dlpo)

                # Update the look guard
                max_dlpo = np.amax(np.absolute(dlpo)) / emd.R
                # Count iterations
                iterations += 1

                # Check for check for overrun
                if iterations == MAX_GA_ITERATIONS:
                    raise Exception('The maximum-a-posterior gradient-ascent '+\
                        'algorithm did not converge before reaching the maximum '+\
                        'number iterations.')

            emd.sigma_f[t,i] = ddlpo_i
            emd.sigma_f_i[t,i] = ddlpo

    return emd.theta_f, emd.sigma_f, emd.sigma_f_i, emd.sigma_o, emd.sigma_o_i





def smoothing_function(emd):
    # Initialization
    emd. theta_s = np.zeros((emd.T, emd.N, emd.N+1))
    emd.theta_o = np.zeros((emd.T, emd.N, emd.N+1))
    emd.A = np.zeros((emd.T-1, emd.N, emd.N+1, emd.N+1))
    emd.sigma_s = np.ones((emd.T, emd.N,emd.N+1,emd.N+1))
    emd.lag_one_covariance = np.zeros((emd.T, emd.N, emd.N+1, emd.N+1))


    emd. theta_s[emd.T-1] = emd.theta_f[emd.T-1]
    emd.sigma_s[emd.T-1] = emd.sigma_f[emd.T-1]

    #Calculation for tt=0~time-2,
    for tt in range(emd.T-1):
        t = emd.T-2-tt
        for i in range(emd.N):
            # Compute the A matrix
            emd.A[t,i] = np.dot(emd.sigma_f[t,i], emd.sigma_o_i[t+1,i])
            # Compute the backward-smoothed means
            emd.theta_o[t+1,i] = emd.theta_f[t,i]
            tmp = np.dot(emd.A[t,i],emd. theta_s[t+1,i]-emd.theta_o[t+1,i])
            emd. theta_s[t,i] = emd.theta_f[t,i] + tmp
            # Compute the backward-smoothed covariances
            tmp = np.dot(emd.A[t,i], emd.sigma_s[t+1,i]-emd.sigma_o[t+1,i])
            tmp = np.dot(tmp, emd.A[t,i].T)
            emd.sigma_s[t][i] = emd.sigma_f[t,i] + tmp

            emd.lag_one_covariance[t,i]=(np.dot(emd.A[t,i],emd.sigma_s[t+1,i]))

    return emd.sigma_s,emd.theta_o,emd.theta_s,emd.lag_one_covariance,emd.A
