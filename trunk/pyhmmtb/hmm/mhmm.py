'''
Created on 12.06.2013

@author: christian
'''

import numpy as np
from pyhmmtb.tools import normalise, em_converged
from pyhmmtb.stats.mixgauss import mixgauss_Mstep, mixgauss_prob
from pyhmmtb.hmm.misc import mc_sample
from pyhmmtb.stats.misc import sample_discrete
from pyhmmtb.stats.gaussian import gaussian_sample
from pyhmmtb.tools import max_mult

def mhmm_sample(T, numex, initial_prob, transmat, mu, Sigma, mixmat=None):
    '''
    % SAMPLE_MHMM Generate random sequences from an HMM with (mixtures of) Gaussian output.
    % [obs, hidden] = sample_mhmm(T, numex, initial_prob, transmat, mu, Sigma, mixmat)
    %
    % INPUTS:
    % T - length of each sequence
    % numex - num. sequences
    % init_state_prob(i) = Pr(Q(1) = i)
    % transmat(i,j) = Pr(Q(t+1)=j | Q(t)=i)
    % mu(:,j,k) = mean of Y(t) given Q(t)=j, M(t)=k
    % Sigma(:,:,j,k) = cov. of Y(t) given Q(t)=j, M(t)=k
    % mixmat(j,k) = Pr(M(t)=k | Q(t)=j) : set to ones(Q,1) or omit if single mixture
    %
    % OUTPUT:
    % obs(:,t,l) = observation vector at time t for sequence l
    % hidden(t,l) = the hidden state at time t for sequence l
    '''
    
    assert initial_prob.ndim == 1
    
    Q = len(initial_prob);
    if mixmat==None:
        mixmat = np.ones((Q,1))
    O = mu.shape[0]
    hidden = np.zeros((T, numex))
    obs = np.zeros((O, T, numex))
    
    hidden = mc_sample(initial_prob, transmat, T, numex).T
    for i in range(0,numex):
        for t in range(0,T):
            q = hidden[t,i]
            m = np.asscalar(sample_discrete(mixmat[q,:], 1, 1))
            obs[:,t,i] = gaussian_sample(mu[:,q,m], Sigma[:,:,q,m], 1)
    
    return obs, hidden

def mhmm_em(data, prior, transmat, mu, Sigma, mixmat=None, **kwargs):
    '''
    % LEARN_MHMM Compute the ML parameters of an HMM with (mixtures of) Gaussians output using EM.
    % [ll_trace, prior, transmat, mu, sigma, mixmat] = learn_mhmm(data, ...
    %   prior0, transmat0, mu0, sigma0, mixmat0, ...) 
    %
    % Notation: Q(t) = hidden state, Y(t) = observation, M(t) = mixture variable
    %
    % INPUTS:
    % data{ex}(:,t) or data(:,t,ex) if all sequences have the same length
    % prior(i) = Pr(Q(1) = i), 
    % transmat(i,j) = Pr(Q(t+1)=j | Q(t)=i)
    % mu(:,j,k) = E[Y(t) | Q(t)=j, M(t)=k ]
    % Sigma(:,:,j,k) = Cov[Y(t) | Q(t)=j, M(t)=k]
    % mixmat(j,k) = Pr(M(t)=k | Q(t)=j) : set to [] or ones(Q,1) if only one mixture component
    %
    % Optional parameters may be passed as 'param_name', param_value pairs.
    % Parameter names are shown below; default values in [] - if none, argument is mandatory.
    %
    % 'max_iter' - max number of EM iterations [10]
    % 'thresh' - convergence threshold [1e-4]
    % 'verbose' - if 1, print out loglik at every iteration [1]
    % 'cov_type' - 'full', 'diag' or 'spherical' ['full']
    %
    % To clamp some of the parameters, so learning does not change them:
    % 'adj_prior' - if 0, do not change prior [1]
    % 'adj_trans' - if 0, do not change transmat [1]
    % 'adj_mix' - if 0, do not change mixmat [1]
    % 'adj_mu' - if 0, do not change mu [1]
    % 'adj_Sigma' - if 0, do not change Sigma [1]
    %
    % If the number of mixture components differs depending on Q, just set  the trailing
    % entries of mixmat to 0, e.g., 2 components if Q=1, 3 components if Q=2,
    % then set mixmat(1,3)=0. In this case, B2(1,3,:)=1.0.
    '''
    
    max_iter = kwargs.pop('max_iter', 10)
    thresh = kwargs.pop('thresh', 1e-4)
    verbose = kwargs.pop('verbose', True)
    cov_type = kwargs.pop('cov_type', 'full')
    adj_prior = kwargs.pop('adj_prior', True)
    adj_trans = kwargs.pop('adj_trans', True)
    adj_mix = kwargs.pop('adj_mix', True)
    adj_mu = kwargs.pop('adj_mu', True)
    adj_Sigma = kwargs.pop('adj_Sigma', True)
      
    previous_loglik = -np.Inf
    loglik = 0
    converged = False
    num_iter = 1
    LL = []

    if not isinstance(data, list):
        data = [data[:,:,i] for i in range(data.shape[2])]
    numex = len(data)
    
    O = data[0].shape[0]
    Q = len(prior)
    if mixmat==None:
        mixmat = np.ones((Q,1))
    M = mixmat.shape[1]
    if M == 1:
        adj_mix = False
    
    while (num_iter <= max_iter) and not converged:
        # E step
        loglik, exp_num_trans, exp_num_visits1, postmix, m, ip, op = ess_mhmm(prior, transmat, mixmat, mu, Sigma, data)
      
        # M step
        if adj_prior:
            prior, _ = normalise(exp_num_visits1)
        if adj_trans:
            transmat, _ = mk_stochastic(exp_num_trans)
        if adj_mix:
            mixmat, _ = mk_stochastic(postmix)
        if adj_mu or adj_Sigma:
            postmixx = np.reshape(np.transpose(postmix), (M*Q,))
            mm = np.reshape(np.transpose(m,(0,2,1)), (O, M*Q))
            opp = np.reshape(np.transpose(op, (0,1,3,2)), (O*O, M*Q))
            ipp = np.reshape(np.transpose(ip), (M*Q,))
            mu2, Sigma2 = mixgauss_Mstep(postmixx, mm, opp, ipp, cov_type=cov_type)
            if adj_mu:
                mu = np.transpose(np.reshape(mu2, (O, M, Q)), (0,2,1))
            if adj_Sigma:
                Sigma = np.transpose(np.reshape(Sigma2, (O, O, M, Q)), (0, 1, 3, 2))
      
        if verbose:
            print 'iteration %d, loglik = %f' % (num_iter, loglik)
        num_iter =  num_iter + 1
        converged, _ = em_converged(loglik, previous_loglik, thresh)
        previous_loglik = loglik;
        LL.append(loglik);
        
    return LL, prior, transmat, mu, Sigma, mixmat

def ess_mhmm(prior, transmat, mixmat, mu, Sigma, data):
    '''
    % ESS_MHMM Compute the Expected Sufficient Statistics for a MOG Hidden Markov Model.
    %
    % Outputs:
    % exp_num_trans(i,j)   = sum_l sum_{t=2}^T Pr(Q(t-1) = i, Q(t) = j| Obs(l))
    % exp_num_visits1(i)   = sum_l Pr(Q(1)=i | Obs(l))
    %
    % Let w(i,k,t,l) = P(Q(t)=i, M(t)=k | Obs(l))
    % where Obs(l) = Obs(:,:,l) = O_1 .. O_T for sequence l
    % Then 
    % postmix(i,k) = sum_l sum_t w(i,k,t,l) (posterior mixing weights/ responsibilities)
    % m(:,i,k)   = sum_l sum_t w(i,k,t,l) * Obs(:,t,l)
    % ip(i,k) = sum_l sum_t w(i,k,t,l) * Obs(:,t,l)' * Obs(:,t,l)
    % op(:,:,i,k) = sum_l sum_t w(i,k,t,l) * Obs(:,t,l) * Obs(:,t,l)'
    '''

    verbose = False

    # [O T numex] = size(data);
    numex = len(data)
    O = data[0].shape[0]
    Q = len(prior)
    M = mixmat.shape[1]
    exp_num_trans = np.zeros((Q, Q));
    exp_num_visits1 = np.zeros((Q, 1));
    postmix = np.zeros((Q, M));
    m = np.zeros((O, Q, M));
    op = np.zeros((O, O, Q, M));
    ip = np.zeros((Q, M));

    mix = M > 1

    loglik = 0
    if verbose:
        print 'forwards-backwards example # '
    for ex in range(0, numex):
        if verbose:
            print '%d ' % ex
        # obs = data(:,:,ex);
        obs = data[ex]
        T = obs.shape[1]
        if mix:
            B, B2 = mixgauss_prob(obs, mu, Sigma, mixmat)
            alpha, beta, gamma, current_loglik, xi_summed, gamma2 = fwdback(prior, transmat, B, obslik2=B2, mixmat=mixmat, compute_xi=True, compute_gamma2=True)
        else:
            B, B2 = mixgauss_prob(obs, mu, Sigma)
            alpha, beta, gamma, current_loglik, xi_summed, _ = fwdback(prior, transmat, B)
        loglik = loglik + current_loglik
        if verbose:
            print 'll at ex %d = %f\n' % (ex, loglik)
        
        exp_num_trans = exp_num_trans + xi_summed  # sum(xi,2)
        exp_num_visits1 = exp_num_visits1 + gamma[:, 0]
        
        if mix:
            postmix = postmix + np.sum(gamma2, 2)
        else:
            postmix = postmix + np.sum(gamma, 1) 
            gamma2 = np.reshape(gamma, (Q, 1, T))  # gamma2(i,m,t) = gamma(i,t)
        for i in range(0, Q):
            for k in range(0, M):
                w = np.reshape(gamma2[i, k, :], (1, T))  # w(t) = w(i,k,t,l)
                wobs = np.multiply(obs,w)  # np.repmat(w, [O 1]) # wobs(:,t) = w(t) * obs(:,t)
                m[:, i, k] = m[:, i, k] + np.sum(wobs, 1)  # m(:) = sum_t w(t) obs(:,t)
                op[:, :, i, k] = op[:, :, i, k] + np.dot(wobs, obs.T)  # op(:,:) = sum_t w(t) * obs(:,t) * obs(:,t)'
                ip[i, k] = ip[i, k] + np.sum(np.sum(np.multiply(wobs, obs), 1))  # ip = sum_t w(t) * obs(:,t)' * obs(:,t)
    if verbose:
        print

    return loglik, exp_num_trans, np.asarray(exp_num_visits1)[:,0], postmix, m, ip, op

def mk_stochastic(T):
    '''
    % MK_STOCHASTIC Ensure the argument is a stochastic matrix, i.e., the sum over the last dimension is 1.
    % [T,Z] = mk_stochastic(T)
    %
    % If T is a vector, it will sum to 1.
    % If T is a matrix, each row will sum to 1.
    % If T is a 3D array, then sum_k T(i,j,k) = 1 for all i,j.
    
    % Set zeros to 1 before dividing
    % This is valid since S(j) = 0 iff T(i,j) = 0 for all j
    '''
    
    T = np.asfarray(T)

    if T.ndim==1 or (T.ndim==2 and (T.shape[0]==1 or T.shape[1]==1)): # isvector
        T,Z = normalise(T)
    elif T.ndim==2: # matrix
        T = np.asmatrix(T)
        Z = np.sum(T,1) 
        S = Z + (Z==0)
        norm = np.tile(S, (1, T.shape[1]))
        T = np.divide(T, norm)
    else: # multi-dimensional array
        ns = T.shape
        T = np.asmatrix(np.reshape(T, (np.prod(ns[0:-1]), ns[-1])))
        Z = np.sum(T,1)
        S = Z + (Z==0)
        norm = np.tile(S, (1, ns[-1]))
        T = np.divide(T, norm)
        T = np.reshape(np.asarray(T), ns)

    return T,Z

def fwdback(init_state_distrib, transmat, obslik, **kwargs):
    '''
    % FWDBACK Compute the posterior probs. in an HMM using the forwards backwards algo.
    %
    % [alpha, beta, gamma, loglik, xi, gamma2] = fwdback(init_state_distrib, transmat, obslik, ...)
    %
    % Notation:
    % Y(t) = observation, Q(t) = hidden state, M(t) = mixture variable (for MOG outputs)
    % A(t) = discrete input (action) (for POMDP models)
    %
    % INPUT:
    % init_state_distrib(i) = Pr(Q(1) = i)
    % transmat(i,j) = Pr(Q(t) = j | Q(t-1)=i)
    %  or transmat{a}(i,j) = Pr(Q(t) = j | Q(t-1)=i, A(t-1)=a) if there are discrete inputs
    % obslik(i,t) = Pr(Y(t)| Q(t)=i)
    %   (Compute obslik using eval_pdf_xxx on your data sequence first.)
    %
    % Optional parameters may be passed as 'param_name', param_value pairs.
    % Parameter names are shown below; default values in [] - if none, argument is mandatory.
    %
    % For HMMs with MOG outputs: if you want to compute gamma2, you must specify
    % 'obslik2' - obslik(i,j,t) = Pr(Y(t)| Q(t)=i,M(t)=j)  []
    % 'mixmat' - mixmat(i,j) = Pr(M(t) = j | Q(t)=i)  []
    %  or mixmat{t}(m,q) if not stationary
    %
    % For HMMs with discrete inputs:
    % 'act' - act(t) = action performed at step t
    %
    % Optional arguments:
    % 'fwd_only' - if 1, only do a forwards pass and set beta=[], gamma2=[]  [0]
    % 'scaled' - if 1,  normalize alphas and betas to prevent underflow [1]
    % 'maximize' - if 1, use max-product instead of sum-product [0]
    %
    % OUTPUTS:
    % alpha(i,t) = p(Q(t)=i | y(1:t)) (or p(Q(t)=i, y(1:t)) if scaled=0)
    % beta(i,t) = p(y(t+1:T) | Q(t)=i)*p(y(t+1:T)|y(1:t)) (or p(y(t+1:T) | Q(t)=i) if scaled=0)
    % gamma(i,t) = p(Q(t)=i | y(1:T))
    % loglik = log p(y(1:T))
    % xi(i,j,t-1)  = p(Q(t-1)=i, Q(t)=j | y(1:T))  - NO LONGER COMPUTED
    % xi_summed(i,j) = sum_{t=}^{T-1} xi(i,j,t)  - changed made by Herbert Jaeger
    % gamma2(j,k,t) = p(Q(t)=j, M(t)=k | y(1:T)) (only for MOG  outputs)
    %
    % If fwd_only = 1, these become
    % alpha(i,t) = p(Q(t)=i | y(1:t))
    % beta = []
    % gamma(i,t) = p(Q(t)=i | y(1:t))
    % xi(i,j,t-1)  = p(Q(t-1)=i, Q(t)=j | y(1:t))
    % gamma2 = []
    %
    % Note: we only compute xi if it is requested as a return argument, since it can be very large.
    % Similarly, we only compute gamma2 on request (and if using MOG outputs).
    %
    % Examples:
    %
    % [alpha, beta, gamma, loglik] = fwdback(pi, A, multinomial_prob(sequence, B));
    %
    % [B, B2] = mixgauss_prob(data, mu, Sigma, mixmat);
    % [alpha, beta, gamma, loglik, xi, gamma2] = fwdback(pi, A, B, 'obslik2', B2, 'mixmat', mixmat);
    
    '''

    obslik2 = kwargs.pop('obslik2', None)
    mixmat = kwargs.pop('mixmat', None)
    fwd_only = kwargs.pop('fwd_only', False)
    scaled = kwargs.pop('scaled', True)
    act = kwargs.pop('act', None)
    maximize = kwargs.pop('maximize', False)
    compute_xi = kwargs.pop('compute_xi', obslik2!=None)
    compute_gamma2 = kwargs.pop('compute_gamma2', obslik2!=None and mixmat!=None)
    
    init_state_distrib = np.asmatrix(init_state_distrib)
    obslik = np.asmatrix(obslik)
    
    Q, T = obslik.shape;
    
    if act==None:
        act = np.zeros((T,))
        transmat = transmat[np.newaxis,:,:]
    
    scale = np.ones((T,))
    
    # scale(t) = Pr(O(t) | O(1:t-1)) = 1/c(t) as defined by Rabiner (1989).
    # Hence prod_t scale(t) = Pr(O(1)) Pr(O(2)|O(1)) Pr(O(3) | O(1:2)) ... = Pr(O(1), ... ,O(T))
    # or log P = sum_t log scale(t).
    # Rabiner suggests multiplying beta(t) by scale(t), but we can instead
    # normalise beta(t) - the constants will cancel when we compute gamma.
    
    loglik = 0
    
    alpha = np.asmatrix(np.zeros((Q,T)))
    gamma = np.asmatrix(np.zeros((Q,T)))
    if compute_xi:
        xi_summed = np.zeros((Q,Q));
    else:
        xi_summed = None
    
    ######## Forwards ########
    
    t = 0
    alpha[:,t] = np.multiply(init_state_distrib, obslik[:,t].T).T
    if scaled:
        #[alpha(:,t), scale(t)] = normaliseC(alpha(:,t));
        alpha[:,t], scale[t] = normalise(alpha[:,t])
    #assert(approxeq(sum(alpha(:,t)),1))
    for t in range (1, T):
        #trans = transmat(:,:,act(t-1))';
        trans = transmat[act[t-1]]
        if maximize:
            m = max_mult(trans.T, alpha[:,t-1])
            #A = repmat(alpha(:,t-1), [1 Q]);
            #m = max(trans .* A, [], 1);
        else:
            m = np.dot(trans.T,alpha[:,t-1])
        alpha[:,t] = np.multiply(m, obslik[:,t])
        if scaled:
            #[alpha(:,t), scale(t)] = normaliseC(alpha(:,t));
            alpha[:,t], scale[t] = normalise(alpha[:,t])
        if compute_xi and fwd_only:  # useful for online EM
            #xi(:,:,t-1) = normaliseC((alpha(:,t-1) * obslik(:,t)') .* trans);
            xi_summed = xi_summed + normalise(np.multiply(np.dot(alpha[:,t-1], obslik[:,t].T), trans))[0];
        #assert(approxeq(sum(alpha(:,t)),1))

    if scaled:
        if np.any(scale==0):
            loglik = -np.Inf;
        else:
            loglik = np.sum(np.log(scale), 0)
    else:
        loglik = np.log(np.sum(alpha[:,T], 0))
    
    if fwd_only:
        gamma = alpha;
        beta = None;
        gamma2 = None;
        return alpha, beta, gamma, loglik, xi_summed, gamma2
    
    ######## Backwards ########
    
    beta = np.asmatrix(np.zeros((Q,T)))
    if compute_gamma2:
        if isinstance(mixmat, list):
            M = mixmat[0].shape[1]
        else:
            M = mixmat.shape[1]
        gamma2 = np.zeros((Q,M,T))
    else:
        gamma2 = None
    
    beta[:,T-1] = np.ones((Q,1))
    #%gamma(:,T) = normaliseC(alpha(:,T) .* beta(:,T));
    gamma[:,T-1], _ = normalise(np.multiply(alpha[:,T-1], beta[:,T-1]))
    t=T-1
    if compute_gamma2:
        denom = obslik[:,t] + (obslik[:,t]==0) # replace 0s with 1s before dividing
        if isinstance(mixmat, list): #in case mixmax is an anyarray
            gamma2[:,:,t] = np.divide(np.multiply(np.multiply(obslik2[:,:,t], mixmat[t]), np.tile(gamma[:,t], (1, M))), np.tile(denom, (1, M)));
        else:
            gamma2[:,:,t] = np.divide(np.multiply(np.multiply(obslik2[:,:,t], mixmat), np.tile(gamma[:,t], (1, M))), np.tile(denom, (1, M))) #TODO: tiling and asmatrix might be slow. mybe remove
        #gamma2(:,:,t) = normaliseC(obslik2(:,:,t) .* mixmat .* repmat(gamma(:,t), [1 M])); % wrong!

    for t in range(T-2, -1, -1):
        b = np.multiply(beta[:,t+1], obslik[:,t+1])
        #trans = transmat(:,:,act(t));
        trans = transmat[act[t]]
        if maximize:
            B = np.tile(b.T, (Q, 1))
            beta[:,t] = np.max(np.multiply(trans, B), 1)
        else:
            beta[:,t] = np.dot(trans, b)
        if scaled:
            #beta(:,t) = normaliseC(beta(:,t));
            beta[:,t], _ = normalise(beta[:,t])
        #gamma(:,t) = normaliseC(alpha(:,t) .* beta(:,t));
        gamma[:,t], _ = normalise(np.multiply(alpha[:,t], beta[:,t]))
        if compute_xi:
            #xi(:,:,t) = normaliseC((trans .* (alpha(:,t) * b')));
            xi_summed = xi_summed + normalise(np.multiply(trans, np.dot(alpha[:,t],b.T)))[0]
        if compute_gamma2:
            denom = obslik[:,t] + (obslik[:,t]==0) # replace 0s with 1s before dividing
            if isinstance(mixmat, list): #in case mixmax is an anyarray
                gamma2[:,:,t] = np.divide(np.multiply(np.multiply(obslik2[:,:,t], mixmat[t]), np.tile(gamma[:,t], (1, M))), np.tile(denom,  (1, M)))
            else:
                gamma2[:,:,t] = np.divide(np.multiply(np.multiply(obslik2[:,:,t], mixmat), np.tile(gamma[:,t], (1, M))), np.tile(denom,  (1, M)))
            #gamma2(:,:,t) = normaliseC(obslik2(:,:,t) .* mixmat .* repmat(gamma(:,t), [1 M]));
    
    # We now explain the equation for gamma2
    # Let zt=y(1:t-1,t+1:T) be all observations except y(t)
    # gamma2(Q,M,t) = P(Qt,Mt|yt,zt) = P(yt|Qt,Mt,zt) P(Qt,Mt|zt) / P(yt|zt)
    #                = P(yt|Qt,Mt) P(Mt|Qt) P(Qt|zt) / P(yt|zt)
    # Now gamma(Q,t) = P(Qt|yt,zt) = P(yt|Qt) P(Qt|zt) / P(yt|zt)
    # hence
    # P(Qt,Mt|yt,zt) = P(yt|Qt,Mt) P(Mt|Qt) [P(Qt|yt,zt) P(yt|zt) / P(yt|Qt)] / P(yt|zt)
    #                = P(yt|Qt,Mt) P(Mt|Qt) P(Qt|yt,zt) / P(yt|Qt)
    
    return alpha, beta, gamma, loglik, xi_summed, gamma2

def mhmm_logprob(data, prior, transmat, mu, Sigma, mixmat=None):
    '''
    % LOG_LIK_MHMM Compute the log-likelihood of a dataset using a (mixture of) Gaussians HMM
    % [loglik, errors] = log_lik_mhmm(data, prior, transmat, mu, sigma, mixmat)
    %
    % data{m}(:,t) or data(:,t,m) if all cases have same length
    % errors  is a list of the cases which received a loglik of -infinity
    %
    % Set mixmat to ones(Q,1) or omit it if there is only 1 mixture component
    '''

    Q = len(prior);
    if mixmat.shape[0] != Q: # trap old syntax
        raise Exception, 'mixmat should be QxM'
    
    if mixmat==None:
        mixmat = np.ones((Q,1))
    
    if not isinstance(data, list):
        data = [data[:,:,i] for i in range(data.shape[2])]
        
    ncases = len(data);
    
    loglik = 0
    errors = []
    
    for m in range(ncases):
        obslik, _ = mixgauss_prob(data[m], mu, Sigma, mixmat);
        alpha, beta, gamma, ll, _, _ = fwdback(prior, transmat, obslik, fwd_only=True)
        if ll==-np.Inf:
            errors.append(m)
        loglik = loglik + ll
    
    return loglik, errors