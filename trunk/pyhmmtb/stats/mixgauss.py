'''
Created on 13.06.2013

@author: christian
'''

import numpy as np
from pyhmmtb.tools import sqdist, approxeq, logdet, isposdef, normalise
from scipy.constants.constants import pi
from numpy.dual import inv
import logging
from pyhmmtb.stats.gaussian import gaussian_prob
from sklearn.mixture.gmm import GMM

def mixgauss_prob(data, mu, Sigma, mixmat = None, unit_norm = False):
    '''
    % EVAL_PDF_COND_MOG Evaluate the pdf of a conditional mixture of Gaussians
    % function [B, B2] = eval_pdf_cond_mog(data, mu, Sigma, mixmat, unit_norm)
    %
    % Notation: Y is observation, M is mixture component, and both may be conditioned on Q.
    % If Q does not exist, ignore references to Q=j below.
    % Alternatively, you may ignore M if this is a conditional Gaussian.
    %
    % INPUTS:
    % data(:,t) = t'th observation vector 
    %
    % mu(:,k) = E[Y(t) | M(t)=k] 
    % or mu(:,j,k) = E[Y(t) | Q(t)=j, M(t)=k]
    %
    % Sigma(:,:,j,k) = Cov[Y(t) | Q(t)=j, M(t)=k]
    % or there are various faster, special cases:
    %   Sigma() - scalar, spherical covariance independent of M,Q.
    %   Sigma(:,:) diag or full, tied params independent of M,Q. 
    %   Sigma(:,:,j) tied params independent of M. 
    %
    % mixmat(k) = Pr(M(t)=k) = prior
    % or mixmat(j,k) = Pr(M(t)=k | Q(t)=j) 
    % Not needed if M is not defined.
    %
    % unit_norm - optional; if 1, means data(:,i) AND mu(:,i) each have unit norm (slightly faster)
    %
    % OUTPUT:
    % B(t) = Pr(y(t)) 
    % or
    % B(i,t) = Pr(y(t) | Q(t)=i) 
    % B2(i,k,t) = Pr(y(t) | Q(t)=i, M(t)=k) 
    %
    % If the number of mixture components differs depending on Q, just set the trailing
    % entries of mixmat to 0, e.g., 2 components if Q=1, 3 components if Q=2,
    % then set mixmat(1,3)=0. In this case, B2(1,3,:)=1.0.
    '''

    if mu.ndim == 1:
        d = len(mu)
        Q = 1
        M = 1
    elif mu.ndim == 2:
        d, Q = mu.shape
        M = 1
    else:
        d, Q, M = mu.shape;
    
    d, T = data.shape
    
    if mixmat == None:
        mixmat = np.asmatrix(np.ones((Q,1)))
    
    # B2 = zeros(Q,M,T); % ATB: not needed allways
    # B = zeros(Q,T);
    
    if np.isscalar(Sigma):
        mu = np.reshape(mu, (d, Q * M))
        if unit_norm:  # (p-q)'(p-q) = p'p + q'q - 2p'q = n+m -2p'q since p(:,i)'p(:,i)=1
            # avoid an expensive repmat
            print('unit norm')
            # tic; D = 2 -2*(data'*mu)'; toc 
            D = 2 - 2 * np.dot(mu.T * data)
            # tic; D2 = sqdist(data, mu)'; toc
            D2 = sqdist(data, mu).T
            assert(approxeq(D,D2)) 
        else:
            D = sqdist(data, mu).T
        del mu 
        del data  # ATB: clear big old data
        # D(qm,t) = sq dist between data(:,t) and mu(:,qm)
        logB2 = -(d / 2.) * np.log(2 * pi * Sigma) - (1 / (2. * Sigma)) * D  # det(sigma*I) = sigma^d
        B2 = np.reshape(np.asarray(np.exp(logB2)), (Q, M, T))
        del logB2  # ATB: clear big old data
      
    elif Sigma.ndim == 2:  # tied full
        mu = np.reshape(mu, (d, Q * M))
        D = sqdist(data, mu, inv(Sigma)).T
        # D(qm,t) = sq dist between data(:,t) and mu(:,qm)
        logB2 = -(d/2)*np.log(2*pi) - 0.5*logdet(Sigma) - 0.5*D;
        #denom = sqrt(det(2*pi*Sigma));
        #numer = exp(-0.5 * D);
        #B2 = numer/denom;
        B2 = np.reshape(np.asarray(np.exp(logB2)), (Q, M, T))
      
    elif Sigma.ndim==3: # tied across M
        B2 = np.zeros((Q,M,T))
        for j in range(0,Q):
            # D(m,t) = sq dist between data(:,t) and mu(:,j,m)
            if isposdef(Sigma[:,:,j]):
                D = sqdist(data, np.transpose(mu[:,j,:], (0, 2, 1)), inv(Sigma[:,:,j])).T
                logB2 = -(d / 2) * np.log(2 * pi) - 0.5 * logdet(Sigma[:, :, j]) - 0.5 * D;
                B2[j, :, :] = np.exp(logB2);
            else:
                logging.error('mixgauss_prob: Sigma(:,:,q=%d) not psd\n' % j)
      
    else:  # general case
        B2 = np.zeros((Q, M, T))
        for j in range(0, Q):
            for k in range(0, M):
                # if mixmat(j,k) > 0
                B2[j, k, :] = gaussian_prob(data, mu[:, j, k], Sigma[:, :, j, k]);
    
    # B(j,t) = sum_k B2(j,k,t) * Pr(M(t)=k | Q(t)=j) 
    
    # The repmat is actually slower than the for-loop, because it uses too much memory
    # (this is true even for small T).
    
    # B = squeeze(sum(B2 .* repmat(mixmat, [1 1 T]), 2));
    # B = reshape(B, [Q T]); % undo effect of squeeze in case Q = 1
      
    B = np.zeros((Q, T))
    if Q < T:
        for q in range(0, Q):
            # B(q,:) = mixmat(q,:) * squeeze(B2(q,:,:)); % squeeze chnages order if M=1
            B[q, :] = np.dot(mixmat[q, :], B2[q, :, :])  # vector * matrix sums over m #TODO: had to change this. Is this correct?
    else:
        for t in range(0, T):
            B[:, t] = np.sum(np.asarray(np.multiply(mixmat, B2[:, :, t])), 1)  # sum over m
    # t=toc;fprintf('%5.3f\n', t)
    
    # tic
    # A = squeeze(sum(B2 .* repmat(mixmat, [1 1 T]), 2));
    # t=toc;fprintf('%5.3f\n', t)
    # assert(approxeq(A,B)) % may be false because of round off error

    return B, B2

def mixgauss_Mstep(w, Y, YY, YTY, **kwargs):
    '''
    % MSTEP_COND_GAUSS Compute MLEs for mixture of Gaussians given expected sufficient statistics
    % function [mu, Sigma] = Mstep_cond_gauss(w, Y, YY, YTY, varargin)
    %
    % We assume P(Y|Q=i) = N(Y; mu_i, Sigma_i)
    % and w(i,t) = p(Q(t)=i|y(t)) = posterior responsibility
    % See www.ai.mit.edu/~murphyk/Papers/learncg.pdf.
    %
    % INPUTS:
    % w(i) = sum_t w(i,t) = responsibilities for each mixture component
    %  If there is only one mixture component (i.e., Q does not exist),
    %  then w(i) = N = nsamples,  and 
    %  all references to i can be replaced by 1.
    % YY(:,:,i) = sum_t w(i,t) y(:,t) y(:,t)' = weighted outer product
    % Y(:,i) = sum_t w(i,t) y(:,t) = weighted observations
    % YTY(i) = sum_t w(i,t) y(:,t)' y(:,t) = weighted inner product
    %   You only need to pass in YTY if Sigma is to be estimated as spherical.
    %
    % Optional parameters may be passed as 'param_name', param_value pairs.
    % Parameter names are shown below; default values in [] - if none, argument is mandatory.
    %
    % 'cov_type' - 'full', 'diag' or 'spherical' ['full']
    % 'tied_cov' - 1 (Sigma) or 0 (Sigma_i) [0]
    % 'clamped_cov' - pass in clamped value, or [] if unclamped [ [] ]
    % 'clamped_mean' - pass in clamped value, or [] if unclamped [ [] ]
    % 'cov_prior' - Lambda_i, added to YY(:,:,i) [0.01*eye(d,d,Q)]
    %
    % If covariance is tied, Sigma has size d*d.
    % But diagonal and spherical covariances are represented in full size.
    '''

    cov_type = kwargs.pop('cov_type', 'full')
    tied_cov = kwargs.pop('tied_cov', 0)
    clamped_cov = kwargs.pop('clamped_cov', [])
    clamped_mean = kwargs.pop('clamped_mean', None)
    cov_prior = kwargs.pop('cov_prior', None)
    
    Y = np.asmatrix(Y)
    
    Ysz, Q = Y.shape
    N = np.sum(w, 0)
    if cov_prior==None:
        # cov_prior = zeros(Ysz, Ysz, Q);
        # for q=1:Q
        #  cov_prior(:,:,q) = 0.01*cov(Y(:,q)');
        # end
        cov_prior = np.transpose(np.tile(0.01 * np.eye(Ysz), (Q, 1, 1)), (1,2,0))
    # YY = reshape(YY, [Ysz Ysz Q]) + cov_prior; % regularize the scatter matrix
    YY = np.reshape(YY, (Ysz, Ysz, Q))
    
    # Set any zero weights to one before dividing
    # This is valid because w(i)=0 => Y(:,i)=0, etc
    w = w + (w == 0);
                
    if clamped_mean!=None:
        mu = np.asmatrix(clamped_mean)
    else:
        # eqn 6
        # mu = Y ./ repmat(w(:)', [Ysz 1]);% Y may have a funny size
        mu = np.asmatrix(np.zeros((Ysz, Q)))
        for i in range(0, Q):
            mu[:, i] = Y[:, i] / w[i]
    
    if not len(clamped_cov) == 0:
        Sigma = clamped_cov
        return mu, Sigma
    
    if not tied_cov:
        Sigma = np.zeros((Ysz, Ysz, Q))
        for i in range(0, Q):
            if cov_type[0] == 's':
                # eqn 17
                s2 = (1 / Ysz) * ((YTY[i] / w[i]) - np.dot(mu[:, i].T, mu[:, i]))
                Sigma[:, :, i] = s2 * np.eye(Ysz)
            else:
                # eqn 12
                SS = YY[:, :, i] / w[i] - np.dot(mu[:, i], mu[:, i].T)
                if cov_type[0] == 'd':
                    SS = np.diag(np.diag(SS))
                Sigma[:, :, i] = SS
    else:  # tied cov
        if cov_type[0] == 's':
            # eqn 19
            s2 = (1 / (N * Ysz)) * (np.sum(YTY, 1) + np.sum(np.multiply(np.diag(np.dot(mu.T, mu)), w)))
            Sigma = s2 * np.eye(Ysz)
        else:
            SS = np.zeros((Ysz, Ysz))
            # eqn 15
            for i in range(1, Q):  # probably could vectorize this...
                SS = SS + YY[:, :, i] / N - np.dot(mu[:, i], mu[:, i].T)
            if cov_type[0] == 'd':
                Sigma = np.diag(np.diag(SS))
            else:
                Sigma = SS
    
    if tied_cov:
        Sigma = np.tile(Sigma, (1, 1, Q))
    Sigma = Sigma + cov_prior

    return np.asarray(mu), Sigma

def mixgauss_init(M, data, cov_type, method='kmeans'):
    '''
    % MIXGAUSS_INIT Initial parameter estimates for a mixture of Gaussians
    % function [mu, Sigma, weights] = mixgauss_init(M, data, cov_type. method)
    %
    % INPUTS:
    % data(:,t) is the t'th example
    % M = num. mixture components
    % cov_type = 'full', 'diag' or 'spherical'
    % method = 'rnd' (choose centers randomly from data) or 'kmeans' (needs netlab)
    %
    % OUTPUTS:
    % mu(:,k) 
    % Sigma(:,:,k) 
    % weights(k)
    '''
    
    if isinstance(data, list):
        data = np.hstack(data)
    elif data.ndim==3:
        O, T, N = data.shape
        data = np.reshape(np.transpose(data, (0, 2, 1)), (O, T*N))
    d, T = data.shape
    
    if method=='rnd':
        C = np.atleast_2d(np.cov(data))
        Sigma = np.transpose(np.tile(np.diag(np.diag(C))*0.5, (M, 1, 1)), (2, 1, 0))
        # Initialize each mean to a random data point
        indices = np.arange(T)
        np.random.shuffle(indices)
        mu = data[:,indices[0:M]]
        weights, _ = normalise(np.ones((M,1)))
    elif method=='kmeans':
        
        gmm = GMM(n_components=M, covariance_type=cov_type,
                  thresh=1e-2, min_covar=1e-3,
                  n_iter=5, n_init=1, params='wmc', init_params='wmc')
        
        gmm.fit(data.T)
        
        mu = gmm.means_.T
        weights = np.asmatrix(gmm.weights_).T
        covars = gmm.covars_
        
        Sigma = np.zeros((d,d,M))
        for m in range(M):
            if cov_type=='diag':
                Sigma[:,:,m] = np.diag(covars[m,:])
            elif cov_type=='full':
                Sigma[:,:,m] = covars[:,:,m]
            elif cov_type=='spherical':
                Sigma[:,:,m] = covars[m] * np.eye(d)
    
    return mu, Sigma, weights