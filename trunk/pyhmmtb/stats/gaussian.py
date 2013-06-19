'''
Created on 13.06.2013

@author: christian
'''

import numpy as np
from scipy.constants.constants import pi
from numpy.dual import det, inv, eig
import logging
from pyhmmtb.tools import EPS

def gaussian_prob(x, m, C, use_log=False):
    '''
    % GAUSSIAN_PROB Evaluate a multivariate Gaussian density.
    % p = gaussian_prob(X, m, C)
    % p(i) = N(X(:,i), m, C) where C = covariance matrix and each COLUMN of x is a datavector
    
    % p = gaussian_prob(X, m, C, 1) returns log N(X(:,i), m, C) (to prevents underflow).
    %
    % If X has size dxN, then p has size Nx1, where N = number of examples
    '''

    m = np.asmatrix(m)
    if m.shape[0]==1:
        m = m.T
    
    d, N = x.shape

    M = np.tile(m, (1,N)) # replicate the mean across columns
    denom = (2*pi)**(0.5*d)*np.sqrt(np.abs(det(C)))
    mahal = np.sum(np.multiply(np.dot((x-M).T, inv(C)),(x-M).T),1) # Chris Bregler's trick
    if np.any(mahal<0):
        logging.warning('mahal < 0 => C is not psd')
    if use_log:
        p = -0.5*mahal - np.log(denom);
    else:
        p = np.exp(-0.5*mahal) / (denom+EPS);
    
    return np.asarray(p)[:,0]

def gaussian_sample(mu, covar, nsamp):
    '''
    %GSAMP    Sample from a Gaussian distribution.
    %
    %    Description
    %
    %    X = GSAMP(MU, COVAR, NSAMP) generates a sample of size NSAMP from a
    %    D-dimensional Gaussian distribution. The Gaussian density has mean
    %    vector MU and covariance matrix COVAR, and the matrix X has NSAMP
    %    rows in which each row represents a D-dimensional sample vector.
    %
    %    See also
    %    GAUSS, DEMGAUSS
    %
    
    %    Copyright (c) Ian T Nabney (1996-2001)
    '''
    
    d = covar.shape[0]
    
    mu = np.reshape(mu, (1, d)) # Ensure that mu is a row vector
    
    eigval, evec = eig(covar)
    
    eigval = np.diag(eigval)
    
    coeffs = np.dot(np.random.randn(nsamp, d), np.sqrt(eigval))
    
    x = np.dot(np.ones((nsamp, 1)),mu) + np.dot(coeffs, evec.T)
    
    return x
