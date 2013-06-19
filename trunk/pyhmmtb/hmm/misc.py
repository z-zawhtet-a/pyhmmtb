'''
Created on 12.06.2013

@author: christian
'''

import numpy as np
from pyhmmtb.stats.misc import sample_discrete

def mc_sample(prior, trans, len, numex=1):
    '''
    % SAMPLE_MC Generate random sequences from a Markov chain.
    % STATE = SAMPLE_MC(PRIOR, TRANS, LEN) generates a sequence of length LEN.
    %
    % STATE = SAMPLE_MC(PRIOR, TRANS, LEN, N) generates N rows each of length LEN.
    '''
    
    trans = np.asarray(trans)
    
    S = np.zeros((numex,len), dtype=int);
    for i in range(0,numex):
        S[i, 0] = sample_discrete(prior)
        for t in range(1, len):
            S[i, t] = sample_discrete(trans[S[i,t-1],:])
    return S

def mk_leftright_transmat(Q, p):
    '''
    % MK_LEFTRIGHT_TRANSMAT Q = num states, p = prob on (i,i), 1-p on (i,i+1)
    % function transmat = mk_leftright_transmat(Q, p)
    '''
    
    transmat = p*np.diag(np.ones((Q,))) + (1-p)*np.diag(np.ones((Q-1,)),1)
    transmat[Q-1,Q-1]=1
    
    return transmat