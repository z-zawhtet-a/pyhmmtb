'''
Created on 13.06.2013

@author: christian
'''

import numpy as np

def sample_discrete(prob, r=1, c=None):
    '''
    % SAMPLE_DISCRETE Like the built in 'rand', except we draw from a non-uniform discrete distrib.
    % M = sample_discrete(prob, r, c)
    %
    % Example: sample_discrete([0.8 0.2], 1, 10) generates a row vector of 10 random integers from {1,2},
    % where the prob. of being 1 is 0.8 and the prob of being 2 is 0.2.
    '''
    
    assert prob.ndim == 1
    
    n = len(prob)
    
    if c==None:
        c = r
    
    R = np.random.rand(r, c)
    M = np.zeros((r, c), dtype=int)
    cumprob = np.cumsum(prob[:])
    
    if n < r*c:
        for i in range(0, n):
            M = M + (R > cumprob[i])
    else:
        # loop over the smaller index - can be much faster if length(prob) >> r*c
        cumprob2 = cumprob[0:-1]
        for i in range(0,r):
            for j in range(0,c):
                M[i,j] = np.sum(R[i,j] > cumprob2)
    
    # Slower, even though vectorized
    #cumprob = reshape(cumsum([0 prob(1:end-1)]), [1 1 n]);
    #M = sum(R(:,:,ones(n,1)) > cumprob(ones(r,1),ones(c,1),:), 3);
    
    # convert using a binning algorithm
    #M=bindex(R,cumprob);
    
    return M