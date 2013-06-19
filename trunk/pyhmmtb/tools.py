'''
Created on 12.06.2013

@author: christian
'''

import numpy as np
from numpy.dual import cholesky
from numpy.linalg.linalg import LinAlgError

EPS = 2.2204e-16

def normalise(A, dim=None):
    '''
    % NORMALISE Make the entries of a (multidimensional) array sum to 1
    % [M, c] = normalise(A)
    % c is the normalizing constant
    %
    % [M, c] = normalise(A, dim)
    % If dim is specified, we normalise the specified dimension only,
    % otherwise we normalise the whole array.
    '''
    
    if dim==None:
        z = np.sum(A[:])
        # Set any zeros to one before dividing
        # This is valid, since c=0 => all i. A(i)=0 => the answer should be 0/1=0
        s = z + (z==0)
        M = A / (1.*s)
    elif dim==0: # normalize each column
        z = np.sum(A, axis=0)
        s = z + (z==0)
        #M = A ./ (d'*ones(1,size(A,1)))';
        M = np.multiply(A, np.tile(1.*s, A.shape[0], 1))
    else:
        # Keith Battocchi - v. slow because of repmat
        z=np.sum(A,dim)
        s = z + (z==0)
        L=A.shape[dim]
        d=A.ndim
        v=np.ones((d,))
        v[dim]=L;
        #c=repmat(s,v);
        c=np.tile(1.*s,v.T)
        M=np.divide(A, c)

    return M, z

def em_converged(loglik, previous_loglik, threshold=1e-4, check_increased=True):
    '''
    % EM_CONVERGED Has EM converged?
    % [converged, decrease] = em_converged(loglik, previous_loglik, threshold)
    %
    % We have converged if the slope of the log-likelihood function falls below 'threshold', 
    % i.e., |f(t) - f(t-1)| / avg < threshold,
    % where avg = (|f(t)| + |f(t-1)|)/2 and f(t) is log lik at iteration t.
    % 'threshold' defaults to 1e-4.
    %
    % This stopping criterion is from Numerical Recipes in C p423
    %
    % If we are doing MAP estimation (using priors), the likelihood can decrase,
    % even though the mode of the posterior is increasing.
    '''

    converged = False
    decrease = True
    
    if check_increased:
        if loglik - previous_loglik < -1e-3: # allow for a little imprecision
            print '******likelihood decreased from %6.4f to %6.4f!\n' % (previous_loglik, loglik)
            decrease = True
            converged = False
            return converged, decrease
            
    delta_loglik = np.abs(loglik - previous_loglik)
    if delta_loglik == np.inf:
        converged = False
    else:
        avg_loglik = (np.abs(loglik) + np.abs(previous_loglik) + EPS)/2.;
        if (delta_loglik / avg_loglik) < threshold:
            converged = True

    return converged, decrease

def sqdist(p, q, A=None):
    '''
    % SQDIST      Squared Euclidean or Mahalanobis distance.
    % SQDIST(p,q)   returns m(i,j) = (p(:,i) - q(:,j))'*(p(:,i) - q(:,j)).
    % SQDIST(p,q,A) returns m(i,j) = (p(:,i) - q(:,j))'*A*(p(:,i) - q(:,j)).
    
    %  From Tom Minka's lightspeed toolbox
    '''
    
    d, pn = p.shape;
    d, qn = q.shape;
    
    if A==None:
        pmag = np.sum(np.multiply(p, p), 0)
        qmag = np.sum(np.multiply(q, q), 0)
        m = np.tile(qmag, (pn, 1)) + np.tile(pmag.T, (1, qn)) - 2 * np.dot(p.T,q)
        #m = ones(pn,1)*qmag + pmag'*ones(1,qn) - 2*p'*q;
    else:
        Ap = np.dot(A,p)
        Aq = np.dot(A,q)
        pmag = np.sum(np.multiply(p, Ap), 0)
        qmag = np.sum(np.multiply(q, Aq), 0)
        m = np.tile(qmag, (pn, 1)) + np.tile(pmag.T, (1, qn)) - 2 * np.dot(p.T,Aq)
    return m

def approxeq(a, b, tol=1e-2, rel=False):
    '''
    % APPROXEQ Are a and b approximately equal (to within a specified tolerance)?
    % p = approxeq(a, b, thresh)
    % 'tol' defaults to 1e-3.
    % p(i) = 1 iff abs(a(i) - b(i)) < thresh
    %
    % p = approxeq(a, b, thresh, 1)
    % p(i) = 1 iff abs(a(i)-b(i))/abs(a(i)) < thresh
    '''
    
    a = a[:]
    b = b[:]
    d = np.abs(a-b);
    if rel:
        p = not np.any( (np.divide(d, (abs(a)+EPS))) > tol)
    else:
        p = not np.any(d > tol)
    
    return p

def logdet(A):
    '''
    % log(det(A)) where A is positive-definite.
    % This is faster and more stable than using log(det(A)).
    
    %  From Tom Minka's lightspeed toolbox
    '''
    
    U = cholesky(A).T;
    y = 2*np.sum(np.log(np.diag(U)),0)
    
    return y

def isposdef(a):
    '''
    % ISPOSDEF   Test for positive definite matrix.
    %    ISPOSDEF(A) returns 1 if A is positive definite, 0 otherwise.
    %    Using chol is much more efficient than computing eigenvectors.
    
    %  From Tom Minka's lightspeed toolbox
    '''
    
    try:
        cholesky(a)
        return True
    except LinAlgError:
        return False
    
def max_mult(A,x):
    '''
    % MAX_MULT Like matrix multiplication, but sum gets replaced by max
    % function y=max_mult(A,x) y(i) = max_j A(i,j) x(j)
    
    %X=ones(size(A,1),1) * x(:)'; % X(j,i) = x(i)
    %y=max(A.*X, [], 2);

    % This is faster
    '''
    
    if x.shape[1]==1:
        X=x*np.ones((1,A.shape[0])) # X(i,j) = x(i)
        y=np.max(np.multiply(A.T, X), 0).T
    else:
        #%this works for arbitrarily sized A and x (but is ugly, and slower than above)
        X=np.tile(x, (1, 1, A.shape[0]))
        B=np.tile(A, (1, 1, x.shape[1]))
        C=np.transpose(B,(1, 2, 0))
        y=np.transpose(np.max(np.multiply(C, X), 0),[2, 1, 0])
        # this is even slower, as is using squeeze instead of permute
        #Y=permute(X, [3 1 2]);
        #y=permute(max(Y.*B, [], 2), [1 3 2]);
    
    return y