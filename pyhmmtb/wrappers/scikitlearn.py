'''
Created on 17.06.2013

@author: christian
'''

import numpy as np
import string
from pyhmmtb.hmm.mhmm import mhmm_em, mk_stochastic, mhmm_logprob
from pyhmmtb.tools import normalise
from pyhmmtb.stats.mixgauss import mixgauss_init

class _BaseHMM(object): 
    
    def __init__(self, n_components=1, startprob=None, transmat=None,
                 startprob_prior=None, transmat_prior=None,
                 algorithm="viterbi", random_state=None,
                 n_iter=10, thresh=1e-2, params=string.ascii_letters,
                 init_params=string.ascii_letters):

        self.n_components = n_components
        self.n_iter = n_iter
        self.thresh = thresh
        self.params = params
        self.init_params = init_params
        self.startprob_ = startprob
        self.startprob_prior = startprob_prior
        self.transmat_ = transmat
        self.transmat_prior = transmat_prior
        self._algorithm = algorithm
        self.random_state = random_state

class GMMHMM(_BaseHMM):
    
    def __init__(self, n_components=1, n_mix=1, startprob=None, transmat=None,
                 startprob_prior=None, transmat_prior=None,
                 algorithm="viterbi", gmms=None, covariance_type='diag',
                 covars_prior=1e-2, random_state=None, n_iter=10, thresh=1e-2,
                 params=string.ascii_letters,
                 init_params=string.ascii_letters):
        
        _BaseHMM.__init__(self, n_components, startprob, transmat,
                          startprob_prior=startprob_prior,
                          transmat_prior=transmat_prior,
                          algorithm=algorithm,
                          random_state=random_state,
                          n_iter=n_iter,
                          thresh=thresh,
                          params=params,
                          init_params=init_params)
        
        self.n_mix = n_mix
        self._covariance_type = covariance_type
        self.covars_prior = covars_prior
        self.gmms = gmms
        if self.gmms:
            raise NotImplementedError, 'Providing gmms is not implemented yet'
        
        self.LL = None
        
    def fit(self, obs):
        obs = self._convertObs(obs)

        O = obs[0].shape[0]
        M = self.n_mix
        Q = self.n_components
        
        if 's' in self.init_params:
            self.startprob_, _ = normalise(self.startprob_)
            
        if 't' in self.init_params:
            self.transmat_, _ = mk_stochastic(self.transmat_)
        
        if 'm' in self.init_params or 'c' in self.init_params:
            mu0, Sigma0, weights0 = mixgauss_init(Q*M, obs, cov_type=self._covariance_type)
            
            if 'm' in self.init_params:
                self.means_ = np.transpose(np.reshape(mu0, (O, M, Q)), (0,2,1))
            
            if 'c' in self.init_params:
                self.covar_ = np.transpose(np.reshape(Sigma0, (O, O, M, Q)), (0, 1, 3, 2))
        
        mixmat0, _ = mk_stochastic(np.random.rand(Q,M))
        
        self.LL, prior1, transmat1, mu1, Sigma1, mixmat1 = mhmm_em(data=obs, 
                                                              prior=self.startprob_, 
                                                              transmat=self.transmat_, 
                                                              mu=self.means_, 
                                                              Sigma=self.covar_, 
                                                              mixmat=mixmat0,
                                                              max_iter=self.n_iter,
                                                              thresh=self.thresh,
                                                              cov_type=self._covariance_type,
                                                              adj_trans='t' in self.params,
                                                              adj_mix='w' in self.params,
                                                              adj_mu='m' in self.params,
                                                              adj_Sigma='c' in self.params)
        
        self.startprob_ = prior1
        self.transmat_ = transmat1
        self.means_ = mu1
        self.covar_ = Sigma1
        self.weights_ = mixmat1

    def score(self, obs):
        obs = self._convertObs(obs)
        
        lp = np.empty((len(obs),))
        for i in range(len(obs)):
            lp[i], err = mhmm_logprob(data=[obs[i]], prior=self.startprob_, transmat=self.transmat_, mu=self.means_, Sigma=self.covar_, mixmat=self.weights_)
        return lp
    
    def _convertObs(self, obs):
        assert isinstance(obs[0], np.ndarray)
        assert obs[0].ndim==2
        
        obs = [obs[i].T for i in range(len(obs))]
        
        return obs
    
    def getLL(self):
        return self.LL