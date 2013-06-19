'''
Created on 17.06.2013

@author: christian
'''

import unittest
from pyhmmtb.wrappers.scikitlearn import GMMHMM
import numpy as np
from pyhmmtb.hmm.mhmm import mhmm_sample, mk_stochastic
from scipy.io.matlab.mio import savemat
from pickle import load, dump

class GMMHMMTest(unittest.TestCase):
    
    def setUp(self):
        
        self.prior = np.array([1, 0, 0])
        self.transmat = np.matrix([[0, 1, 0],
                              [0, 0, 1],
                              [0, 0, 1]])
        
        self.mu = np.zeros((2,3,2))
        self.mu[:,:,0] = np.array([[1, 2, 3],
                              [1, 2, 3]])
        self.mu[:,:,1] = np.array([[4, 5, 6],
                              [4, 5, 6]])
        
        self.Sigma = np.zeros((2, 2, 3, 2))
        for i in range(3):
            self.Sigma[:,:,i,0] = np.diag(np.ones((2,))*0.01)
            self.Sigma[:,:,i,1] = np.diag(np.ones((2,))*0.01)
        
        self.mixmat = np.array([[.5,.5],
                            [.5,.5],
                            [.5,.5]])
    
        try:
            
            with open('GMMHMMTest.cache', 'rb') as f:
                cache = load(f)

            self.obs = cache['obs']
            self.prior0 = cache['prior0']
            self.transmat0 = cache['transmat0']
            self.mu0 = cache['mu0']
            self.Sigma0 = cache['Sigma0']
            self.mixmat0 = cache['mixmat0']
            
        except:
            
            print 'generating data'
        
            obs, hidden = mhmm_sample(T=4, 
                                      numex=100,
                                      initial_prob=self.prior,
                                      transmat=self.transmat,
                                      mu=self.mu, 
                                      Sigma=self.Sigma,
                                      mixmat = self.mixmat)
            
            self.obs = [obs[:,:,i].T for i in range(obs.shape[2])] 

            self.prior0, _ = mk_stochastic(np.random.rand(3))
            self.transmat0, _ = mk_stochastic(np.random.rand(3,3))
            
            self.mu0 = np.zeros((2,3,2))
            self.mu0[:,:,0] = np.array([[1.5, 2.5, 3.5],
                                  [1.5, 2.5, 3.5]])
            self.mu0[:,:,1] = np.array([[4.5, 5.5, 6.5],
                                  [4.5, 5.5, 6.5]])
            
            self.Sigma0 = np.zeros((2, 2, 3, 2))
            for i in range(3):
                self.Sigma0[:,:,i,0] = np.diag(np.ones((2,))*1.0)
                self.Sigma0[:,:,i,1] = np.diag(np.ones((2,))*1.0)
            
            self.mixmat0= np.array([[.2,.8],
                                [.2,.8],
                                [.2,.8]])
            
            cache = {'obs':self.obs,
                     'prior0':self.prior0,
                     'transmat0':self.transmat0,
                     'mu0':self.mu0,
                     'Sigma0':self.Sigma0,
                     'mixmat0':self.Sigma0}
            
            with open('GMMHMMTest.cache', 'wb') as f:
                dump(cache, f)
            
            savemat('GMMHMMTest.mat', cache)
    
    def testTrainStateParams(self):
        
        hmm = GMMHMM(n_components=3, 
                     n_mix=2, 
                     startprob=self.prior0, 
                     transmat=self.transmat0, 
                     covariance_type='diag')
        
        hmm.fit(self.obs)
        
        print hmm.score(self.obs)

if __name__=='__main__':
    unittest.main()