'''
Created on 12.06.2013

@author: christian
'''

import unittest
import numpy as np
from pyhmmtb.hmm.mhmm import mk_stochastic, mhmm_sample, mhmm_em
import os
from scipy.io.matlab.mio import loadmat, savemat
from pickle import load, dump
from pyhmmtb.stats.mixgauss import mixgauss_init

@unittest.skip('')
class MkstochasticTest(unittest.TestCase):
    
    def testVector(self):
        T = np.random.rand(1,3)[0,:]
        
        M, S = mk_stochastic(T)
        
        assert np.abs(sum(M)-1) < 1e-3
    
    def testMatrix(self):
        T = np.asmatrix(np.random.rand(3,3))
        M, S = mk_stochastic(T)
        
        assert np.all(np.abs(np.sum(M,1)-np.matrix([[1],
                                              [1],
                                              [1]])) < 1e-3)
        
    def testTensor(self):
        T = np.random.rand(3,3,3)
        M, S = mk_stochastic(T)
        
        assert np.all(np.abs(np.sum(M, 2)-np.ones((3,3)))<1e-3)

@unittest.skip('')
class MhmmSampleTest(unittest.TestCase):
    
    def test(self):
        
        mu = np.zeros((1,3,1))
        mu[0,:,0] = [1, 2, 3]
        Sigma = np.zeros((1, 1, 3, 1))
        Sigma[0,0,:,0] = 0.01
        
        obs, hidden = mhmm_sample(T=4, 
                                  numex=1000,
                                  initial_prob=np.array([1, 0, 0]),
                                  transmat=np.matrix([[0, 1, 0],
                                                      [0, 0, 1],
                                                      [0, 0, 1.]]),
                                  mu=mu, 
                                  Sigma=Sigma)
        
        assert np.all(np.abs(np.mean(np.asmatrix(obs), axis=1)-np.array([[1],
                                                                         [2],
                                                                         [3],
                                                                         [3]])) < 1e-1)

@unittest.skip('')
class MhmmEM1DTest(unittest.TestCase):
    
    def setUp(self):
        
        self.prior = np.array([1, 0, 0])
        self.transmat = np.matrix([[0, 1, 0],
                              [0, 0, 1],
                              [0, 0, 1]])
        
        self.mu = np.zeros((1,3,2))
        self.mu[0,:,0] = [1, 2, 3]
        self.mu[0,:,1] = [4, 5, 6]
        self.Sigma = np.zeros((1, 1, 3, 2))
        self.Sigma[0,0,:,0] = 0.01
        self.Sigma[0,0,:,1] = 0.01
        self.mixmat = np.array([[.5,.5],
                            [.5,.5],
                            [.5,.5]])
        
        self.obs, self.hidden = mhmm_sample(T=4, 
                                  numex=100,
                                  initial_prob=self.prior,
                                  transmat=self.transmat,
                                  mu=self.mu, 
                                  Sigma=self.Sigma,
                                  mixmat = self.mixmat)
        
        self.prior0, _ = mk_stochastic(np.random.rand(3))
        
        self.transmat0, _ = mk_stochastic(np.random.rand(3,3))
        
        self.mu0 = np.zeros((1,3,2))
        self.mu0[0,:,0] = [1.5, 2.5, 3.5]
        self.mu0[0,:,1] = [4.5, 5.5, 6.5]
        
        self.Sigma0 = np.zeros((1, 1, 3, 2))
        self.Sigma0[0,0,:,0] = 1.0
        self.Sigma0[0,0,:,1] = 1.0
        
        self.mixmat0 = np.array([[.3,.7],
                            [.3,.7],
                            [.3,.7]])
    
    def testStateHiddenParams(self):
        
        LL, prior1, transmat1, mu1, Sigma1, mixmat1 = mhmm_em(data=self.obs, prior=self.prior0, transmat=self.transmat0, mu=self.mu, Sigma=self.Sigma,
                                  mixmat=self.mixmat)
        
        assert np.all(np.diff(LL)>=-1e-3) # Likelihood is increasing
        assert np.all(np.abs(self.prior-prior1)<1e-3)
        assert np.all(np.abs(self.transmat-transmat1)<1e-3)
        
    def testTrainEmissionParams(self):
         
        LL, prior1, transmat1, mu1, Sigma1, mixmat1 = mhmm_em(data=self.obs, prior=self.prior, transmat=self.transmat, mu=self.mu0, Sigma=self.Sigma0,
                                  mixmat=self.mixmat0)
         
        assert np.all(np.diff(LL)>=-1e-3) # Likelihood is increasing
        assert np.all(np.abs(mu1-self.mu)<1e-1)
        assert np.all(np.abs(Sigma1-self.Sigma)<2e-2)
        assert np.all(np.abs(mixmat1-self.mixmat) < 1e-1)
        
@unittest.skip('')
class MhmmEM2DTest(unittest.TestCase):
    
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
            with open('MhmmEM2DTest.cache', 'rb') as f:
                cache = load(f)
            
            self.obs = cache['obs']
            self.prior0 = cache['prior0']
            self.transmat0 = cache['transmat0']
            self.mu0 = cache['mu0']
            self.Sigma0 = cache['Sigma0']
            self.mixmat0 = cache['mixmat0']
            
        except:
        
            self.obs, hidden = mhmm_sample(T=4, 
                                      numex=100,
                                      initial_prob=self.prior,
                                      transmat=self.transmat,
                                      mu=self.mu, 
                                      Sigma=self.Sigma,
                                      mixmat = self.mixmat)
        

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
        
            cache =  {'obs':self.obs,
                      'prior0':self.prior0,
                      'transmat0':self.transmat0,
                      'mu0':self.mu0,
                      'Sigma0':self.Sigma0,
                      'mixmat0':self.mixmat0}
            
            with open('MhmmEM2DTest.cache', 'wb') as f:
                dump(cache, f)
            savemat('MhmmEM2DTest.mat', cache)
    
    def testStateHiddenParams(self):
        
        LL, prior1, transmat1, mu1, Sigma1, mixmat1 = mhmm_em(data=self.obs, prior=self.prior0, transmat=self.transmat0, mu=self.mu, Sigma=self.Sigma,
                                  mixmat=self.mixmat)
        
        assert np.all(np.diff(LL)>=-1e-3) # Likelihood is increasing
        assert np.all(np.abs(self.prior-prior1)<1e-3)
        assert np.all(np.abs(self.transmat-transmat1)<1e-3)
        
    def testStateEmissionParams(self):
        
        LL, prior1, transmat1, mu1, Sigma1, mixmat1 = mhmm_em(data=self.obs, prior=self.prior, transmat=self.transmat, mu=self.mu0, Sigma=self.Sigma0,
                                  mixmat=self.mixmat0)
        
        assert np.all(np.diff(LL)>=-1e-3) # Likelihood is increasing
        assert np.all(np.abs(mu1-self.mu)<1e-1)
        assert np.all(np.abs(Sigma1-self.Sigma)<1e-1)
        assert np.all(np.abs(mixmat1-self.mixmat)<1e-1)
        
class MhmmEM2DTestGaussInit(unittest.TestCase):
    
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
            with open('MhmmEM2DTestGaussInit.cache', 'rb') as f:
                cache = load(f)
            
            self.obs = cache['obs']
            self.prior0 = cache['prior0']
            self.transmat0 = cache['transmat0']
            self.mu0 = cache['mu0']
            self.Sigma0 = cache['Sigma0']
            self.mixmat0 = cache['mixmat0']
            
        except:
        
            self.obs, hidden = mhmm_sample(T=4, 
                                      numex=100,
                                      initial_prob=self.prior,
                                      transmat=self.transmat,
                                      mu=self.mu, 
                                      Sigma=self.Sigma,
                                      mixmat = self.mixmat)
        

            self.prior0, _ = mk_stochastic(np.random.rand(3))
            self.transmat0, _ = mk_stochastic(np.random.rand(3,3))
            
            O = self.obs.shape[0]
            M = 2
            Q = 3
            
            mu0, Sigma0, weights0 = mixgauss_init(Q*M, self.obs, cov_type='diag')
            
            self.mu0 = np.transpose(np.reshape(mu0, (O, M, Q)), (0,2,1))
            self.Sigma0 = np.transpose(np.reshape(Sigma0, (O, O, M, Q)), (0, 1, 3, 2))
            
            self.mixmat0, _ = mk_stochastic(np.random.rand(Q,M))
            
            cache =  {'obs':self.obs,
                      'prior0':self.prior0,
                      'transmat0':self.transmat0,
                      'mu0':self.mu0,
                      'Sigma0':self.Sigma0,
                      'mixmat0':self.mixmat0}
            
            with open('MhmmEM2DTestGaussInit.cache', 'wb') as f:
                dump(cache, f)
            savemat('MhmmEM2DTestGaussInit.mat', cache)
        
    def testWithMixgaussInit(self):
        
        LL, prior1, transmat1, mu1, Sigma1, mixmat1 = mhmm_em(data=self.obs, 
                                                              prior=self.prior0, 
                                                              transmat=self.transmat0, 
                                                              mu=self.mu0, 
                                                              Sigma=self.Sigma0, 
                                                              mixmat=self.mixmat0)

    
if __name__=='__main__':
    unittest.main()