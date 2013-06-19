'''
Created on 13.06.2013

@author: christian
'''

import unittest
import numpy as np
from pyhmmtb.stats.mixgauss import mixgauss_prob, mixgauss_init

class MixgaussProbTest(unittest.TestCase):
    
    def test_1D_M1_Spherical(self):
        m = np.matrix([1])
        C = 1
        x = np.matrix([1])
        
        B, B2 = mixgauss_prob(x, m, C)
        
        assert np.all(np.abs(B==np.matrix([[0.3989]])) < 1e-3)
        assert np.all(np.abs(B2==np.matrix([[0.3989]])) < 1e-3)
        
    def test_1D_M2_Spherical(self):
        m = np.array([[1, 2]])
        C = 1
        x = np.matrix([1])
        
        B, B2 = mixgauss_prob(x, m, C)
        
        assert np.all(np.abs(B==np.matrix([[0.3989],[0.2420]])) < 1e-3)
        assert np.all(np.abs(B2==np.array([[[0.3989]],[[0.2420]]])) < 1e-3)
        
    def test2D_M2_tied_ind_MQ(self):
        m = np.matrix([[1],[2]])
        C = np.matrix([[1,0],
                       [0,1]])
        x = np.matrix([[1], 
                       [2]])
        
        B, B2 = mixgauss_prob(x, m, C)
        
        assert np.all(np.abs(B==np.matrix([[0.1592]])) < 1e-3)
        assert np.all(np.abs(B2==np.matrix([[0.1592]])) < 1e-3)
        
class MixgaussInit1D(unittest.TestCase):
    
    def test(self):
        
        obs = np.concatenate((np.random.randn(10000, 1), 10 + np.random.randn(30000, 1))).T
        
        mu, Sigma, weights = mixgauss_init(M=2, data=obs, cov_type='diag', method='kmeans')
        
        assert np.all(np.abs(mu-np.array([[10, 0]])) < 1e-1) or np.all(np.abs(mu-np.array([[0, 10]])) < 1e-1)
        assert np.abs(Sigma[:,:,0]-1)<1e-1
        assert np.abs(Sigma[:,:,1]-1)<1e-1
        assert np.all(np.abs(weights-np.array([[0.75], [0.25]])) < 0.2) or np.all(np.abs(weights-np.array([[0.25], [0.75]])) < 0.2)

class MixgaussInit2D(unittest.TestCase):
    
    def test(self):
        
        mean0 = [0, 10]
        cov = [[.1, 0],
               [0, .1]]
        
        obs = np.random.multivariate_normal(mean0, cov, 10000)
        
        mean1 = [5, 15]
        cov = [[.1, 0],
               [0, .1]]
        
        obs = np.vstack((obs, np.random.multivariate_normal(mean1, cov, 30000)))

        mu, Sigma, weights = mixgauss_init(M=2, data=obs.T, cov_type='diag', method='kmeans')
        
        assert np.all(np.abs(mu[:,0]-mean0) < 1e-1) or np.all(np.abs(mu[:,1]-mean0) < 1e-1)
    
if __name__=='__main__':
    unittest.main()