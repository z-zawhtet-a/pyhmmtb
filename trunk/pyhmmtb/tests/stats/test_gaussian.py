'''
Created on 13.06.2013

@author: christian
'''

import unittest
import numpy as np
from pyhmmtb.stats.gaussian import gaussian_sample, gaussian_prob
from numpy.oldnumeric.mlab import cov

class GaussianSampleTest(unittest.TestCase):
    
    def testRow(self):
        mu = np.matrix([[1],[2]])
        covar = np.matrix([[1,0],
                           [0,1]])
        nsamp = 1
        assert gaussian_sample(mu, covar, nsamp).shape == (1,2)
    
    def test2D(self):
        mu = np.matrix([[1],[2]])
        covar = np.matrix([[1,.5],
                           [.5,2]])
        nsamp = 10000
        M = gaussian_sample(mu, covar, nsamp)
        assert np.all(np.abs(np.mean(M, 0)-mu.T)<1e-1)
        assert np.all(np.abs(cov(M)-covar)<1e-1)
        
class GaussianProbTest(unittest.TestCase):
    
    def test1(self):
        m = np.matrix([[1],[2]])
        C = np.matrix([[1,0],
                       [0,1]])
        x = np.matrix([[1], 
                       [2]])

        assert np.all(np.abs(gaussian_prob(x, m, C, use_log=False)==np.matrix([[0.1592]])) < 1e-3)
        assert np.all(np.abs(gaussian_prob(x, m, C, use_log=True)==np.matrix([[-1.8379]])) < 1e-3)
        
    def testMany(self):
        m = np.matrix([[1],[2]])
        C = np.matrix([[1,0],
                       [0,1]])
        x = np.matrix([[1, 0], 
                       [2, 0]])
        
        assert np.all(np.abs(gaussian_prob(x, m, C, use_log=False)==np.matrix([[0.1592],[0.0131]])) < 1e-3)
        assert np.all(np.abs(gaussian_prob(x, m, C, use_log=True)==np.matrix([[-1.8379],[-4.3379]])) < 1e-3)
        
if __name__=='__main__':
    unittest.main()