'''
Created on 13.06.2013

@author: christian
'''

import unittest
import numpy as np
from pyhmmtb.stats.misc import sample_discrete

class SamplediscreteTest(unittest.TestCase):
    
    def testRow(self):
        assert sample_discrete([0.8, 0.2], 1, 10).shape == (1,10)
    
    def testDistSmall(self):
        n = 10000
        M = np.zeros((n,2))
        for i in range(n):
            M[i,:] = sample_discrete([0.8, 0.1, 0.1], 1, 2)
        dist = np.bincount(M.ravel().astype(int)) / (n*2.)
        assert np.all(np.abs(dist-[0.8, 0.1, 0.1]) < 1e-2)
        
    def testDistLarge(self):
        n = 10000
        M = sample_discrete([0.8, 0.2], n, 10)
        dist = np.bincount(M.ravel().astype(int)) / (n*10.)
        assert np.all(np.abs(dist-[0.8, 0.2]) < 1e-2)
        
if __name__=='__main__':
    unittest.main()