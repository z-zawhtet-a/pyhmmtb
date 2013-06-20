'''
Created on 13.06.2013

@author: christian
'''

import unittest
import numpy as np
from pyhmmtb.hmm.misc import mc_sample

class MCSampleTest(unittest.TestCase):
    
    def test(self):
        prior = np.array([1, 0, 0])
        trans = np.matrix([[0, 1, 0],
                           [0, 0, 1],
                           [0, 0, 1]])
        len = 4
        numex = 2
    
        M = mc_sample(prior, trans, len, numex)
        assert np.all(np.abs(M-np.array([[0, 1, 2, 2],
                                         [0, 1, 2, 2]]))) < 1e-3
    
if __name__=='__main__':
    unittest.main()