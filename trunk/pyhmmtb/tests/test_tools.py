'''
Created on 12.06.2013

@author: christian
'''

import unittest

import numpy as np
from pyhmmtb.tools import normalise, sqdist, approxeq, logdet, isposdef,\
    max_mult

class NormaliseTest(unittest.TestCase):
    
    def testVectorRow(self):
        v = np.array([1, 2])
        vn, s = normalise(v)
        assert np.all(np.abs(vn-np.array([0.33333, 0.66666])) < 1e-3)
        assert s==3
        
    def testVectorCol(self):
        v = np.array([[1, 2]])
        vn, s = normalise(v)
        assert np.all(np.abs(vn-np.array([0.33333, 0.66666])) < 1e-3)
        assert s==3
        
    def testMatrix(self):
        v = np.array([[1, 2], 
                      [1, 2]])
        vn, s = normalise(v)
        assert np.all(np.abs(vn-np.array([[ 0.16666667,  0.33333333],
                                          [ 0.16666667,  0.33333333]])) < 1e-3)
        assert s==6
        
    def testTensor(self):
        v = np.array([[[1, 2], 
                       [1, 2]], 
                      [[1, 2], 
                       [1, 2]]])
        vn, s = normalise(v)
        assert np.all(np.abs(vn-np.array([[[ 0.08333333, 0.16666667],
                                           [ 0.08333333, 0.16666667]],
                                          [[ 0.08333333,  0.16666667],
                                           [ 0.08333333,  0.16666667]]])) < 1e-3)
        assert s==12
        
class SqdistTest(unittest.TestCase):
    
    def test1(self):
        p = np.matrix([[1,2],
                       [1,2]])
        q = np.matrix([[1,2],
                       [1,2]])
        d = sqdist(p, q)
        assert np.all(d==np.array([[0,2],[2,0]]))
        
    def test2(self):
        p = np.matrix([[1,2],
                       [1,2]])
        q = np.matrix([[1,2],
                       [1,2]])
        A = np.matrix([[1,0],
                       [.5,0]])
        d = sqdist(p, q, A)
        assert np.all(d==np.array([[0,1.5],[1.5,0]]))
        
class ApproxeqTest(unittest.TestCase):
    
    def testVector(self):
        a = np.matrix([1, 2])
        b = np.matrix([1, 2])
        
        assert approxeq(a,b)
        
        b = np.matrix([1, 2.1])
        
        assert not approxeq(a,b)
        
        assert approxeq(a,b,tol=1e-1,rel=True)
        
        assert not approxeq(a,b,tol=1e-2,rel=True)
    
    def testMatrix(self):
        a = np.matrix([[1, 2],[1, 2]])
        b = np.matrix([[1, 2],[1, 2]])
        
        assert approxeq(a,b)
        
        b = np.matrix([[1, 2],[1, 2.1]])
        
        assert not approxeq(a,b)
        
        assert approxeq(a,b,tol=1e-1,rel=True)
        
        assert not approxeq(a,b,tol=1e-2,rel=True)
        
class LogdetTest(unittest.TestCase):
    
    def test(self):
        A = np.matrix([[1,0],
                       [0,1]])
        assert logdet(A)==0
        
        A = np.matrix([[2,0],
                       [0,2]])
        assert np.abs(logdet(A)-1.3862)<1e-4
        
class IsposdefTest(unittest.TestCase):
    
    def test(self):
        A = np.matrix([[1,0],
                       [0,1]])
        assert isposdef(A)
        
        A = np.matrix([[1,0],
                       [0,-1]])
        
        assert not isposdef(A)
        
        A = np.matrix([[1,0],
                       [1,-1]])
        
        assert not isposdef(A)
        
class MaxmultTest(unittest.TestCase):
    
    def test(self):
        A = np.matrix([[2,1],
                       [0,1]])
        x = np.matrix([[1],[1]])
        assert np.all(max_mult(A, x)==np.matrix([[2],[1]]))
    
if __name__=='__main__':
    unittest.main()