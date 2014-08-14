#! /usr/bin/env python
import numpy as np
import numpy.random as rand

import unittest
import os,logging

import pmfcalculator
import pmfcalculator.potentials as bpot



class testPotentials(unittest.TestCase):
    
    def setUp(self):
        aa=np.arange(36).reshape(3,3,4)
        
        kx = [33,44]
        ky = [11,22]
        x0 = [-1,-2]
        y0 = [-3,-4]
        mx=np.array([0.0,1.0,2.0])
        my=np.array([3.0,4,5])
        
        self.histedges = [mx,my]
        self.biasType = ["harmonic","harmonic"]
        self.collvars = ['x','y']
        self.vardict = {}
        self.vardict['x'] = x0
        self.vardict['xfc'] = kx
        self.vardict['y'] = y0
        self.vardict['yfc'] = ky


    def runTest(self):
        
        U = bpot.biasPotential(self.biasType,self.collvars,self.vardict,self.histedges)
        u0 = bpot.harmonic(0.5, 33, -1) + bpot.harmonic(3.5,11,-3)
        u114 = bpot.harmonic(1.5, 44, -2) + bpot.harmonic(4.5,22,-4)
        self.assertEqual(U[0,0,0],u0)
        self.assertEqual(U[1,1,3],u114)




if __name__ == "__main__":
    unittest.main()        