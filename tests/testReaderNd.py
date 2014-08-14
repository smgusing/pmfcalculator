#! /usr/bin/env python

import numpy as np
import unittest
import pmfcalculator
import os
from pmfcalculator.readerNd import ReaderNd


class test_readerNd(unittest.TestCase):
    
    def setUp(self):
        file_string = """xvgdir: /home/gurpreet/Documents/prog/workspace/pmfcalculator/tests/test1d/1D
xtcdir: /home/gurpreet/Documents/prog/workspace/pmfcalculator/tests/test1d/1D
stride: 1
trials: ['R1']
systems: ['dummy']
xvgsuffix: dh
xtcsuffix: 1ns
collvars: ['var1','var2']
var1: [0,1,2,3]
var2: [1.0,2.0,3.0,4.0]
var1fc: [50,50,50,50]
var2fc: [10,10,10,10]
bias: ["harmonic","harmonic"]
"""
        self.yamlfile = "test.yaml"
        with open(self.yamlfile,'w') as inpf:
            inpf.write(file_string)
        
        inpf.close()
        print("%s created"%self.yamlfile)
        
    def tearDown(self):
        if os.path.isfile(self.yamlfile): os.remove(self.yamlfile)
        
 
    def runTest(self):
        self.assertTrue(os.path.isfile(self.yamlfile))
        rd = ReaderNd(self.yamlfile)
        print rd.xvgfiles





if __name__ == "__main__":
    unittest.main()        