#! /usr/bin/env python

import abc
import numpy as np

class Bias1D(object):
    ''' Abstract base class for baising potentials
    
    '''
    
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def compute_potential_1D(self,params,x):
        return

class Bias2D(object):

    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def compute_potential_2D(self,paramsX,paramsY,x,y):
        return
    
 
 
class HarmonicBias(Bias1D):
    
    def compute_potential_1D(self,params,x):
        ''' compute harmonic potential
        
        Parameters: params: tuple
                        should contain force constants and Xopt
                        both could be vector or scalar
                    
                    x: array or scalar
                    
        Returns: U: array or scalar
                if either of forcek or x or xopt is vector then it will be
                vector
        
        '''
        
        forceConst,xopt = params
        
        U = 0.5 * forceConst * np.square(x - xopt)
        
        return U
    
class CosineBias(Bias1D):
    
    def compute_potential_1D(self,params,x):
       ''' cosine potential for angle
       
       '''

       forceConst,xopt = params
       U = forceConst * (1.0 - np.cos(np.radians(x - xopt)))
       
       return U
    
class Harmonic_cosine(Bias2D):
    ''' Both Harmonic and Cosine potentials
    '''
    
    def compute_potential_2D(self,paramsX,paramsY,x,y):
        
        posB = HarmonicBias()
        angleB = CosineBias()
        Ux = posB.compute_potential_1D(paramsX,x)
        Uy = angleB.compute_potential_1D(paramsY, y)

        return Ux + Uy
         
        
        
        
    
