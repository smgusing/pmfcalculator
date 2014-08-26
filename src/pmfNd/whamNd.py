import numpy as np 
np.seterr(all='raise',under='warn',over='warn')
import logging,sys,os


import pmfcalculator
from pmfNd import PmfNd

logger = logging.getLogger(__name__)


class WhamNd(PmfNd):
    
    def __init__(self,temperature=None):
        
        super(WhamNd,self).__init__(temperature)
        
    
    def estimateWeights(self,maxSteps = 10000, tolerance = 1e-5) :
        '''Compute 2D pmf using the given project and input arguments
            Parameters:
            
        '''
        curErr = 1.0 + tolerance
        step = 0
        feconst = 1.0
        
        f = np.copy(self.f)
        f = f - f.min()
        zerof = np.where(f==0)
        f[zerof] = feconst
        c = np.exp(-self.beta* self.Ub)
        while (curErr > tolerance) and ( step < maxSteps):
            p =  self.unbiasedProb(f,c)
            new_f = self.calcWeights(p,c)
            new_f = new_f - new_f.min() + feconst # Avoid all zero solution
            avg_err=np.sum(np.abs(new_f - f))
            f = new_f
            step += 1
            logger.info("Step %s : Sum change %s",step,avg_err)
        
        self.prob = self.unbiasedProb(f, c)
            
        self.f = f

    def unbiasedProb(self,f,c):
        ''' use wham equations, equation 15
        '''
         
        p = np.zeros(self.hist.shape)
         
        for i,j in np.ndenumerate(p):
            denom = ( self.sim_samples_used * f * c[i]).sum()
     
            if denom > np.finfo(float).eps:
                p[i] = self.hist[i]/ denom 
            else:
                p[i] = np.inf
     
        return p

    def calcWeights(self,p,c):
        ''' wham equations, equation 7 
        ''' 
        
        nsim = c.shape[-1]
        f = np.zeros(nsim)
        
        for i in range(nsim):
            denom = (p * c[...,i]).sum()
            if denom > np.finfo(float).eps:
                f[i] = 1./ denom
            else:
                f[i] = np.inf
        
        return f
        








# def calcP(M,N,f,c):
#      
#     p = np.zeros(M.shape)
#      
#     for i in range(M.shape[0]):
#         for j in range(M.shape[1]):
#             denom = (N * f * c[i,j,:]).sum()
#  
#             if denom > np.finfo(float).eps:
#                 p[i,j] = M[i,j]/ denom 
#             else:
#                 p[i] = np.inf
#  
#     return p
#  
# def calcF(p,c):
#      
#     nsim = c.shape[-1]
#     f = np.zeros(nsim)
#     denom = np.zeros(nsim) 
#     for i in range(p.shape[0]):
#         for j in range(p.shape[1]):
#             denom =  denom + p[i,j] * c[i,j,:]
#      
#     f = 1./denom    
#     return f
        
