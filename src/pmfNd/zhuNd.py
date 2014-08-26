#! /usr/bin/env python
import numpy as np 
import logging, sys, os
#import scipy


#import pmfcalculator
from pmfcalculator import StatsUtils
from pmfNd import PmfNd
np.seterr(all='raise',under='warn',over='warn')
logger = logging.getLogger(__name__)

# select minimizer
try:
    import scipy.optimize as opt 
    globalVar_useScipy = True
    logger.info("Using scipy.optimize for minimization.")
except ImportError:
    from naiveMinimizer import naiveMinimize
    globalVar_useScipy = False
    logger.info("Could not import scipy.optimize; falling back to a naive minimization implementation.")


def compute_logsum(numpyArray):
    ''' return log of sum of exponent of array
        taken from MBAR
    '''
    
    ArrayMax = numpyArray.max()
    
    return np.log(np.exp(numpyArray - ArrayMax ).sum() ) + ArrayMax


def calcA(g, M, log_c, beta, N):
    ''' use equation 19 
    
    M : array_type
        ND array with frequency in each bin.
    
    log_c : array_type
       ND+1 array [nbins,nsims], baising potential at bin i evaluated using bias from simulation k

     N : array_type
        Total number of samples from simulation k
   
    
    '''
    
    nbins = M.size
    logN = np.log(N)
           
    part2 = 0.0
    for i,j in np.ndenumerate(M):
        if M[i] > 0:
            denom =  compute_logsum( logN + log_c[i] + g ) 
            part2 = part2 + ( M[i] * (np.log(M[i]) - denom ) ) 
        else:
            pass
              
     
    A = (-(N * g).sum() ) - part2
    
    #print "A", A 
    
    return A
     
def calcAder(g,M, log_c, beta, N):
    ''' use equation 20
    '''
    nbins = M.size
    nsims = N.size
    
    p = np.zeros(M.shape)
    
    for i,j in np.ndenumerate(p):
        
        denom = np.exp(compute_logsum( np.log(N) + g + log_c[i] ) )
        
        if denom > np.finfo(float).eps:
            p[i] = M[i]/ denom 
        else:
            p[i] = np.inf
                
        
    derv = np.zeros(nsims)
    
    for i in range(nsims):
        derv[i] = N[i] * (np.exp(g[i]) *  (p * np.exp(log_c[...,i]) ).sum() - 1.)
        
    return derv

    



class ZhuNd(PmfNd):
    ''' Class to compute nD wham
    
     Parameters
    --------------
        bias: array of functions to compute biasing potentials
       temperature: float
              temperature in kelvin
                       
    '''
    def __init__(self, temperature=None ):
        
        super(ZhuNd,self).__init__(temperature)
        
        logger.info("ZhuNd successfully initialized")

        
      
    def estimateWeights(self) :
        '''Compute 2D pmf using the given project and input arguments
        '''
        nonzero = self.f != 0.0
        g = np.zeros_like(self.f)
        g[nonzero] = np.log(self.f[nonzero])
        bounds = [(None,None) for i in range(g.size)]
        #bounds = [(-100,100) for i in range(g.size)]
        bounds[self.windowZero] = (0,0)
        log_c = -self.beta*self.Ub
        
        if globalVar_useScipy == True:
            
            res = opt.fmin_l_bfgs_b(calcA,x0=g,fprime=calcAder,args=(self.hist, log_c,
                                self.beta, self.sim_samples_used),factr=1, disp=10, bounds = bounds)
            g = res[0]
        else:
            calcA_partial    = lambda g: calcA(g, self.hist, log_c, self.beta, self.sim_samples_used)
            calcAder_partial = lambda g: calcAder(g, self.hist, log_c, self.beta, self.sim_samples_used)
            arbitraryTolerance = 1e-7 
            g = naiveMinimize(calcA_partial, calcAder_partial, g, arbitraryTolerance)
        
        f = np.exp(g)
        self.prob = self.unbiasedProb(f,np.exp(log_c))
    
        return np.exp(g)
        


#     def _compute_prob(self):
#         ''' compute probabilites once final F_k are known
#             Does proper normalization for angle
#         '''
#         
#         F_k = self.F_k
#         prob = np.zeros_like(self.hist,dtype=np.float)
#         for i,j in np.ndenumerate(self.hist):
#             num = self.hist[i]
#             U = self.Ub[i]
#             logbf = F_k - self.beta * U + np.log(self.sim_samples_used)
#             denom = compute_logsum(logbf)
#             if num == 0:
#                 #prob[i] = np.NAN
#                 prob[i] = 0
#             else:    
#                 prob[i] = np.exp (np.log(num) - denom)
# 
#         self.prob = prob





#         for i in xrange(self.midp_xbins.size):
#             for j in xrange(self.midp_ybins.size):
#                  num = self.hist[i, j]
#                  U_b = self.U_bij[i, j, :]
#                  logbf = self.beta * (F_k - U_b) + np.log(self.N_k)
#                  #logbf[notzero] = self.beta * (F_k[notzero] - U_b[notzero]) + np.log(self.N_k[notzero])
#                  denom = StatsUtils.compute_logsum(logbf)
#                  if num == 0:
#                      self.prob[i,j] = np.NAN
#                  else:    
#                      self.prob[i,j] = np.exp ( np.log(num) - denom )
#         
        
        
 