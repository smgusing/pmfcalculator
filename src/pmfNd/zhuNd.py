#! /usr/bin/env python
import numpy as np
import logging, sys, os
#import scipy


import pmfcalculator
from pmfcalculator import StatsUtils
from pmfNd import PmfNd
from whamNd import WhamNd
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
    def __init__(self, temperature=None, scIterations = 0, tolerance=1e-5):

        super(ZhuNd,self).__init__(temperature)
        self.scIterations = scIterations
        logger.info("ZhuNd successfully initialized")
        self.tolerance=tolerance


    def estimateWeights(self) :
        '''Compute 2D pmf using the given project and input arguments
        '''
        log_c = -self.beta*self.Ub

        if self.scIterations > 0:
            wham = WhamNd(temperature=self.temperature,maxSteps=self.scIterations, tolerance=self.tolerance)

            wham.setParams(self.Ub,histogramfile=self.histFN)
            wham.estimateWeights()
            self.f = wham.f

        nonzero = self.f != 0.0
        g = np.zeros_like(self.f)
        g[nonzero] = np.log(self.f[nonzero])
        bounds = [(None,None) for i in range(g.size)]

        if globalVar_useScipy == True:

            res = opt.fmin_l_bfgs_b(calcA,x0=g,fprime=calcAder,args=(self.hist,
                                log_c, self.beta, self.sim_samples_used),factr=1,
                                    disp=10, bounds = bounds)

#             res = opt.fmin_bfgs(calcA,x0=g,fprime=calcAder,args=(self.hist,
#                                 log_c, self.beta, self.sim_samples_used),
#                                 epsilon=1e-9,full_output=True)


            g = res[0]

        else:
            calcA_partial    = lambda g: calcA(g, self.hist, log_c, self.beta, self.sim_samples_used)
            calcAder_partial = lambda g: calcAder(g, self.hist, log_c, self.beta, self.sim_samples_used)
            arbitraryTolerance = 1e-7
            g = naiveMinimize(calcA_partial, calcAder_partial, g, arbitraryTolerance)

        f = np.exp(g)
        self.f = f

        #self.prob = self.unbiasedProb(f,np.exp(log_c))
        if self.scIterations > 0:
            wham.setParams(self.Ub,histogramfile=self.histFN,f=f)
            wham.estimateWeights()
            self.f = wham.f

        self.prob = self.unbiasedProb(self.f,np.exp(log_c))

        logger.info("checkpointing")
        np.savez(self.chkpointfile,self.f)


