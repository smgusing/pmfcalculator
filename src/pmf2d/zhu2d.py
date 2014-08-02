#! /usr/bin/env python
import numpy as np 
import logging, sys, os
import time
import pmfcalculator
import scipy
import scipy.optimize as opt 
from pmf2d import Pmf2d
np.seterr(all='raise',under='warn')
logger = logging.getLogger(__name__)


         

def calcA(dG,nmidpx,nmidpy, hist, U_bij,
                                beta, N_k):
    S = N_k.size
    dGcumPadded = np.zeros(S,dtype=np.float)
    dGcumPadded[1:] = np.cumsum(dG)
      
    part1 = (N_k[1:]*dGcumPadded[1:]).sum() 
            
    
    
    part2 = 0
    for ii in xrange(nmidpx):
        for jj in xrange(nmidpy):
             num = hist[ii, jj]
             if num > 0:
                 denom = N_k*np.exp(-beta*U_bij[ii,jj,:]) * np.exp(dGcumPadded)
                 try:
                     part2 += num * np.log(num/denom.sum())
                 except FloatingPointError:
                     pass
                 
    
    A = -part1 -part2
    print "A", A              
    return A
    
def calcAder(dG,nmidpx,nmidpy, hist, U_bij,
                                beta, N_k):
    S = N_k.size
    dGcumPadded = np.zeros(S,dtype=np.float)
    dGcumPadded[1:] = np.cumsum(dG)
    derv = np.zeros_like(dG)

    for i in range(1,S):
        N_k[1:] * dGcumPadded[1:]
        part2 = 0   
         
        for ii in xrange(nmidpx):
            for jj in xrange(nmidpy):
                num = hist[ii, jj] * np.exp(-beta*U_bij[ii,jj,i]) 
                if num > 0:
                    denom = N_k*np.exp(-beta*U_bij[ii,jj,:]) * np.exp(dGcumPadded)
                    try:
                        part2 += num/denom.sum() - 1 
                    except FloatingPointError:
                        pass
        
        derv[i-1] = N_k[i]*np.exp(dGcumPadded[i])*part2 
    
    print "DERV",derv
    
    return derv
    
    
    
    
# def error_func(F_k,nmidpx,nmidpy, hist, U_bij,
#                                 beta, N_k):
#     A = calcA(F_k,nmidpx,nmidpy, hist, U_bij,
#                                 beta, N_k)
#     
#     err_k = np.abs(F_knew - F_k)
#     av_err = np.average(err_k)
#     print "Err",av_err
#     print F_knew -F_k
#     print 
#     return av_err

    
def minimize2d(F_k,nmidpx,nmidpy, hist, U_bij,
                                beta, N_k,g_k,chkdur):
    
    
    # convert fe to dG
    dG = np.zeros(F_k.size - 1, dtype = np.float)
    for i in range(dG.size):
        try:
            dG[i] = F_k[i+1]/ F_k[i]
            dG[i] = np.log(dG[i])
        except FloatingPointError:
            dG[i] = 0
   
   
   
   
#                                 beta, N_k)
#     for i in range(20):
#         print "self cons", i
#         F_knew = iterate_Fe(F_k,nmidpx,nmidpy, hist, U_bij,beta, N_k)
#         err_k = np.abs(F_knew - F_k)
#         av_err = np.average(err_k)
#         print "Err",av_err
#         F_k = F_knew
                         
    res=opt.minimize(calcA,x0=dG,args=(nmidpx,nmidpy, hist, U_bij,
                                beta, N_k),method='BFGS',jac=calcAder)
    
    #res=opt.fmin_cg(error_func,x0=F_k,args=(nmidpx,nmidpy, hist, U_bij,
    #                            beta, N_k),epsilon=2)
    #res=opt.fmin_cg(error_func,x0=F_k,args=(nmidpx,nmidpy, hist, U_bij,
    #                            beta, N_k),fprime=iterate_Fe_der)
    #F_knew = iterate_Fe(F_k,nmidpx,nmidpy, hist, U_bij,
    #                            beta, N_k)
    
    #print "NEW",F_knew
    
    dG = res.x0
    
    
    # convert dG to Fe
    F_knew = np.zeros_like(F_k)
    F_knew[:-1] = np.cumsum(dG)
    F_knew[:-1] = np.exp(F_knew[:-1])
    F_knew = F_knew- F_knew[0]
     
    
    
    
    return F_knew
    



class Zhu2d(Pmf2d):
    ''' Class to compute 2d wham
    
    Parameters: bias: object of class derived from Bias class
                    Must have calculate_potential_2d method
                maxiter: float
                    Number of iterations
                tol: float
                    tolerance
                nbins: list
                    number of bins in each dimension
                temperature: float
                    temperature in kelvin
                x0: array
                y0: array
                fcx: array
                fcy: array
                g_k: array
                chkdur: integer
                       
    '''
    def __init__(self, bias, maxiter=10e5, tol=10e-6, nbins=[100, 100],
                  temperature=300, x0=None, y0=None,fcx=None,
                  fcy=None,g_k=None,chkdur=None):
        
        super(Hummer2d,self).__init__(bias, maxiter, tol, nbins,
                  temperature, x0, y0,fcx,
                  fcy,g_k,chkdur)
        
        logger.info("Wham2d successfully initialized")

        
      
    def estimateFreeEnergy(self,F_k = None, histogramfile = None, 
                                  fefile = None,g_k = None, chkpointfile = ".fe.npz") :
        '''Compute 2D pmf using the given project and input arguments'''
        
        #nxbins =self.nbins[0]
        #nybins = self.nbins[1]
        beta = self.beta
        if os.path.isfile(histogramfile):
            self.hist, self.midp_xbins, self.midp_ybins, self.N_k = self.load_histogram(histogramfile)
        
        elif self.hist is None:
            logger.debug("histogram not initialized") 
            raise SystemExit("No histogram file and no histogram values ... Exiting")
            
        
        if self.hist.flags["C_CONTIGUOUS"] == False:
            self.hist = np.ascontiguousarray(self.hist, self.hist.dtype)
        
        nmidpx,nmidpy=self.midp_xbins.size,self.midp_ybins.size

        if (F_k is None):
            if (fefile is not None) and os.path.isfile(fefile):
                logger.info("loading free energies from  %s", fefile)
                self.F_k = np.load(fefile)['arr_0']
            elif os.path.isfile(chkpointfile):
                logger.info("loading free energies from chkpointfile %s", chkpointfile)
                self.F_k = np.load(chkpointfile)['arr_0']
            else:
                logger.info("Initial Free energies set to zeros")
        else:        
            logger.info("Initial Free energies to user supplied values")
            self.set_freeEnergies(F_k)
        
        if g_k is None:
            g_k = self.g_k
        
        bconv = False
        itr = 0
        self.U_bij=np.zeros((nmidpx,nmidpy,self.K),dtype=np.float64)
        for i in xrange(nmidpx):
            for j in xrange(nmidpy):
                #self.U_bij[i, j, :] = self._compute_biasing_pot(self.midp_xbins[i],self.midp_ybins[j])
                self.U_bij[i, j, :] = self.bias.compute_potential_2D( paramsX=(self.fcxy[:,0], self.xyopt[:,0]),
                                                                      paramsY=(self.fcxy[:,1], self.xyopt[:,1]),
                                                                      x=self.midp_xbins[i],
                                                                      y=self.midp_ybins[j] )
                
        logger.debug("Biasing potentials stored")
        
        #self._self_consistent_iterations()
        F_k = self.F_k.copy()
        self.hist = self.hist.astype(np.int64)
        prob = np.zeros([nmidpx,nmidpy],np.float64)
        st = time.clock()
        av_err = -1
        while (bconv == False):
            if (itr >= self.maxiter):
                logger.warn("Maximum iterations reached (%s). Bailing out" % self.maxiter)
                break
            else:
                itr += self.chkdur
            
            F_knew = minimize2d(F_k,nmidpx,nmidpy,
                                self.hist,self.U_bij,
                                self.beta, self.N_k,g_k,self.chkdur)
            
            err_k = np.abs(F_knew - F_k)
            av_err = np.average(err_k)
            F_k = F_knew
            
            logger.info("Iteration %s Average Error %s" % (itr, av_err))
            logger.info("saving fe chkpoint")
            np.savez(chkpointfile, F_k)
            if (av_err < self.tol):
                bconv = True
                
        logger.info("Error %s", av_err)
        logger.debug("Iter: %d  Time taken sec: %s", itr, time.clock() - st)
        self.F_k = F_k    
        self._compute_prob()
        

    def _compute_prob(self):
        ''' compute probabilites once final F_k are known
            Does proper normalization for angle
        '''
        
        F_k = self.F_k
        for i in xrange(self.midp_xbins.size):
            for j in xrange(self.midp_ybins.size):
                 num = self.hist[i, j]
                 U_b = self.U_bij[i, j, :]
                 logbf = self.beta * (F_k - U_b) + np.log(self.N_k)
                 #logbf[notzero] = self.beta * (F_k[notzero] - U_b[notzero]) + np.log(self.N_k[notzero])
                 denom = self._compute_logsum(logbf)
                 if num == 0:
                     self.prob[i,j] = np.NAN
                 else:    
                     self.prob[i,j] = np.exp ( np.log(num) - denom )
        
        
        
    def _compute_logsum(self,numpyArray):
        ''' return log of sum of exponent of array
        '''
        
        ArrayMax = numpyArray.max()
        return np.log(np.exp(numpyArray - ArrayMax ).sum() ) + ArrayMax
    
        
            