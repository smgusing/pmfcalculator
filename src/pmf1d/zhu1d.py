#! /usr/bin/env python
import numpy as np 
import logging,sys,os,time
import scipy
import scipy.optimize as opt 

import pmfcalculator
from pmfcalculator import cwham
from pmf1d import Pmf1d

np.seterr(all='raise',under='warn')

logger = logging.getLogger(__name__)


# ################################################33
# def calcA(dG,nmidp, hist, U_b, beta, N_k):
#     S = N_k.size
#     lg = np.zeros(S,dtype=np.float)
#     lg[1:] = np.cumsum(dG)
#     g = np.exp(lg)
#        
#     part1 = (N_k*lg).sum() 
#              
#      
#     #print "dG",dG
#     part2 = 0
#     for i in xrange(nmidp):
#          num = hist[i]
#          if num > 0:
#              denom = N_k*U_b[i,:] * g
#              try:
#                  part2 += num * np.log(num/denom.sum())
#              except FloatingPointError:
#                  pass
#                   
#      
#     A = -( part1 + part2)
#     print "A", A              
#     return A
# #     
# def calcAder(dG,nmidp, hist, U_b, beta, N_k):
#     S = N_k.size
#     lg = np.zeros(S,dtype=np.float)
#     lg[1:] = np.cumsum(dG)
#     derv = np.zeros_like(dG)
#     g = np.exp(lg)
#     
#     for i in range(1,S):
#         part2 = 0   
#           
#         for ii in xrange(nmidp):
#             num = hist[ii] * U_b[ii,i] 
#             if num > 0:
#                 denom = N_k*U_b[ii,:] * g
#                 try:
#                     part2 += num/denom.sum()  
#                 except FloatingPointError:
#                     pass
#          
#         derv[i-1] = N_k[i]*((g[i]*part2) -1) 
#      
#     dervcum = np.cumsum(derv)
#      
#     dervcum = dervcum[::-1]
#     print "DERV",dervcum
#      
#     return dervcum

# def minimize1d(F_k,nmidp, hist, U_b,
#                                 beta, N_k,g_k,chkdur):
#     
#     # convert fe to dG
#     dG = np.zeros(F_k.size-1)
#     F_knew = np.zeros_like(F_k)
#     U_b = np.exp(-beta*U_b)
#     res = opt.fmin_ncg(calcA,x0=dG,fprime=calcAder,args=(nmidp, hist, U_b,
#                        beta, N_k),avextol=1e-07)
#     
#     F_knew[1:] = np.exp(np.cumsum(res)) -1
#     return F_knew
#     

# def compute_logsum(numpyArray):
#     ''' return log of sum of exponent of array
#     '''
#     
#     ArrayMax = numpyArray.max()
#     
#     return np.log(np.exp(numpyArray - ArrayMax ).sum() ) + ArrayMax

    
def calcA(x, nmidp, hist, U_bexp, beta, N_k):
    S = N_k.size
    lf = np.zeros(S)
    lf[1:] += x  
    f = np.exp(lf)
    part1 = (N_k * lf).sum() 
    # print "G",g
    part2 = 0
    for i in xrange(nmidp):
        num = hist[i]
        if num > 0:
            # denom = np.log(N_k) + (-beta*U_b[i,:]) + g
            try:
                denom = N_k * U_bexp[i, :] * f
                part2 += num * np.log(num / denom.sum())
                # part2 += num * (np.log(num)-compute_logsum(denom))
            except FloatingPointError:
                pass
                  
     
    A = (-1 * part1) + (-1* part2)
    print "A", A              
    return A
     
def calcAder(x,nmidp, hist, U_bexp, beta, N_k):
    S = N_k.size
    derv = np.zeros(S-1)
    lf = np.zeros(S)
    lf[1:] += x  
    f = np.exp(lf)
    #print "G",g
    for i in range(1,S):
        part2 = 0   
          
        for ii in xrange(nmidp):
            num = hist[ii] * U_bexp[ii,i] 
            #denom = np.log(N_k) + (-beta*U_b[ii,i]) + g
            try:
                denom = N_k*U_bexp[ii,:] * f
                #part2 += (np.log(num)-compute_logsum(denom))
                part2 += num/denom.sum()    
            except FloatingPointError:
                pass
         
        derv[i-1] = N_k[i]*(f[i]*part2 -1) 
     
     
    #print "DERV",derv
     
    return derv

def minimize1d(F_k,nmidp, hist, U_b,
                                beta, N_k,g_k,chkdur):
     
    # convert fe to dG
    x = np.zeros(F_k.size-1)
    U_bexp = np.exp(-beta*U_b)
    res = opt.fmin_ncg(calcA,x0=x,fprime=calcAder,args=(nmidp, hist, U_bexp,
                       beta, N_k),avextol=1e-07)
    F_knew = np.zeros_like(F_k)
    F_knew[1:] = res
     
    return F_knew
    


class Zhu1d(Pmf1d):

    def __init__(self, bias, maxiter=1e5, tol=1e-6, nbins=None, temperature=None,
                 x0=None, fcx=None, g_k=None, chkdur = None ):
        '''
            Needs bias object, rest is self explainatory
        '''
        super(Zhu1d,self).__init__(bias, maxiter, tol, nbins, temperature,
                 x0, fcx, g_k, chkdur)
 
        logger.info("Wham1d successfully initialized")


    def estimateFreeEnergy(self,F_k = None, histogramfile = None, 
                                  fefile = None,g_k = None, chkpointfile = ".fe.npz",
                                  setToZero=None) :
        '''Compute 1D pmf using the given project and input arguments'''
        
        nbins = self.nbins
        beta = self.beta
        if os.path.isfile(histogramfile):
            self.hist, self.midp_bins, self.pos_bins, self.N_k = self.load_histogram(histogramfile)
            
        elif self.hist is None:
            logger.debug("histogram not initialized") 
            raise SystemExit("No histogram file and no histogram values ... Exiting")
        
        if self.hist.flags["C_CONTIGUOUS"] == False:
            self.hist = np.ascontiguousarray(self.hist, self.hist.dtype)
 
 
        nmidp = self.midp_bins.size

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

        if setToZero is not None:
            # Get umbrella number that correspond to given reaction coordinate
            windowZero = self.getWindow(setToZero)
        else:
            windowZero = 0   

         
        self.U_b=np.zeros((nmidp,self.K),dtype=np.float64)
        for i in xrange(nmidp):
                self.U_b[i,:] = self.bias.compute_potential_1D(params=(self.fcx,self.xopt),x=self.midp_bins[i])
                
        logger.debug("Biasing potentials stored")
        
        #self._self_consistent_iterations()
        F_k=self.F_k.copy()
        bconv = False
        itr = 0
        st = time.clock()
        self.hist = self.hist.astype(np.int64)
        av_err = -1
        while (bconv == False):
            # #  p[i]= number of counts in bin[i]/(total count* exp(beta * Free_energy-bias_potential)
            # # beta*exp(free_energy)=sum(exp(-beta*bias_pot)*prob)
            #F_knew = self._update_fe(F_k)
            if (itr >= self.maxiter):
                logger.warn("Maximum iterations reached (%s). Bailing out" % self.maxiter)
                break
            else:
                itr += self.chkdur
             
            
            F_knew = minimize1d(F_k,nmidp,
                                self.hist,self.U_b,
                                self.beta, self.N_k,g_k, self.chkdur)

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

    def _update_fe(self,F_k):
        
        F_knew = np.zeros_like(F_k)
        for i in xrange(self.midp_bins.size):
            num = self.hist[i]
            U_b = self.U_b[i, :]
            logbf = self.beta * (F_k - U_b) + np.log(self.N_k)
            denom = self._compute_logsum(logbf)
            if num == 0:
                logbf = (-self.beta * U_b) 
            else:
                logbf = (-self.beta * U_b) + np.log(num) - denom
                
                 
            F_knew += np.exp(logbf)
            
        F_knew = -1.*np.log(F_knew) / self.beta        
        F_knew = F_knew - F_knew[0]
        
        return F_knew
    
    def _compute_logsum(self,numpyArray):
        ''' return log of sum of exponent of array
        '''
        
        ArrayMax = numpyArray.max()
        
        return np.log(np.exp(numpyArray - ArrayMax ).sum() ) + ArrayMax
        
        
    def _compute_prob(self):
        
        F_k=self.F_k
        
        for i in xrange(self.midp_bins.size):
            num = self.hist[i]
            U_b = self.U_b[i, :]
            logbf = self.beta * (F_k - U_b) + np.log(self.N_k)
            denom = self._compute_logsum(logbf)
            if num == 0:
                self.prob[i] = 0
            else:
                self.prob[i] = np.exp ( np.log(num)- denom )
                
    def getWindow(self,rcoords):
        ''' get state number from reaction coordinate values
        
        Parameters: 
            rcoords: umbrella coord
        
        Returns:
            state: integer
            
        '''
        
        umbNo = np.digitize([rcoords], self.xopt) 
        logger.info("state with x0: %s will be set to zero",self.xopt[umbNo])
        
        return umbNo
