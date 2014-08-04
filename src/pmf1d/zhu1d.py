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



    
def calcA(g, M, c, beta, N):
    ''' use equation 19 '''
    
    nbins = M.size

        
    part2 = 0.0
    for i in xrange(nbins):
        if M[i] > 0:
            # denom = np.log(N_k) + (-beta*U_b[i,:]) + g
            try:
                denom = ( N * c[i, :] * np.exp(g) ).sum()
                part2 += M[i] * np.log(M[i] / denom ) 
                # part2 += num * (np.log(num)-compute_logsum(denom))
            except FloatingPointError:
                pass
        else:
            pass
              
     
    A = (-1. * (N * g).sum() ) + (-1. * part2)
    
    print "A", A 
    #print "G", g             
    
    return A
     
def calcAder(g,M, c, beta, N):
    ''' use equation 20
    '''
    nbins = M.size
    nsims = N.size
    
    # calculate p
    p = np.zeros(nbins)
    for i in range(nbins):
        denom = ( N * np.exp(g) * c[i,:] ).sum() 
        try:
            p[i] = M[i]/ denom 
        
        except:
            p[i] = 0.
        
        
    derv = np.zeros(nsims)
    #print "P", p
    
    for i in range(nsims):
        derv[i] = N[i] * np.exp(g[i]) * ( (p * c[:,i] ).sum() - 1.)
        
    #print "DER", derv    
    return derv
        

def minimize1d(F_k, M, U_b, beta, N, chkdur):
    ''' use equation 18, 19, 20 
    
    Parameters
    -------------
    
    F_k : array type
        Free energy of simulation k
    
    M : array_type
        1D array with frequency in each bin.
    
    U_b : array_type
       2D array [nbins,nsims], baising potential at bin i evaluated using bias from simulation k
       
    N : array_type
        Total number of samples from simulation k
        
    
    '''
    
     
    # convert f to g = log(f)
    g = np.copy(F_k)
    g[np.where(g == 0.0)] = 1.0
    g = np.log(g) 
    c = np.exp(-beta*U_b)

    res = opt.fmin_cg(calcA,x0=g,fprime=calcAder,args=(M, c,
                        beta, N),full_output=True,maxiter = 2)
    F_k = np.exp(res.x)
#     print "sss", g
    
#     res = opt.fmin_powell(calcA,x0=g,args=(M, c,
#                        beta, N),full_output=True,)
    
#    print "old" , F_k
    print "new",  res 
    return F_k


def calcA_dg(dg, M, c, beta, N):
    ''' using equation 22a  '''
    
    nbins = M.size
    nsims = N.size
    
    g = np.zeros(nsims)
    g[1:] = np.cumsum(dg)
    
        
    part2 = 0.0
    for i in xrange(nbins):
        if M[i] > 0:
            # denom = np.log(N_k) + (-beta*U_b[i,:]) + g
            try:
                denom = ( N * c[i, :] * np.exp(g) ).sum()
                part2 += M[i] * np.log(M[i] / denom ) 
                # part2 += num * (np.log(num)-compute_logsum(denom))
            except FloatingPointError:
                pass
        else:
            pass
              
     
    A = (-1. * (N * g).sum() ) + (-1. * part2)
    
    print "A", A 
    #print "G", g             
    
    return A

def calcA_dg_der(dg,M, c, beta, N):
    ''' use equation 20
    '''

    nbins = M.size
    nsims = N.size

    
    g = np.zeros(nsims)
    g[1:] = np.cumsum(dg)

    
    # calculate p
    p = np.zeros(nbins)
    for i in range(nbins):
        denom = ( N * np.exp(g) * c[i,:] ).sum() 
        try:
            p[i] = M[i]/ denom 
        
        except:
            p[i] = 0.
        
        
    derv = np.zeros(nsims)
    #print "P", p
    
    for i in range(nsims):
        derv[i] = N[i] * np.exp(g[i]) * ( (p * c[:,i] ).sum() - 1.)
    
    derv_cum = np.cumsum(derv[1:])    
    #print "DER", derv    
    return derv_cum



def minimize1d_1(F_k, M, U_b, beta, N, chkdur):
    ''' use equation 21a -- 22b 
    
    Parameters
    -------------
    
    F_k : array type
        Free energy of simulation k
    
    M : array_type
        1D array with frequency in each bin.
    
    U_b : array_type
       2D array [nbins,nsims], baising potential at bin i evaluated using bias from simulation k
       
    N : array_type
        Total number of samples from simulation k
        
    
    '''
    
    # convert f to g = log(f)
    g = np.copy(F_k)
    g[np.where(g == 0.0)] = 1.0
    g = np.log(g)
    c = np.exp(-beta*U_b)
    
    dg = g[1:] - g[:-1]
    
    res = opt.fmin_cg(calcA_dg,x0=dg,fprime=calcA_dg_der,args=(M, c,
                        beta, N),full_output=True)
    print "new",  res 

    sys.exit()
    F_k = np.exp(res[0])
#     print "sss", g
    
#     res = opt.fmin_powell(calcA,x0=g,args=(M, c,
#                        beta, N),full_output=True,)
    
#    print "old" , F_k
    return F_k







    


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
        '''Estimates F_k 
        
        Parameters
        ---------------
                F_k : array type
                Free energy for simulation k

                histogramfile : npz file 
                
                fefile : npz file
                
                g_k : array type
                    statistical inefficiency

                chkpointfile: npz file

                setToZero: float
                    collective variable where PMF profile should be set to zero
        '''
        
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
             
            
            F_knew = minimize1d_1(F_k, self.hist,self.U_b,
                                self.beta, self.N_k, self.chkdur)

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
