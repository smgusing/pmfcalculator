#! /usr/bin/env python
import numpy as np 
import logging, sys, os
#import scipy
import scipy.optimize as opt 


#import pmfcalculator
from pmfcalculator import StatsUtils
from pmfNd import PmfNd
np.seterr(all='raise',under='warn')
logger = logging.getLogger(__name__)

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
            denom =  compute_logsum( logN + log_c[i, :] + g ) 
            part2 = part2 + ( M[i] * (np.log(M[i]) - denom ) ) 
        else:
            pass
              
     
    A = (-(N * g).sum() ) - part2
    
    print "A", A 
    
    return A
     
def calcAder(g,M, log_c, beta, N):
    ''' use equation 20
    '''
    nbins = M.size
    nsims = N.size
    
    p = np.zeros(M.shape)
    
    for i,j in np.ndenumerate(p):
        denom = np.exp(compute_logsum( np.log(N) + g + log_c[i,:] ) ) 
        try:
            p[i] = M[i]/ denom 
        
        except:
            p[i] = 0.
        
        
    derv = np.zeros(nsims)
    
    for i in range(nsims):
        derv[i] = N[i] * (np.exp(g[i]) *  (p * np.exp(log_c[:,i]) ).sum() - 1.)
        
    #print "DER", derv    
    return derv




def minimizeNd(F_k, M, U_b, beta, g_k, N, windowZero):
    ''' use equation 18, 19, 20 
    
    Parameters
    -------------
    
    F_k : array type
        Free energy of simulation k
    
    M : array_type
        ND array with frequency in each bin.
    
    U_b : array_type
       ND+1 array [nbins,nsims], baising potential at bin i evaluated using bias from simulation k
       
    N : array_type
        Total number of samples from simulation k
        
    
    '''
    
    g = np.copy(F_k)
    bounds = [(None,None) for i in range(g.size)]
    bounds[windowZero] = (0,0)
    
    log_c = -beta*U_b
    res = opt.fmin_l_bfgs_b(calcA,x0=g,fprime=calcAder,args=(M, log_c,
                        beta, N),factr =10, bounds = bounds)
    F_k = res[0]
    
    return F_k


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

        
      
    def estimateFreeEnergy(self,Ub,F_k = None, histogramfile = None, 
                                  fefile = None,g_k = None, chkpointfile = "chkp.fe.npz",
                                  setToZero=None) :
        '''Compute 2D pmf using the given project and input arguments
            Parameters:
                F_k: Array
                    Initial guess of free energies
                histogramfile: npz file
                fefile: npz file
                g_k: Array
                chkpointfile: npzfile
                setToZero: list 
                    reaction coordinate value where fe should be shifted to zero
        '''
        
        self.Ub = Ub
        if os.path.isfile(histogramfile):
            self.load_histogram(histogramfile)
        
        elif self.hist is None:
            logger.debug("histogram not initialized") 
            raise SystemExit("No histogram file and no histogram values ... Exiting")
            
        
        if self.hist.flags["C_CONTIGUOUS"] == False:
            self.hist = np.ascontiguousarray(self.hist, self.hist.dtype)
        
        self.ndim = len(self.histEdges)
        if (F_k is None):
            if (fefile is not None) and os.path.isfile(fefile):
                logger.info("loading free energies from  %s", fefile)
                self.F_k = np.load(fefile)['arr_0']
            elif os.path.isfile(chkpointfile):
                logger.info("loading free energies from chkpointfile %s", chkpointfile)
                self.F_k = np.load(chkpointfile)['arr_0']
            else:
                logger.info("Initial Free energies set to zeros")
                self.F_k = np.zeros(self.sim_samples.size)
        else:        
            logger.info("Initial Free energies to user supplied values")
            self.set_freeEnergies(F_k)
        
        if g_k is None:
            g_k = np.ones_like(self.F_k)
            self.g_k = g_k
            
            
        if setToZero is not None:
            # Get umbrella number that correspond to given reaction coordinate
            windowZero = setToZero
        else:
            windowZero = 0   
        
                        
        logger.debug("Biasing potentials stored")
        
        #self._self_consistent_iterations()
        F_k = self.F_k.copy()
        self.hist = self.hist.astype(np.int64)
        F_knew = minimizeNd(F_k,self.hist,Ub,self.beta,
                                self.g_k,self.sim_samples_used,windowZero)
            
        logger.info("saving fe chkpoint")
        np.savez(chkpointfile, F_knew)
                
        self.F_k = F_knew  
        self._compute_prob()
        
        

    def _compute_prob(self):
        ''' compute probabilites once final F_k are known
            Does proper normalization for angle
        '''
        
        F_k = self.F_k
        prob = np.zeros_like(self.hist,dtype=np.float)
        for i,j in np.ndenumerate(self.hist):
            num = self.hist[i]
            U = self.Ub[i]
            logbf = F_k - self.beta * U + np.log(self.sim_samples_used)
            denom = compute_logsum(logbf)
            if num == 0:
                #prob[i] = np.NAN
                prob[i] = 0
            else:    
                prob[i] = np.exp (np.log(num) - denom)

        self.prob = prob
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
        
        
    
    def calcFrameWeight(self,x,y):
        ''' compute unnormalized weight of an individual frame
        '''
        
        U_b = self.bias.compute_potential_2D( paramsX=(self.fcxy[:,0], self.xyopt[:,0]),
                                              paramsY=(self.fcxy[:,1], self.xyopt[:,1]),
                                              x=x, y=y )
        logbf = self.beta * (self.F_k - U_b) + np.log(self.N_k)
        denom = StatsUtils.compute_logsum(logbf)
        try:
            w = 1./np.exp(denom)
        except ZeroDivisionError:
            logger.critical("weight w is infinite! ... I will quit")
            raise SystemExit
        
        return w
        
    def getWindow(self,rcoords):
        ''' get state number from reaction coordinate values
        
        Parameters: 
            rcoords: list with two elements
        
        Returns:
            state: integer
            
        '''
        
        rcx,rcy = rcoords
        
        sindexx = np.digitize([rcx], self.x0) 
        sindexy = np.digitize([rcy], self.y0)
        
        umbNo = sindexx * self.y0.size + sindexy
        
        logger.info("state with x0,y0: %s,%s will be set to zero",self.xyopt[umbNo,0],self.xyopt[umbNo,1])
        
        return umbNo
        
         
        
        
        
        