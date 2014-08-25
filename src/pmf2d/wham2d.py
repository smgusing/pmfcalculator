#! /usr/bin/env python
import numpy as np 
import logging, sys, os
import time

import pmfcalculator
from pmfcalculator import cwham,StatsUtils
from pmf2d import Pmf2d
np.seterr(all='raise',under='warn')
logger = logging.getLogger(__name__)


class Wham2d(Pmf2d):
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
        
        super(Wham2d,self).__init__(bias, x0, y0,fcx, fcy,maxiter, tol, nbins,
                  temperature, g_k,chkdur)
        
        logger.info("Wham2d successfully initialized")

        
      
    def estimateFreeEnergy(self,F_k = None, histogramfile = None, 
                                  fefile = None,g_k = None, chkpointfile = ".fe.npz",
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
       
        beta = self.beta
        if os.path.isfile(histogramfile):
            self.hist, self.midp_xbins, self.midp_ybins, self.N_k, xedges, yedges = self.load_histogram(histogramfile)
        
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
            
            
        if setToZero is not None:
            # Get umbrella number that correspond to given reaction coordinate
            windowZero = self.getWindow(setToZero)
        else:
            windowZero = 0   
        
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
        bconv = False
        itr = 0
        while (bconv == False):
            if (itr >= self.maxiter):
                logger.warn("Maximum iterations reached (%s). Bailing out" % self.maxiter)
                break
            else:
                itr += self.chkdur
            
            F_knew = cwham.minimize2d(F_k,nmidpx,nmidpy,
                                self.hist,self.U_bij,
                                self.beta, self.N_k,g_k,self.chkdur,windowZero)
            
            err_k = np.abs(np.exp(F_knew) - np.exp(F_k))
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
                 denom = StatsUtils.compute_logsum(logbf)
                 if num == 0:
                     self.prob[i,j] = np.NAN
                 else:    
                     self.prob[i,j] = np.exp ( np.log(num) - denom )
        
        
        
    
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
        
         
        
        
        
        