#! /usr/bin/env python
import numpy as np 
import logging,sys,os,time
import pmfcalculator
from pmfcalculator import cwham
from pmf1d import Pmf1d
np.seterr(all='raise',under='warn')

logger = logging.getLogger(__name__)


class Wham1d(Pmf1d):

    def __init__(self, bias, maxiter=1e5, tol=1e-6, nbins=None, temperature=None,
                 x0=None, fcx=None, g_k=None, chkdur = None ):
        '''
            Needs bias object, rest is self explainatory
        '''
        super(Wham1d,self).__init__(bias, maxiter, tol, nbins, temperature,
                 x0, fcx, g_k, chkdur)
 
        logger.info("Wham1d successfully initialized")


    def estimateFreeEnergy(self,F_k = None, histogramfile = None, 
                                  fefile = None,g_k = None, chkpointfile = ".fe.npz",
                                  setToZero=None) :
        '''Compute 1D pmf using the given project and input arguments

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
             
            
            F_knew = cwham.minimize1d(F_k,nmidp,
                                self.hist,self.U_b,
                                self.beta, self.N_k,g_k, self.chkdur,windowZero)

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
        
