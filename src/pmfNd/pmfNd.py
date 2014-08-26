#! /usr/bin/env python
import abc
import numpy as np 
import logging, sys, os
import time
from pmfcalculator import minimize


np.seterr(all='raise',under='warn')
logger = logging.getLogger(__name__)

def compute_logsum(numpyArray):
    ''' return log of sum of exponent of array
        taken from MBAR
    '''
    ArrayMax = numpyArray.max()
    return np.log(np.exp(numpyArray - ArrayMax ).sum() ) + ArrayMax




class PmfNd(object):
    ''' Base Class to compute N dimensional PMF 
    
     Parameters
    --------------
       temperature: float
              temperature in kelvin
    '''

    __metaclass__ = abc.ABCMeta

    
    def __init__(self, temperature=None ):
        
        R = 8.3144621 / 1000.0  # Gas constant in kJ/mol/K
        
        if temperature is None:
            self.beta = None
        else:
            self.beta = 1.0 / (R * temperature)  # inverse temperature of simulations (in 1/(kJ/mol))
        
        logger.info("PmfNd successfully initialized")
        
    def make_ndhistogram(self, observ=None, cv_ranges=None,number_bins=None):
        ''' Construct histogram
        
        Parameters
        ------------
             observ: list of arrays
                        consisting of observation of  n reaction coordinate
             number_bins: 1D array
                    Number of observation in each window
            cv_ranges: list of tuples
                    range for each dimension
        Attributes
        ------------
            hist: array nd
                histogram containing frequencies
            edges: list of arrays
                edges of the histogram
            sim_samples: array
                number of samples from each simulation
            sim_samples_used: array
                number of samples after considering range
        '''
        
        
        logger.info("Will make histogram now")
        num_observ = [ np.shape(i)[0] for i in observ ] # number of observations per simulations
        num_observ = np.array(num_observ,dtype=np.int)
        # Make 2D array joining all the files. ncols =  number of cv, nrows= total number of observations
        cv = np.vstack(observ)

        if cv_ranges is None:

            logger.info("Using automatic determination of ranges")
            cv_ranges = [ (i,j)   for i,j in zip(cv.min(axis=0),cv.max(axis=0))]
            logger.debug("ranges %s ", cv_ranges)
            
        else:
            logger.info("Using provided ranges %s",cv_ranges)
            
        # create mask for data outside the range in ANY dimension
        nsamples,ncv = cv.shape
        cv_mask = np.zeros(cv.shape[0], dtype=np.bool)
        idx =  np.ones(nsamples,dtype=np.bool)

        for i in range(ncv):
            
            idx= idx * ( cv[:,i] >= cv_ranges[i][0]) * (cv[:,i] <= cv_ranges[i][1])
        npoints = np.count_nonzero(idx)
        
        if  npoints == 0:
            logger.error("No samples considered. Check your ranges \n ")
            raise SystemExit("Exiting .. Sorry!")
        else:
            cv_mask[idx] = True
                                
        # determine number of samples considered from each simulation
        observCum = np.insert(np.cumsum(num_observ),0,0)
        num_observ1=[]
        for beg,end in zip(observCum[:-1],observCum[1:]):
            num_observ1.append( np.count_nonzero(cv_mask[beg:end]) )
        num_observ1 = np.array(num_observ1,dtype=np.int)
        logger.debug("Number of samples considered from each simulations %s",num_observ1)
        logger.debug("total number of simulations %s",num_observ.size)
        if number_bins == None:
            number_bins = [50 for i in range(ncv)]
        elif (len(number_bins) != ncv ):
            logger.error("number of bin sizes do not match the dimensions of histogram")
            raise SystemExit("Exiting")
        else:
            pass
        
        hist_ranges=[]
        for i in range(ncv):
            hist_ranges.append( np.linspace(cv_ranges[i][0],cv_ranges[i][1],number_bins[i]) )
            logger.debug("Bin Width for Dim %s : %s ", i, hist_ranges[i][1]-hist_ranges[i][0])

        
        
        histNd,edges = np.histogramdd(cv[cv_mask],bins=hist_ranges)
        
        checkhist = np.where(histNd < sys.float_info.epsilon)
        if checkhist[0].size > 0:
            logger.warn("%s", checkhist)
            logger.warn("Some bins have no data .. please adjust the bins")
            logger.warn("\n....I will not quit though ....\n")
            ##raise SystemExit("Quitting over this")
            
        self.hist = histNd
        self.histEdges = edges
        self.sim_samples = num_observ
        self.sim_samples_used = num_observ1
        self.cv_mask = cv_mask
        
        logger.info("%sd Histogram computed",ncv)
        

    def setParams(self,Ub,f = None, histogramfile = None, fefile = None,
                        ineff = None, chkpointfile = "chkp.fe.npz",
                                  setToZero=None):
        ''' set Ub and various other parameters
        
                Ub: array  shape [nbin_collvar1,nbin_collvar2... nbin_ncolvarN,Nsimulations]
                    baising potential evalulated on histogram
                
                f: Array
                    Initial estimate of weights
                    
                histogramfile: npz file
                
                fefile: npz file
                
                ineff: Array
                    statistical inefficiency of each simulation
                
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
            
        self.hist = self.hist.astype(np.int64)
        self.ndim = len(self.histEdges)
        
        if (f is None):
            if (fefile is not None) and os.path.isfile(fefile):
                logger.info("loading free energies from  %s", fefile)
                self.f = np.load(fefile)['arr_0']
            elif os.path.isfile(chkpointfile):
                logger.info("loading free energies from chkpointfile %s", chkpointfile)
                self.f = np.load(chkpointfile)['arr_0']
            else:
                logger.info("Initial Weights set to one")
                self.f = np.ones(self.sim_samples.size)
        else:        
            logger.info("Initializing Free energies to user supplied values")
            self.f = f
        
        if ineff is None:
            self.ineff = np.ones_like(self.f)
        else:
            self.ineff = ineff
            
            
        if setToZero is not None:
            self.windowZero = setToZero
        else:
            self.windowZero = 0   
           

#     def _compute_prob(self):
#         ''' compute probabilites once final weights are known
# 
#         '''
#         
#         prob = np.zeros_like(self.hist,dtype=np.float)
#         for i,j in np.ndenumerate(self.hist):
#             num = self.hist[i]
#             U = self.Ub[i]
#             logbf = self.f - self.beta * U + np.log(self.sim_samples_used)
#             denom = compute_logsum(logbf)
#             if num == 0:
#                 #prob[i] = np.NAN
#                 prob[i] = 0
#             else:    
#                 prob[i] = np.exp (np.log(num) - denom)
# 
#         self.prob = prob

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







#     
#    
#     def divideProbwithSine(self,dim):
#         ''' Divide by sine for  normalization
#         
#         Parameters:
#             Dim: Either 'x' or 'y'
#         '''
#         
#         if dim == 'x':
#             logger.info("Diving 1st dimension with sine")
#             for i in range(self.midp_xbins.size):
#                 self.prob[i,:] = self.prob[i,:]/np.sin(np.radians(self.midp_xbins[i]))
#         elif dim == 'y':             
#             logger.info("Diving 2nd dimension with sine")
#             for i in range(self.midp_ybins.size):
#                 self.prob[:,i] = self.prob[:,i]/np.sin(np.radians(self.midp_ybins[i]))
#         else:
#             logger.critical("%s not recognized")

        
    def write_probabilities(self,filename):
        ''' write npz files with probabilites
        '''
        # returns boolean array with zero set to false and rest true
    

        np.savez(filename, self.histEdges, self.prob)
        logger.info("probabilites written to %s" % filename)

    def probtopmf(self):
        ''' convert probabilities to pmf  
        '''
        
        p = self.prob.copy()
        p[np.isnan(p)] = 0.0
        p[np.where(p < sys.float_info.epsilon)] = np.NAN    
        #mask_notzero = ~(p < sys.float_info.epsilon)
        mask_notnan = ~ np.isnan(p)
        #p[~mask_notzero] = np.NAN
        p[mask_notnan] = -1.* np.log(p[mask_notnan]) / self.beta
        p[mask_notnan] = p[mask_notnan] - p[mask_notnan].min()
        self.pmf = p

    
    def write_pmf(self,filename):
        ''' write npz file with PMF profile. 
        '''
        
        np.savez(filename, self.histEdges, self.pmf)
        logger.info("pmf written to %s " % filename)
        
    def writeWeights(self,fefilename):
        ''' write free energies to file
        '''
        logger.info("Saving free energies in file %s",fefilename)
        np.savez(fefilename,self.f)

    
    def load_histogram(self,histfile):
        ''' loads histogram 
        '''
        logger.info("loading data from histogram file: %s", histfile)
        a = np.load(histfile)
            
        hist, histEdges, sim_samples, sim_samples_used, cv_mask  = a['arr_0'], a['arr_1'], a['arr_2'], a['arr_3'],a['arr_4']
        self.hist = hist
        self.histEdges = histEdges
        self.sim_samples = sim_samples
        self.sim_samples_used = sim_samples_used 
        self.cv_mask = cv_mask

    def load_pmf(self,filename):
        ''' load PMF profile from file. 
        
        '''
        
        a = np.load(filename)
        self.histEdges, self.pmf = a['arr_0'],a['arr_1']
        logger.info("pmf loaded from %s " % filename)
        
    def write_histogram(self,histfile):
        ''' saves histogram to histfile
        '''
        logger.info("Saving histogram in %s",histfile)
        
        np.savez(histfile, self.hist, self.histEdges, self.sim_samples, self.sim_samples_used, self.cv_mask )
        
        logger.info("histogram file %s saved", histfile)
        
    ################################################
    # old code, works with 2d, Need to fix for Nd
     
#     def calcFrameWeight(self,x,y):
#         ''' compute unnormalized weight of an individual frame
#         '''
#         
#         U_b = self.bias.compute_potential_2D( paramsX=(self.fcxy[:,0], self.xyopt[:,0]),
#                                               paramsY=(self.fcxy[:,1], self.xyopt[:,1]),
#                                               x=x, y=y )
#         logbf = self.beta * (self.F_k - U_b) + np.log(self.N_k)
#         denom = StatsUtils.compute_logsum(logbf)
#         try:
#             w = 1./np.exp(denom)
#         except ZeroDivisionError:
#             logger.critical("weight w is infinite! ... I will quit")
#             raise SystemExit
#         
#         return w
#         
#     def getWindow(self,rcoords):
#         ''' get state number from reaction coordinate values
#         
#         Parameters: 
#             rcoords: list with two elements
#         
#         Returns:
#             state: integer
#             
#         '''
#         
#         rcx,rcy = rcoords
#         
#         sindexx = np.digitize([rcx], self.x0) 
#         sindexy = np.digitize([rcy], self.y0)
#         
#         umbNo = sindexx * self.y0.size + sindexy
#         
#         logger.info("state with x0,y0: %s,%s will be set to zero",self.xyopt[umbNo,0],self.xyopt[umbNo,1])
#         
#         return umbNo
#   
 
        
        
    
