#! /usr/bin/env python
import abc
import numpy as np 
import logging, sys, os
import time


np.seterr(all='raise',under='warn')
logger = logging.getLogger(__name__)


class PmfNd(object):
    ''' Class to compute N dimensional PMF
    
     Parameters
    --------------
        bias: array of functions to compute biasing potentials
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


    @abc.abstractmethod
    def estimateFreeEnergy(self, **args ) :
        ''' Abstract method for estimating free energy
        '''
        return
    
   
    def divideProbwithSine(self,dim):
        ''' Divide by sine for  normalization
        
        Parameters:
            Dim: Either 'x' or 'y'
        '''
        
        if dim == 'x':
            logger.info("Diving 1st dimension with sine")
            for i in range(self.midp_xbins.size):
                self.prob[i,:] = self.prob[i,:]/np.sin(np.radians(self.midp_xbins[i]))
        elif dim == 'y':             
            logger.info("Diving 2nd dimension with sine")
            for i in range(self.midp_ybins.size):
                self.prob[:,i] = self.prob[:,i]/np.sin(np.radians(self.midp_ybins[i]))
        else:
            logger.critical("%s not recognized")

        
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
        
    def write_FreeEnergies(self,fefilename):
        ''' write free energies to file
        '''
        logger.info("Saving free energies in file %s",fefilename)
        np.savez(fefilename,self.F_k)

    
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
        self.midp_xbins, self.midp_ybins, self.pmf = a['arr_0'],a['arr_1'],a['arr_2']
        logger.info("pmf loaded from %s " % filename)
        
    def write_histogram(self,histfile):
        ''' saves histogram to histfile
        '''
        logger.info("Saving histogram in %s",histfile)
        
        np.savez(histfile, self.hist, self.histEdges, self.sim_samples, self.sim_samples_used, self.cv_mask )
        
        logger.info("histogram file %s saved", histfile)
        
    ################################################
    
    def set_freeEnergies(self,F_k):
        '''
        '''
        if (self.F_k.shape == F_k.shape) and (self.F_k.dtype == F_k.dtype):
            self.F_k = F_k
        else:
            logger.critical("Cannot set free energies .. check shape and type")  
   
        
        
    
