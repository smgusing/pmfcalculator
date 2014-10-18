#! /usr/bin/env python
import abc
import numpy as np
import logging, sys, os



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
            self.temperature = temperature
        logger.debug("PmfNd successfully initialized")

    @abc.abstractmethod
    def estimateWeights(self,**args):
        '''Abstract method for estimating weights'''
        return

    def mask_outofrange(self, cv, cv_ranges=None ):
        ''' Generate mask for cv where cv is out of cv_range
        
        Parameters
        ------------
            cv : array 
                array with each column as component of vector
            cv_ranges: list of tuples
                each tuple contains min,max for the dimension
        Returns
        ------------
            cv_mask: array dtype(bool)
                mask for rows where all components that are 
                within range are set to true
            cv_ranges: list of tuples
                each tuple contains min,max for the dimension
       '''
        idx =  np.ones(cv.shape[0],dtype=np.bool)
        cv_mask = np.zeros(cv.shape[0], dtype=np.bool)

        if cv_ranges is None:
            logger.info("Using automatic determination of ranges")
            cv_ranges = [ (i, j)   for i, j in zip(cv.min(axis=0), cv.max(axis=0))]
            logger.debug("ranges %s ", cv_ranges)
        
        else:
            logger.debug("Using provided ranges %s", cv_ranges)
            # create mask for data outside the range in ANY dimension
            for i in range(cv.shape[1]):
                idx = idx * (cv[:, i] >= cv_ranges[i][0]) * (cv[:, i] <= cv_ranges[i][1])
           
            npoints = np.count_nonzero(idx)
            if  npoints == 0:
                logger.error("No samples considered. Check your ranges \n ")
                raise SystemExit("Exiting .. Sorry!")
            else:
                pass
        
        cv_mask[idx] = True
        
        return cv_mask
       

    def make_ndhistogram(self, observ, cv_ranges=None, number_bins=None, ineff = None):
        ''' Construct histogram

        Parameters
        ------------
             observ: list of arrays
                        consisting of observation of  n reaction coordinate
             number_bins: 1D array
                    Number of observation in each window
            cv_ranges: list of tuples
                    range for each dimension
            ineff: statistical inefficiency in each observation
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
        nsamples,ncv = cv.shape
        
        if number_bins == None:
            number_bins = [50 for i in range(ncv)]
        elif (len(number_bins) != ncv ):
            logger.error("number of bin sizes do not match the dimensions of histogram")
            raise SystemExit("Exiting")
        else:
            pass
        
        cv_mask = self.mask_outofrange(cv,cv_ranges)

        # determine number of samples considered from each simulation
        observCum = np.insert(np.cumsum(num_observ),0,0)
        num_observ1=[]
        simRanges=[]
        for beg,end in zip(observCum[:-1],observCum[1:]):
            num_observ1.append( np.count_nonzero(cv_mask[beg:end]) )
            simRanges.append((beg,end))


        num_observ1 = np.array(num_observ1,dtype=np.int)
        logger.debug("Number of samples considered from each simulations %s",num_observ1)
        logger.debug("total number of simulations %s",num_observ.size)



        hist_ranges=[]
        for i in range(ncv):
            hist_ranges.append( np.linspace(cv_ranges[i][0],cv_ranges[i][1],number_bins[i]) )
            logger.debug("Bin Width for Dim %s : %s ", i, hist_ranges[i][1]-hist_ranges[i][0])


        ## check whether ineff is provided, and scale N and hist accordingly
        histNd = np.zeros([i-1 for i in number_bins],dtype=np.float)
        edges = None
        if (ineff != None ) and (np.any(ineff != 1)):
            logger.info("Using ineff ")
            num_observ1 = num_observ1/ineff
            for i,j in enumerate(simRanges):
                beg, end = j
                simSamples = observ[i][cv_mask[beg:end],...]
                h,edges = np.histogramdd(simSamples,bins=hist_ranges)
                histNd = histNd + h/ineff[i]
        else:
            logger.info("Will not use ineff ")
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
        self.chkpointfile = chkpointfile
        self.histFN=histogramfile

        if os.path.isfile(histogramfile):
            self.load_histogram(histogramfile)

        elif self.hist is None:
            logger.debug("histogram not initialized")
            raise SystemExit("No histogram file and no histogram values ... Exiting")


        if self.hist.flags["C_CONTIGUOUS"] == False:
            self.hist = np.ascontiguousarray(self.hist, self.hist.dtype)

        ###self.hist = self.hist.astype(np.int64)
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



    def divideProbwithSine(self,dim):
        ''' Divide by sine for  normalization

        Parameters:
            Dim: Either 'x' or 'y'
        '''


        logger.info("Diving %s dimension with sine",dim)
        midpts = (self.histEdges[dim][1:] + self.histEdges[dim][:-1]) * 0.5
        p = np.rollaxis(self.prob,dim,-1)
        for i in range(midpts.size):
                p[...,i] = p[...,i]/np.sin(np.radians(midpts[i]))


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
        self.hist, self.histEdges, self.sim_samples, self.sim_samples_used, self.cv_mask = a['arr_0'], a['arr_1'], a['arr_2'], a['arr_3'],a['arr_4']

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



