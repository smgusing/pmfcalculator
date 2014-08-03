#! /usr/bin/env python
import abc
import numpy as np 
import logging,sys,os,time

np.seterr(all='raise',under='warn')

logger = logging.getLogger(__name__)


################################################33

class Pmf1d(object):
    
    __metaclass__ = abc.ABCMeta


    def __init__(self, bias=None, maxiter=10e5, tol=10e-6, nbins=None, temperature=None,
                 x0=None, fcx=None, g_k=None, chkdur = None ):
        ''' Class to compute 1d PMF
        
        Parameters: bias: object of class derived from Bias class
                        Must have calculate_potential_1d method
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

        R = 8.3144621 / 1000.0  # Gas constant in kJ/mol/K
        self.bias = bias # bias object containing methods to computing biasing potential
        self.maxiter = maxiter  # Maximum iterations before bailing out
        self.tol = tol  # tolerance for convergence
        self.chkdur = chkdur

        if temperature is None:
            self.beta = None
        else:
            self.beta = 1.0 / (R * temperature)  # inverse temperature of simulations (in 1/(kJ/mol))
        
        self.nbins = nbins  # number of bins
        if nbins is None:
            self.prob = None
            self.pmf  = None
        else:
            nxmidp=nbins-1
            self.prob = np.zeros(nxmidp, dtype=np.float64)
            self.pmf = np.zeros(nxmidp, dtype=np.float64)
            
        if  x0 != None :
            K = np.size(x0)
            self.K=K
            # vector with each component as a state with two force constants 
            self.fcx = np.array(fcx, dtype=np.float32)
            # same but with optimum spring positions
            self.xopt = np.array(x0, dtype=np.float32)
            # I can allocate the memory now, but better to compute histogram and then
            # fill up U_bij
            self.U_b = None
            self.F_k = np.zeros(K, dtype=np.float64)
        else:
            self.K = None
    
        if g_k is None:
            self.g_k = np.ones(K, dtype=np.float64)
        else:
            self.g_k = g_k
            
        self.hist,self.midp_bins,self.N_k=None,None,None
        self.edges = None
 
        logger.info("Pmf1d successfully initialized")

    def make_histogram(self, pos_kn = None, N_k = None, binrange = None):
        ''' Construct histogram
        
        Parameters
        ------------
            pos_kn: list of arrays
                 n observations of k simulations
                 
            N_k: array
                Number of observation in each window
                
            binrange: list of lists
                binrange of histogram
        
        Returns
        ------------
        None
        
        attributes
        ------------
            histogram: hist[k,nbins] number of enteries in bin i from simulation k 
         
                           
        '''
        
        K=self.K
        logger.info("Will make histogram now")
            
        nbins = self.nbins       
        N_max = N_k.max()
        # Create a list of indices of all configurations in kn-indexing.
        # As Kx and Ky are same either can be used here. 
        # Also N_k is symmetric (number of samples should be same for x and y)
        # 
        sample_indices = np.zeros([K, N_max], dtype=np.bool)
        for k in range(0, K):
            sample_indices[k, 0:N_k[k]] = True
        
        if binrange is None:
            logger.info("Using automatic determination of bin ranges")
            pos_min, pos_max = pos_kn[sample_indices].min(), pos_kn[sample_indices].max()
            logger.debug("data min %s , data max %s", pos_min, pos_max)
        else:
            logger.info("Using provided bin ranges %s",binrange)
            pos_min, pos_max = binrange[0], binrange[1]
            
            # Update N_k according to range
            for k in range(0, K):
                # posmin  and posmax are included 
                idx=( pos_kn[k, 0:N_k[k]] >= pos_min) * (pos_kn[k, 0:N_k[k]] <= pos_max)
                npoints = np.count_nonzero(idx)
                if  npoints == 0:
                    logger.warn("No samples are considered from window %s",k)
                    #msg = ("Handling of such situation is not implemented",
                    #       "Either remove the corresponding files, or change bin range")
                    #logger.warn("%s","\n".join(msg))
                    logger.warn("Dim1 min %s, max %s, n %s ",pos_kn[k, 0:N_k[k] ].min(),pos_kn[k, 0:N_k[k]].max(),N_k[k])
                    #raise SystemExit("Exiting .. Sorry!")
                else:
                    N_k[k] = npoints
                    logger.debug("New sample numbers %s %s",k,N_k[k]) 
                
        
        
        #pos_max += (pos_max - pos_min) / (nbins)
        # pos_min-=(pos_max-pos_min)/nbins
        logger.debug("Bin min %s , Bin max %s", pos_min, pos_max)
        pos_bins = np.linspace(pos_min, pos_max, nbins)
        # The last bin is half open. So pos_max is in the last bin 
        hist,edges = self.hist1d(pos_kn[sample_indices], pos_bins)
        
        midp_bins = (pos_bins[:-1] + pos_bins[1:]) * 0.5
        delta_pos = pos_bins[1] - pos_bins[0]
        # logger.debug("E min %s , E max %s",U_min,U_max)
        logger.debug("Bin Width  %s  ", delta_pos)

        
        checkhist = np.where(hist < sys.float_info.epsilon)
        if checkhist[0].size > 0:
            logger.warn("%s", edges[checkhist])
            logger.warn("Some bins have no data .. please adjust the bins")
            logger.warn("\n....I will not quit though ....\n")
            #raise SystemExit()
            
            
        self.hist,self.midp_bins,self.N_k=hist, midp_bins, N_k
        self.edges = edges
        self.pos_bins = pos_bins
        logger.info("1d Histogram computed")

     
    @abc.abstractmethod
    def estimateFreeEnergy(self, **args):
        '''Compute free energies using the given project and input arguments'''
        
        return
            
    def write_probabilities(self,filename):
        ''' write npz files with probabilites
        '''
        # returns boolean array with zero set to false and rest true
        np.savez(filename, self.midp_bins, self.prob)
        logger.info("probabilites written to %s" % filename)
       
    def probtopmf(self):
        ''' convert probability to pmf 
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
        ''' write pmf profile to file
        '''
        np.savetxt(filename, np.array([self.midp_bins,self.pmf]).transpose())
        logger.info("pmf written to %s " % filename)
        
    def load_pmf(self,filename):
        ''' load PMF profile from file. 
        
        '''
        
        a = np.loadtxt(filename)
        self.midp_bins, self.pmf = a[:,0],a[:,1]
        logger.info("pmf loaded from %s " % filename)

        
    def write_FreeEnergies(self,fefilename):
        '''
        '''
        logger.info("Saving free energies in file %s",fefilename)
        np.savez(fefilename,self.F_k)


    def hist1d(self, x,  bins):
        ''' Create the histogram. The values outside the bin ranges are dropped
        '''

        H, xe = np.histogram(x, bins=bins, normed=False)
        return H,xe
    
        
    def load_histogram(self,histfile):
        
        logger.info("loading data from histogram file: %s", histfile)
        a = np.load(histfile)
        hist, midp, edges, n_k = a['arr_0'], a['arr_1'], a['arr_2'], a['arr_3']
        return hist, midp, edges, n_k    

    def write_histogram(self,histfile):
        ''' saves histogram to histfile
        '''
        logger.info("Saving histogram in %s",histfile)
        np.savez(histfile, self.hist, self.midp_bins, 
                 self.pos_bins, self.N_k)
        logger.info("histogram file %s saved", histfile)
    

    
    ################################################
    
    def set_freeEnergies(self,F_k):
        '''
        '''
        if (self.F_k.shape == F_k.shape) and (self.F_k.dtype == F_k.dtype):
            self.F_k = F_k
        else:
            logger.critical("Cannot set free energies .. check shape and type")  

