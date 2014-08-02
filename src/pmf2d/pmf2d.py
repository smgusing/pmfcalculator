#! /usr/bin/env python
import abc
import numpy as np 
import logging, sys, os
import time


np.seterr(all='raise',under='warn')
logger = logging.getLogger(__name__)


class Pmf2d(object):
    ''' Class to compute 2d PMF
    
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
    __metaclass__ = abc.ABCMeta

    
    def __init__(self, bias=None, x0=None, y0=None,fcx=None, fcy=None, maxiter=10e5, tol=10e-6, nbins=None,
                  temperature=None, g_k=None,chkdur=100):
        
        R = 8.3144621 / 1000.0  # Gas constant in kJ/mol/K
        self.bias = bias
        self.maxiter = maxiter  # Maximum iterations before bailing out
        self.tol = tol  # tolerance for convergence
        
        if temperature is None:
            self.beta = None
        else:
            self.beta = 1.0 / (R * temperature)  # inverse temperature of simulations (in 1/(kJ/mol))
            
        self.chkdur = chkdur
        

        self.nbins = nbins  # number of bins
        if nbins is None:
            self.prob = None
            self.pmf  = None
        else:
            nxmidp,nymidp=nbins[0]-1,nbins[1]-1
            self.prob = np.zeros([nxmidp,nymidp], dtype=np.float64)
            self.pmf = np.zeros([nxmidp,nymidp], dtype=np.float64)
            
        if (x0 != None) and (y0 != None):
            Kx = np.size(x0)
            Ky = np.size(y0)
            K = Kx * Ky
            self.K = K
            self.x0 = x0
            self.y0 = y0
            
            # vector with each component as a state with two force constants 
            self.fcxy = np.zeros((K, 2), dtype=np.float32)
            # same but with optimum spring positions
            self.xyopt = np.zeros((K, 2), dtype=np.float32)
            self.fcxy[:, 0] = fcx.repeat(Ky)
            self.fcxy[:, 1] = fcy.reshape(1, Ky).repeat(Kx, axis=0).flatten()
            self.xyopt[:, 0] = x0.repeat(Ky)
            self.xyopt[:, 1] = y0.reshape(1, Ky).repeat(Kx, axis=0).flatten()

            # I can allocate the memory now, but better to compute histogram and then
            # fill up U_bij
            self.U_bij = None
            self.F_k = np.zeros(K, dtype=np.float64)
        else:
            self.K = None
            
        
        if g_k is None:
            self.g_k = np.ones(K, dtype=np.float64)
        else:
            self.g_k = g_k
            
        self.hist,self.midp_xbins,self.midp_ybins,self.N_k=None,None,None,None
        self.xedges,self.yedges = None,None
        
        logger.info("Pmf2d successfully initialized")
        
    def make_2dhistogram(self, pos_kn=None, N_k=None, binrange=None):
        ''' Construct histogram
        
        Parameters: pos_kn: list of arrays
                        consisting of observation of reaction coordinate
                    N_k: array
                        Number of observation in each window
                    binrange: list of lists
                        binrange of histogram
                           
        '''
        
        
        K=self.K
        logger.info("Will make histogram now")
            
            
        nxbins, nybins = self.nbins       
        pos_xkn, pos_ykn = pos_kn[0], pos_kn[1]
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
            
            pos_xmin, pos_xmax = pos_xkn[sample_indices].min(), pos_xkn[sample_indices].max()
            pos_ymin, pos_ymax = pos_ykn[sample_indices].min(), pos_ykn[sample_indices].max()
            
            logger.debug("dataX min %s , data max %s", pos_xmin, pos_xmax)
            logger.debug("dataY min %s , data max %s", pos_ymin, pos_ymax)
            
        else:
            logger.info("Using provided bin ranges %s",binrange)
            pos_xmin, pos_xmax = binrange[0][0], binrange[0][1]
            pos_ymin, pos_ymax = binrange[1][0], binrange[1][1]
            
            # update N_k accordingly
            for k in range(0, K):
               xidx=( pos_xkn[k, 0:N_k[k]] >= pos_xmin) * (pos_xkn[k, 0:N_k[k]] <= pos_xmax)
               yidx=( pos_ykn[k, 0:N_k[k]] >= pos_ymin) * (pos_ykn[k, 0:N_k[k]] <= pos_ymax)
               idx=xidx*yidx
               npoints = np.count_nonzero(idx)
               
               if  npoints == 0:
                   logger.warn("No samples are considered from window %s",k)
                   #msg = ("Handling of such situation is not implemented",
                   #       "Either remove the corresponding files, or change bin range")
                   #logger.error("%s","\n".join(msg))
                   logger.warn("Dim1 min %s, max %s, n %s ",pos_xkn[k, 0:N_k[k] ].min(),pos_xkn[k, 0:N_k[k]].max(),N_k[k])
                   logger.warn("Dim2 %s,%s ",pos_ykn[k, 0:N_k[k]].min(),pos_ykn[k, 0:N_k[k]].max())
                   #raise SystemExit("Exiting .. Sorry!")
               
               else:
                   N_k[k] = npoints
                   logger.debug("New sample numbers %s %s",k,N_k[k]) 

        pos_xbins = np.linspace(pos_xmin, pos_xmax, nxbins)
        pos_ybins = np.linspace(pos_ymin, pos_ymax, nybins)
        
        
        midp_xbins = (pos_xbins[:-1] + pos_xbins[1:]) * 0.5
        midp_ybins = (pos_ybins[:-1] + pos_ybins[1:]) * 0.5
        delta_posx = pos_xbins[1] - pos_xbins[0]
        delta_posy = pos_ybins[1] - pos_ybins[0]
        logger.debug("Bin Width X %s  Y %s", delta_posx, delta_posy)
        logger.debug("Bins %s %s", pos_xbins, pos_ybins)

        hist,xedges,yedges = self._hist2d(pos_xkn[sample_indices], pos_ykn[sample_indices],
                             pos_xbins, pos_ybins)
        
        checkhist = np.where(hist < sys.float_info.epsilon)
        if checkhist[0].size > 0:
            logger.warn("%s", checkhist)
            logger.warn("Some bins have no data .. please adjust the bins")
            logger.warn("\n....I will not quit though ....\n")
            
            ##raise SystemExit("Quitting over this")
            
        self.hist = hist
        self.xedges = xedges
        self.yedges = yedges
        self.midp_xbins,self.midp_ybins = midp_xbins, midp_ybins
        self.N_k = N_k
        logger.info("2d Histogram computed")

            
    def _hist2d(self, x, y, Xin, Yin):
        ''' Create the histogram. The values outside the bin ranges are dropped
        '''
        bins = [Xin, Yin]
    

        H, xe, ye = np.histogram2d(x, y, bins=bins, normed=False)
        return H, xe, ye


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
    

        np.savez(filename, self.midp_xbins, self.midp_ybins, self.prob)
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
        
        np.savez(filename, self.midp_xbins, self.midp_ybins, self.pmf)
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
            
        hist, x, y, n_k, xedges, yedges = a['arr_0'], a['arr_1'], a['arr_2'], a['arr_3'],a['arr_4'], a['arr_5']
        
        return hist, x, y, n_k, xedges, yedges    

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
        np.savez(histfile, self.hist, self.midp_xbins, self.midp_ybins, self.N_k,self.xedges,self.yedges)
        logger.info("histogram file %s saved", histfile)
        
    ################################################
    
    def set_freeEnergies(self,F_k):
        '''
        '''
        if (self.F_k.shape == F_k.shape) and (self.F_k.dtype == F_k.dtype):
            self.F_k = F_k
        else:
            logger.critical("Cannot set free energies .. check shape and type")  
   
        
        
    
