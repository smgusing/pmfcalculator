import numpy as np
import numpy.random as nprand
#import pymbar.timeseries as timeseries # from pymbar
import timeseries # from pymbar
import logging,sys,os

logger = logging.getLogger(__name__)


def compute_stat_inefficiency(observ):
    ''' computes iacts
    '''
    logger.info("computing IACTS")
    ineff = []

    for sim in observ:
        simIneff = []
        for j in range(sim.shape[1]):
            simIneff.append( timeseries.statisticalInefficiency( sim[:,j] ) )
        
        ineff.append(simIneff)
    
    ineff = np.array(ineff,dtype=np.float)
    
    logger.info("IACTS computed")
    return ineff

def subsample(observ,maxIneff):
    ''' subsample according to largest inefff
    
    Parameters
    -------------
        observ: list of arrays
        ineff: array with ineff for each column of observ
        
    Return
    -----------
        newObserv: list of arrays subsampled according to ineff
    '''
    logger.info("Subsampling using given ICATS")
    newObserv = [] 
    for i,sim in enumerate(observ):
        
        indices = timeseries.subsampleCorrelatedData(sim[:,0], g = maxIneff[i])
        newsim = sim[indices,...]
        newObserv.append(newsim)
    logger.debug("Original %s \nNew %s",[i.shape[0] for i in observ],[i.shape[0]  for i in newObserv])
    logger.info("Subsampled using given ICATS")
    return newObserv

def bootstrap(observ):
    ''' boot strap given list of arrays
    
    Parameters
    ------------
        observ : list of arrays
            observations from simulations
    
    Return
    --------
        bootobserv : list of arrays
    
    '''
    logger.info("Generating subsamples by random picking")
    bootObserv = []
    for sim in observ:
        ridx = nprand.choice(sim.shape[0],sim.shape[0],replace = True)
        newsim = sim[ridx,...]
        bootObserv.append(newsim)

    logger.info("Random subsample generated")    
    return bootObserv



def compute_stat_inefficiency2D(pos_xkn,pos_ykn,N_k):
    ''' computes iacts
    '''
    logger.info("computing IACTS")
    K = pos_xkn.shape[0]
    ineff = np.zeros(K,dtype=np.float)
    ineffx = np.zeros(K,dtype=np.float)
    ineffy = np.zeros(K,dtype=np.float)
    for i in range(K):
        ineffx[i] = timeseries.statisticalInefficiency( pos_xkn[i,0:N_k[i]] ) 
        ineffy[i] = timeseries.statisticalInefficiency( pos_ykn[i,0:N_k[i]] )
        
        if ineffx[i] > ineffy[i]:
            ineff[i] = ineffx[i]
        else:
            ineff[i] = ineffy[i]
        #logger.debug("IACT X and Y %s %s %s",iactx,iacty,i )
    logger.info("IACTS computed")
    return ineff,ineffx,ineffy
        
def compute_stat_inefficiency1D(pos_kn,N_k):
    ''' computes iacts
    
    '''
    
    logger.info("computing IACTS")
    K = pos_kn.shape[0]
    ineff = np.zeros(K,dtype=np.float)
    for i in range(K):
        ineff[i] = timeseries.statisticalInefficiency( pos_kn[i,0:N_k[i]] ) 
        logger.debug("%d %f\n",i,ineff[i])
    logger.info("IACTS computed")
    return ineff
 
             
def subsample2D(pos_xkn,pos_ykn,N_k,ineff):
    ''' Modifies pos_xkn,pos_ykn,N_k inplace
    '''
    logger.info("Subsampling using given ICATS")
    K = pos_xkn.shape[0]
    newNk = np.zeros_like(N_k)
    for i in range(K):
        indices = timeseries.subsampleCorrelatedData(pos_xkn[i,0:N_k[i]], g = ineff[i])
        newN = len(indices)
        pos_xkn[i,0:newN] = pos_xkn[i,indices]
        pos_ykn[i,0:newN] = pos_ykn[i,indices]
        newNk[i] = newN
#         if newN < 10:
#             logger.warn("Very few independent samples %s",newN)
 
    logger.debug("Original %s New %s",N_k,newNk)
    logger.info("Subsampled using given ICATS")
    return pos_xkn,pos_ykn,newNk

def subsample1D(pos_kn,N_k,ineff):
    ''' Modifies pos_xkn,pos_ykn,N_k inplace
    
    '''
    
    logger.info("Subsampling using given ICATS")
    K = pos_kn.shape[0]
    for i in range(K):
        indices = timeseries.subsampleCorrelatedData(pos_kn[i,0:N_k[i]], g = ineff[i])
        newN = len(indices)
        pos_kn[i,0:newN] = pos_kn[i,indices]
        logger.debug("Original %s New %s",N_k[i],newN)
        N_k[i] = newN
        if newN < 10:
            logger.warn("Very few independant samples %s",newN)
 
    logger.info("Subsampled using given ICATS")
    return pos_kn,N_k

    
def generate_bootstrapsample2D(pos_xkn,pos_ykn,N_k):
    '''
    '''
    logger.info("Generating subsamples by random picking")
    K = pos_xkn.shape[0]
    new_pos_xkn=np.empty_like(pos_xkn)
    new_pos_ykn=np.empty_like(pos_xkn)
    for i in range(K):
        new_pos_xkn[i,0:N_k[i]] = nprand.choice(pos_xkn[i,0:N_k[i]],
                                                size = N_k[i], replace = True)
        new_pos_ykn[i,0:N_k[i]] = nprand.choice(pos_ykn[i,0:N_k[i]],
                                                size = N_k[i], replace = True)
    
    logger.info("Random subsample generated")    
    return new_pos_xkn,new_pos_ykn

def generate_bootstrapsample1D(pos_kn,N_k):
    ''' Generate Subsamples
    
    '''
    
    logger.info("Generating subsamples by random picking")
    K = pos_kn.shape[0]
    new_pos_kn=np.empty_like(pos_kn)
    for i in range(K):
        new_pos_kn[i,0:N_k[i]] = nprand.choice(pos_kn[i,0:N_k[i]],
                                                size = N_k[i], replace = True)
    
    logger.info("Random subsample generated")    
    return new_pos_kn



def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    Copied from python cookbook
    from: http://scipy.org/Cookbook/SignalSmooth

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string   
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[2*x[0]-x[window_len:1:-1],x,2*x[-1]-x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='same')
    return y[window_len-1:-window_len+1]


def smooth2D(pos_xkn,pos_ykn,ineffx,ineffy):
    ''' smooth data according to statistical inefficiencies
    
    '''
    logger.info("Smoothing data")
    K = pos_xkn.shape[0]
    for i in range(K):
        logger.info("%s %s",np.ceil(ineffx[i]),np.ceil(ineffy[i]))
        newx = smooth(pos_xkn[i,:],window_len = np.ceil(ineffx[i]),
                       window = 'flat')
        pos_xkn[i,:] = newx
        
        newy = smooth(pos_ykn[i,:],window_len = np.ceil(ineffy[i]),
                      window = 'flat')
        pos_ykn[i,:] = newy

    logger.info("Smoothing Done")
    
    return pos_xkn,pos_ykn

def smooth1D(pos_kn, ineff):
    ''' smooth data according to statistical inefficiencies
    
    '''
    logger.info("Smoothing data")
    K = pos_kn.shape[0]
    for i in range(K):
        logger.info("sim %s window size %s",i, np.ceil(ineff[i]))
        new_pos = smooth(pos_kn[i,:],window_len = np.ceil(ineff[i]),
                       window = 'flat')
        pos_kn[i,:] = new_pos

    logger.info("Smoothing Done")
    
    return pos_kn

def load_ineff2D(infile):
    ''' return inefficiencies from file 

    '''

    logger.info("loading data from file: %s", infile)
    a = np.load(infile)
    return  a['arr_0'], a['arr_1'], a['arr_2']

def compute_logsum(numpyArray):
    ''' return log of sum of exponent of array
    '''
    
    ArrayMax = numpyArray.max()
    return np.log(np.exp(numpyArray - ArrayMax ).sum() ) + ArrayMax
