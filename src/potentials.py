#! /usr/bin/env python

import numpy as np
import itertools
import sys


def harmonic(x,forceConst,xopt):
    ''' Harmonic potential
    
    Parameters
    ------------
    forceConst : array
    xopt:    array
    x:    array
    
    Returns
    ----------
    U : array
    '''
    
    U = 0.5 * forceConst * np.square(x - xopt)
    
    return U
    

def cosine(x,forceConst,xopt):
    ''' Cosine potential
    
    Parameters
    ------------
    forceConst : array
    xopt:    array
    x:    array
    
    Returns
    ----------
    U : array
    '''
    
    U = forceConst * (1.0 - np.cos(np.radians(x - xopt)))
    
    return U


def biasPotential(biasType,collvars,vardict,histEdges):
    ''' calculate biasing potential on every bin
    
     Parameters
    -------------
        biasType: list
            potential functions to use [ "harmonic" or "cosine"]
        collvars: list
            keys of collvars
        vardict: dict
            collvars as keys and parameter list as values
        histEdges: 
            list of edges of histogram
    Returns
    ------------
        Ub: array 
            shape [nbin_collvar1,nbin_collvar2... nbin_ncolvarN,Nsimulations]
    
    Notes
    -------
        
    '''
    
    biasFuncts=[]
    for i in biasType:
        
        if i == "harmonic":
            biasFuncts.append(harmonic)
        elif i == "cosine":
            biasFuncts.append(cosine)
        else:
            errstr = "%s Not implimented"%i
            raise SystemExit(errstr)
    
    histMidPoints = [(edge[1:] + edge[:-1])*0.5 for edge in histEdges]
    
    varZipped = [zip(vardict[i+'_fc'],vardict[i + '_x0']) for i in collvars]
        
    # for every point evaluate function with these paramaters    
    fargs = list(itertools.product(*varZipped))
    
    #Dimensions of Ub is dimension of histogram + 1
    # size of each dimenstion is number of gridpoints in that dimension
    # the last dimension is total number of simulations
        
    UbDim = [len(midpoints) for midpoints in histMidPoints ]
    # Total number of simulations
    nsim = 1
    
    for j in collvars:
        nsim = nsim * len(vardict[j])
    
    
    UbDim.append(nsim)
    Ub = np.zeros(UbDim,dtype = np.float64)
    
    for i,j in np.ndenumerate(Ub):
        Ub[i] = _eval_potential(i,biasFuncts,fargs,histMidPoints)
        #print i, Ub[i]
        #sys.exit()
    
    return Ub
    
            
        
        
def _eval_potential(ind,biasFuncts,fargs,histMidpoints):
    ''' evaluate potential for a given point with given functions and arguments
    
    parameters
    -------------
    ind: tuple
        index of Ub array
    biasFuncts:
        list of functions
    fargs:
        function arguments for each function
    histmidpoints: list of arrays
    
    returns
    ---------
    U: float
        potential energy
        
    '''
    
    ndim = len(ind) - 1
    fargInd = ind[-1]
    U = 0.0
    for i in range(ndim):
        point = histMidpoints[i][ind[i]]
        U = U + biasFuncts[i](point,*fargs[fargInd][i])
        #print point,fargs[fargInd][i], biasFuncts[i](point,*fargs[fargInd][i])
    return U
        
    
    
    
    