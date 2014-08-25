#! /usr/bin/env python

import numpy as np
import itertools
import sys


# ----------------------------------------------------------------
# Interface

def biasPotential(biasTypes, collvars, vardict, histogramEdgeLists):
    return _biasPotential( _readFunctions(biasTypes)
                         , _readArgLists(collvars, vardict)
                         , _readEdgeLists(histogramEdgeLists)
                         )
            
def _readFunctions(functionNames):
    return [ lookupDict_potentials[fn] for fn in functionNames ]

def _readArgLists(collvars, vardict):
    return [ zip(vardict[i+'_fc'],vardict[i + '_x0']) for i in collvars ]

def _readEdgeLists(edgeLists):
    # identity function
    return edgeLists 

def _testBiasPotential(biasTypes, collvars, vardict, histogramEdgeLists):
    new = _biasPotential( _readFunctions(biasTypes)
                         , _readArgLists(collvars, vardict)
                         , _readEdgeLists(histogramEdgeLists)
                         )

    old = biasPotentialOld(biasTypes, collvars, vardict, histogramEdgeLists)

    tol = 1e-9
    err = 0
    for i,j in np.ndenumerate(new):
        if not (old[i]*(1-tol) <= new[i] <= old[i]*(1+tol)):
            err = 1
            print "error:", i, new[i], old[i]
    if err == 0:
        print "No difference exceeding a relative error of {}.".format(tol)
    raw_input("...")

# ----------------------------------------------------------------
# Implementation

# :: [(Arg -> Double -> Double)] -> [[Arg]] -> [[Double]] -> [[Double]]
def _biasPotential(fs, argLists, edgeLists):
    """
    Argument types:
        fs:         a list of functions (fs[dim])
        argLists:   a list of lists of argument tuples (argLists[dim][i]), and
        edgeLists:  a list of lists of scalars (edgeLists[dim][i]),
        where indexing is shown by e.g. [dim] (reaction coordinate dimension index), or
                                        [i]   (unnamed index).
    """
    simulationParameters = list(itertools.product(*argLists))
    simulationPotentials = map(lambda p: _combinePotentials(fs, p), simulationParameters)
    binCenters = map(_binCenters, edgeLists)
    binDimensions = map(len, binCenters)
    binCoordinates = list(itertools.product(*binCenters))
    return np.reshape( np.array([ f(p) for p in binCoordinates for f in simulationPotentials ])
                     , binDimensions + [len(simulationPotentials)]
                     )

# :: [(Arg -> Double -> Double)] -> [Arg] -> (Vector Double -> Double)
def _combinePotentials(fs, argTuples):
    def combinedPotential(xs):
        acc = 0.0
        for i in xrange(len(xs)):
            acc += fs[i](xs[i], *argTuples[i])
        return acc
    return combinedPotential

# :: [Double] -> [Double]
def _binCenters(edgeList):
    return  0.5*(edgeList[1:] + edgeList[:-1])


# ----------------------------------------------------------------
# Potential functions.
# Note: potential functions must have call signatures of the type (Double, *args).

def potential_harmonic(x, k, x0):
    """Evaluate at 'x' a harmonic potential with center 'x0' and force constant 'k'". """
    return 0.5 * k * (x - x0)*(x - x0)

def potential_cosine(x, k, x0):
    """Evaluate at 'x' a cosine function with center 'x0' and force constant 'k'.
    Note: x has units of degrees."""
    return k * (1. - np.cos(np.radians(x - x0)))
    
lookupDict_potentials = { "harmonic"  : potential_harmonic
                        , "cosine"    : potential_cosine
                        }

# ----------------------------------------------------------------
# Old version
# (Keeping it in case it is needed, may remove later).

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


def biasPotentialOld(biasType,collvars,vardict,histEdges):
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
