#! /usr/bin/env python

import numpy as np
import itertools


# ----------------------------------------------------------------
# Interface

def biasPotential(functionNames, collvars, vardict, histogramEdgeLists):
    return _biasPotential( _readFunctions(functionNames)
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

def potential_harmonic(x, x0, k):
    """Evaluate at 'x' a harmonic potential with center 'x0' and force constant 'k'". """
    return 0.5 * k * (x - x0)*(x - x0)

def potential_cosine(x, x0, k):
    """Evaluate at 'x' a cosine function with center 'x0' and force constant 'k'.
    Note: x has units of degrees."""
    return k * (1. - np.cos((x - x0)*(np.pi/180.0)))
    
lookupDict_potentials = { "harmonic"  : potential_harmonic
                        , "cosine"    : potential_cosine
                        }

