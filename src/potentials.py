#! /usr/bin/env python

import numpy as np
import itertools
from operator import mul


# Expose only this function:
def biasingPotentials(?):
    return _combinePotentials( _readFunctions(?)
                             , _readArgLists(?)
                             , _readEdgeLists(?)
                             )
            
# Transform functions, argLists, edgeLists from whatever form in which they are available.
def _readFunctions(functionNames):
    return [ lookupDict_potentials[fn] for fn in functionNames ]

def _readArgLists(vardict, collvars):
    return [ zip(vardict[i+'_fc'],vardict[i + '_x0']) for i in collvars ]

def _readEdgeLists(?):
    pass

# I am not yet sure in what forms the arguments of this function are available.
# At the moment, it takes a list of functions (fs[dim])
#                         a list of lists of argument tuples (argLists[dim][i]), and
#                         a list of lists of scalars (edgeLists[dim][i]),
#     where indexing is shown by e.g. [dim] (reaction coordinate dimension index), or
#                                     [i]   (unnamed index).
# :: [(Arg -> Double -> Double)] -> [[Arg]] -> [[Double]] -> [[Double]]
def _biasingPotentials(fs, argLists, edgeLists):
    simulationParameters = list(itertools.product(*argLists))
    simulationPotentials = map(lambda p: _combinePotentials(fs, p), simulationParameters)
    binCoordinates = list(itertools.product(*map(_binCenters, edgeLists)))
    return np.reshape((len(simulationPotentials), len(binCoordinates))
                      np.array([ f(p) for f in simulationPotentials for p in binCoordinates ])
                      )

# :: [(Arg -> Double -> Double)] -> [Arg] -> (Vector Double -> Double)
def _combinePotentials(fs, argTuples):
    def combinedPotential(xs):
        acc = 0.0
        for i in xrange(len(xs)):
            acc += fs[i](xs[i], *argTuples[i])
        return acc
    return combinedPotentials

# :: [Double] -> [Double]
def _binCenters(edgeList):
    return  0.5*(edgeList[1:] + edgeList[:-1])


# ----------------------------------------------------------------
# Potential functions.
# Note: potential functions must have call signatures of the type (Double, *args).

lookupDict_potentials = { "harmonic"  : potential_harmonic
                        , "cosine"    : potential_cosine
                        }

def potential_harmonic(x, x0, k):
    """Evaluate at 'x' a harmonic potential with center 'x0' and force constant 'k'". """
    return 0.5 * k * (x - xopt)*(x - xopt)

def potential_cosine(x, x0, k):
    """Evaluate at 'x' a cosine function with center 'x0' and force constant 'k'.
    Note: x has units of degrees."""
    return k * (1. - np.cos((x - x0)*(np.pi/180.0)))
    
