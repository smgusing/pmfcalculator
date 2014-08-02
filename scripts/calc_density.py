#! /usr/bin/env python

import numpy as np
import argparse 
import logging, sys, os
import pmfcalculator
from pmfcalculator.grid3d import Grid3D

logger = logging.getLogger("pmfcalculator")


des = '''calc_density.py: Calculate 3d density.  
    Note: The script does not perform error checking on command line arguments'''

parser = argparse.ArgumentParser(description=des    
    , formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("-weightfile", dest='weightInFn',
        help='Weight file',
        default="weight.txt")

parser.add_argument("-xtcfile", dest='xtcInFn',
        help='Input xtc file',
        default="fitted.xtc")

parser.add_argument("-ndxfile", dest='ndxInFn',
        help='Input index file',
        default="index.ndx")

parser.add_argument("-tprfile", dest='tprInFn',
        help='Input tpr file',
        default="sim.tpr")

parser.add_argument("-nbins",dest='nbins',
        help='number of bins in x  and y. Input two numbers seperated by space ',
        default=[100, 100, 100],nargs=3, type=int)

parser.add_argument("-gridwidth",dest='gridwidth',
        help=' Input 3 numbers seperated by spaces',
        default=[2.0, 2.0, 2.0],nargs=3, type=float)

parser.add_argument("-shiftcenter",dest='shiftCenter',
        help=' 3 numbers seperated by spaces',
        default=[2.0, 2.0, 2.0],nargs=3, type=float)


parser.add_argument("--setcenterxy",action="store_true", dest="bCenterxy", default=False,
              help="set grid center from command line")


parser.add_argument("-gridcenterxy",dest='gridCenter',
        help=' Input 3 numbers seperated by spaces',
        default=[0, 0],nargs=2, type=float)

parser.add_argument("-out", dest='histOutFn',
        help='Output histogram',
        default="density.npz")

parser.add_argument("-l", dest='loglevel', help='level of logging (info,warn,debug)',
        default="debug")




def main(args):
    
    nbins = np.array(args.nbins,dtype=np.int32) + 1
    gridwidth = np.array(args.gridwidth,dtype=np.float32)
    # Read the groups from which grid center will be estimated
    gridCenter,grp1,grp2,grp3 = Grid3D.calcGridCenterGmx(xtcInFn=args.xtcInFn,
                                                  tprInFn=args.tprInFn,
                                                  ndxInFn=args.ndxInFn)
    if args.bCenterxy == True:
        logger.info("setting gridxy from command line arguments")
        gridCenter = tuple([args.gridCenter[0],args.gridCenter[1],gridCenter[2]])
        
    gridCentShifted = tuple( [gridCenter[i] + args.shiftCenter[i] for i in range(3)] )
    
    
    frweights = np.loadtxt(args.weightInFn,dtype=np.float)
    
    grid = Grid3D(gridcenter=gridCentShifted, nbins=nbins,
                           gridwidth=gridwidth)
    
    grid.calcDensityGmx(xtcInFn=args.xtcInFn,atomindices=grp3.indices,frameWeight=frweights)
    grid.writeHistogram(args.histOutFn)
    
    
        
    logger.info("Finished successfully")
            
if __name__ == '__main__':
    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    logger.setLevel(numeric_level)
    print "#################################"
    print "## Program version",pmfcalculator.__version__
    print "## Invoked with Following Arguments " 
    for key, value in vars(args).items():
        print "# %s = %s"%(key,value)
    print "#################################"
   
    main(args)
