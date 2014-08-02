#! /usr/bin/env python
import numpy as np
import argparse 
import logging, sys, os
import pmfcalculator 
from pmfcalculator.frameselector import Frameselector2d
from pmfcalculator.pmf2d import Wham2d
import pmfcalculator.StatsUtils as utils
import pickle as pk
 
########################################
logger = logging.getLogger("pmfcalculator")


des = '''dump_frames.py: Dump frames corrosponding to selected bins  
    Note: The script does not perform error checking on command line arguments'''

parser = argparse.ArgumentParser(description=des    
    , formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-projfile", dest='prjfile', default='proj.yaml', help='Project file')


parser.add_argument("-ener", dest='ener', help='Energy (kJ/mol) to add to minimum',
        default=0.0, type=float)

parser.add_argument("-fefile",dest='fefile', 
        help='free energy file name. Will be used to store free energies',
        default="fe2d.npz")

parser.add_argument("-temperature",dest='temperature', help='Temerature in kelvin',
        default=300.0,type=float)


parser.add_argument("-pmffile", dest='pmffile',
        help='pmf file name. Will be used to store pmf',
        default="pmf2d.npz")


parser.add_argument("-range",dest='binrange',
        help='xrange(start,end) and yrange(start,end) Input 4 numbers seperated by spaces',
        default=[0.0, 0.0, 0.0, 0.0],nargs=4, type=float)

parser.add_argument("-l", dest='loglevel', help='level of logging (info,warn,debug)',
        default="debug")

parser.add_argument("-xtcfile", dest='xtcout', default='sel.xtc', help='output xtc file')



def main(args):
    
    # if all elements in binrange are zero then set args.binrange to None
    if all(x == 0 for x in args.binrange):
        args.binrange = None
    else:
        binrange = ( (args.binrange[0],args.binrange[1] ), 
                     (args.binrange[2],args.binrange[3] ) )
        args.binrange = binrange
    
    ## Read the project file containing information about the pmf project
    prj = pmfcalculator.Reader2d(infile=args.prjfile)
    x0=np.array(prj.x0,dtype=np.float32)
    y0=np.array(prj.y0,dtype=np.float32)
    kx=np.array(prj.kx,dtype=np.float32)
    ky=np.array(prj.ky,dtype=np.float32)
    
    
    bias = pmfcalculator.Harmonic_cosine()
    pmfobj = Wham2d(bias,temperature=args.temperature, x0=x0, y0=y0,fcx=kx,fcy=ky)
    F_k = np.load(args.fefile)['arr_0']
    
    pmfobj.F_k = F_k
    pmfobj.load_pmf(args.pmffile)

    frsel = Frameselector2d(prjobj=prj)
    chkpFn = 'sel.pkl'
    
    if not os.path.isfile(chkpFn):
        logger.info("checkpoint file 'sel.pkl' not found. Will calculate selections")
        frsel.selectwithinPmf(args.ener,pmfobj,binrange=args.binrange)
        pk.dump(frsel.seldict,open(chkpFn,'wb'))
    else:
        logger.info("checkpoint file 'sel.pkl' found. Will load data from it")
        seldict = pk.load(open(chkpFn,'rb'))
        frsel.seldict = seldict
    	
    frsel.writeXtcfromselection(xtcOutFn=args.xtcout,weightOutFn='weight.txt')
    
    
    
    
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
