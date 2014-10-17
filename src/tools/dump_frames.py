#! /usr/bin/env python
import numpy as np
import argparse 
import logging, os

import pmfcalculator 
from pmfcalculator.frameselector import Frameselector
# from pmfcalculator.pmfNd import WhamNd
# import pmfcalculator.StatsUtils as utils
import pmfcalculator.potentials as bpot 
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
        default="feNd.npz")

parser.add_argument("-temperature",dest='temperature', help='Temerature in kelvin',
        default=300.0,type=float)


parser.add_argument("-pmffile", dest='pmffile',
        help='pmf file name. Will be used to store pmf',
        default="pmfNd.npz")


#parser.add_argument("-range",dest='binrange',
#        help='xrange(start,end) and yrange(start,end) Input 4 numbers seperated by spaces',
#        default=[0.0, 0.0, 0.0, 0.0],nargs=4, type=float)

parser.add_argument("-l", dest='loglevel', help='level of logging (info,warn,debug)',
        default="debug")

parser.add_argument("-xtcfile", dest='xtcout', default='sel.xtc', help='output xtc file')

def main(args):
    observFN = "observNd_time.npz"
    prj = pmfcalculator.ReaderNd(infile=args.prjfile)
    
    if not os.path.isfile(observFN):
        logger.info("Reading files ...")
        observ = prj.read_xvgfiles(btime=True)
        logger.info("files read. Now saving %s",observFN)
        np.savez(observFN,observ)
    else:
        logger.info("loading %s",observFN)
        observ = np.load(observFN)['arr_0']
        
    selector = Frameselector(temperature = args.temperature)
    selector.load_pmf(args.pmffile)
    
    selRange = [selector.pmf.min(),selector.pmf.min()+args.ener]
    logger.debug("Range %s",selRange)
    selector.observations_within_gridvalues(observ,selRange)
    gridValues = []    
    for runId,cvInfoList in selector.selectionItr:
        for cvInfo in cvInfoList:
            #print runId, cvInfo.gridValue
            gridValues.append(cvInfo.gridValue)

    np.savetxt("feFrames.txt",np.array(gridValues).transpose())
    #selector.writeXtcfromselection(prj.xtcfiles, xtcOutFn=args.xtcout)
    
def _getBiasPot(prj,edges):
    ubpotfile = "Ubpot.npz"
    if not os.path.isfile(ubpotfile):
        Ub = bpot.biasPotential(prj.biasType, prj.collvars, prj.vardict, edges)
        np.savez(ubpotfile,Ub)
    else:
        logger.info("loading Ub from %s",ubpotfile)
        Ub = np.load(ubpotfile)["arr_0"]
    
            
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
    logger.info("Finished successfully")
