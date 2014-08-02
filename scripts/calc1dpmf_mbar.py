#! /usr/bin/env python
import numpy as np
import argparse 
import logging, sys, os
import numpy.random as nprand
import pmfcalculator 
from pmfcalculator.pmf1d import Wham1d
import pymbar  # multistate Bennett acceptance ratio
import pmfcalculator.StatsUtils as utils
 
########################################
logger = logging.getLogger("pmfcalculator")


des = '''calc1dpmf.py: Calculate 1d pmf.  
    Note: The script does not perform error checking on command line arguments'''

parser = argparse.ArgumentParser(description=des    
    , formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-projfile", dest='prjfile', default='proj.yaml', help='Project file')

parser.add_argument("-maxiter", dest='maxiter', help='Maximum self consistent iteration',
    default=100, type=int)

parser.add_argument("-stride", dest='stride', help='read every stride line for histogram',
    default=1, type=int)

parser.add_argument("-readupto", dest='readupto', help='number of lines to read',
    default=0, type=int)

parser.add_argument("-temperature", dest='temperature', help='Temerature in kelvin',
        default=300.0, type=float)

parser.add_argument("-tolerance", dest='tol', help='Tolerance for self consistent iterations',
        default=1e-4, type=float)

parser.add_argument("-fsuffix", dest='filesuffix', help='suffix used in file name containing data',
        default="dh")

parser.add_argument("-histfile", dest='histfile',
        help='histogram file name. Will be used to store histogram',
        default="hist1d.npz")

parser.add_argument("-probfile", dest='probfile',
        help='probability file name. Will be used to store probabilities',
        default="prob1d.npz")

parser.add_argument("-pmffile", dest='pmffile',
        help='pmf file name. Will be used to store pmf',
        default="pmf1d.txt")

parser.add_argument("-fefile", dest='fefile',
        help='free energy file name. Will be used to store free energies',
        default="fe1d.npz")

parser.add_argument("-nbootstrap", dest='nbootstrap',
        help='number of bootstraps', type=int,
        default=0)

parser.add_argument("-chkdur", dest='chkdur', help='Number of iteration after to save chkpoint files',
        default=10, type=int)

parser.add_argument("-nbins", dest='nbins',
        help='number of bins', default=100, type=int)

parser.add_argument("-range", dest='binrange',
        help='range(start,end). Input 2 numbers seperated by spaces',
        default=[0.0, 0.0], nargs=2, type=float)

parser.add_argument("-inefffile", dest='inefffile',
        help='inefficiency file name. Will be used to store inefficiencies',
        default="stat_ineff.npz")

parser.add_argument("-l", dest='loglevel', help='level of logging (info,warn,debug)',
        default="debug")

parser.add_argument("--average_subsample", action="store_true", dest="bAverage",
                    default=False, help="averaging over correlated sample")



def main(args):
        # if all elements in binrange are zero then set args.binrange to None
    if all(x == 0 for x in args.binrange):
        args.binrange = None
    else:
        binrange = (args.binrange[0], args.binrange[1])
        args.binrange = binrange

    
    prj = pmfcalculator.Reader1d(infile=args.prjfile, logger=logger)
    x0 = np.array(prj.x0, dtype=np.float32)
    kx = np.array(prj.kx, dtype=np.float32)
    if len(x0) != len(kx):
        raise SystemExit("x0 {0} and kx {1} of different size".format(len(x0), len(kx))) 

    logger.info("Reading files ...")    
    pos_kn, N_k = prj.read_files(dir=prj.pdir, suffix=args.filesuffix,
                                             skipline=args.stride, readupto=args.readupto)
    logger.info("files read.")    
    
    ### Compute pmf with stat inefficiencies
    if not os.path.isfile(args.inefffile):
        ineff = utils.compute_stat_inefficiency1D(pos_kn, N_k)
        np.savez(args.inefffile,ineff)
    else:
        ineff = np.load(args.inefffile)['arr_0']
    
    bias = pmfcalculator.HarmonicBias()
    N_max = np.max(N_k)
    K = np.size(x0)
    u_kln = np.zeros([K,K,N_max], dtype=np.float)
    for k in range(K):
        for l in range(K):
            print k,l,K
            u_kln[k,l,:N_k[k]] = bias.compute_potential_1D(params=(kx[l],x0[l]), x=pos_kn[k,:N_k[k]])

    logger.info("Using MBAR")
    print "Running MBAR..."
    mbar = pymbar.MBAR(u_kln, N_k, verbose = True, method = 'adaptive')
#     (f_i, df_i) = mbar.computePMF(u_kn, bin_kn, nbins,uncertainties='from-specified',
#                               pmf_reference = nbins-5)
    
    calc = Wham1d(bias,maxiter=0, tol=args.tol, nbins=args.nbins,
                   temperature=args.temperature,
                   x0=x0, fcx=kx, g_k=ineff, chkdur=args.chkdur)
     
 
    if not os.path.isfile(args.histfile):
        logger.info("histogram file %s not found", args.histfile)
        calc.make_histogram(pos_kn=pos_kn, N_k=N_k, binrange=args.binrange)
        calc.write_histogram(args.histfile)
    else:
        logger.info("Will use %s", args.histfile)
#         
    calc.estimateFreeEnergy(histogramfile=args.histfile,
                                    F_k=mbar.f_k)
#     
    calc.write_FreeEnergies(args.fefile)
    calc.write_probabilities(args.probfile)
    calc.write_pmf(args.pmffile)
#     ####
#     # do averaging according to correlations
#     if args.bAverage:
#         pos_kn = utils.smooth1D(pos_kn, ineff)
#     else:
#         logger.info("Subsampling without averaging")
#         
#     pos_kn,N_k = utils.subsample1D(pos_kn,N_k,ineff)
#     g_k = np.zeros_like(ineff) + 1.0
#     for i in range(args.nbootstrap):
#         logger.info(" Bootstrap iteration: %d ", i + 1)
#         histfilename = args.histfile.replace(".npz", str(i) + ".npz")
#         probfilename = args.probfile.replace(".npz", str(i) + ".npz")
#         pmffilename = args.pmffile.replace(".txt", str(i) + ".txt")
#         fefilename = args.fefile.replace(".npz", str(i) + ".npz")
#         
#         if not os.path.isfile(histfilename):
#             logger.info("histogram file %s not found", histfilename)
#             sub_pos_kn = utils.generate_bootstrapsample1D(pos_kn, N_k)
#             calc.make_histogram(pos_kn=sub_pos_kn, N_k=N_k,
#                            binrange=args.binrange)
#             calc.write_histogram(histfilename)
#         else:
#             logger.info("Will use %s", histfilename)
#             
#         calc.calc1dpmf_harmonic(histogramfile=histfilename,
#                                        fefile=fefilename,
#                                        g_k=g_k)
#         calc.write_FreeEnergies(fefilename)
#         calc.write_probabilities(probfilename)
#         calc.write_pmf(pmffilename)
        
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
