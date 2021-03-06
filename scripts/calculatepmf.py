#! /usr/bin/env python
import numpy as np
import argparse
import logging, sys, os
import numpy.random as nprand
import pmfcalculator
from pmfcalculator import ReaderNd
from pmfcalculator.pmfNd import ZhuNd,WhamNd
import pmfcalculator.StatsUtils as utils
import pmfcalculator.potentials as bpot


def doBootstrap(args,prj,calc,subObserv,iteration,binranges,Ub,setToZero):

    i = iteration
    logger.info(" Bootstrap iteration: %d ", i + 1)
    histfilename = args.histfile.replace(".npz", str(i) + ".npz")
    probfilename = args.probfile.replace(".npz", str(i) + ".npz")
    pmffilename = args.pmffile.replace(".npz", str(i) + ".npz")
    fefilename = args.fefile.replace(".npz", str(i) + ".npz")

    if not os.path.isfile(histfilename):
        logger.info("histogram file %s not found", histfilename)
        bootObserv = utils.bootstrap(subObserv)
        calc.make_ndhistogram(observ=bootObserv, cv_ranges=binranges, number_bins=prj.nbins)
        calc.write_histogram(histfilename)
    else:
        logger.info("Will use %s", histfilename)

    calc.setParams(Ub,histogramfile=histfilename, fefile = fefilename,
                    ineff=None , chkpointfile=args.chkpfile,
                              setToZero=setToZero)


    calc.estimateWeights()
    calc.writeWeights(fefilename)

    try:
        idxCos = prj.biasType.index("cosine")
        calc.divideProbwithSine(dim=idxCos)
    except ValueError:
        pass
    
    calc.write_probabilities(probfilename)
    calc.probtopmf()
    calc.write_pmf(pmffilename)
    



########################################
logger = logging.getLogger("pmfcalculator")


des = '''calc1dpmf.py: Calculate 1d pmf.
    Note: The script does not perform error checking on command line arguments'''

parser = argparse.ArgumentParser(description=des
    , formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-method", dest='method', default='WHAM', help='WHAM | ZHU ')

parser.add_argument("-projfile", dest='prjfile', default='proj.yaml', help='Project file')

parser.add_argument("-sciter", dest='maxiter', help='Maximum self consistent iteration. Ignored in ZHU method',
    default=100, type=int)


parser.add_argument("-readupto", dest='readupto', help='number of lines to read',
    default=0, type=int)

parser.add_argument("-temperature", dest='temperature', help='Temerature in kelvin',
        default=300.0, type=float)

parser.add_argument("-tolerance", dest='tol', help='Tolerance for self consistent iterations. Ignored with ZHU',
        default=1e-5, type=float)

parser.add_argument("-histfile", dest='histfile',
        help='histogram file name. Will be used to store histogram',
        default="histNd.npz")

parser.add_argument("-observfile", dest='observFN',
        help='npz file with observations',
        default="observNd.npz")

parser.add_argument("-probfile", dest='probfile',
        help='probability file name. Will be used to store probabilities',
        default="probNd.npz")

parser.add_argument("-pmffile", dest='pmffile',
        help='pmf file name. Will be used to store pmf',
        default="pmfNd.npz")

parser.add_argument("-fefile", dest='fefile',
        help='free energy file name. Will be used to store free energies',
        default="feNd.npz")

parser.add_argument("-chkpfile", dest='chkpfile',
        help='checkpoint filename',
        default="chkp.fe.npz")


parser.add_argument("-nbootstrap", dest='nbootstrap',
        help='number of bootstraps', type=int,
        default=0)

parser.add_argument("-bootbegin",dest='bootbegin',
        help='Begin bootstrap iteration and numbering from',type=int,
        default=0)

parser.add_argument("-chkdur", dest='chkdur', help='Number of iteration after to save chkpoint files',
        default=10, type=int)

parser.add_argument("-inefffile", dest='inefffile',
        help='inefficiency file name. Will be used to store inefficiencies',
        default="stat_ineff.npz")

parser.add_argument("-l", dest='loglevel', help='level of logging (info,warn,debug)',
        default="info")

parser.add_argument("--average_subsample", action="store_true", dest="bAverage",
                    default=False, help="averaging over correlated sample")

parser.add_argument("--noIneff", action="store_true", dest="bNoIneff",
                    default=False, help="do not use inefficiency factors")



def main(args):

    ## Read Project file
    prj = ReaderNd(infile=args.prjfile)

    # if all elements in binrange are zero then set args.binrange to None
    if all(x == 0 for x in prj.binranges):
        binranges = None
    else:
        binranges = prj.binranges

    if prj.setToZero == None:
        setToZero = None
    else:
        setToZero = prj.setToZero

    # Read Data
    if not os.path.isfile(args.observFN):
        logger.info("Reading files ...")
        observ = prj.read_xvgfiles(readupto=args.readupto)
        logger.info("files read. Now saving %s",args.observFN)
        np.savez(args.observFN,observ)
    else:
        logger.info("loading %s",args.observFN)
        observ = np.load(args.observFN)['arr_0']

    if args.bNoIneff == False:
        ### Compute pmf with stat inefficiencies
        if not os.path.isfile(args.inefffile):
            ineff = utils.compute_stat_inefficiency(observ)
            np.savez(args.inefffile,ineff)

        else:
            ineff = np.load(args.inefffile)['arr_0']

        maxIneff = ineff.max(axis=1)
        maxIneff = np.around(maxIneff,decimals=0)

    else:
        maxIneff = None
        logger.warn(('Will not calculate IACTs. Assuming uncorrelated Data'))

    if args.method == "WHAM":
        calc = WhamNd(temperature=args.temperature,maxSteps=args.maxiter,
                       tolerance=args.tol)
    elif args.method == "ZHU":
        calc = ZhuNd(temperature=args.temperature,scIterations=args.maxiter,
                     tolerance=args.tol)
    else:
        logger.error("method %s not implemented yet, please choose between WHAM or ZHU")
        raise SystemExit()

    if not os.path.isfile(args.histfile):
        logger.info("histogram file %s not found", args.histfile)
        calc.make_ndhistogram(observ = observ, cv_ranges=binranges,
                              number_bins=prj.nbins, ineff=maxIneff)
        calc.write_histogram(args.histfile)
    else:
        logger.info("Will use %s", args.histfile)
        calc.load_histogram(args.histfile)
    ubpotfile = "Ubpot.npz"
    if not os.path.isfile(ubpotfile):
        Ub = bpot.biasPotential(prj.biasType,prj.collvars,prj.vardict,calc.histEdges)
        np.savez(ubpotfile,Ub)
    else:
        logger.info("loading Ub from %s",ubpotfile)
        Ub = np.load(ubpotfile)["arr_0"]


    calc.setParams(Ub,histogramfile=args.histfile, fefile = args.fefile,
                        ineff = maxIneff, chkpointfile = args.chkpfile,
                                  setToZero=setToZero)

    calc.estimateWeights()

    calc.writeWeights(args.fefile)
    try:
        idxCos = prj.biasType.index("cosine")
        calc.divideProbwithSine(dim=idxCos)
    except ValueError:
        pass

    calc.write_probabilities(args.probfile)
    calc.probtopmf()

    calc.write_pmf(args.pmffile)
    os.remove(args.chkpfile)


    ####
    # do averaging according to correlations
#     if args.bAverage:
#         pos_kn = utils.smooth1D(pos_kn, ineff)
#     else:
#         logger.info("Subsampling without averaging")

    if args.nbootstrap > 0:
        subObserv = utils.subsample(observ,maxIneff)
    else:
        pass
    
    for i in range(args.bootbegin,args.bootbegin + args.nbootstrap):
        doBootstrap(args,prj,calc,subObserv,i,binranges,Ub,setToZero)
        
        
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
