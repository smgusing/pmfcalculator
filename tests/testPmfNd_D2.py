#! /usr/bin/env python

import numpy as np
import numpy.random as rand

import unittest
import os,logging

from multiprocessing import Pool
#import pmfcalculator
import smooth as sm



from pmfcalculator.readerNd import ReaderNd
from pmfcalculator.pmfNd import ZhuNd
import pmfcalculator.potentials as bpot

logger = logging.getLogger("pmfcalculator")
logger.setLevel("DEBUG")



def gendist(prob,n):
    cdf=np.cumsum(prob)
    binindex=np.arange(cdf.size)
    #eps=10e-8
    randn=rand.random_sample(size=n)
    bins=np.digitize(randn,cdf)
    return binindex[bins]

def get_random2d(Xin,Yin,dist_in,n):
    val=np.empty((n,2),dtype=np.float32)
    col_dist=np.sum(dist_in,axis=0)
    col_dist=col_dist/np.sum(col_dist)
    ind1=gendist(col_dist,n)
    val[:,0]=Xin[ind1]
    row_dist=dist_in[:,ind1]
    row_distsum=np.sum(row_dist,axis=0)
    row_dist=row_dist/row_distsum
    for i in xrange(row_dist.shape[1]):
       ind2=gendist(row_dist[:,i],n=1)
       val[i,1]=Yin[ind2]
    return val


def add_smooth_harmonic_well(x,x0,well_width,forcek,smwindow):

    xb=x0-well_width
    xe=x0+well_width
    j=0
    y=np.zeros_like(x)
    hindex=np.where(x>xb)[0]
    hindex=hindex[np.where(x[hindex]<xe)[0]]
    xh=x[hindex]
    yh=0.5*forcek*(xh-x0)**2
    yh=yh-yh.max()
    y[hindex]=yh
    ysm=sm.smooth(y,window_len=smwindow)
    return y

def add_sine_potential(x,forcek,mult):
    y=np.zeros_like(x,dtype=np.float)
    y=forcek*np.sin(mult*np.radians(x))
    return y-y.max()

def create_2d_pmf(x,y,Xparams,Yparams,outf="2dpmf.npz"):
    
    print "creating 2d pmf"
    well_minpos=4
    well_width=2
    well_forcek=5
    smwindow=10
    angle_forcek=5
    angle_mult=4
    X,Y=np.meshgrid(x,y)
    Z=np.zeros_like(X, dtype=np.float64)
    print X.shape,Z.shape
    v1=add_smooth_harmonic_well(x,well_minpos,well_width,well_forcek,smwindow)
    v2=add_sine_potential(y,angle_forcek,angle_mult)
    for i in range(Z.shape[0]):   Z[i,:]=Z[i,:]+v1
    for i in range(Z.shape[1]):   Z[:,i]=Z[:,i]+v2
    Z=Z-Z.min()
    print "Saving %s"%outf
    np.savez(outf,x,y,Z.transpose())
    print "Saved"
    return X,Y,Z
    
def gen_harmonic_potentials(x,ks,x0s):
    """ 
    Calculate harmonic potential as (0.5 * k * (x-x0)^2)
    
    Parameters:
        x: position array
        ks: array of force constants
        x0s: array of equilibrium positions
     Returns:
         U_kn: 2D array of potential with x as rows and potential as columns
    """
    
    U_kn=np.zeros((len(ks),x.size),dtype=np.float)
    for i in range(len(ks)):
        u=0.5* ks[i] * (x-x0s[i])**2
        U_kn[i,:]=u
    return U_kn


def gen_cosine_potentials(x,ks,x0s):
    
    U_kn=np.zeros((len(ks),x.size),dtype=np.float)
    for i in range(len(ks)):
        u=ks[i] * (1- np.cos(np.radians(x-x0s[i])))
        U_kn[i,:]=u
    return U_kn

def convert_pot_to_prob(V,RT):
    """
    Convert potential to probability p=exp(-beta * U) 
    """
    
    p=np.exp(-1*V*RT)
    p=p/p.sum()
    return p

def generate_samples_from_2dprob(P,x,y,n=500000):
    """
     Generate numbers with a given probability distribution
     
     Parameters:
     
         x: array along which number will be generated
         prob: array of probability distribution of x such that sum(x)=1
         n: number of samples to generate
         
     Returns:
         randn: array of samples of size n 
         
    """
    #vals=np.zeros((n,2))
    #for i in range(n):
        #vals[i,0],vals[i,1]=get_random2d(x,y,P,n)
    vals=get_random2d(x,y,P,n)
    

    return vals

def write_2dsamples(fn,data):
    OF=open(fn,'w')
    for i in range(data.shape[0]):
        OF.write("%10d%10.5f%10.5f\n" % (i,data[i,0],data[i,1]))
    OF.close()
    print "%s written"%fn

def gen_yaml(yamlfile,x0s,y0s,kx,ky):
    'generate Yaml file'
    
    trials=['R1']
    systems=["dummy"]
    positions=["%3.2f"%x0s[i] for i in range(x0s.size)]
    angles=["%3.2f"%y0s[i] for i in range(y0s.size)]
    OF=open(yamlfile,'w')
    OF.write("xvgdir: %s/2D\n"%os.getcwd())
    OF.write("xtcdir: %s/2D\n"%os.getcwd())
    OF.write("xvgsuffix: %s\n"%"dh")
    OF.write("xtcsuffix: %s\n"%"dh")
    OF.write("stride: %s\n"%1)
    OF.write("trials: %s\n"%trials)
    OF.write("systems: %s\n"%systems)
    OF.write("collvars: %s\n"%"['pos', 'angles']")
    OF.write("pos: %s\n"%positions)
    OF.write("angles: %s\n"%angles)
    OF.write("pos_x0: %s\n"%x0s.tolist())
    OF.write("angles_x0: %s\n"%y0s.tolist())
    OF.write("pos_fc: %s\n"%kx.tolist())
    OF.write("angles_fc: %s\n"%ky.tolist())
    OF.write("bias: %s\n"%"['harmonic', 'cosine']")
    OF.close()
    
def gen_samples(args):
    #i,Ux,npts1,npts2,PMF,RT,ny0,U_kn2,x,y,x0s,y0s=args[0],args[1],args[2],args[3],args[4],args[5],args[6],args[7],args[8],args[9],args[10],args[11]
    i,Ux,npts1,npts2,PMF,RT,ny0,U_kn2,x,y,x0s,y0s = args
    print ny0
    for j in range(ny0):
        Uy=U_kn2[j,:]
        Uy=Uy.reshape(npts2,1).repeat(npts1,1)
        U_xy=Ux+Uy+PMF
        P=convert_pot_to_prob(U_xy,RT)
        sample=generate_samples_from_2dprob(P,x,y,n=10000)
        fn="2D/R1_dummy_%3.2f_%3.2f_dh.xvg"%(x0s[i],y0s[j])
        write_2dsamples(fn,sample)
        #print i,j,fn
 
    

  



    
def createTestData(pool):
  
    T=300.0 # temperature in kelvin
    RT=8.3144621*T/1000.
    rc1_beg=2
    rc1_end=6
    npts1=80
    rc2_beg=40
    rc2_end=180
    npts2=45
    
    nx0=40 # number of points along rc where samples will be generated
    ny0=45
    #nx0=40 # number of points along rc where samples will be generated
    #ny0=20
    fcx0=100  # force constant for harmonic potential
    fcy0=100
    
    datadir = "2D"
    
    if not os.path.isdir(datadir):
        os.mkdir(datadir)
    
    yamlfile = "dummy.yaml"
    refpmf_file="refpmf2d.npz"

    x=np.linspace(rc1_beg,rc1_end,npts1)
    y=np.linspace(rc2_beg,rc2_end,npts2)
    Xparams=None
    Yparams=None
    x0s=np.linspace(rc1_beg,rc1_end,nx0+2)[1:-1]
    y0s=np.linspace(rc2_beg,rc2_end,ny0+2)[1:-1]
    fcsx=np.zeros(nx0)+fcx0
    fcsy=np.zeros(ny0)+fcy0
 
    #size is npts2,npts1

    X,Y,PMF = create_2d_pmf(x,y,Xparams,Yparams,outf=refpmf_file)
    
    gen_yaml(yamlfile,x0s,y0s,fcsx,fcsy)
    
    #plot_2dpmf(refpmf_file)
    # size is nx0,npts1
    U_kn1 = gen_harmonic_potentials(x,fcsx,x0s)
    # size is ny0,npts2
    U_kn2 = gen_cosine_potentials(y,fcsy,y0s)
    #np.savez("potx.npz",U_kn1,x)
    #np.savez("potx.npz",U_kn2,y)
    #U_xy=np.zeros((nx0,ny0,npts2,npts1))
    
    P=np.zeros_like(PMF)
    ## QUICK AND DIRTY
    ## !should write it properly!
    arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10,arg11,arg12=([],[],[],[],
                                                                   [],[],[],[],[],[],[],[])
    for i in range(nx0):
        arg1.append(i)
        Ux=U_kn1[i,:]
        Ux=Ux.reshape(1,npts1).repeat(npts2,0)
        arg2.append(Ux)
        arg3.append(npts1)
        arg4.append(npts2)
        arg5.append(PMF)
        arg6.append(RT)
        arg7.append(ny0)
        arg8.append(U_kn2)
        arg9.append(x)
        arg10.append(y)
        arg11.append(x0s)
        arg12.append(y0s)
        
    args=zip(arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10,arg11,arg12)
    pool.map(gen_samples,args)
    


class testpmfNd_D2():
    
    def __init__(self,pool):
        print "\n Creating Test Data \n"
        if not os.path.isdir("2D"): 
            createTestData(pool)
        else:
            print("Directory 2D Exists ... will not regenerate")
            
        self.yamlfile = "dummy.yaml"
        
    def genpmf(self):
        prj = ReaderNd(self.yamlfile)
        obsFN="observ.npz"
        if not os.path.isfile(obsFN):
            print("Reading xvgs ..")
            observ = prj.read_xvgfiles()
            np.savez(obsFN,observ)
        else:
            print("loading %s"%obsFN)
            observ = np.load(obsFN)['arr_0']
            
        number_bins = [40, 20] 
        temperature = 300
        cv_ranges = [(2.0, 6.0),( 40.0, 180.0)] 
        setZero = [6.0,90.0] 
        
        calc = ZhuNd(temperature = temperature)  
        histFN = "hist.npz"
        
        if not os.path.isfile(histFN):
            calc.make_ndhistogram(observ=observ,cv_ranges=cv_ranges,number_bins=number_bins)
            calc.write_histogram(histFN)
        else:
            print("loading histogram")
            calc.load_histogram(histFN)
        
        Ub = bpot.biasPotential(prj.biasType,prj.collvars,prj.vardict,calc.histEdges)
        calc.estimateFreeEnergy(Ub,histogramfile=histFN,setToZero=None)
        calc.write_FreeEnergies("feNd.npz")
        calc.write_probabilities("probNd.npz")
        calc.probtopmf()
        calc.write_pmf("pmfNd.npz")
        self.plotpmf("pmfNd.npz")

        
         


if __name__ == "__main__":
    
    pool=Pool(processes=12)
    test = testpmfNd_D2(pool)
    test.genpmf()
    
