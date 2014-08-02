#! /usr/bin/env python
# -*- coding: utf-8 -*-
## Author Gurpreet Singh
## Script to generate dummy PMF data
import os, sys, glob
import numpy as np
#from CUST import *
import numpy.random as rand
import smooth as sm


def create_pmf(rc_beg=0,rc_end=8,npts=8001,hp=(4,2,5),smw=300,outf="refpmf.dat"):
    """
    Create a test pmf by adding a harmonic potential of given width at a given point
    NOTE: The potential is smoothed afterwards
    
    Parameters:
        rc_beg: begining of reaction coordinate (rc)
        rc_end: end of reaction coordinate
        npts: number of points along the rc
        hp: a tuple containing equilibrium position of harmonic potential,width, and force constant
        outf: file to which pmf will be written
        smw: smoothing window
    Returns:
        pmf: 2D numpy array with rc as its 1st column and PMF as its 2nd column
    """
    
    x0=hp[0]
    xb=x0-hp[1]
    xe=x0+hp[1]
    k=hp[2]
    j=0
    x=np.linspace(rc_beg,rc_end,npts)
    y=np.zeros(x.size)
    pmf=np.zeros((x.size,2))
    pmf[:,0]=x
    hindex=np.where(x>xb)[0]
    hindex=hindex[np.where(x[hindex]<xe)[0]]
    xh=x[hindex]
    yh=0.5*k*(xh-x0)**2
    yh=yh-yh.max()
    y[hindex]=yh
    ysm=sm.smooth(y,window_len=smw)
    OF=open(outf,'w')
    for i in range(ysm.size):
        OF.write("%s %s\n"%(x[i],ysm[i]))
    pmf[:,1]=ysm
    
    print "File %s successfully written"%outf
    return pmf

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


def convert_pot_to_prob(x,pot,RT):
    """
    Convert potential to probability p=exp(-beta * U)
     
    """
    
    p=np.exp(-1*pot*RT)
    p=p/p.sum()
    return p

def write_probs(fn,x,p):
    OF=open(fn,'w')
    for i in range(x.size):
        OF.write("%10.5f%10.5f\n" % (x[i],p[i]))
    OF.close()
    print "%s written"%fn

def generate_samples_from_prob(x,prob,n=50000):
    """
     Generate numbers with a given probability distribution
     
     Parameters:
     
         x: array along which number will be generated
         prob: array of probability distribution of x such that sum(x)=1
         n: number of samples to generate
         
     Returns:
         randn: array of samples of size n 
         
    """
    
    cdf=np.cumsum(prob)
    binindex=np.arange(cdf.size)
    eps=10e-8
    nzindex=np.where(cdf >eps)[0]
    nzbins=binindex[nzindex]
    randn=rand.random_sample(size=n)
    bins=np.digitize(randn,cdf[nzindex])
    randn=x[nzbins[bins]]
    return randn

def write_samples(fn,x0,data):
    
    OF=open(fn,'w')
    for i in range(data.size):
        OF.write("%10d%10.5f%10.5f\n" % (i,x0,data[i]))
    OF.close()
    print "%s written"%fn
    
def write_samples_alan(fn,x0,data):
    OF=open(fn,'w')
    for i in range(data.size):
        OF.write("%10d%10.5f\n" % (i,data[i]))
    OF.close()
    print "%s written"%fn

def gen_yaml(yamlfile,datadir,x0s,ks):
    'generate Yaml file'
    
    trials=['R1']
    systems=["dummy"]
    nested=['trails','systems','positions']
    Vars=["%3.2f"%x0s[i] for i in range(x0s.size)]
    OF=open(yamlfile,'w')
    OF.write("xvgdir: %s/%s\n"%(os.getcwd(),datadir) )
    OF.write("xtcdir: %s/%s\n"%(os.getcwd(),datadir) )
    OF.write("stride: %s\n"%1)
    OF.write("trials: %s\n"%trials)
    OF.write("systems: %s\n"%systems)
    OF.write("xvgsuffix: %s\n"%"dh")
    OF.write("xtcsuffix: %s\n"%"1ns")
    OF.write("nested: %s\n"%nested)
    OF.write("positions: %s\n"%Vars)
    OF.write("x0: %s\n"%x0s.tolist())
    OF.write("kx: %s\n"%ks.tolist())
    OF.close()
    
def convert_pmf_to_force(pmf):
  
    x=pmf[:,0]
    y=pmf[:,1]
    force=np.zeros((x.size-1,y.size-1))
    yder=np.diff(y)/np.diff(x)
    force[:,0]=x[:-1]
    force[:,1]=yder
    return force

def make_tabulated_pot(force,pmf):
    x=force[:,0]
    y=pmf[:,1]
    f=force[:,1]
    table='table_b0.xvg'
    OF=open(table,'w')
    for i in range(x.size):
        OF.write("%5.3f%10.5f%10.5f\n" % (x[i],y[i],-1.0*f[i]))
    OF.close()
def gen_alan_metadata(x0s,fcs):
    outf='alan_inp.txt'
    OF=open(outf,'w')
    for i in range(x0s.size):
        # Add PMF potential to the harmonic potential
        fn="%s/data1d/R1_xa1_%3.2f_dh.xvg"%(os.getcwd(),x0s[i])
        OF.write("%s %s %s\n"%(fn,x0s[i],fcs[i]))
    OF.close()
    

    
def main():
  
    # create pmf and writes it to pmf.dat file.
    # Add PMF potential to the harmonic potential
    # generates harmonic potentials
    # Generate distribution from a given potential
    # Generate samples that follow the given distribution
    T=300.0 # temperature in kelvin
    RT=8.3144*T/1000.
      
    nx0=160 # number of points along rc where samples will be generated
    fc=500  # force constant for harmonic potential
    rc_beg=0
    rc_end=8
    datadir = "1D"
    yamlfile = "dummy.yaml"
    if not os.path.isdir(datadir):
        os.mkdir(datadir)
    
    
    pmf=create_pmf(rc_beg=rc_beg,rc_end=rc_end,npts=8001,hp=(4,2,5),smw=300,outf="refpmf.dat")
    x=pmf[:,0]
    x0s=np.linspace(rc_beg,rc_end,nx0+2)[1:-1]
    fcs=np.zeros(nx0)+fc
    U_kn=gen_harmonic_potentials(x,fcs,x0s)
    
    for i in range(nx0):
        # Add PMF potential to the harmonic potential
        U_kn[i,:]=U_kn[i,:]+pmf[:,1]
        p=convert_pot_to_prob(x,U_kn[i,:],RT)
        fn="%s/R1_dummy_%3.2f_prob.dat"%(datadir,x0s[i])
        write_probs(fn,x,p)
        s=generate_samples_from_prob(x,p,n=10000)
        fn="%s/R1_dummy_%3.2f_dh.xvg"%(datadir,x0s[i])
        write_samples(fn,x0s[i],s)
        #write_samples_alan(fn,x0s[i],s)

    gen_yaml(yamlfile,datadir,x0s,fcs)
    #gen_alan_metadata(x0s,fcs)
    #force=convert_pmf_to_force(pmf)    
    #make_tabulated_pot(force,pmf)
    
    
main()
