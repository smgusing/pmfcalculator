#! /usr/bin/env python
# -*- coding: utf-8 -*-
## Author Gurpreet Singh
## Script to generate dummy PMF data
## Author Gurpreet Singh
import os, sys, glob,math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


##############################################
def plot_ref(ax,infile,shiftx=1):
    a=np.loadtxt(infile)
    x=a[:,0]
    y=a[:,1]
    shiftindex=np.digitize([shiftx],x)
    y-=y[shiftindex]
    ax.plot(x,y,color="g",lw=3, label = "Ref")


def plot_res(ax,infiles,shiftx=1):
    
    for label,infile in enumerate(infiles):
        a=np.loadtxt(infile)
        x=a[:,0]
        y=a[:,1]
        shiftindex=np.digitize([shiftx],x)
        y-=y[shiftindex]
        ax.plot(x,y,color="r",lw=2, ls="--", label=label)





    
##############################################

def main():
    fig=plt.figure()
    ax=fig.add_subplot(111)
    plot_ref(ax,"refpmf.dat")
    
    #results = ["pmf1d%s.txt"%i for i in range(4)]
    results = ["pmf1d.txt"]
    plot_res(ax,results)
    
    ax.legend(loc="lower right")
    #ax.axis([0,7.8,0,15])
    plt.savefig("1dpmf.png")
    plt.clf()
    
    #plot_1da("pmf1d.npz")
main()
