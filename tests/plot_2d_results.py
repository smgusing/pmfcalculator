#! /usr/bin/env python
# -*- coding: utf-8 -*-
# # Author Gurpreet Singh
import os, sys, glob, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import txtreader, setplotformat
#import pmfcalculator
from pmfcalculator import Reader2d

##############################################
setplotformat.doublecol()

def plot_1dpmf(pmf1dfiles, shiftx=3.8):
    ''' plot 1d pmf profiles from files
    
    '''
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for pmf1dfile in pmf1dfiles:    
        data = txtreader.readcols(pmf1dfile)
    
        shiftindex = np.digitize([shiftx], data[:, 0])
        data[:, 1] -= data[shiftindex, 1]
    
        xmin, xmax = data[:, 0].min(), data[:, 0].max()
        ymin, ymax = data[:, 1].min(), data[:, 1].max()
        ymin -= (ymax - ymin) / len(data)
        ymax += (ymax - ymin) / len(data)
    
        ax.plot(data[:, 0], data[:, 1], lw=2)
        ax.set_xlim([0, 4])
        ax.set_ylim([ymin, ymax])
    # ax.axis([0,4,ymin,ymax])
        ax.grid()
        
    ax.set_xlabel("Distance (nm)")
    ax.set_ylabel("PMF (kJ/mol)")
    plt.savefig("1dpmf_from2d.png")

def plot_ineff(infile, prjfile):
    ''' plot stat ineff from files
    
    '''
    
    prj = Reader2d(infile=prjfile)
    a = np.load(infile)
    nx, ny = len(prj.x0), len(prj.y0)
    X, Y = np.meshgrid(prj.y0, prj.x0)
    zx = a['arr_1']
    zy = a['arr_2']
    Zx = zx.reshape(nx, ny)
    Zy = zy.reshape(nx, ny)
    cmap = mpl.cm.cool
    cmap.set_under('y')
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    # levels=np.linspace(0,Z.max(),50)
    # p1=ax.contourf(X,Y,Z,cmap=cmap,levels=levels)
    p1 = ax1.pcolor(X, Y, Zx, cmap=cmap, vmin=1, vmax=Zx.max())
    p2 = ax2.pcolor(X, Y, Zy, cmap=cmap, vmin=1, vmax=Zy.max())
    cbar1 = plt.colorbar(p1, ax=ax1, format="%4.1f", extend='min')
    cbar2 = plt.colorbar(p2, ax=ax2, format="%4.1f", extend='min')
    axs = [ax1, ax2]
    for ax in axs:
    # ax.axis([0,180,0,4])
        ax.grid()
        ax.set_xlabel("Angle (Degrees)")
        ax.set_ylabel("Distance (nm)")
    cbar1.set_label("IneffX")
    cbar2.set_label("IneffY")
    plt.savefig(infile.replace(".npz", ".png"))

def getsubindex(pmf2dfile, shiftx, shifty):
    ''' return indicies of bin where values will fall
    
    '''
    
    a = np.load(pmf2dfile)
    x = a['arr_0']
    y = a['arr_1']
    # X,Y=np.meshgrid(y,x)
    Z = a['arr_2']
    mask = np.isnan(Z)
    Z = np.ma.masked_array(data=Z, mask=mask)
    sindexx = np.digitize([shiftx], x) 
    sindexy = np.digitize([shifty], y) 
    return sindexx, sindexy, x, y, Z
    
def calc2dpmf(pmf2dfiles, shiftx, shifty):
    ''' compute 2d pmf with errorbars determined using bootstrapping data
    
        INPUT:
            pmf2dfiles: list of files containing 2d pmfs
        OUTPUT:
            pmf2dav.npz: contaning two arrays, average and errorbars
    
    '''
    sindexx, sindexy, x, y, Z = getsubindex(pmf2dfiles[0], shiftx, shifty)
    X, Y = np.meshgrid(y, x)
    Zs = []
    print Z.shape
    for pmf2dfile in pmf2dfiles:
        a = np.load(pmf2dfile)
        x = a['arr_0']
        y = a['arr_1']
        Z = a['arr_2']
        Zs.append(Z)
        print pmf2dfile
    Zs = np.array(Zs)
    subs = Zs[:, sindexx, sindexy].repeat(Zs.shape[1] * Zs.shape[2]).reshape(Zs.shape)
    Zs = Zs - subs
    Zsmasked = np.ma.masked_array(data=Zs, mask=np.isnan(Zs))
    Zav = Zsmasked.mean(axis=0)
    Zerr = Zsmasked.std(axis=0, ddof=1) * 2.0
    
    np.savez("results/pmf2dav.npz", x, y, Zav.filled(fill_value=np.NAN),
             Zerr.filled(fill_value=np.NAN))
    
    #     print subs
    # cmap=mpl.cm.gist_earth
    cmap = mpl.cm.jet
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    levels1 = np.linspace(Zav.min(), Zav.max(), 100)
    levels2 = np.linspace(Zerr.min(), Zerr.max(), 100)
    p1 = ax1.contourf(X, Y, Zav, cmap=cmap, levels=levels1)
    p2 = ax2.contourf(X, Y, Zerr, cmap=cmap, levels=levels2)
    cbar1 = plt.colorbar(p1, ax=ax1, format="%4.1f")
    cbar2 = plt.colorbar(p2, ax=ax2, format="%4.1f")
    axs = [ax1, ax2]
    for ax in axs:
        ax.axis([0, 180, 0, 4])
        ax.grid()
        ax.set_xlabel("Angle (Degrees)")
        ax.set_ylabel("Distance (nm)")
    cbar1.set_label("PMF (kJ/mol)")
    cbar2.set_label("PMF (kJ/mol)")
    plt.savefig("results/pmf2dav.png")
    plt.clf()

def calc2dfe(fe2dfiles, shiftx, shifty,prjfile):
    """ calculate average fe
    """
    fes = []
    prj = Reader2d(infile=prjfile)
    nx, ny = len(prj.x0), len(prj.y0)
    x0, y0 = np.array(prj.x0), np.array(prj.y0)
    K = x0.size * y0.size
    #xy = np.zeros((K,2),dtype=np.float32)
    #xy[:,0] = x0.repeat(y0.size)
    #xy[:,1] = y0.reshape(1,y0.size).repeat(x0.size,axis=0).flatten()
    
    a = np.where(x0 == shiftx)[0][0]
    b = np.where(y0 == shifty)[0][0]
    print a,b
    s = a * y0.size + b 
    print "reference state", s
   
    for fe2dfile in fe2dfiles:
        print "loading", fe2dfile
        fe = np.load(fe2dfile)['arr_0'] 
        fe = fe - fe[s]
        fes.append(fe)
    
    fes = np.array(fes,dtype=np.float)
    fes = np.ma.masked_array(data=fes, mask=np.isnan(fes) )
    
    avfes = fes.mean(axis=0)
    errfes = fes.std(axis=0,ddof=1) * 2.0
    print avfes
    print errfes
    np.savez('fe2dav.npz',avfes.filled(fill_value=np.NAN), errfes.filled(fill_value=np.NAN))
    
    
    
def plot_2dpmf(pmf2dfile):
    '''plot 2d pmf from files
    
    '''
    
    a = np.load(pmf2dfile)
    x = a['arr_0']
    y = a['arr_1']
    X, Y = np.meshgrid(y, x)
    Z = a['arr_2']
    print x.shape,y.shape,X.shape,Y.shape,Z.shape
    mask = np.isnan(Z)
    Z = np.ma.masked_array(data=Z, mask=mask)
    
    # cmap=mpl.cm.gist_earth
    cmap = mpl.cm.jet
    fig = plt.figure()
    ax = fig.add_subplot(111)
    levels = np.linspace(0, Z.max(), 100)
    # levels=np.linspace(17,28,100)
    p1 = ax.contourf(X, Y, Z, cmap=cmap, levels=levels)
    # p1=ax.pcolor(X,Y,Z,cmap=cmap)
    cbar = plt.colorbar(p1, ax=ax, format="%4.1f")

    ax.axis([0, 180, 0, 8])
    ax.grid()
    ax.set_xlabel("Angle (Degrees)")
    ax.set_ylabel("Distance (nm)")
    cbar.set_label("PMF (kJ/mol)")
    plt.savefig(pmf2dfile.replace(".npz", ".png"))

    # plt.show()

    # plt.show()
def plot_2dhist(histfile):
    ''' plot 2d histgram from files
    
    '''
    a = np.load(histfile)
    Z, x, y, n_k = a['arr_0'], a['arr_1'], a['arr_2'], a['arr_3']
    X, Y = np.meshgrid(y, x)
    # cmap=mpl.cm.gist_earth
    cmap = mpl.cm.spectral_r
    cmap.set_under('w')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # levels=np.linspace(0,Z.max(),50)
    # p1=ax.contourf(X,Y,Z,cmap=cmap,levels=levels)
    p1 = ax.pcolor(X, Y, Z, cmap=cmap, vmin=5, vmax=Z.max())
    cbar = plt.colorbar(p1, ax=ax, format="%4.1f", extend='min')

    # ax.axis([0,180,0,4])
    ax.grid()
    ax.set_xlabel("Angle (Degrees)")
    ax.set_ylabel("Distance (nm)")
    cbar.set_label("Counts")
    plt.savefig(histfile.replace(".npz", ".png"))
                 
def plot_prob2d(histfile):
    ''' plot 2d probabilites from files
    
    '''
    a = np.load(histfile)
    x, y, Z = a['arr_0'], a['arr_1'], a['arr_2']
    X, Y = np.meshgrid(y, x)
    print Z
    # cmap=mpl.cm.gist_earth
    cmap = mpl.cm.cool
    cmap.set_under('y')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # levels=np.linspace(0,Z.max(),50)
    # p1=ax.contourf(X,Y,Z,cmap=cmap,levels=levels)
    p1 = ax.pcolor(X, Y, Z, cmap=cmap, vmin=0.0, vmax=Z.max())
    cbar = plt.colorbar(p1, ax=ax, format="%4.1f", extend='min')

    # ax.axis([0,180,0,4])
    ax.grid()
    ax.set_xlabel("Angle (Degrees)")
    ax.set_ylabel("Distance (nm)")
    cbar.set_label("Counts")
    plt.savefig(histfile.replace(".npz", ".png"))
    
def plot_2db(infile):
    ''' plot with binary color showing minima +2.5 kj along one dimenison
    '''
    
    a = np.load(infile)
    x = a['arr_0']
    y = a['arr_1']
    X, Y = np.meshgrid(y, x)
    Z = a['arr_2']
    mask = np.isnan(Z)
    Z = np.ma.masked_array(data=Z, mask=mask)
    cmap = mpl.cm.jet
    fig = plt.figure()
    ax = fig.add_subplot(111)
    levels = np.linspace(0, Z.max(), 100)
    a = Z.min(axis=1) + 2.5
    Z1 = np.zeros_like(Z)
    for i in range(len(a)):
        b = np.where(Z[i, :] <= a[i])
        Z1[i, b] = 10
    p2 = ax.pcolor(X, Y, Z1)
    plt.axis([0, 180, 0, 4])
    plt.savefig("a.png")
    # plt.show()

def convert_2dpmftoprob(pmfav, pmferr, beta):
    ''' converts pmf to probabilites and propogates errors accordingly
    
    '''
    notnan = ~np.isnan(pmfav)
    pav = np.zeros_like(pmfav) + np.NAN
    perr = np.zeros_like(pmferr) + np.NAN
    
    # # set min pmf to zero, this will make the largest value to be 1
    pmfav[notnan] -= pmfav[notnan].min()
    pav[notnan] = np.exp(-beta * pmfav[notnan])
    # # for y = ae^x, sigma_y = Sigma_x * (a*e^x)         
    perr[notnan] = pmferr[notnan] * np.exp(-beta * pmfav[notnan])
    
    pavMasked = np.ma.masked_array(data=pav, mask= ~notnan)   
    # perrMasked = np.ma.masked_array(data = perr,mask = ~notnan )   
    return pavMasked, perr
    
    
def convert_2dpmfto1dpmf(pmf2davfile,pmf2dfile, beta):
    ''' convert 2d pmf to 1d and propogate uncertainties

    '''
    
    a = np.load(pmf2davfile)
    b = np.load(pmf2dfile)
    x, y, pmfav, pmferr = (b['arr_0'], b['arr_1'], b['arr_2'], a['arr_3'])

    pav, perr = convert_2dpmftoprob(pmfav, pmferr, beta)
    # take the sum and then convert to normal array
    pav1d = pav.sum(axis=1)
    pav1d = pav1d.filled(fill_value=np.NAN)
    pmf1d = np.copy(pav1d)
    
    pav1d[pav1d == 0.0] = np.NAN
    notnan = ~np.isnan(pav1d)
    ## if x = (a + b)/n then Sx = sqrt(( Sa^2 + Sb^2)/n^2)
    ## if x = (a + b) then Sx = sqrt(( Sa^2 + Sb^2))
    notnan1 = ~np.isnan(perr)
    perrsq = np.zeros_like(perr) + np.NAN
    perrsq[notnan1] = perr[notnan1] * perr[notnan1]
    perrsq = np.ma.masked_array(data=perrsq, mask= ~notnan1)
   # pav1derr = perrsq.mean(axis=1)/perrsq.shape[1]
    
    perr1d = np.sqrt(perrsq.sum(axis=1))
    pmf1d[notnan] = -1. * np.log(pav1d[notnan]) / beta
    pmf1derr = np.copy(perr1d)
    
    # if x = a*log(b) then Sx = a*Sb/b
    pmf1derr = (1. / beta) * (perr1d / pav1d) 
    np.savez("pmf1dfrom2d_new.npz",x, pmf1d, pmf1derr.filled(fill_value=np.NAN))
    plt.errorbar(x, pmf1d, yerr=pmf1derr, marker='o')
    plt.savefig("pmf1dfrom2d_new.png")
    plt.clf()
    #plt.show()
    
#     pmf1d[nonzero1d] = -1.* np.log(p1d[nonzero1d]) / beta
#     p1d[nonzero1d] = p1d[nonzero1d] - p1d[nonzero1d].min()
#     pmf1d[:,1]=p1d
#     
#     txtreader.writecols(pmf1dfile,pmf1d)
    
    
##############################################
def plot_multiples(histfiles, probfiles, pmf1dfiles, pmf2dfiles):
   
    for i in range(len(histfiles)):
        # plot_2dpmf(pmf2dfiles,shiftx = 3.8,shifty)
        plot_2dhist(histfiles[i])
        # plot_prob2d(probfiles[i])
        #plot_2dpmf(pmf2dfiles[i])
        # convert_2dprobto1dpmf(probfiles[i], pmf1dfiles[i],beta)
        # plot_2dpmf(pmf2dfiles[i])
        print i
        # plot_2db(pmf2dfiles[i])




def main():
    N = 2
    R = 8.3144621 / 1000.0  # Gas constant in kJ/mol/K
    temperature = 300
    beta = 1.0 / (R * temperature)  # inverse temperature of simulations (in 1/(kJ/mol))
    histfiles = ["hist2d%s.npz" % i for i in range (N)]
    probfiles = ["prob2d%s.npz" % i for i in range (N)]
    pmf1dfiles = ["pmf1d%s.txt" % i for i in range (N)]
    pmf2dfiles = ["pmf2d%s.npz" % i for i in range (N)]
    fe2dfiles = ["fe2d%s.npz" % i for i in range (N)]
    ineff_file = "stat_ineff.npz"
    prjfile = "dummy.yaml"
    pmf2davfile = "pmf2dav.npz"
    pmf2dfile = "pmf2d.npz"
    fe2davfile = "fe2dav.npz"
    #plot_multiples(histfiles,probfiles,pmf1dfiles,pmf2dfiles)
    plot_2dpmf("refpmf2d.npz")
    plot_2dpmf("pmf2d.npz")
    #plot_2dhist("results/hist2d.npz")
    #calc2dpmf(pmf2dfiles, shiftx=3.5, shifty=90.0)
    #plot_ineff(ineff_file,prjfile)
    #convert_2dpmfto1dpmf(pmf2davfile,pmf2dfile, beta)
    #calc2dfe(fe2dfiles, shiftx=3.5, shifty=90.0,prjfile=prjfile)
    
main()
