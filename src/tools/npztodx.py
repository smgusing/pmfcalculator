#! /usr/bin/env python
# -*- coding: utf-8 -*-
import sys,glob,re,glob
import numpy as np
import numpy.ma as ma


def writedx(npoints,origin,voxel,H,OutFn):
    
    OutFh=open(OutFn,'w')
    OutFh.write("# Converted from numpy array \n")
    OutFh.write("object 1 class gridpositions counts %d %d %d\n"%(npoints[0],npoints[1],npoints[2]))
    OutFh.write("origin %f %f %f\n"%(origin[0],origin[1],origin[2]))
    OutFh.write("delta %g 0 0\n"%voxel[0])
    OutFh.write("delta 0 %g 0\n"%voxel[1])
    OutFh.write("delta 0 0 %g\n"%voxel[2])
    OutFh.write("object 2 class gridconnections counts %d %d %d\n"%(npoints[0],npoints[1],npoints[2]))
    OutFh.write("object 3 class array type double rank 0 items %d data follows\n"%(npoints[0]*npoints[1]*npoints[2]))
    n=1
    for i in range(npoints[0]):
        print i," of ",npoints[0]
        for j in range(npoints[1]):
            for k in range(npoints[2]):
                OutFh.write("%g"%H[i,j,k])
                if n%3 == 0: 
                    OutFh.write("\n")
                else:
                    OutFh.write(" ")
                n+=1

    OutFh.write("\n")
    OutFh.write('object "density" class field\n')
    OutFh.close()


def main():
    npzInFn=sys.argv[1]
    dxOutFn=sys.argv[2]
    txtOutFn=dxOutFn.replace(".dx",".txt")
    a=np.load(npzInFn)
    H,xe,ye,ze = a['arr_0'],a['arr_1'],a['arr_2'],a['arr_3']
    #midx = (xe[1:] +xe[:-1])*0.5 
    #midy = (ye[1:] +ye[:-1])*0.5 
    #midz = (ze[1:] +ze[:-1])*0.5 
    voxel = [xe[1]-xe[0], ye[1]-ye[0], ze[1]-ze[0] ]
    origin = [ xe.min() + voxel[0] *0.5, ye.min() + voxel[1] *0.5, ze.min() + voxel[2] *0.5 ]
    npoints = [ xe.size - 1, ye.size -1 , ze.size -1 ]
    voxel = np.array(voxel) * 10
    vol_voxel = np.prod(voxel)
    H = H/vol_voxel # number density in A^3
    origin = np.array(origin) * 10
    writedx(npoints,origin,voxel,H,dxOutFn)
    #nonzeroH = H[ np.nonzero(H) ]
    # This is contentious. The density should be based on a grid or on voxels visited?
    #ie those that were visited atleast ones, or on the volume of grid irrespective of visits.
    #nonzeroH = H[ np.nonzero(H) ]
    nonzeroH = H
    print "Average number density in A^3: ", np.mean(nonzeroH)
    print "Std Deviation:", np.std(nonzeroH,ddof=1)
    y,x = np.histogram(nonzeroH,bins=100,normed="False")
    x=np.array(x[:-1])
    print y.shape, x.shape
    np.savetxt(txtOutFn,np.vstack([x,y]).transpose())
    
main()
