

import numpy as np
import logging
from collections import defaultdict
import itertools
import sys

from gp_grompy import Gmxtc,Gmstx,Gmndx

logger = logging.getLogger(__name__)


######### Helper functions#######################
def getFirstXtcFrame(xtcInFn):
    ''' Returns first frame of xtc file as numpy array
    
    '''
    xtc = Gmxtc()
    xtc.open_xtc(xtcInFn, 'r')
    ret = xtc.read_first_xtc()
    frame = xtc.x_to_array()
    xtc.close_xtc()
    return frame



class AtomGroup():
    
    def __init__(self,id=None,size=None,ndx=None):
        '''
        '''
        self.id = id
        self.size = size
        self.indices = None
        self.masses = None
        self.centerOfMass = None
        
        if ndx is not None:
            self.indicies=self.setIndicies(ndx)
            
    
    def setIndicies(self,ndx):
        self.indices = np.empty(self.size,dtype=np.int32)
        for i in range(self.size):
            self.indices[i] = ndx.index[self.id][i]
            
    def setMasses(self,tpr):
        self.masses = np.zeros(self.size,dtype=np.float32)
        for i in range(self.size):
            j = self.indices[i]
            self.masses[i]= tpr.top.atoms.atom[j].m
        
    def calcCenterOfMass(self,frame):
        coords=frame[self.indices,:]
        self.centerOfMass = coords.T.dot(self.masses)/self.masses.sum()
            
        
    

class Grid3D():
    
    def __init__(self,gridcenter=None,nbins=None,gridwidth=None):
        '''
        '''
        self.gridcenter = gridcenter
        self.nbins = nbins  
        self.gridwidth = gridwidth
        self.edges = None
        if all(a is not None for a in [gridcenter,nbins,gridwidth]):
            self.edges = self.calcedges()
            self.histogram = np.zeros(nbins-1,dtype=np.float32)
        else:
            self.bins = None
            
    def calcedges(self):
        '''
        
        ''' 
        binsmin = self.gridcenter - self.gridwidth * 0.5
        binsmax = self.gridcenter + self.gridwidth * 0.5
        
        bins = []
        for i in range(3):
            bin = np.linspace(binsmin[i],binsmax[i],self.nbins[i]).astype(np.float32)
            bins.append( bin )
            binw = bin[1] -bin[0]
            logger.info("Dim: %d Bin width: %f Min: %f Max: %f",
                        i,binw,binsmin[i],binsmax[i])
            
        
        return tuple(bins)
            
        
    def calcDensityGmx(self,xtcInFn,atomindices,frameWeights):
        ''' Calculate density of a given groupindex on the grid from xtc
        
        '''
        xtc=Gmxtc()
        xtc.open_xtc(xtcInFn, 'r')
        ret = xtc.read_first_xtc()
        if frameWeights:
            frameWeights = frameWeights/frameWeights.sum()
            frameWeight = iter(frameWeights)
        else:
            frameWeight = itertools.repeat(1)
        frno=0
        while ret == 1:
            logger.info("Frame %d",frno)
            frame = xtc.x_to_array()
            selatoms = frame[atomindices,:]
            H = self._callHistogram(selatoms) * frameWeight.next()
            self.histogram += H 
            ret = xtc.read_next_xtc()
            frno+=1
        self.histogram /= frno
        logger.info("Histogram calculated")
        
    def writeHistogram(self,npzOutFn="hist.npz"):
        ''' Write histogram
            
        '''
        np.savez(npzOutFn,self.histogram,self.edges[0],self.edges[1],self.edges[2])
        
    def _callHistogram(self,frame):
        '''
        '''
        
        H,edges = np.histogramdd(frame,bins=self.edges)
        H=H.astype(np.float32)
        return H
        

    @staticmethod
    def calcGridCenterGmx(xtcInFn,tprInFn,ndxInFn):
        ''' Calculate center of the grid based on molecules of interest
        
            The center of grid is based on center of mass of two groups.
            The x,y is centered at com of molecule2 and z is center at
            com of molecule1. The com are computed from first xtc frame 
        
            Parameters:
                    xtcInFn
                    tpxInFn
                    ndxInFn
            Returns:
                    tuple containing x,y,z
        '''
        
        stx = Gmstx()
        ndx = Gmndx()
        stx.read_tpr(tprInFn)
        # Get index of group1 and group 2 
        ngroups = 3
        msg='''Specify three groups: 
                First Group will be used to set z
                Second Group will be used to set x and y
                Density of the 3rd group will be calculated on this grid'''
        print(msg)
        ndx.read_index(ndxInFn,ngroups)
        grp1 = AtomGroup(id=0,size=ndx.isize[0],ndx=ndx)
        grp2 = AtomGroup(id=1,size=ndx.isize[1],ndx=ndx)
        grp3 = AtomGroup(id=2,size=ndx.isize[2],ndx=ndx)
        grp1.setMasses(tpr=stx)
        grp2.setMasses(tpr=stx)
        grp3.setMasses(tpr=stx)
        frame = getFirstXtcFrame(xtcInFn)
        grp1.calcCenterOfMass(frame)
        grp2.calcCenterOfMass(frame)
        
        gridcenter = [grp2.centerOfMass[0],grp2.centerOfMass[1],grp1.centerOfMass[2]]
        gridcenter = np.array(gridcenter,dtype=np.float32)
        logger.info("Grid Center %s",gridcenter)
        
        return gridcenter,grp1,grp2,grp3
    
        
        
        
        
        