#! /usr/bin/env python
import numpy as np
import logging
from collections import defaultdict
import sys
from collections import namedtuple
from ctypes import c_float
import pickle as pk

from pmfcalculator.pmfNd import PmfNd
from gp_grompy import Gmxtc,matrix

logger = logging.getLogger("pmfcalculator")



class Frameselector(PmfNd):
    ''' Class to select and write multiple frames from multiple of trajectories
        
    '''
    
    def __init__(self, temperature=None):
        super(Frameselector,self).__init__(temperature)
        self._cvInfoType = namedtuple('CvInfo',['time','cv','idx','gridValue'])
    
    
    def estimateWeights(self):
        pass
            
    def _gridpoints_with_gridvalues(self,valueRange):
        # Get bins from pmf that matches the criteria
        # all the bins within width of minimum
        pmf = np.copy(self.pmf)
        notnan = ~np.isnan(pmf)
        vmin,vmax = valueRange
        ## wiered behaviour as nan values gives floating error
        pmf[~notnan]=np.inf
        withinRange = np.nonzero( (pmf >= vmin) * (pmf < vmax) )
        selGridIdx = zip(*(withinRange))
        logger.debug("Selected gridIndex %s",selGridIdx)
            
        return selGridIdx
        

    def _mapObservToGrid(self,observ,bTimeColumn=True):
        ''' returns the bin index of the observation
       
        Parameters
        ------------
           observ :
           edges :
           mask :
           bTimeColumn:
                   
        ''' 
        if bTimeColumn == True:
            time = observ[:,0]
            cv = observ[:,1:]
        else:
            time = np.arange(observ.shape[0])
            cv = observ
        
        cv_ranges = [(edge.min(),edge.max()) for edge in self.histEdges]
        
        cv_mask = self.mask_outofrange(cv,cv_ranges=cv_ranges)
        
        
        indices=[]
        for i,edge in enumerate(self.histEdges):
            indices.append( np.digitize(cv[cv_mask,i],edge) ) 
        
        observMap = {}
        for i,idx in enumerate(zip(*indices)):
            try:
                observMap[idx].append((time[i],cv[i]))
            except KeyError:
                observMap[idx] = [(time[i],cv[i])] 
        
        return observMap
    
    def observations_within_gridvalues(self, observations, selRange):
        ''' Get bin numbers as a tuple that matches the criteria

            Parameters
            -------------
                grid: array_like
                    Nd grid 
                edges: list of arrays
                    each array is an edge of grid
                    
                width: float
                    energy range is pmf.min() + width
                
            
            Returns
            --------------
            selection: dict
                run number as key and list of timeframes as value
            refbins: list
              list of tuples consisting of binids that match the criteria
                
        '''
        # Read Data

        selection = defaultdict(list)

        selGridIdx = self._gridpoints_with_gridvalues(selRange)
                
        for i,sim in enumerate(observations):
            mappedObserv = self._mapObservToGrid(sim, bTimeColumn=True)
            for idx in selGridIdx:
                if idx in mappedObserv:
                    for time,cv in mappedObserv[idx]:
                        cvInfo = self._cvInfoType(time=time, cv=cv, idx=idx, gridValue=self.pmf[idx])
                        selection[i].append(cvInfo)
        
        self._selection = selection
    
    @property
    def selectionItr(self):
        for key in sorted(self._selection):
            yield key, self._selection[key]
        
    
    def writeToFile(self,chkpFn):
        pk.dump(self._selection,open(chkpFn,'wb'))
    
    def loadFromFile(self,chkpFn):
        self._selection = pk.load(open(chkpFn,'rb'))
                     
    def writeXtcfromselection(self, xtcfiles, xtcOutFn):
        ''' Write xtc file containing frames that matched criteria
        
        '''
        
        xtcRead = Gmxtc()
        xtcWrite = Gmxtc()
        frno = 0
        xtcWrite.open_xtc(xtcOutFn, 'w')
        for runid, cvInfoList in self.selectionItr:
            xtcInFn = xtcfiles[runid]
            for cvInfo in cvInfoList:
                logger.info("time %s",cvInfo.time)
                xtcRead.read_timeframe(xtcInFn,time=cvInfo.time)
                xtcWrite.copy(xtcRead)
                xtcWrite.time = c_float(frno)
                ret = xtcWrite.write_xtc()
                xtcWrite.clear()
                frno +=1
        xtcWrite.close_xtc()
        logger.info("xtc file %s written",xtcOutFn)
        

        
#     def writeXtcfromselection(self,xtcOutFn='sel.xtc'):
#         ''' Write xtc file containing frames that matched criteria
#         
#         '''
#         
#         xtcRead = Gmxtc()
#         xtcWrite = Gmxtc()
#         frno = 0
#         xtcWrite.open_xtc(xtcOutFn, 'w')
#         for runid, frameVars in self.selectionItr:
#             xtcInFn = self.prjobj.xtcfiles[runid]
#             for frtime,w in frameVars:
#                 logger.info("time %s",frtime)
#                 xtcRead.read_timeframe(xtcInFn,time=frtime)
#                 xtcWrite.copy(xtcRead)
#                 xtcWrite.time = c_float(frno)
#                 ret = xtcWrite.write_xtc()
#                 frno +=1
#         xtcWrite.close_xtc()
#         logger.info("xtc file %s written",xtcOutFn)
    










class Frameselector2d():
    ''' Class to select and write multiple frames from multiple of trajectories
        
    '''
    
    def __init__(self, prjobj=None, seldict=None, refidx=None):
        ''' 
        '''
        # dictionary with trajno as key, and a list with
        # frame time,and weight as tuple
        if seldict is None:
            self.seldict = defaultdict(list)
        else:
            self.seldict = seldict
        # list of tuples, where tuple is a binid 
        self.refidx = refidx
        self.prjobj = prjobj

    def selectwithinPmf(self,width,pmfobj,binrange=None):
        ''' Get bin numbers as a tuple that matches the criteria

            Parameters: pmffile: npz file
                        width: float
                             from minimum for bin selection
                        binrange:
                            if provided, data within the binrange is considered
                            should be the same as used in making pmf profile
            Returns: sel: dict
                        run number as key and list of timeframes as value
                     refbins: list
                         list of tuples consisting of binids that match the criteria
                
        '''
        # Read Data
        logger.info("Reading xvg files ...")    
        pos_xkn, pos_ykn, N_k, timestamp = self.prjobj.read_xvgfiles(btime=True)
        pmfobj.N_k = N_k
        logger.info("xvgfiles read.")

        edgeX,edgeY=self._midpToEdges(pmfobj.midp_xbins,pmfobj.midp_ybins)
        pmf = pmfobj.pmf
        K = pmfobj.K
        N_max = N_k.max()
        
        # Get bins from pmf that matches the criteria
        # all the bins within witdth of minimum
        notnan = ~np.isnan(pmf)
        pmfmin =pmf[notnan].min()
        logger.info("Minumum %f",pmfmin)
        ## wiered behaviour as nan values gives floating error
        pmf[~notnan]=np.inf
        refidx = zip(*np.where(pmf<=(pmfmin+width) ))
        

        sample_indices = np.zeros([K, N_max], dtype=np.bool)
        for k in range(K):
           sample_indices[k, 0:N_k[k]] = True
        
        # If the bin range is not given, we can use the whole data.
        # Other wise we have to update the sample_indicies for the values
        # that are within the binrange
        if binrange is None:
            logger.info("Using automatic determination of bin ranges")
            
            pos_xmin, pos_xmax = pos_xkn[sample_indices].min(), pos_xkn[sample_indices].max()
            pos_ymin, pos_ymax = pos_ykn[sample_indices].min(), pos_ykn[sample_indices].max()
            
            logger.debug("dataX min %s , data max %s", pos_xmin, pos_xmax)
            logger.debug("dataY min %s , data max %s", pos_ymin, pos_ymax)
            
        else:
            logger.info("Using provided bin ranges %s",binrange)
            pos_xmin, pos_xmax = binrange[0][0], binrange[0][1]
            pos_ymin, pos_ymax = binrange[1][0], binrange[1][1]
            
            for k in range(0, K):
               xidx=( pos_xkn[k, 0:N_k[k]] >= pos_xmin) * (pos_xkn[k, 0:N_k[k]] <= pos_xmax)
               yidx=( pos_ykn[k, 0:N_k[k]] >= pos_ymin) * (pos_ykn[k, 0:N_k[k]] <= pos_ymax)
               idx=xidx*yidx
               sample_indices[k, 0:N_k[k]] = idx
        
        for k in range(K):
            idx = sample_indices[k, 0:N_k[k]]
            xindex=np.digitize(pos_xkn[k,idx],edgeX)
            yindex=np.digitize(pos_ykn[k,idx],edgeY)
            
            # bin indices as key and positions,timestamp as values
            trajdict={}
            for key,value in zip( zip(xindex,yindex), zip(pos_xkn[k,idx],pos_ykn[k,idx],timestamp[k,idx]) ):
                trajdict[key] = value
                
            for key in refidx:
                if key in trajdict:
                    posx,posy,frtime = trajdict[key]
                    w = pmfobj.calcFrameWeight(posx,posy)
                    self.seldict[k].append((frtime,w))
        
        self.refidx = refidx
                     
    def writeXtcfromselection(self,xtcOutFn='sel.xtc',weightOutFn='weigth.txt'):
        ''' Write xtc file containing frames that matched criteria
        
        '''
        
        xtcRead = Gmxtc()
        xtcWrite = Gmxtc()
        frWeights = []
        frno = 0
        xtcWrite.open_xtc(xtcOutFn, 'w')
        for runid,frlist in self.seldict.items():
            xtcInFn = self.prjobj.xtcfiles[runid]
            for frtime,w in frlist:
                logger.info("time %s",frtime)
                xtcRead.read_timeframe(xtcInFn,time=frtime)
                xtcWrite.copy(xtcRead)
                xtcWrite.time = c_float(frno)
                ret = xtcWrite.write_xtc()
                frWeights.append(w)
                frno +=1
        xtcWrite.close_xtc()
        frWeights = np.array(frWeights)
        np.savetxt(weightOutFn,frWeights)
        logger.info("xtc file %s written",xtcOutFn)

#     def writeXtcfromselection(self,xtcOutFn='sel.xtc',weightOutFn='weigth.txt'):
#         ''' Write xtc file containing frames that matched criteria
#         
#         '''
#         
#         xtcRead = Gmxtc()
#         xtcWrite = Gmxtc()
#         traj = []
#         boxs = []
#         frWeights = []
#         prec = 0
#         for runid,frlist in self.seldict.items():
#             xtcInFn = self.prjobj.xtcfiles[runid]
#             print xtcInFn
#             for frtime,w in frlist:
#                 logger.info("time %s",frtime)
#                 xtcRead.read_timeframe(xtcInFn,time=frtime)
#                 fr = xtcRead.x_to_array()
#                 prec = xtcRead.prec
#                 traj.append(fr.reshape(1,fr.shape[0],3))
#                 boxs.append(self.copybox(xtcRead.box))
#                 frWeights.append(w)
#         
#         frWeights = np.array(frWeights)
#         np.savetxt(weightOutFn,frWeights)
# 
#         traj = np.vstack(traj)
#         timelist = [ c_float(i) for i in range(len(traj))]
#         xtcWrite.write_array_as_traj(xtcOutFn,traj,boxs,timelist,prec)
#         logger.info("xtc file %s written",xtcOutFn)

    def _midpToEdges(self,midpX,midpY):
        ''' load pmf file and convert midpoint to binedges
        '''

        xw=midpX[1]-midpX[0]
        yw=midpY[1]-midpY[0]
        edgeX=np.zeros(midpX.size+1)
        edgeY=np.zeros(midpY.size+1)
        edgeX[:-1] = midpX - (xw*0.5) 
        edgeY[:-1] = midpY - (yw*0.5) 
        edgeX[-1] = edgeX[-2] + xw
        edgeY[-1] = edgeY[-2] + yw
        
        return edgeX,edgeY

    def copybox(self,box):
        ''' convert coordinates to numpy array
        '''
#         buffer_from_memory = ctypes.pythonapi.PyBuffer_FromMemory
#         buffer_from_memory.restype = ctypes.py_object
#         buf = buffer_from_memory(self.box, 4 * 3 * 3)
#         box=np.ndarray((3, 3),dtype=np.float32, order='C',
#                      buffer=buf)
#         newbox = np.copy(box)
#         newboxp = newbox.ctypes.data_as(matrix)
#         return newboxp
        newbox = matrix()
        itr = ((x,y) for x in range(3) for y in range(3))
        for x,y in itr:
            newbox[x][y] = box[x][y] 
        return newbox


