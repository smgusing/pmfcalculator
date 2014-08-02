#! /usr/bin/env python
import os,sys
import numpy as np
import numpy.ma as ma
from numpy.random import randint,random_sample
import tpt
import itertools as it
import logging
#################################
logger = logging.getLogger("Grid")
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(fmt='[%(name)s] %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)
############################################################

class Grid(object):
    def __init__(self,gridfile=None,dims=None,values=None):
        ''' Holds scalar field on a 2d grid
            Parameters:
                gridfile: npz file containing x,y,and H (x,y)
                dims: list of arrays, each array is a dimension
                values: ndarray
                
            Returns:
                grid obj
        '''
        if gridfile is not None:
            self._init_from_file(gridfile)
        else:
             if np.any([dims,values] is None):
                 helpstring=''' provide data for dims and values'''
                 raise SystemExit(helpstring)
             else:
                 logger.debug("Initalizing from given dims and values")
                 self.dims = dims
                 self.values = values
                 
        self.ndims = len(self.dims)
        self.dimsizes = np.array([dim.size for dim in self.dims])
                
        self.nvalues = np.array(self.values.shape).prod()
        
        delta_dims = np.zeros(self.ndims)
        for i in range(self.ndims):
            delta_dims[i] = np.abs(self.dims[i][1] - self.dims[i][0])

        self.delta_dims = delta_dims
        edges = []
        for i in range(self.ndims):
            edge = self.dims[i] - ( self.delta_dims[i] * 0.5 )
            np.append(edge,[self.dims[i][-1] + ( self.delta_dims[i] * 0.5 )])
            edges.append(edge)
        self.edges = edges
        
        ## Need to change this for N dim grid
        dim1dir = [-1,0,1]
        dim2dir = [-1, 0 , 1]
        self.directions =[ (i,j) for i in dim1dir for j in dim2dir ]
        self.nmoves = len(self.directions)
        
        logger.debug("dim shapes %s values: %s",[i.shape for i in self.dims],self.values.shape)
        logger.debug("Number of Dims: %d",self.ndims)
        logger.debug("Delta dims: %s ", self.delta_dims)
        logger.debug("neighbours: %d", self.nmoves)
        logger.info("Grid object Initialized")
            
    def _init_from_file(self,gridfile):
        '''
        '''
        if os.path.isfile(gridfile):
            a=np.load(gridfile)
        else:
               raise SystemExit("{0} nonexistent .. quitting".format(gridfile))
        
        dim1 = a["arr_0"]
        dim2 = a["arr_1"]
        self.dims = [dim1,dim2]
        self.values = a["arr_2"]
        
        
        
        
    def get_value(self,idx):
        ''' Returns value at a given idx. If out of bounds returns inf
        '''
        bound,idx = self.check_index_bounds(idx)
        if bound == True:
            value = self.values[tuple(idx)]
        else:
            value = np.inf
        
        return value    
        
    def check_index_bounds(self,idx ):
        ''' check if the tuple is out of the boundary
            Parameters:
                idx: tuple
                    indices of dim1 and dim2
            Returns:
                bound: boolean
                    True if indices are within bound
                idx: tuple
                    nearest idx in case out_of_bounds
        '''
        
        dims = self.dims
        new_idx = [0,0]
        bounds = np.zeros(self.ndims,dtype=np.bool)
        
        for i in range(self.ndims):
            if idx[i] < 0:
                new_idx[i] = 0
                bounds[i] = False
            elif idx[i] >= dims[i].size:
                new_idx[i] = dims[i].size -1
                bounds[i] = False
            else:
                new_idx[i] = idx[i]
                bounds[i] = True
                
        bound = np.all(bounds == True)
        
        return bound,new_idx
        
        
    def gridpoint_to_index(self,gridpoint):
        ''' Returns indices for a given gridpoint
            Parameters:
                gridpoint: tuple
                    values for dims
            Returns:
                bound: bool
                    True if gridpoint within bounds of grid
                idx: tuple
                    index for given gridpoint
        
        '''
        
        dims = self.dims
        idx = np.zeros(self.ndims,dtype=np.int)
        
        bounds = np.zeros(self.ndims,dtype=np.bool)
        for i in range(self.ndims):
            if gridpoint[i] < dims[i].min():
                bounds[i] = False
                idx[i] = 0
            elif gridpoint[i] > dims[i].max():
                bounds[i] = False
                idx[i] = dims[i].size - 1
            else:
                idx[i] = np.digitize([pdim1],self.edges[i])
                bounds[i] = True

        bound = np.all(bounds == True)

        return bound,idx
        
    def to_transition_matrix(self,beta):
        ''' convert grid to transition matrix
            
        '''
        logger.debug("Making transition Matrix ...")
        T = np.zeros((self.nvalues,self.nvalues))
        ind = []
        for i in range(self.ndims):
            ind.append(np.arange(self.dims[i].size))
            
        itr = it.product(*ind)
        stateno = 0
        for i in itr:
            T_vec = self.get_prob_vector(i,beta)
            T[stateno,:] = self.get_prob_vector(i,beta)
            stateno+=1
        
        logger.debug("... Done")
        return T
        
        
    def get_prob_vector(self,idx,beta):
        ''' Returns probability vector for a given gridpoint index
            for points outside the grid, the prob is zero
            NOTE: the edges are still treated as if they have same number of neighbours
            as any other point
        '''
        
        neighbours = self.get_neighbours(idx)
        T_vec = np.zeros(self.nvalues)
        ener_i = self.values[idx]
        for neighidx in neighbours:
            ener_j = self.get_value(neighidx)
            if ener_j == np.inf:
                T_ij = 0
            else:
                T_ij = np.min([ np.exp(-beta * (ener_j - ener_i)),1.0])

                T_ij = T_ij/self.nmoves
                
            stateno = 0
            for i in range(self.ndims-1):
                stateno += self.dimsizes[i+1]*neighidx[i]
            
            stateno = stateno + neighidx[-1]   
            T_vec[stateno] = T_ij
        
        stateno = self.ind_to_state([idx])
        T_vec[stateno]= 1.0 - T_vec.sum()  
        
        return T_vec

    def get_neighbours(self,idx):
        ''' Returns list containing indices of neighbours for a given gridpoint index
            Parameters: 
                idx:tuple
                    indices
            Returns:
                neighbours: list
                    list of nearest neighbours excluding itself.
        '''
        neighbours=[]
        for move in self.directions:
            new_idx = []
            for i in range(self.ndims):
                new_idx.append(idx[i]+move[i])
            bound,tmp=self.check_index_bounds(new_idx)
            new_idx = tuple(new_idx)
            if (bound == True) and (new_idx != idx):
                neighbours.append(new_idx)
                
        return neighbours
    
    def get_statepop(self,beta):
        ''' returns equilibrium probability for each gridpoint
            PE minimum is shifted to 0
        '''
        values = self.values - self.values.min()
        pop = np.exp(-beta * values)
        C = pop.sum()
        pop = pop/C
        return C,pop
    
    def ind_to_state(self,idxlist):
        ''' convert indices to state number
        '''
        states = []
        for idx in idxlist:
            state =0
            for i in range(self.ndims-1):
                state = state + idx[i]*self.dimsizes[i+1]
            state = state + idx[-1]
            states.append(state)
        
        return states

def example_energy():
    # number of partition points and contour window.
    # interface could be nopart, and window [-3.0,3.0,-2.5,3.5]
    nopart = 100  # number of mesh points in x-direction
    XB = -3.0     # starting coordinate in x-direction
    XE = 3.0      # ending coordinate in x-direction
    YB = -3.0     # starting coordinate in y-direction
    YE = 3.0      # ending coordinate in y-direction
    epsilon = 0.001
    xe,ye = (np.linspace(XB,XE,nopart),np.linspace(YB,YE,nopart))
    [x,y] = np.meshgrid(np.linspace(XB,XE,nopart),np.linspace(YB,YE,nopart));
    aa = np.array([-4, -1, -1, -1])
    bb = np.array([0, 0, 0, 0])
    cc = np.array([-1, -1, -1, -1])
    AA = np.array([-4, -5, -5, 8])
    XX = np.array([0, 1, -1, 0])
    YY = np.array([2.75, 0.15, 0, -0.5])
    # calculation of the potential on grid points
    V = AA[0]*np.exp(aa[0]*pow(x-XX[0],2)+bb[0]*(x-XX[0])*(y-YY[0])+cc[0]*pow(y-YY[0],2));
    for j in range(1,4):
        V =  V + AA[j]*np.exp(aa[j]*pow(x-XX[j],2)+bb[j]*(x-XX[j])*(y-YY[j])+cc[j]*pow(y-YY[j],2))
    V = V + epsilon*(np.power(x,4)+np.power(y,4))
    return [x,y,V,xe,ye]

def make_source_sink(x,y,V):
    xind1 = np.where(x > 3.6)[0]
    yind  = np.arange(y.size)
    s = it.product(xind1,yind)
    source=[]
    for i in s:
        source.append(i)
    x,y=np.where(V<3.0)        
    sink=[]
    for i,j in zip(x,y):
        sink.append(tuple([i,j]))
    return source,sink


# def make_source_sink(x,y,V):
#     xind1 = np.where(x < -0.5)[0]
#     xind2 = np.where(x > 0.5)[0]
#     yind  = np.arange(y.size)
#     s = it.product(yind,xind1)
#     source=[]
#     for i in s:
#         if V[i] < -2.9:
#             #V[i] = 1
#             source.append(i)
#             
#     s = it.product(yind,xind2)
#     sink=[]
#     for i in s:
#         if V[i] < -2.9:
#             #V[i] = 1
#             sink.append(i)
#             
#     return source,sink

def calculate_prob_current(T,pop,Q):
    '''
    '''
    F = np.zeros_like(T)
    ind = []
    for i in range(len(F.shape)):
        ind.append(np.arange(F.shape[i]))
        
    itr = it.product(*ind)
    for i,j in itr:
        F[i,j] = pop[i] * T[i,j] * (Q[j] - Q[i])
    Fabs=np.abs(F)
    Fs= 0.5 * np.sum(Fabs,axis=1)
    return Fs
    

def plot_main(X,Y,Q,V,pc,source,sink):
    vshape=V.shape
    Vflat= V.flatten()
    Vflat[source_states]=np.nan
    V1=Vflat.reshape(vshape)
    source_mask = np.isnan(V1)
    Vflat= V.flatten()
    Vflat[sink_states]=np.nan
    V1=Vflat.reshape(vshape)
    sink_mask = np.isnan(V1)
    
    #V[sink_states]=1
    #V=V.reshape(vshape)
    Q=Q.reshape(vshape)
    pc=pc.reshape(vshape)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    nlevel=80
    ax1 = fig.add_subplot(221)
    levels = np.linspace(V.min(),V.max(),nlevel)
    #V=ma.masked_array(V,mask=sink_mask)
    p1=ax1.contourf(X,Y,V,levels=levels)
    plt.colorbar(p1,ax=ax1)
             
    ax2 = fig.add_subplot(222)
    levels = np.linspace(Q.min(),Q.max(),nlevel)
    Q=ma.masked_array(Q,mask=sink_mask)
    p2=ax2.contourf(X,Y,Q,levels=levels)
    plt.colorbar(p2,ax=ax2)
    
    ax3 = fig.add_subplot(223)
    #ax3 = fig.add_subplot(111)
    levels = np.linspace(pc.min(),pc.max(),nlevel)
    pc=ma.masked_array(pc,mask=sink_mask)
    p3=ax3.contourf(X,Y,pc,levels=levels)
    plt.colorbar(p3,ax=ax3)
    plt.show()

        
if __name__ == "__main__":
    
    temperature = 298.15
    R = 8.3144621 / 1000.0  # Gas constant in kJ/mol/K
    beta =  1.0 / (R * temperature)
    tprobInFn = "tprob.npz"
    committorInFn="commit.npz"
    pcInFn="probcurr.npz"
    popInFn="pop.npz"
    #X,Y,V,x,y = example_energy()
    grid = Grid(gridfile="pmf2d.npz")
    x,y,V= grid.dims[0],grid.dims[1],grid.values
    #V[:80,30:37]=V[:80,30:37]-10
    X,Y = np.meshgrid(y,x)

    if not os.path.isfile(tprobInFn):
        T = grid.to_transition_matrix(beta)
        np.savez(tprobInFn,T)
    else:
        T = np.load(tprobInFn)["arr_0"]
        
    C,pop = grid.get_statepop(beta)
    pop=pop.ravel()

    source,sink = make_source_sink(x,y,V)
    source_states=grid.ind_to_state(source)
    sink_states=grid.ind_to_state(sink)
    
    if not os.path.isfile(committorInFn):
        Q = tpt.calculate_committors(source_states,sink_states,T)
        np.savez(committorInFn,Q)
    else:
        Q = np.load(committorInFn)["arr_0"]
        
    if not os.path.isfile(popInFn):
        Pcurr = calculate_prob_current(T,pop,Q)
        np.savez(pcInFn,Pcurr)    
    else:
        Pcurr = np.load(pcInFn)["arr_0"]

    plot_main(X,Y,Q,V,Pcurr,source,sink)
    
#     net_flux = tpt.calculate_net_fluxes(source_states,sink_states,
#                                         T,populations=pop,committors=Q)
#     paths,bottlenecks,fluxes = tpt.find_top_paths(source_states,sink_states,
#                                                   T,net_flux=net_flux)
#     print paths
#     print bottlenecks
#     print fluxes

    #print net_flux.shape
    #plot1(X,Y,Q,V)
  
  #grid = Grid2d("pmf2d.npz")

    
    
    
  


  
