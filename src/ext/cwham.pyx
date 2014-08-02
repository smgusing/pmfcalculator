###
#Note to self: this is same speed as pure numpy!
#Also returning value is broken 

###
import cython
import numpy as np
cimport numpy as np
cdef extern from "helperwham.h":

    int c_minimize2d(long N, long nmidpx, long nmidpy,
          double* F_k, long* hist, 
          double* U_bij, double beta,
          long* N_k, double* g_k,  double* F_knew, long chkdur, long windowZero)

    int c_minimize1d(long N, long nmidp,double* F_k,
        long* hist, double* U_bi,
        double beta, long* N_k, double* g_k, 
          double* F_knew, long chkdur,long windowZero)

#     int c_update_F_k2d(long N, long nmidpx, long nmidpy,
#           double* F_k, long* hist, 
#           double* U_bij, double beta,
#           long* N_k, double* g_k,  double* F_knew,long windowZero) 
#     
#     int c_update_F_k1d(long N, long nmidp,double* F_k,
#         long* hist, double* U_bi,
#         double beta, long* N_k,  double* F_knew)
#    void c_compute_logsum(double* inparray, int N, double* logsum)




@cython.boundscheck(False)
@cython.wraparound(False)
def minimize2d(np.ndarray[double,ndim=1,mode='c'] F_k, 
                 long nmidp_xbins, long nmidp_ybins,
                 np.ndarray[long,ndim=2,mode='c']hist,
                 np.ndarray[double,ndim=3,mode='c'] U_bij,
                 double beta,
                 np.ndarray[long,ndim=1,mode='c'] N_k,
                 np.ndarray[double,ndim=1,mode='c'] g_k,
                 long chkdur, long windowZero):

    #print F_k,"INP"
    cdef long N = F_k.shape[0]
    cdef: 
        int i,j,ret
        np.ndarray[double,ndim=1,mode='c'] F_knew = np.empty_like(F_k)
    
    
    ret = c_minimize2d(N, nmidp_xbins, nmidp_ybins,
                        &F_k[0], &hist[0,0], 
                        &U_bij[0,0,0], beta, &N_k[0],
                        &g_k[0], &F_knew[0], chkdur, windowZero)
    
    #print F_knew,"OUT",ret
    
    return F_knew



@cython.boundscheck(False)
@cython.wraparound(False)
def minimize1d(np.ndarray[double,ndim=1,mode='c'] F_k, 
                 long nmidp,
                 np.ndarray[long,ndim=1,mode='c']hist,
                 np.ndarray[double,ndim=2,mode='c'] U_b,
                 double beta,
                 np.ndarray[long,ndim=1,mode='c'] N_k,
                 np.ndarray[double,ndim=1,mode='c'] g_k,
                 long chkdur,long windowZero):

    #print F_k,"INP"
    cdef long N = F_k.shape[0]
    cdef: 
        int i,j,ret
        np.ndarray[double,ndim=1,mode='c'] F_knew = np.empty_like(F_k)
    
    
    ret=c_minimize1d(N, nmidp, &F_k[0], &hist[0], 
                  &U_b[0,0], beta,&N_k[0],
                  &g_k[0], &F_knew[0], chkdur,windowZero)
    
    #print F_knew,"OUT",ret
    
    return F_knew


##################OLD############################
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def update_F_k2d(np.ndarray[double,ndim=1,mode='c'] F_k, 
#                  long nmidp_xbins, long nmidp_ybins,
#                  np.ndarray[long,ndim=2,mode='c']hist,
#                  np.ndarray[double,ndim=3,mode='c'] U_bij,
#                  double beta,
#                  np.ndarray[long,ndim=1,mode='c'] N_k,
#                  np.ndarray[double,ndim=1,mode='c'] g_k,
#                  long windowZero):
# 
#     #print F_k,"INP"
#     cdef long N = F_k.shape[0]
#     cdef: 
#         int i,j,ret
#         np.ndarray[double,ndim=1,mode='c'] F_knew = np.zeros_like(F_k)
#     
#     
#     ret=c_update_F_k2d(N, nmidp_xbins, nmidp_ybins,
#                         &F_k[0], &hist[0,0], 
#                         &U_bij[0,0,0], beta, &N_k[0],
#                         &g_k[0], &F_knew[0],windowZero)
#     
#     #print F_knew,"OUT",ret
#     
#     return F_knew
# 
# 
# 
# @cython.boundscheck(False)
# @cython.wraparound(False)
# def update_F_k1d(np.ndarray[double,ndim=1,mode='c'] F_k, 
#                  long nmidp,
#                  np.ndarray[long,ndim=1,mode='c']hist,
#                  np.ndarray[double,ndim=2,mode='c'] U_b,
#                  double beta,
#                  np.ndarray[long,ndim=1,mode='c'] N_k):
# 
#     #print F_k,"INP"
#     cdef long N = F_k.shape[0]
#     cdef: 
#         int i,j,ret
#         np.ndarray[double,ndim=1,mode='c'] F_knew = np.zeros_like(F_k)
#     
#     
#     ret=c_update_F_k1d(N, nmidp, &F_k[0], &hist[0], 
#                   &U_b[0,0], beta,&N_k[0],  &F_knew[0])
#     
#     #print F_knew,"OUT",ret
#     
#     return F_knew

# def _update_F_k2d(np.ndarray[double,ndim=1,mode='c'] F_k, 
#                  long nmidp_xbins, long nmidp_ybins,
#                  np.ndarray[long,ndim=2,mode='c']hist,
#                  np.ndarray[double,ndim=3,mode='c'] U_bij,
#                  double beta,
#                  np.ndarray[long,ndim=1,mode='c'] N_k):
#  
#     
#     F_knew = np.zeros_like(F_k)
#     logbf  = np.zeros_like(F_k)
#     nonzeroN_k = N_k != 0
#     for i in xrange(nmidp_xbins):
#         for j in xrange(nmidp_ybins):
#              num = hist[i, j]
#              U_b = U_bij[i, j, :]
#               
#              logbf[nonzeroN_k] = beta * (F_k[nonzeroN_k] - U_b[nonzeroN_k]) + np.log(N_k[nonzeroN_k])
#              #denom = _compute_logsum(logbf)
#              denom = compute_logsum(logbf)
#              logbf = (-beta * U_b) + np.log(num) - denom
#               
#              F_knew += np.exp(logbf)
#                
#                
#     nonzero = F_knew != 0
#     F_knew[nonzero] = -1.*np.log(F_knew[nonzero]) / beta
#     F_knew = F_knew - F_knew[nonzero][0]
#     return F_knew
# 
# def _update_fe(self,F_k):
#     F_knew = np.zeros_like(F_k)
#     for i in xrange(self.midp_bins.size):
#         num = self.hist[i]
#         U_b = self.U_b[i, :]
#         logbf = self.beta * (F_k - U_b) + np.log(self.N_k)
#         denom = self._compute_logsum(logbf)
#         if num == 0:
#             logbf = (-self.beta * U_b) 
#         else:
#             logbf = (-self.beta * U_b) + np.log(num) - denom
#             
#              
#         F_knew += np.exp(logbf)
#         
#     F_knew = -1.*np.log(F_knew) / self.beta        
#     F_knew = F_knew - F_knew[0]
#     
#     return F_knew

# @cython.boundscheck(False)
# @cython.wraparound(False)
# def compute_logsum( np.ndarray[double,ndim=1,mode='c'] numpyArray):
#      cdef:
#          int N
#          double ArrayMax
#          double logsum
#      N=numpyArray.shape[0]
#      c_compute_logsum(&numpyArray[0], N,&logsum)
#      return logsum





