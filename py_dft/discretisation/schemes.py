from numba import jit
import numpy as np

@jit(nopython=True)
def shift(A,num):
    E = np.empty_like(A)
    F = np.empty_like(A)
    G = np.empty_like(A)
    H = np.empty_like(A)
    
    E[:num] = A[-num:]
    E[num:] = A[:-num]
    F[:-num] = A[num:]
    F[-num:] = A[:num]
    
    G[:,:num] = A[:,-num:]
    G[:,num:] = A[:,:-num]
    H[:,:-num] = A[:,num:]
    H[:,-num:] = A[:,:num]
    return E + F + G + H

@jit(nopython=True)
def centralDiffShift(A,h,Nx):
    A = A**3
#     D = np.empty_like((Nx,Nx))
    D = -1.*shift(A,2)+ 16.*shift(A,1) - 60.*A
    return D/(12.*h**2)

@jit(nopython=True)
def centralDiff(A,h,delta):
    A = A**3
    D = -1.*shift(A,2)+ 16.*shift(A,1) - 60.*A
    return 1./delta * D/(12.*h**2)