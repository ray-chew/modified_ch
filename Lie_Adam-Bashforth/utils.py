import numpy as np
from scipy import integrate
import schemes
import checker
import os

# function to generate wavenumber grid.
# ref: https://stackoverflow.com/questions/7161417/how-to-calculate-wavenumber-domain-coordinates-from-a-2d-fft
def wavenumber(Nx,dim):
    freqs = np.fft.fftfreq(Nx,(1.0/float(Nx)))

    # populate wavenumber grid
    if dim == 1:
        k = freqs
        k2 = k**2
        k4 = k2**2

    if dim == 2:
        kx, ky = np.meshgrid(freqs,freqs)
        k = kx + ky
        k2 = kx**2+ky**2
        k4 = k2**2

    if dim == 3:
        kx, ky, kz = np.meshgrid(freqs,freqs,freqs)
        k = kx + ky
        k2 = kx**2+ky**2
        k4 = k2**2

    return k,k2,k4


# function to calculate unitless energy based on energy functional.
# T1, T2 and T3 are the three terms of the energy functional.
def energyFunc(u,eps,sigma,m,dx,k,tpiL,dim):
    T1 = -1.j * k * tpiL * np.fft.fftn(u)
    T1 = np.real(np.fft.ifftn(T1))
    T1 = 0.5*eps**2*abs(T1)**2

    T2 = 1./4*(np.real(u)**2-1.)**2

    up = np.real(u)-m
    for d in range(dim):
        up = integrate.trapz(up,dx)
    T3 = -1.*up
    T3 = 0.5*sigma*abs(T3)**2

    en = T1 + T2 + T3
    en = np.real(en)
    for d in range(dim):
        en = integrate.trapz(en,dx)

    return en

def avgConc(u,dx,L,dim):
    u = np.real(u)
    for d in range(dim):
        u = integrate.trapz(u,dx)
    return u / L**dim


# define path to store diagrams and arrays.
def getPath(Nx,L,dt,it,eps,sigma,m,seed,lv,imgDir,tag):
    folder = '/Nx='+str(Nx)+',L='+str(L)+',dt='+str(dt)+ \
                ',it='+str(it)+',eps='+str(eps)+',sigma='+str(sigma)+',m='+str(m)+ \
                ',seed='+str(seed)+',lv='+str(lv)+tag
    checker.checkDir(imgDir,folder)
    return imgDir + folder


# define how to save arrays.
def outputArray(data,path,filename):
    #assert(len(data.shape) <= 3, "Output array has 4 dimensions or more. Not supoprted!")

    # ref: https://stackoverflow.com/questions/3685265/how-to-write-a-multidimensional-array-to-a-text-file
    if len(data.shape) == 3:
        with file(path+'/'+filename+'.txt', 'w') as outfile:
            # Write array shape
            outfile.write('# Array shape: {0}\n'.format(data.shape))

            # iterate through each layer of data
            for data_slice in data:

                # The formatting string indicates that I'm writing out
                # the values in left-justified columns 7 characters in width
                # with 2 decimal places.
                np.savetxt(outfile, data_slice, fmt='%-7.2f')

                # Writing out a break to indicate different slices...
                outfile.write('# New slice\n')
    else:
        np.savetxt(path+'/'+filename+'.txt', data)


# generate (Nx)^n-dimensional arrays.
def ndArr(Nx,dim):
    ndArr = np.empty(Nx)
    shape = [Nx]
    for d in range(dim-1):
        shape += [1]
        ndArr = np.tile(ndArr,shape)
    return ndArr.shape
