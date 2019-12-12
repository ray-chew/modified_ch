from utils import writer, funcs
from numerics import time_stepping
import numpy as np
import pyfftw
import time

from numba import jit
from scipy import integrate
import matplotlib.pyplot as plt


def strang(options, wisdom=None):
    L = options['L']
    Nx = options['Nx']
    # eps = options['eps']
    # sigma = options['sigma']
    delta = options['delta']
    m = options['m']
    dt = options['dt']
    it = options['it']
    chiN = options['chiN']
    prefix= options['prefix']
    
    io = writer.writer(prefix,chiN,delta)
    io.create_output_file(options)
    
    h = L/Nx
    dx = np.linspace(0,L,Nx)
    
    # lv defines the levels for which sampling of the systems are made, e.g. sampling of energy and average concentration.
    lv = int(it/100)

    # define seed and uniform distribution.
    # seed = 693
    # np.random.seed(seed)
    r = np.random.uniform(-1.,1.,(Nx,Nx))

    # zeroing the mean of the uniform distribution.
    if r.mean()<0:
        r -= r.mean()
    else:
        r += r.mean()

    # a scaling factor
    C = 0.01

    # a constant
    tpiL = 2.*np.pi/L

    # initialise arrays for pyFFTW.
    u = pyfftw.empty_aligned((Nx,Nx), dtype='complex128', n=16)
    v = pyfftw.empty_aligned((Nx,Nx), dtype='complex128', n=16)
    vp = pyfftw.empty_aligned((Nx,Nx), dtype='complex128', n=16)
    vpp = pyfftw.empty_aligned((Nx,Nx), dtype='complex128', n=16)

    # define initial conditions.
    u[:,:] = m + C*r

    # get wavenumber/frequency grids (k2 == k^2; k4 == k^4)
    freqs = np.fft.fftfreq(Nx,(1.0/float(Nx)))
    kx, ky = np.meshgrid(freqs,freqs)
    k = kx + ky
    k2 = kx**2+ky**2
    k4 = k2**2
    k = np.sqrt(k2)

    # initialise arrays to store avg. concentration and energy values.
    avgConc = np.zeros(int(it/lv+1))
    energy = np.zeros(int(it/lv+1))

    # if there is wisdom for FFT plans, import it.
    if wisdom != None:
        pyfftw.import_wisdom(wisdom)

    # define pyFFTW plans for the forward and inverse FFTs.
    fft_object = pyfftw.FFTW(u, vpp, threads=4, axes=(0,1))
    ifft_object = pyfftw.FFTW(vpp, v, direction='FFTW_BACKWARD', threads=4, axes=(0,1))

    fft_object = pyfftw.FFTW(v, vp, threads=4, axes=(0,1))
    ifft_object = pyfftw.FFTW(vp, u, direction='FFTW_BACKWARD', threads=4, axes=(0,1))

    # initialise some counters.
    j = 0
    t1 = 0
    t2 = 0
    t3 = 0

    for i in range(it+1):
        # the first Strang-splitting substep (eqn 4.5a): Fourier-PS.
        tt1, v[...] = time_stepping.fourier_ps(delta, k4, k2, tpiL, dt, u, fft_object, ifft_object, v, vpp, scheme='Strang')
        t1 += tt1

        # the second Strang-splitting substep (eqn 4.5b): SSP-RK3.
        tt2, v = time_stepping.ssp_rk3(v,h,Nx,delta,m,dt)
        t2 += tt2

        # the third Strang-splitting substep (eqn 4.5c): Fourier-PS.
        tt3, u[...] = time_stepping.fourier_ps(delta, k4, k2, tpiL, dt, v, fft_object, ifft_object, u, vp, scheme='Strang')
        t3 += tt3

        # plot intermediate morphologies.
        # if i%int((it-1)/10)==0:
        #     fig1 = plt.figure(figsize=(3,3))
        #     plt.imshow(u,origin='left', extent=[0,L,0,L])

        # sample energy and average concentration at levels.
        if i%lv==0:
            dim = 2
            u0 = u.real

            avg_conc = funcs.get_avg_conc(u0,dim,dx,L)
            en = funcs.get_energy(k,tpiL,u,delta,dx,m,dim)
            avgConc[j] = avg_conc
            energy[j] = en

            j += 1
            
            data = [avg_conc, en, u.real]
            io.write_data(i, data)
        
        if it % 1000 == 0:
            print("iteration = %i" %it)

    return pyfftw.export_wisdom()



def lie(options, wisdom=None):
    L = options['L']
    Nx = options['Nx']
    # eps = options['eps']
    # sigma = options['sigma']
    delta = options['delta']
    m = options['m']
    dt = options['dt']
    it = options['it']
    chiN = options['chiN']
    prefix= options['prefix']
    
    io = writer.writer(prefix,chiN,delta)
    io.create_output_file(options)
    
    h = L/Nx
    dx = np.linspace(0,L,Nx)

    # lv = levels to capture average concentration and energy changes.
    it += 1
    lv = int(it/100)

    # define empty arrays for pyFFTW
    u = pyfftw.empty_aligned((Nx,Nx), dtype='complex128', n=16)
    uhat = pyfftw.empty_aligned((Nx,Nx), dtype='complex128', n=16)
    v = pyfftw.empty_aligned((Nx,Nx), dtype='complex128', n=16)

    # a constant.
    tpiL = 2.*np.pi/L

    # initial condition
    C = 0.1 # scaling factor
    r = np.random.uniform(-1.,1.,(Nx,Nx))

    # zeroing the mean of the uniform distribution.
    if r.mean()<0:
        r -= r.mean()
    else:
        r += r.mean()
        
    # generate wavenumber / frequency grid.
    freqs = np.fft.fftfreq(Nx,(1.0/float(Nx)))
    kx, ky = np.meshgrid(freqs,freqs)
    k = kx + ky
    k2 = kx**2+ky**2
    k4 = k2**2
    k = np.sqrt(k2)

    # empty arrays to store avg. conc. and energy.
    avgConc = np.zeros(int(it/lv+1))
    energy = np.zeros(int(it/lv+1))

    # if there is pyFFTW wisdom, import it.
    if wisdom != None:
        pyfftw.import_wisdom(wisdom)

    # define pyFFTW plans.
    fft_object = pyfftw.FFTW(u, uhat, threads=4, axes=(0,1))
    ifft_object = pyfftw.FFTW(v, uhat, direction='FFTW_BACKWARD', threads=4, axes=(0,1))

    j = 0
    t1 = 0
    t2 = 0

    u[:,:] = m + C*r

    # iterations.
    for i in range(it+1):
        # the Fourier-PS substep.
        tt1 = time_stepping.fourier_ps(delta, k4, k2, tpiL, dt, u, fft_object, ifft_object, v, uhat)
        t1 += tt1

        # the strong-stability preserving runge-kutta substep.
        tt2, v = time_stepping.ssp_rk2(v,h,Nx,delta,m,dt)
        t2 += tt2

        u[:,:] = v[:,:]

        # capture average concentration and energy of system at current level.
        if i%lv==0:
            dim = 2
            u0 = u.real

            avg_conc = funcs.get_avg_conc(u0,dim,dx,L)
            en = funcs.get_energy(k,tpiL,u0,delta,dx,m,dim)
            avgConc[j] = avg_conc
            energy[j] = en

            j += 1
            
            data = [avg_conc, en, u0]
            io.write_data(i, data)
        
        if i % 1000 == 0:
            print("iteration = %i" %i)

    wisdom = pyfftw.export_wisdom()


