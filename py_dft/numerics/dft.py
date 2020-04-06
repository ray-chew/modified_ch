# from utils import writer, funcs
# from numerics import time_stepping
import numpy as np
import pyfftw
import time
import h5py
from utils import writer

from numba import jit
from scipy import integrate
import matplotlib.pyplot as plt

from skimage.transform import resize

def strang(options,wisdom=None):
    Lx = options['Lx']
    Ly = options['Ly']
    Nx = options['Nx']
    Ny = options['Ny']
    delta = options['delta']
    m = options['m']
    dt = options['dt']
    it = options['it']
    prefix = options['prefix']
    plot_initial = options['plot_initial']
    ic_path = options['ic_path']
    
    io = writer.writer(prefix,Lx,Ly)
    io.create_output_file(options)
    
    # h = spatial step-size; dx = 1D spatial grid.
    hx = Lx/Nx
    hy = Ly/Ny
    dx = np.linspace(0,Lx,Nx)
    dy = np.linspace(0,Ly,Ny)
    dxy = [dx,dy]

    # lv = levels to capture average concentration and energy changes.
    it += 1
    lv = int(it/100)

    # define empty arrays for pyFFTW
    u = pyfftw.empty_aligned((Nx,Ny), dtype='complex64', n=16)
    
    uhat = pyfftw.empty_aligned((Nx,Ny), dtype='complex64', n=16)
    v = pyfftw.empty_aligned((Nx,Ny), dtype='complex64', n=16)

    # a constant.
    tpiLx = 2.*np.pi/Lx
    tpiLy = 2.*np.pi/Ly

    # initial condition
    np.random.seed(111)
    C = 1.0 # scaling factor
    r = np.random.uniform(-1.,1.,(Nx,Ny))

    # zeroing the mean of the uniform distribution.
    if r.mean()<0:
        r += r.mean()
    else:
        r -= r.mean()

    u[:,:] = m + C*r

    file_ic = h5py.File(ic_path, 'r')
    if Ly >= 11.0:
        r = file_ic['ic_2p'][:,:]
    else:
        r = file_ic['ic'][:,:]

    r = resize(r, (Nx,Ny))
    
    u[:,:] = (m - 0.5 * r.mean()) + 0.5 * r
    
    print(u.real.mean())
    
    if plot_initial == True:
        ic = np.copy(u.real)
        io.write_data(0, ic)
        
    # generate wavenumber / frequency grid.
    freqX = np.fft.fftfreq(Nx,(1.0/float(Nx))) * tpiLy
    freqY = np.fft.fftfreq(Ny,(1.0/float(Ny))) * tpiLx
    kx, ky = np.meshgrid(freqY,freqX)
    k = kx + ky
    k2 = kx**2+ky**2
    kx2 = k2
    ky2 = k2
    kx4 = k2**2
    ky4 = k2**2

    k = np.sqrt(k2)

    # empty arrays to store avg. conc. and energy.
    avgConc = np.zeros(int(it/lv+1))
    energy = np.zeros(int(it/lv+1))

    # if there is pyFFTW wisdom, import it.
    if wisdom != None:
        pyfftw.import_wisdom(wisdom)

    # define pyFFTW plans.
    fft_obj = pyfftw.FFTW(u, uhat, threads=1, axes=(0,1))

    ifft_obj = pyfftw.FFTW(v, uhat, direction='FFTW_BACKWARD', threads=1, axes=(0,1))

    j = 0
    t1 = 0
    t2 = 0
    
    # iterations.
    for i in range(it+1):
        # the Fourier-PS substep.
        tic1 = time.time()
        fac=0.5

        uhat = np.exp((-1.*delta*(kx4*fac + ky4*fac) + (1./delta)*(kx2*fac + ky2*fac) -1.)*dt)*fft_obj(u)
        v = ifft_obj(uhat)

        toc1 = time.time()
        t1 += (toc1-tic1)

        # the strong-stability preserving runge-kutta substep.
        tic2 = time.time()
        v = ssp_rk2(v,hx,hy,dt,delta,m)
        toc2 = time.time()
        t2 += (toc2-tic2)

        u[:,:] = v[:,:]

        # capture average concentration and energy of system at current level.
        if i%lv==0:
            dim = 2
            u0 = u.real
            for d in range(dim):
                dd = dxy[d]
#                 print(dd.shape)
#                 print(u0.shape)
                u0 = integrate.trapz(u0,dd,axis=0)
            avg_conc = u0 / (Lx*Ly)
            avgConc[j] = avg_conc

            T1 = -1.j * k * np.fft.fft(u)
            T1 = np.real(np.fft.ifft(T1))
            T1 = 0.5*delta*abs(T1)**2

            T2 = 1./(delta*4.)*(np.real(u)**2-1.)**2

            up = np.real(u)-m
            for d in range(dim):
                dd = dxy[d]
                up = integrate.trapz(up,dd,axis=0)
            T3 = -1.*up
            T3 = 0.5*abs(T3)**2

            en = T1 + T2 + T3
            en = np.real(en)
            for d in range(dim):
                dd = dxy[d]
                en = integrate.trapz(en,dd,axis=0)
            energy[j] = en
            j += 1
            
            data = [avg_conc, en, u.real]
            io.write_data(i, data)
            
    wisdom = pyfftw.export_wisdom()

    # normalise energy to initial energy, possible due to unitlessness of the energy.
    energy /= energy[0]
    
    # print("End:")
    # plt.figure()
    # plt.imshow(u.real, extent=[0,Lx,0,Ly])
    # plt.colorbar()
    # plt.show()
    
    print("===================================\n")
    return avgConc, energy, u




@jit(nopython=True)
def shifty(A,num):
    E = np.empty_like(A)
    F = np.empty_like(A)
    
    E[:num] = A[-num:]
    E[num:] = A[:-num]
    F[:-num] = A[num:]
    F[-num:] = A[:num]

    return E + F

@jit(nopython=True)
def shiftx(A,num):
    G = np.empty_like(A)
    H = np.empty_like(A)
    
    G[:,:num] = A[:,-num:]
    G[:,num:] = A[:,:-num]
    H[:,:-num] = A[:,num:]
    H[:,-num:] = A[:,:num]
    return G + H

@jit(nopython=True)
def centralDiff(A,hx,hy,delta):
    A = A**3
#     D = -1.*shift(A,2)+ 16.*shift(A,1) - 60.*A
    Dx = -1. * shiftx(A,2) + 16. * shiftx(A,1) - 30.*A
    Dx /= 12.*hx**2
    Dy = -1. * shifty(A,2) + 16. * shifty(A,1) - 30.*A
    Dy /= 12.*hy**2
    return 1./delta * (Dx+Dy)

@jit(nopython=True)
def ssp_rk2(v,hx,hy,dt,delta,m):
    v1 = v + dt * (centralDiff(v,hx,hy,delta) + m)
    v = 1./2. * v + 1./2. * v1 + 1./2. * dt * (centralDiff(v1,hx,hy,delta) + m)
    return v

def ic_hex(Nx,Ny):
    arr = np.ones((Ny,Nx)) * -0.0
    mi = 3
    mi += 0
    eix = 6
    eix += 1
    eiy = 1
    eiy += 0
    bs = 19
    br = int(bs/2)
#     print(br)

    bidx0 = (slice(0,br),slice(eix,eix+bs))
    bidx1 = (slice(-br,None),slice(eix,eix+bs))

    bidx2 = (slice(0,br),slice(-eix-bs,-eix))
    bidx3 = (slice(-br,None),slice(-eix-bs,-eix))

    bidx4 = (slice(eiy+bs+mi,eiy+mi+2*bs),slice(-br,None))
    bidx5 = (slice(eiy+bs+mi,eiy+mi+2*bs),slice(0,br))
    bidx6 = (slice(eiy+bs+mi,eiy+mi+2*bs),slice(int(Nx/2) - br,int(Nx/2) + br+1))

    ball = np.ones((19,19)) * -0.0
    xx = np.arange(-9,10)
    X,Y = np.meshgrid(xx,xx)
    r = np.sqrt(X**2 + Y**2)
#     print(r)
    ball[np.where(r <= 9.)] = 1.0
    ball_left = ball[:,:br]
    ball_right = ball[:,br+1:]
    ball_top = ball[:br,:]
    ball_bottom = ball[br+1:,:]

    arr[bidx0] = ball_bottom * 0.1
    arr[bidx1] = ball_top * 0.1
    arr[bidx2] = ball_bottom * 0.1
    arr[bidx3] = ball_top * 0.1
    arr[bidx4] = ball_left * 0.1
    arr[bidx5] = ball_right * 0.1
    arr[bidx6] = ball
    return arr



# def lie(options, wisdom=None):
#     L = options['L']
#     Nx = options['Nx']
#     # eps = options['eps']
#     # sigma = options['sigma']
#     delta = options['delta']
#     m = options['m']
#     dt = options['dt']
#     it = options['it']
#     chiN = options['chiN']
#     prefix= options['prefix']
    
#     io = writer.writer(prefix,chiN,delta)
#     io.create_output_file(options)
    
#     h = L/Nx
#     dx = np.linspace(0,L,Nx)

#     # lv = levels to capture average concentration and energy changes.
#     it += 1
#     lv = int(it/100)

#     # define empty arrays for pyFFTW
#     u = pyfftw.empty_aligned((Nx,Nx), dtype='complex128', n=16)
#     uhat = pyfftw.empty_aligned((Nx,Nx), dtype='complex128', n=16)
#     v = pyfftw.empty_aligned((Nx,Nx), dtype='complex128', n=16)

#     # a constant.
#     tpiL = 2.*np.pi/L

#     # initial condition
#     C = 0.1 # scaling factor
#     r = np.random.uniform(-1.,1.,(Nx,Nx))

#     # zeroing the mean of the uniform distribution.
#     if r.mean()<0:
#         r -= r.mean()
#     else:
#         r += r.mean()
        
#     # generate wavenumber / frequency grid.
#     freqs = np.fft.fftfreq(Nx,(1.0/float(Nx)))
#     kx, ky = np.meshgrid(freqs,freqs)
#     k = kx + ky
#     k2 = kx**2+ky**2
#     k4 = k2**2
#     k = np.sqrt(k2)

#     # empty arrays to store avg. conc. and energy.
#     avgConc = np.zeros(int(it/lv+1))
#     energy = np.zeros(int(it/lv+1))

#     # if there is pyFFTW wisdom, import it.
#     if wisdom != None:
#         pyfftw.import_wisdom(wisdom)

#     # define pyFFTW plans.
#     fft_object = pyfftw.FFTW(u, uhat, threads=4, axes=(0,1))
#     ifft_object = pyfftw.FFTW(v, uhat, direction='FFTW_BACKWARD', threads=4, axes=(0,1))

#     j = 0
#     t1 = 0
#     t2 = 0

#     u[:,:] = m + C*r

#     # iterations.
#     for i in range(it+1):
#         # the Fourier-PS substep.
#         tt1 = time_stepping.fourier_ps(delta, k4, k2, tpiL, dt, u, fft_object, ifft_object, v, uhat)
#         t1 += tt1

#         # the strong-stability preserving runge-kutta substep.
#         tt2, v = time_stepping.ssp_rk2(v,h,Nx,delta,m,dt)
#         t2 += tt2

#         u[:,:] = v[:,:]

#         # capture average concentration and energy of system at current level.
#         if i%lv==0:
#             dim = 2
#             u0 = u.real

#             avg_conc = funcs.get_avg_conc(u0,dim,dx,L)
#             en = funcs.get_energy(k,tpiL,u0,delta,dx,m,dim)
#             avgConc[j] = avg_conc
#             energy[j] = en

#             j += 1
            
#             data = [avg_conc, en, u0]
#             io.write_data(i, data)
        
#         if i % 1000 == 0:
#             print("iteration = %i" %i)

#     wisdom = pyfftw.export_wisdom()


