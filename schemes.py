import utils
import splittings
import numpy as np
import pyfftw
from scipy import integrate
import matplotlib.pyplot as plt

TWOPI = 2. * np.pi

# strang-splitting + adambashforth in 2D.
def strangSplitting(Nx,L,dt,it,eps,sigma,m,seed,wisdom,lv,dim=2):
    assert ((dim==1) or (dim==2) or (dim==3)), "Dimensions not 1, 2, or 3: Input dim=1, dim=2, or dim=3."

    # get 1D spatial grid
    dx = np.linspace(0,L,Nx)

    # get the shape of the order parameter array, e.g. (64x64) for 2D.
    shape = utils.ndArr(Nx,dim)

    # random initial condition.
    np.random.seed(seed)
    r = np.random.uniform(-1.,1.,shape)

    # zero the mean of the initial random value
    if r.mean()<0:
        r -= r.mean()
    else:
        r += r.mean()

    # a scaling factor
    C = 0.01

    # two-pi/L
    tpiL = TWOPI/L

    # if there is pyfftw wisdom, import it
    if wisdom != None:
        pyfftw.import_wisdom(wisdom)

    # initialize empty arrays for pyfftw
    u = pyfftw.empty_aligned(shape, dtype='complex128', n=16)
    v = pyfftw.empty_aligned(shape, dtype='complex128', n=16)
    v0 = pyfftw.empty_aligned(shape, dtype='complex128', n=16)
    v1 = pyfftw.empty_aligned(shape, dtype='complex128', n=16)

    # populate the order parameter array with the random initial values with zero-mean
    u = m+C*r

    # array to store values of u at every capture
    us = np.zeros((shape+(it/lv+1,)))

    # take a snapshot of avg. conc and energy at every 10 iterations
    avgConc = np.zeros(it/10+1)
    energy = np.zeros(it/10+1)

    # counter to populate lists of avg. conc and energy, and of array 'us'.
    j = 0
    jj = 0

    # path and filename to store arrays and diagrams
    imgDir = 'imgs'
    tag = ',strangFFTW_'+str(dim)+'D'
    path = utils.getPath(Nx,L,dt,it,eps,sigma,m,seed,lv,imgDir,tag)

    # get frequency grid
    k,k2,k4 = utils.wavenumber(Nx,dim)

    # run the number of iterations
    for i in xrange(it+1):
        u = splittings.strangFFTW(dt,eps,sigma,m,u,k,k2,k4,tpiL)

        # for every 10-iterations, capture the avg. conc and energy for error analysis
        if (i%10)==0:
            avgConc[j] = utils.avgConc(u,dx,L,dim)
            energy[j] = utils.energyFunc(u,eps,sigma,m,dx,k,tpiL,dim)
            j += 1

        # at every lv, capture a snapshot of the morphology and store an array of u
        if (i%lv)==0:
            u = np.real(u)

            # generate filename based on current iteration.
            filename  = 'it='+str(i)

            # generate diagram.
            if dim == 1:
                fig1 = plt.figure(figsize=(5,3))
                plt.xlabel('x')
                plt.ylabel('u(x)')
                plt.plot(dx,u)

            if dim == 2:
                fig1 = plt.figure(figsize=(3,3))
                plt.imshow(u,origin='left', extent=[0,L,0,L])

            # save diagrams in pdf.
            plt.savefig(path+'/'+filename+'.pdf')
            plt.close();

            # save order parameter array without imaginary part.
            uFloat = np.array(u,dtype='float64')
            utils.outputArray(uFloat,path,filename)

            # append order parameter array to us.
            us[...,jj] = u
            jj += 1

    # export, if there is pyfftw wisdom.
    wisdom = pyfftw.export_wisdom()

    # as energy is unitless, rescale energy as a fraction of the initial energy.
    energy /= energy[0]

    # save the average conc. and energy arrays.
    utils.outputArray(energy,path,'energy')
    utils.outputArray(avgConc,path,'avgConc')

    return us, avgConc, energy, wisdom
