import numpy as np
import pyfftw
from discretisation import grids
from numerics import mde


def scft(options):
    Nx = options['Nx']
    fa = options['f']
    gam = options['gam']
    XN = options['chiN']
    it = options['it']
    V = options['L']

    # define number of monomers for each type.
    N = 20
    fb = 1.-fa
    Na = int(fa*N)
    Nb = int(fb*N)
    print('fa = %.3f' %fa)
    print('fb = %.3f' %fb)
    print('')

    # define copolymer sequence contour step-sizes (ds), and contour resolution (Ns).
    ds = 1./(N)
    Ns = int(N)

    # define time step-size (gamma_1 and gamma_2 of eqn 6.1a)
    eps = gam

    # define the Flory-Huggins parameter such that X_ab == constant always.
    Xab = XN/N
    
    # random seed for reproducibility.
#     np.random.seed(555)

    # initialize empty arrays and initial conditions.
    w = np.zeros((Nx,Nx,2))
    C = 0.1 # scaling factor
    w[:,:,:] = C*np.random.uniform(-1.,1.,(Nx,Nx,2))
    phi = np.zeros((Nx,Nx,2))
    q = np.zeros((Nx,Nx,Ns+1))
    qt = np.zeros((Nx,Nx,Ns+1))
    
    # import pyfftw wisdom if it is defined.
    # if wisdom != None:
    #     pyfftw.import_wisdom(wisdom)
    
    # initialise pyfftw arrays.
    qpp = pyfftw.empty_aligned((Nx,Nx), dtype='complex128', n=16)
    qppt = pyfftw.empty_aligned((Nx,Nx), dtype='complex128', n=16)
    qp = pyfftw.empty_aligned((Nx,Nx), dtype='complex128', n=16)
    qpt = pyfftw.empty_aligned((Nx,Nx), dtype='complex128', n=16)
    v = pyfftw.empty_aligned((Nx,Nx), dtype='complex128', n=16)
    vt = pyfftw.empty_aligned((Nx,Nx), dtype='complex128', n=16)
    
    # define pyfftw FFT plans.
    fft_object = pyfftw.FFTW(qpp, qp, threads=1, axes=(0,1))
    ifft_object = pyfftw.FFTW(qp, v, direction='FFTW_BACKWARD', threads=1, axes=(0,1))
    
    fft_object = pyfftw.FFTW(qppt, qpt, threads=1, axes=(0,1))
    ifft_object = pyfftw.FFTW(qpt, vt, direction='FFTW_BACKWARD', threads=1, axes=(0,1))

    # get the monomer configuration in the polymer.
    f, ft = mde.get_fselector(fa,fb,Ns)

    # get wavenumber and spatial grid.
    freqs = np.fft.fftfreq(Nx,(1.0/float(Nx)))
    kx, ky = np.meshgrid(freqs,freqs)
    k = kx + ky
    k2 = kx**2+ky**2
    
    # get spatial grid.
    dx = np.linspace(0,V,num=Nx)

    # calculate xi and phi at first iteration. Step 1 of algorithm.
    pressure = (w[:,:,0] + w[:,:,1])/2
    pressure = pressure-pressure.mean()
    xi = Xab*N-pressure

    # calculate the initial phi values.
    phi[:,:,0] = (w[:,:,0] - xi)/(Xab*N)
    phi[:,:,1] = (w[:,:,1] - xi)/(Xab*N)

    # for each subsequent iteration...
    for i in range(it):
        # store the old phi values for equation 6.1c, step 4 of the algorithm.
        phi_old = np.copy(phi)

        # define initial conditions (eqn 5.30)
        q[:,:,0] = 1
        qt[:,:,0] = 1

        # solve the MDE.
        for s in range(1,Ns+1):
            # solving MDE: substep 1 (eqn 6.5a)
            qpp[:,:] = np.exp(-0.5*ds*w[:,:,f[s]])*q[:,:,s-1]
            qppt[:,:] = np.exp(-0.5*ds*w[:,:,ft[s]])*qt[:,:,s-1]
            
            # solving MDE: substep 2 (eqn 6.5b)
            qp[:,:] = np.exp(-4*np.pi**2*ds*k2/(V**2))*fft_object(qpp)
            qpt[:,:] = np.exp(-4*np.pi**2*ds*k2/(V**2))*fft_object(qppt)
            
            # solving MDE: substep 3 (eqn 6.5c)
            q[:,:,s] = np.exp(-0.5*ds*w[:,:,f[s]])*ifft_object(qp).real
            qt[:,:,s] = np.exp(-0.5*ds*w[:,:,ft[s]])*ifft_object(qpt).real
        
        # get the phi values using helper function.
        phi = mde.solve_phi(q,qt,Nx,int(fa*Ns),ds,V)

        # get the pressure term (2nd term of xi, equation 5.47e).
        pressure = (w[:,:,0] + w[:,:,1])/2
        pressure = pressure-pressure.mean()

        # equation 6.1c.
        delta_phi = phi_old.sum(axis=2)-phi.sum(axis=2)
        delta_phi = delta_phi-delta_phi.mean()

        # update w(i) to w(i+1), equation 6.1a. This completes substep 4 of the algorithm.
        w[:,:,0] = w[:,:,0]+gam*(Xab*N*(phi[:,:,1]-0.5)+pressure-w[:,:,0])-eps*(delta_phi)
        w[:,:,1] = w[:,:,1]+gam*(Xab*N*(phi[:,:,0]-0.5)+pressure-w[:,:,1])-eps*(delta_phi)

        # print every 100th iteration.
        if i%100==0:
            print('it = %i' %i)
        
    # wisdom = pyfftw.export_wisdom()

    return phi