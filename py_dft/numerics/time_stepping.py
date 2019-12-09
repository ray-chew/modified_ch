import numpy as np
import numexpr as ne
import time
from discretisation import schemes

def fourier_ps(delta,k4,k2,tpiL,dt,u,fft_object,ifft_object, v, vp):
    tic = time.time()
    vp[...] = np.exp(0.5*(-1.*delta*k4*tpiL**4+(1./delta)*k2*tpiL**2-1.)*dt)*fft_object(u)
    v[...] = ifft_object(vp)
    toc = time.time()
    t = (toc-tic)

    return t, v

def ssp_rk3(v,h,Nx,delta,m,dt):
    tic = time.time()
    Fu = schemes.centralDiffShift(v,h,Nx)
    v1 = ne.evaluate('v + dt * (1./delta * Fu + m)')
    Fu = schemes.centralDiffShift(v1,h,Nx)
    v2 = ne.evaluate('3./4. * v + 1./4. * v1 + 1./4. * dt * (1./delta * Fu + m)')
    Fu = schemes.centralDiffShift(v2,h,Nx)
    v = ne.evaluate('1./3. * v + 2./3. * v2 + 2./3. * dt * (1./delta * Fu + m)')
    toc = time.time()
    t = (toc-tic)

    return t, v