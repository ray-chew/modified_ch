import numpy as np
import pyfftw

# second order Strang-Splitting and Adam-Bashforth time-stepping based on pyFFTW.
def strangFFTW(dt,eps,sigma,m,u,k,k2,k4,tpiL):
    fft_obj1 = pyfftw.builders.fftn(u)
    v = np.exp(0.5*(-1.*eps**2*k4*tpiL**4+k2*tpiL**2-sigma)*dt)*fft_obj1()
    fft_obj2 = pyfftw.builders.ifftn(v)
    v = fft_obj2()
    v = v.real
    fft_obj3 = pyfftw.builders.fftn(v**3)
    v0 = -1. * k2 * tpiL**2 * fft_obj3()
    fft_obj4 = pyfftw.builders.ifftn(v0)
    v0 = fft_obj4()
    v0 = v0.real

    fft_obj7 = pyfftw.builders.fftn(u**3)
    v2 = -1. * k2 * tpiL**2 * fft_obj7()
    fft_obj8 = pyfftw.builders.ifftn(v2)
    v2 = fft_obj8()
    v2 = v2.real

    # Adam-Bashforth
    v0 = 0.5 * dt * (3.*(v0 + sigma*(m)) - (v2 + sigma*(m)) ) + v

    fft_obj5 = pyfftw.builders.fftn(v0)
    v1 = np.exp(0.5*(-1.*eps**2*k4*tpiL**4+k2*tpiL**2-sigma)*dt)*fft_obj5()
    fft_obj6 = pyfftw.builders.ifftn(v1)
    u = fft_obj6()
    return u


# second order Strang-Splitting and Adam-Bashforth time-stepping based on numpy.fft.
def strangNumpy(dt,eps,sigma,m,u,k,k2,k4,tpiL):
    v = np.exp(0.5*(-1.*eps**2*k4*tpiL**2+k2*tpiL-sigma)*dt)*fft.fftn(u)
    v = np.real(fft.ifftn(v))

    v0 = -1. * k2 * tpiL * fft.fftn(v**3)
    v0 = np.real(fft.ifftn(v0))

    # Adam-Bashforth
    v0 = 0.5 * dt * (3.*(v0 + sigma*(m)) - (v2 + sigma*(m)) ) + v

    v1 = np.exp(0.5*(-1.*eps**2*k4*tpiL**2+k2*tpiL-sigma)*dt)*fft.fftn(v0)
    u = np.real(fft.ifftn(v1))
    return u


# first order Lie-Splitting and explicit Euler time-stepping based on pyFFTW.
def lieFFTW(dt,eps,sigma,m,u,k,k2,k4,tpiL):
    fft_obj1 = pyfftw.builders.fftn(u)
    v = np.exp((-1.*eps**2*k4*(tpiL)**4+k2*tpiL**2-sigma)*dt)*fft_obj1()
    fft_obj2 = pyfftw.builders.ifftn(v)
    v = fft_obj2()
    fft_obj3 = pyfftw.builders.fftn(np.power(v,3))
    v0 = -1. * k2 * tpiL**2 * fft_obj3()
    fft_obj4 = pyfftw.builders.ifftn(v0)
    v0 = fft_obj4()

    # explicit euler.
    u = dt * (v0 + sigma*m) + v
    return u
