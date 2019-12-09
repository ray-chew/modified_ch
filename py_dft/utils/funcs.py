import numpy as np
from scipy import integrate

def get_avg_conc(u0,dim,dx,L):
    # get average concentration.
    for _ in range(dim):
        u0 = integrate.trapz(u0,dx)
        
    avg_conc = u0 / L**2
    return avg_conc

def get_energy(k,tpiL,u,delta,dx,m,dim):
    assert(dim==2)
    # get energy.
    T1 = -1.j * k * tpiL * np.fft.fft2(u)
    T1 = np.real(np.fft.ifft2(T1))
    T1 = 0.5*delta*abs(T1)**2

    T2 = 1./(delta*4)*(np.real(u)**2-1.)**2

    up = np.real(u)-m
    for _ in range(dim):
        up = integrate.trapz(up,dx)
    T3 = -1.*up
    T3 = 0.5*abs(T3)**2

    en = T1 + T2 + T3
    en = np.real(en)
    for _ in range(dim):
        en = integrate.trapz(en,dx)
    return en