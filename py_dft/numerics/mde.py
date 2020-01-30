import numpy as np
from scipy import integrate

def get_fselector(fa,fb,Ns):
    ka = int(fa*Ns)
    kb = int(fb*Ns)
    sf = np.concatenate((np.zeros(ka+1),np.ones(kb+1)),axis=0).astype(int)
    sft = np.concatenate((np.ones(kb+1),np.zeros(ka+1)),axis=0).astype(int)
    return sf,sft

def solve_phi(q,qt,Nx,Na,ds,L):
    ax1 = np.linspace(0.,L,num=Nx)
    phi = np.zeros((Nx,Nx,2))
    for j in range(Nx):
        for k in range(Nx):
            # here is where the flipping of the backward propagator occurs.
            tmp1 = q[j,k,:]*qt[j,k,::-1]
            
            # eqn 5.47a and eqn 5.47b
            phi[j,k,0] = integrate.trapz(tmp1[:Na+1],dx=ds)
            phi[j,k,1] = integrate.trapz(tmp1[Na:],dx=ds)
    Q = 1./(L*L) * integrate.trapz(integrate.trapz(q[:,:,-1],x=ax1),x=ax1)

    return phi/Q, Q