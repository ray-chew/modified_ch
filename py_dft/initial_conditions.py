import numpy as np

# L = length of domain; Nx = spatial resolution;
# h = spatial stepsize; dx = 1D spatial grid;
# dt = temporal stepsize

class ic():
    L = 1.0
    Nx = 128
    dt = 0.0005

    chiN = 20.0
    N = 20
    l = 1./N
    f = 1.-0.3
    chi = chiN / N
    omega = (L**2)**(2/3)

    # define small parameters.
    delta = (3./64)**(1/3) * (1./ (chiN**(2/3 * f * (1 - f))) )
    delta = 0.31748021039

    # u-bar == m.
    m = 0.0

    # number of iterations
    it = 10000
    it += 1

    options = {
        'L' : L,
        'Nx' : Nx,
        'delta' : delta,
        'm' : m,
        'dt' : dt,
        'it' : it,
        'prefix' : 'dft',
        'N' : N,
        'l' : l,
        'f' : f,
        'chi' : chi,
        'chiN' : chiN,
        'omega' : omega
    }

    def __init__(self):
        self.options = self.options