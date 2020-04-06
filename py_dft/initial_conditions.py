import numpy as np

# L = length of domain; Nx = spatial resolution;
# h = spatial stepsize; dx = 1D spatial grid;
# dt = temporal stepsize

class ic(object):
    Nx = 256
    Ny = 256
    dt = 0.001
    it = 75000

    m = -0.40
    f = 0.3
    delta = 0.15

    # Lys = np.arange(6.4,18.2,0.4)
    # Lys = np.arange(7.0,8.0,0.1)
    # Lys = np.arange(14.0,15.0,0.1)
    # Lys = np.array([15.0])
    Lys = np.array([14.85])
    # Lys = np.delete(Lys,[4,8])
    # Lys = np.concatenate((Lys,[7.4,7.8]))
    # Lys.sort(kind='mergesort')
    Lxs = Lys * 2.0 / np.sqrt(3)

    Ls = [[lx,ly] for lx,ly in zip(Lxs,Lys)]

    plot_initial = False
    ic_path = '/home/ray/git-projects/modified_ch/ic_64_64.h5'

    options = {
        'Ls' : Ls,
        'Nx' : Nx,
        'Ny' : Ny,
        'delta' : delta,
        'm' : m,
        'dt' : dt,
        'it' : it,
        'f' : f,
        'plot_initial' : plot_initial,
        'ic_path' : ic_path,
        'prefix' : 'dft'
    }

    def __init__(self):
        self.options = self.options
        