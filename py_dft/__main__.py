import initial_conditions
from numerics import dft

import numpy as np
import time

if __name__ == '__main__':

    ic = initial_conditions.ic()

    # Nx, Ny = ic.options['Nx'], ic.options['Ny']
    # delta = ic.options['delta']
    # m = ic.options['m']
    # dt = ic.options['dt']
    # it = ic.options['it']
    # f = ic.options['f']
    # sigma = ic.options['sigma']

    for Li in ic.options['Ls']:
        Lx = Li[0]
        Ly = Li[1]
        rmin = Lx * np.sqrt(ic.options['f']) / 4.0

        print("Lx =", Lx)
        print("Ly =", Ly)

        options = ic.options
        options['Lx'] = Lx
        options['Ly'] = Ly
        options['rmin'] = rmin

        tic = time.time()
        avgConc, energy, u = dft.strang(options)
        toc = time.time()

        print("time taken: %.3fs" %(toc-tic))
        print("====================\n")

