import numpy as np

def get_k(Nx):
    freqs = np.fft.fftfreq(Nx,(1.0/float(Nx)))
    kx, ky = np.meshgrid(freqs,freqs)
    # k = kx + ky
    k2 = kx**2+ky**2
    return k2