# modified_ch
This repository contains code that solves:
1. The Ohta-Kawasaki / modified Cahn-Hilliard equation used in Density Functional Theory (DFT) and
2. the modified diffusion / Fokker-Planck equation used in Self-Consistent Field Theory (SCFT).

The chapters below refer to where the results appear in my Master's thesis (available upon request).

---

<p align="center">
<a href="https://opensource.org/licenses/MPL-2.0">
<img alt="License: MPL 2.0" src=https://img.shields.io/badge/License-MPL_2.0-brightgreen.svg>
</a>
</p>

---

- DFT numerics (Chapter 4, 7):
    1. *1D_Lie_SSP_RK2.ipynb* - 1D DFT scheme based on Lie-splitting involving a Fourier pseudospectral method, and strong-stability preserving 2nd-order accurate Runge-Kutta (SSP-RK3) time-stepping with a 2nd-order accurate central difference scheme.
    2. *2D_Strang_SSP_RK3.ipynb* - 2D DFT scheme based on a Strang-splitting involving two Fourier-PS methods, a 3rd-order SSP-RK3, and a 4th-order accurate central difference scheme.

- SCFT numerics (Chapters 6, 7):
    1. *1D_SCFT* - 1D SCFT scheme based on the paper published by [Rasmussen and Kalosakas (2002)](https://onlinelibrary.wiley.com/doi/abs/10.1002/polb.10238).
    2. *2D_SCFT* - 2D SCFT scheme.
    
- Benchmarking (Appendix F):
    1. *GPU_test.ipynb* - Compares computational costs of CPU- and GPU-based methods. Used alongside *EOC.ipynb* for calculating the order of convergence for the 2D DFT scheme in Chapter 4, Section 4.3.

- Misc:
    1. *lsa.nb* - Mathematica code for calculating the theoretical DFT order-disorder curves.
    2. *Pts1.csv*, *Pts2.csv* - Data points of the theoretical DFT curves.
    3. *SCFT_Us_XN_10-15_V_1-9* and *DFT_Us_delta_0.40-0.64_V_1.0-23.8* - Output of the data arrays used to plot the scatter plots of figure 7.5 (comparison of DFT and SCFT equilibrium homogeneous lamellar periodicity).
    4. *detect_peaks.py* - Peak counter from [here](https://github.com/demotu/BMC/blob/master/functions/detect_peaks.py).
    5. *uArrs* - Various output arrays used in determining the order of convergence of the 2D DFT numerical scheme in chapter 4, section 4.3.
    6. *ETDRK4* - Archived code. An attempt at the 1D exponential time-differencing 4th-order accurate Runge-Kutta method. See [Kassam and Trefethen (2005)](https://people.maths.ox.ac.uk/trefethen/publication/PDF/2005_111.pdf) for more details. The method works for small enough time steps.
    7. *Lie_Adam-Bashforth* - My first attempt at the DFT numerical scheme. It works in 1D, 2D, and 3D. However, the stability analysis of this scheme is not easy.
