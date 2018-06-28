# modified_ch
Some code on the Ohta-Kawasaki / modified Cahn-Hilliard equation.

Todo:
1. Make schemes.py fully independent of the pyfftw unless otherwise specified.
2. Allow continuation of simulation from an imported array.
3. Remove the complex shifting of the ETDRK4 scheme from the loop (since it has to be initialised only once).
4. Make ETDRK4 scheme work for 2D, then put it into schemes.py.
5. Add and commit result files.
