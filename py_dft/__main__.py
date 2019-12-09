import initial_conditions
from numerics import dft

init = initial_conditions.ic()
dft.dft(init.options)

