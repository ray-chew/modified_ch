import initial_conditions
from numerics import dft

if __name__ == '__main__':
    init = initial_conditions.ic()
    dft.lie(init.options)

