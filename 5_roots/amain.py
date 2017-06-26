import numpy as np
import globvar
import root_finding
import systems


def amain():
    """
    Test of the first root finding method
    """
    # Initialization
    x0 = np.array([3, -1], dtype='float64')
    dx = np.array([1e-9, 1e-9], dtype='float64')

    print('Check part A')
    print('Solve the system A*x*y = 1')
    print(', where exp(-x) + exp(-y) = 1 + 1/A, and A=10000.')
    print('Starting point: x0 =\n', x0)
    print('f(x0) =', systems.sys(x0))
    print('The roots using Newton method:')
    roots = root_finding.newton(systems.sys, x0, dx)
    print(roots)
    print('Number of calls:', globvar.ncalls)

    globvar.ncalls = 0
amain()
