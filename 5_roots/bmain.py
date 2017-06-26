import numpy as np
import globvar
import root_finding
import systems


def bmain():
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
    print('f(x0) =\n', systems.sys(x0))
    print('The roots using Newton method:')
    roots = root_finding.newton_jacobian(systems.sys, x0,
                                         systems.jacobian_sys)
    print(roots)
    print('f(roots) =\n', systems.sys(roots))
    print('Number of calls:', globvar.ncalls)
    print('\n\n')

    # Initialization
    globvar.ncalls = 0
    x0 = np.array([1, 400], dtype='float64')
    dx = np.array([1e-9, 1e-9], dtype='float64')

    print('Check part B')
    print('Solve the Rosenbrock valley function')
    print('Starting point: x0 =\n', x0)
    print('f(x0) =\n', systems.rosenbrock(x0))
    print('The roots using Newton method:')
    roots = root_finding.newton_jacobian(systems.rosenbrock, x0,
                                         systems.jacobian_rosenbrock)
    print(roots)
    print('f(roots) =\n', systems.rosenbrock(roots))
    print('Number of calls:', globvar.ncalls)
    print('\n\n')

    # Initialization
    globvar.ncalls = 0
    x0 = np.array([1, -1], dtype='float64')
    dx = np.array([1e-9, 1e-9], dtype='float64')

    print('Check part C')
    print('Solve the Himmelblau function')
    print('Starting point: x0 =\n', x0)
    print('f(x0) =\n', systems.himmelblau(x0))
    print('The roots using Newton method:')
    roots = root_finding.newton_jacobian(systems.himmelblau, x0,
                                         systems.jacobian_himmelblau)
    print(roots)
    print('f(roots) =\n', systems.himmelblau(roots))
    print('Number of calls:', globvar.ncalls)
    print('\n\n')

    # Initialization
    globvar.ncalls = 0
    x0 = np.array([1, -1], dtype='float64')
    dx = np.array([1e-9, 1e-9], dtype='float64')

    print('Check part D: Make some interesting example')
    print('Solve the Matya function')
    print('Starting point: x0 =\n', x0)
    print('f(x0) =\n', systems.matya(x0))
    print('The roots using Newton method:')
    roots = root_finding.newton_jacobian(systems.matya, x0,
                                         systems.jacobian_matya)
    print(roots)
    print('f(roots) =\n', systems.matya(roots))
    print('Number of calls:', globvar.ncalls)
    print('\n\n')

bmain()
