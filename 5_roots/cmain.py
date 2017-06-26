import numpy as np
import globvar
import root_finding
import systems


def cmain():
    """
    Test of the third root finding method using quadratic linesearch
    """
    # Initialization
    x0 = np.array([3, -1], dtype='float64')
    dx = np.array([1e-9, 1e-9], dtype='float64')

    print('Check part A')
    print('Solve the system A*x*y = 1')
    print(', where exp(-x) + exp(-y) = 1 + 1/A, and A=10000.')
    print('Starting point: x0 =\n', x0)
    print('f(x0) =\n', systems.sys(x0))
    print('The roots using Newton method with quadratic linesearch:')
    roots = root_finding.newton_quad(systems.sys, x0, dx)
    print(roots)
    print('f(roots) =\n', systems.sys(roots))
    print('Number of calls:', globvar.ncalls)
    globvar.ncalls = 0
    dx = np.array([1e-9, 1e-9], dtype='float64')
    roots = root_finding.newton(systems.sys, x0, dx)
    print('Using the Newton method with linear linesearch, the roots were')
    print(roots)
    print('Using the method from part A, the number of calls were',
          globvar.ncalls)
    print('\n\n')

    # Initialization
    globvar.ncalls = 0
    x0 = np.array([1, 400], dtype='float64')
    dx = np.array([1e-9, 1e-9], dtype='float64')

    print('Check part B')
    print('Solve the Rosenbrock valley function')
    print('Starting point: x0 =\n', x0)
    print('f(x0) =\n', systems.rosenbrock(x0))
    print('The roots using Newton method with quadratic linesearch:')
    roots = root_finding.newton_quad(systems.rosenbrock, x0, dx)
    print(roots)
    print('f(roots) =\n', systems.rosenbrock(roots))
    print('Number of calls:', globvar.ncalls)
    globvar.ncalls = 0
    dx = np.array([1e-9, 1e-9], dtype='float64')
    roots = root_finding.newton(systems.rosenbrock, x0, dx)
    print('Using the Newton method with linear linesearch, the roots were')
    print(roots)
    print('Using the method from part A, the number of calls were',
          globvar.ncalls)
    print('\n\n')

    # Initialization
    globvar.ncalls = 0
    x0 = np.array([1, -1], dtype='float64')
    dx = np.array([1e-9, 1e-9], dtype='float64')

    print('Check part C')
    print('Solve the Himmelblau function')
    print('Starting point: x0 =\n', x0)
    print('f(x0) =\n', systems.himmelblau(x0))
    print('The roots using Newton method with quadratic linesearch:')
    roots = root_finding.newton_quad(systems.himmelblau, x0, dx)
    print(roots)
    print('f(roots) =\n', systems.himmelblau(roots))
    print('Number of calls:', globvar.ncalls)
    globvar.ncalls = 0
    dx = np.array([1e-9, 1e-9], dtype='float64')
    roots = root_finding.newton(systems.himmelblau, x0, dx)
    print('Using the Newton method with linear linesearch, the roots were')
    print(roots)
    print('Using the method from part A, the number of calls were',
          globvar.ncalls)
    print('\n\n')

    # Initialization
    globvar.ncalls = 0
    x0 = np.array([1, -1], dtype='float64')
    dx = np.array([1e-9, 1e-9], dtype='float64')

    print('Check part D: Make some interesting example')
    print('Solve the Matya function')
    print('Starting point: x0 =\n', x0)
    print('f(x0) =\n', systems.matya(x0))
    print('The roots using Newton method with quadratic linesearch:')
    roots = root_finding.newton_quad(systems.matya, x0, dx)
    print(roots)
    print('f(roots) =\n', systems.matya(roots))
    print('Number of calls:', globvar.ncalls)
    globvar.ncalls = 0
    dx = np.array([1e-9, 1e-9], dtype='float64')
    roots = root_finding.newton(systems.matya, x0, dx)
    print('Using the Newton method with linear linesearch, the roots were')
    print(roots)
    print('Using the method from part A, the number of calls were',
          globvar.ncalls)
    print('\n\n')

cmain()
