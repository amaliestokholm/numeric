import numpy as np
import minimize
import systems
import globvar
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../5_roots'))
import root_finding as root


def bmain():
    """
    Test of the minimization method using the Quasi-Newton method
    """
    # Initialization
    globvar.ncalls = 0
    x0 = np.array([-2, 2], dtype='float64')
    dx = np.array([1e-9, 1e-9], dtype='float64')
    alpha = 1e-4
    
    print('Testing the quasi-Newton method with Broydens update')
    print('Check part A')
    print('Minimize the Rosenbrock valley function')
    print('f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2')
    print('Starting point: x0 =\n', x0)
    print('f(x0) =\n', systems.rosenbrock(x0))
    print('The minimum is found as:')
    mini = minimize.qnewton_minimize(systems.rosenbrock,
                                     systems.grad_rosenbrock,
                                     x0, alpha)
    print(mini)
    print('f(min) =\n', systems.rosenbrock(mini))
    print('Number of steps used:', globvar.ncalls)
    print('We can now compare the number of steps used by different methods')
    globvar.ncalls = 0
    mini = minimize.newton_minimize(systems.rosenbrock,
                                    systems.grad_rosenbrock,
                                    systems.hessian_rosenbrock,
                                    x0, alpha)
    print('Number of steps used in the Newton minimization (A) is',
          globvar.ncalls)

    globvar.ncalls = 0
    mini = root.newton(systems.grad_rosenbrock, x0, dx)
    print('Number of steps used in the Newton root-finding is',
          globvar.ncalls)

    globvar.ncalls = 0
    mini = root.newton_jacobian(systems.grad_rosenbrock, x0,
                                systems.hessian_rosenbrock)
    print('Number of steps using Newton root-finding with the Jacobian is',
          globvar.ncalls)
    print('\n\n')

    # Initialization
    x0 = np.array([2.5, 2.0], dtype='float64')
    alpha = 1e-4
    globvar.ncalls = 0

    print('Check part B')
    print('Minimize the Himmelblau function')
    print('f(x, y) = (x^2 + y -11)^2 + (x + y^2 - 7)^2') 
    print('Starting point: x0 =\n', x0)
    print('f(x0) =\n', systems.himmelblau(x0))
    print('The minimum is found as:')
    mini = minimize.qnewton_minimize(systems.himmelblau,
                                    systems.grad_himmelblau,
                                    x0, alpha)
    print(mini)
    print('f(min) =\n', systems.himmelblau(mini))
    print('Number of steps used:', globvar.ncalls)
    print('\n\n')
    print('We can now compare the number of steps used by different methods')
    globvar.ncalls = 0
    mini = minimize.newton_minimize(systems.himmelblau,
                                    systems.grad_himmelblau,
                                    systems.hessian_himmelblau,
                                    x0, alpha)
    print('Number of steps used in the Newton minimization (A) is',
          globvar.ncalls)

    globvar.ncalls = 0
    mini = root.newton(systems.grad_himmelblau, x0, dx)
    print('Number of steps used in the Newton root-finding is',
          globvar.ncalls)

    globvar.ncalls = 0
    mini = root.newton_jacobian(systems.grad_himmelblau, x0,
                                systems.hessian_himmelblau)
    print('Number of steps using Newton root-finding with the Jacobian is',
          globvar.ncalls)
    print('\n\n')
bmain()
