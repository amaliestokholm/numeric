import numpy as np
import minimize
import systems
import globvar


def amain():
    """
    Test of the minimazation method
    """
    # Initialization
    x0 = np.array([-2, 2], dtype='float64')
    alpha = 1e-4

    print('Check part A')
    print('Minimize the Rosenbrock valley function')
    print('f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2')
    print('Starting point: x0 =\n', x0)
    print('f(x0) =\n', systems.rosenbrock(x0))
    print('The minimum is found as:')
    mini = minimize.newton_minimize(systems.rosenbrock,
                                    systems.grad_rosenbrock,
                                    systems.hessian_rosenbrock,
                                    x0, alpha)
    print(mini)
    print('f(min) =\n', systems.rosenbrock(mini))
    print('Number of steps used:', globvar.ncalls)
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
    mini = minimize.newton_minimize(systems.himmelblau,
                                    systems.grad_himmelblau,
                                    systems.hessian_himmelblau,
                                    x0, alpha)
    print(mini)
    print('f(min) =\n', systems.himmelblau(mini))
    print('Number of steps used:', globvar.ncalls)
    print('\n\n')
amain()
