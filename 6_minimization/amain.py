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
    print('Number of steps used:', globvar.steps)
    print('\n\n')
amain()
