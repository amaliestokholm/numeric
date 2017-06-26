import numpy as np
import globvar


def sys(q):
    """
    System of equations with A = 10000.
    A * x * y = 1
    exp(-y) + exp(-y) = 1 + 1/A
    Returns:

    """
    # Count the number of calls to function
    globvar.ncalls += 1

    # Initialize
    z = np.zeros(2, dtype='float64')
    x = q[0]
    y = q[1]
    A = 10000.0

    # Calculate
    z[0] = A * x * y - 1
    z[1] = np.exp(-x) + np.exp(-y) - 1 - (1 / A)
    return z


def jacobian_sys(q):
    """
    Derivatives of sys in form of the Jacobian
    """
    # Initialize
    J = np.zeros((2, 2), dtype='float64')
    x = q[0]
    y = q[1]
    A = 10000.0

    # Fill the Jacobian
    J[0, 0] = A * y
    J[0, 1] = A * x
    J[1, 0] = -np.exp(-x)
    J[1, 1] = -np.exp(-y)
    return J


def rosenbrock(q):
    """
    The Rosenbrock's valley function (or the gradient of it)
    """
    # Count the number of calls to function
    globvar.ncalls += 1

    # Initialize
    z = np.zeros(2, dtype='float64')
    x = q[0]
    y = q[1]
    
    # Calculate
    z[0] = 2 * (x - 1) - 400 * (y - x * x) * x
    z[1] = 200 * (y - x * x)
    return z


def jacobian_rosenbrock(q):
    """
    Derivatives of Rosenbrock's valley function in form of the Jacobian
    """
    # Initialize
    J = np.zeros((2, 2), dtype='float64')
    x = q[0]
    y = q[1]

    # Fill the Jacobian
    J[0, 0] = 2 - 400 * y + 1200 * x * x
    J[0, 1] = -400 * x
    J[1, 0] = -400 * x
    J[1, 1] = 200
    return J


def himmelblau(q):
    """
    The Himmelblau function (or the gradient of it)
    """
    # Count the number of calls to function
    globvar.ncalls += 1

    # Initialize
    z = np.zeros(2, dtype='float64')
    x = q[0]
    y = q[1]
    
    # Calculate
    z[0] = 4 * (x * x + y - 11) * x + 2 * (x + y * y - 7) 
    z[1] = 2 * (x * x + y - 11) + 4 * (x + y * y -7) * y
    return z


def jacobian_himmelblau(q):
    """
    Derivatives of Himmelblau function in form of the Jacobian
    """
    # Initialize
    J = np.zeros((2, 2), dtype='float64')
    x = q[0]
    y = q[1]

    # Fill the Jacobian
    J[0, 0] = 12 * x * x + 4 * y - 42 
    J[0, 1] = 4 * x + 4 * y
    J[1, 0] = 4 * x + 4 * y
    J[1, 1] = 4 * x + 12 * y *y - 26
    return J


def matya(q):
    """
    Matyas function 
    """
    # Count the number of calls to function
    globvar.ncalls += 1

    # Initialize
    z = np.zeros(2, dtype='float64')
    x = q[0]
    y = q[1]
    
    # Calculate
    z[0] = 2 * 0.26 * x - 0.48 * y 
    z[1] = 2 * 0.26 * y - 0.48 * x
    return z


def jacobian_matya(q):
    """
    Derivatives of Matya function in form of the Jacobian
    """
    # Initialize
    J = np.zeros((2, 2), dtype='float64')
    x = q[0]
    y = q[1]

    # Fill the Jacobian
    J[0, 0] = 0.52
    J[0, 1] = -0.48 
    J[1, 0] =  -0.48
    J[1, 1] = 0.52 
    return J
