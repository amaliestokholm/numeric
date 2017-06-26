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
