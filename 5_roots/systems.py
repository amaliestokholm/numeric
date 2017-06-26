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
