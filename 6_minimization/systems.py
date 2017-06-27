import numpy as np


def rosenbrock(q):
    """
    The Rosenbrock valley function: (1 - x)^2 + 100(y - x^2)^2
    """
    x = q[0]
    y = q[1]
    return (1 - x) ** 2 + 100 * (y - x * x) ** 2


def grad_rosenbrock(q):
    """
    The first derivatives of the Rosenbrock valley function
    """
    # Initialize
    df = np.zeros(2, dtype='float64')
    x = q[0]
    y = q[1]

    # Calculate
    df[0] = 2 * (x - 1) - 400 * (y - x * x) * x
    df[1] = 200 * (y - x * x)
    return df


def hessian_rosenbrock(q):
    """
    The second derivatives of the Rosenbrock valley function
    """
    # Initialize
    H = np.zeros((2, 2), dtype='float64')
    x = q[0]
    y = q[1]

    # Calculate
    H[0, 0] = 2 - 400 * y + 1200 * x * x
    H[0, 1] = -400 * x
    H[1, 0] = -400 * x
    H[1, 1] = 200
    return H


def himmelblau(q):
    """
    The Himmelblau function: (x^2 + y - 11)^2 + (x + y^2 -7)^2
    """
    x = q[0]
    y = q[1]
    return (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2


def grad_himmelblau(q):
    """
    The first derivatives of the Himmelblau function
    """
    # Initialize
    df = np.zeros(2, dtype='float64')
    x = q[0]
    y = q[1]

    # Calculate
    df[0] = 4 * (x * x + y - 11) * x + 2 * (x + y * y - 7)
    df[1] = 2 * (x * x + y - 11) + 4 * (x + y * y - 7) * y
    return df


def hessian_himmelblau(q):
    """
    The second derivatives of the Himmelblau function
    in form of a Hessian matrix
    """
    # Initialize
    H = np.zeros((2, 2), dtype='float64')
    x = q[0]
    y = q[1]

    # Fill the Hessian
    H[0, 0] = 12 * x * x + 4 * y - 42
    H[0, 1] = 4 * x + 4 * y
    H[1, 0] = 4 * x + 4 * y
    H[1, 1] = 4 * x + 12 * y * y - 26
    return H


def decay(t, p):
    """
    The function fitting the activity of a radioactive substance.
    Arguments:
        - 't': Time
        - 'p': Parameters
    """
    return p[0] * np.exp(-t / p[1]) + p[2]


def grad_decay(t, p):
    """
    This function computes the gradient of decay.
    """
    g = np.zeros(3, dtype='float64')
    g[0] = np.exp(-t / p[1])
    g[1] = (p[0] * t * np.exp(-t / p[1])) / (p[1] * p[1])
    g[2] = 1
    return g

    
def master(t, y, s, p):
    """
    The master function or the squared loss to be minimized in order to
    determine the lifetime
    Arguments:
        - 't': Time
        - 'y': Measurement of activity
        - 's': Uncertainty on y
        - 'p': Parameters
    """
    n = len(t)
    sum = 0
    for i in range(n):
        sum += ((decay(t[i], p) - y[i]) ** 2) / (s[i] * s[i])
    return sum


def grad_master(t, y, s, p):
    """
    This function computes the gradient of the squared loss
    """
    n = len(t)
    q = np.zeros(3, dtype='float64')
    for i in range(n):
        dec = decay(t[i], p)
        gd = grad_decay(t[i], p)
        for j in range(3):
            q[j] = gd[j] * (2 * (dec - y[i]) / (s[i] * s[i]))
    return q
