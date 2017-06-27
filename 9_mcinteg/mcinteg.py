import numpy as np


def mc_plain(F, a, b, N):
    """
    This routine is a plain Monte Carlo multi-dimensional integration
    Arguments:
        - 'F': Function to integrate
        - 'a': Starting point in a vector
        - 'b': End point in a vector
        - 'N': Number of points to sample
    Returns:
        - 'res': Result of the integration
        - 'err': Error on result
    """
    # Initialization
    vol = 1
    x = np.zeros(a.shape, dtype='float64')
    sum = 0
    sumsq = 0

    # Calculate volumne
    for i in range(len(a)):
        vol *= b[i] - a[i]

    # Sample N points
    for i in range(N):
        x = np.random.uniform(a, b, len(a))  # just as randomx 
        y = F(x)
        sum += y
        sumsq += y * y

    # Calculate mean and variance
    mean = s / N
    sigma = np.sqrt(sumsq / N - mean * mean)

    # Compute error and result
    err = vol * sigma / np.sqrt(N)
    res = vol * mean
    return res, err
