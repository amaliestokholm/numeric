import numpy as np


def corput(n, b):
    """
    This routine calculates the van der Corput numbers of any base b according
    to chapter 9 the lecture notes.
    Arguments:
        - 'n': The index in the sequence.
        - 'b': The base
    Returns:
        - 'position': Value of n in the sequence of base b
    """
    # Initialization
    position = 0
    stepsize = 1 / b

    # Compute the base-b van der Corput number. This takes log_b(n) iterations
    while n > 0:
        q, r = divmod(n, b)
        position += r * stepsize
        n = q
        stepsize /= b
    return position


def halton(n, d):
    """
    This function computes the Halton sequence, which is a generalization of
    the van der Corput sequence to d-dimensional spaces.
    Arguments:
        - 'n': The index in the sequence
        - 'd': Number of dimensions
    Returns:
        - 'xs': A vector containing the position in the d-dimensional unit cube
    """
    # Assuming d < 20
    b = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31,
                  37, 41, 43, 47, 53, 59, 61, 67])
    assert d <= len(b), 'Not enough primes!'
    return np.array([corput(n, b[i]) for i in range(d)])


def lattice(n, d):
    """
    This function computes the lattice rules.
    Arguments:
        - 'n': The index in the sequence
        - 'd': Number of dimensions
    Returns:
        - 'xs': A vector containing the position in the d-dimensional unit cube
    """
    alpha = np.array([np.sqrt(13 * (i + 1)) % 1 for i in range(d)])
    return np.array([(n * alpha[i]) % 1 for i in range(d)])


def mc_quasi(F, a, b, N, sequence):
    """
    This routine is a Monte Carlo multi-dimensional integration using
    a quasi-random sampling.
    Arguments:
        - 'F': Function to integrate
        - 'a': Starting point in a vector
        - 'b': End point in a vector
        - 'N': Number of points to sample
        - 'sequence': Either 'lattice' or 'halton'
    """
    assert sequence in (lattice, halton)

    # Initialization
    vol = 1
    x = np.zeros(a.shape, dtype='float64')
    sum = 0

    # Calculate volumne
    vol = np.product(b - a)

    # Calculate Halton sampling
    for i in range(N):
        x = a + (b - a) * sequence(i, len(a))
        y = F(x)
        sum += y

    # Calculate mean
    mean = sum / N

    # Compute error and result
    res = vol * mean
    return res


def mc_quasi_err(F, a, b, N):
    """
    This routine computes the Monte Carlo integral using two different methods.
    Arguments:
        - 'F': Function to integrate
        - 'a': Starting point in a vector
        - 'b': End point in a vector
        - 'N': Number of points to sample
    Returns:
        - 'res': The mean of the result from both methods.
        - 'err': The error estimated as the difference between the two results.
    """
    res_halton = mc_quasi(F, a, b, N, halton)
    res_lattice = mc_quasi(F, a, b, N, lattice)

    # Error is estimated as the difference between the two methods
    err = abs(res_halton - res_lattice)

    # The result is given as the mean of the two results
    res = (res_halton + res_lattice) / 2

    return res, err
