import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../2_lineq/'))
assert os.path.exists(sys.path[-1]), sys.path[-1]
from lineqsolver import qr_gv_decomp as decomp
from lineqsolver import qr_gv_solve_return as solve
from lineqsolver import qr_gv_inverse as inverse
sys.path.append(os.path.join(os.path.dirname(__file__), '../3_eigen/'))
import a_eigen, b_eigen


def QR_lsfit(flist, x, y, dy):
    """
    This routine calculates the fit of a linear combinations of a series of
    functions \sum{c_i * f_i(x)} to the data (x, y) with error dy on y, using
    least mean squares and QR-decomposition with Given's rotation. 
    Arguments:
        - 'flist': List of functions to fit the data
        - 'x': Vector with x data
        - 'y': Vector with y data
        - 'dy': Vector with error on y data
    Returns:
        - 'c': Matrix containing the fitting coefficients
        - 'dc': Uncertainties on the coefficients
        - 'S': The covariance matrix
    """
    # Initialization
    n = len(x)
    m = len(flist)
    A = np.zeros((n, m), dtype='float64')
    b = np.zeros(n, dtype='float64')
    c = np.zeros(m, dtype='float64')
    dc = np.zeros(m, dtype='float64')
    Rinv = np.zeros((m, m), dtype='float64')

    # Fill A and c
    for i in range(n):
        # Weight data by error
        b[i] = y[i] / dy[i]

        for j in range(m):
            A[i, j] = flist[j](x[i]) / dy[i]

    # Decompose using Given's rotation and solve by in-place backsub
    decomp(A)
    x = solve(A, b)

    # Save it in c
    for i in range(m):
        c[i] = b[i]

    # Calculate the inverse
    inverse(A, Rinv)

    # Calculate the covariance matrix S
    S = np.dot(Rinv, np.transpose(Rinv))

    # Calculate the uncertainties on the coefficients from S
    for i in range(m):
        dc[i] = np.sqrt(S[i, i])

    return c, dc


def evalfunc(c, flist, x):
    """
    Evaluates the fit of a linear combination \sum{c_i * f_i(x)} at point x.
    """
    return sum([c[i] * flist[i](x) for i in range(len(flist))])
