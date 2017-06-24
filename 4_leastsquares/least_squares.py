import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../2_lineq/'))
assert os.path.exists(sys.path[-1]), sys.path[-1]
import lineqsolver
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
        - 'S': The covariance matrix
        - 'dc': Uncertainties on the coefficients
    """
    # Initialization
    n = len(xs)
    m = len(funcs)
    A = np.zeros((n, m), dtype='float64')
    b = np.zeros(n, dtype='float64')
    c = np.zeros(m, dtype='float64')
    dc = np.zeros(m, dtype='float64')







QR_lsfit(1, 2, 3)
