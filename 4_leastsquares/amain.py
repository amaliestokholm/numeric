import numpy as np
import least_squares as leastsq


# Define functions
def inv(x):
    return 1 / x


def const(x):
    return 1


def lin(x):
    return x


def amain():
    """
    Test of the least-squares fitting
    """
    # Load data and fit 
    x, y, dy = np.loadtxt('data.txt')
    flist = [inv, const, lin]
    c, dc, S = leastsq.QR_lsfit(flist, x, y, dy)

    # Prepare plot

