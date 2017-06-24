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
    c, dc = leastsq.QR_lsfit(flist, x, y, dy)
    n = len(flist)

    # Prepare plot
    xs = 150
    xfit = np.linspace(x[0], x[-1], xs)
    yfit = np.array([leastsq.evalfunc(c, flist, xfit[i]) for i in range(n)])
    yfit_l = np.array([leastsq.evalfunc(c - dc, flist, xfit[i]) for i in range(n)])
    yfit_u = np.array([leastsq.evalfunc(c + dc, flist, xfit[i]) for i in range(n)])

    for i in range(n):
        print(xfit[i], yfit[i], yfit_l[i], yfit_u[i], sep='\t')

amain()

