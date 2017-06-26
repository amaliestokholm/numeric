import numpy as np
import least_squares as leastsq


# Define functions
def inv(x):
    return 1 / x


def const(x):
    return 1


def lin(x):
    return x


def cmain():
    """
    Test of the least-squares fitting
    """
    # Load data and fit
    x, y, dy = np.loadtxt('data.txt')
    flist = [inv, const, lin]
    c, dc = leastsq.singular_lsfit(flist, x, y, dy)

    # Prepare plot
    xs = 150
    xfit = np.linspace(x[0], x[-1], xs)

    for i in range(len(x)):
        print('%s\t%s\t%s' % (x[i], y[i], dy[i]))
    print('\n\n')
    for i in range(xs):
        yfit = leastsq.evalfunc(c, flist, xfit[i])
        yfit_l = leastsq.evalfunc(c - dc, flist, xfit[i])
        yfit_u = leastsq.evalfunc(c + dc, flist, xfit[i])
        print('%s\t%s\t%s\t%s' % (xfit[i], yfit, yfit_u, yfit_l))

cmain()
