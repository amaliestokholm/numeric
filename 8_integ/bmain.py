import numpy as np
import integ
import sys
sys.setrecursionlimit(2000)
import globvar
from scipy import integrate


def f5(x):
    """
    Test function: exp(-x)
    """
    globvar.ncalls += 1
    return np.exp(-(x ** 2))


def f6(x):
    """
    Test function: exp(-4x)
    """
    globvar.ncalls += 1
    return np.exp(-4 * x)


def f7(x):
    """
    Test function: exp(5x)
    """
    globvar.ncalls += 1
    return np.exp(5 * x)


def bmain():
    # Initialization
    acc = 1e-7
    eps = 1e-7
    a = np.NINF
    b = np.PINF
    exact5 = np.sqrt(np.pi)

    globvar.ncalls = 0
    print('Integrating exp(-x^2) from (x) =\n',a)
    print('to (x) =\n', b)
    res5, err5, recmax5 = integ.integ_recursive(f5, a, b, acc, eps)
    print('The integral is', res5)
    print('The error on the integral is', err5)
    print('The actual error is', abs(res5 - exact5))
    print('The number of recursions was', globvar.ncalls)
    print('The depth of the recursion was', recmax5)
    compare, cerr = integrate.quad(f5, a, b)
    print('Scipys integrate quad gives', compare) 
    print('with error', cerr)
    print('\n\n')

    # Initialization
    a = 0
    b = np.PINF
    exact6 = 1 / 4

    globvar.ncalls = 0
    print('Integrating exp(-4x) from (x) =\n',a)
    print('to (x) =\n', b)
    res6, err6, recmax6 = integ.integ_recursive(f6, a, b, acc, eps)
    print('The integral is', res6)
    print('The error on the integral is', err6)
    print('The actual error is', abs(res6 - exact6))
    print('The number of recursions was', globvar.ncalls)
    print('The depth of the recursion was', recmax6)
    compare, cerr = integrate.quad(f6, a, b)
    print('Scipys integrate quad gives', compare) 
    print('with error', cerr)
    print('\n\n')

    # Initialization
    a = np.NINF
    b = 0
    exact7 = 1 / 5

    globvar.ncalls = 0
    print('Integrating exp(5x) from (x) =\n',a)
    print('to (x) =\n', b)
    res7, err7, recmax7 = integ.integ_recursive(f7, a, b, acc, eps)
    print('The integral is', res7)
    print('The error on the integral is', err7)
    print('The actual error is', abs(res7 - exact7))
    print('The number of recursions was', globvar.ncalls)
    print('The depth of the recursion was', recmax7)
    compare, cerr = integrate.quad(f7, a, b)
    print('Scipys integrate quad gives', compare) 
    print('with error', cerr)
    print('\n\n')
bmain()
