import numpy as np
import integ
import sys
sys.setrecursionlimit(2000)
import globvar
from scipy import integrate


def f1(x):
    """
    Test function: sqrt(x)
    """
    globvar.ncalls += 1
    return np.sqrt(x)


def f2(x):
    """
    Test function: 1 / sqrt(x)
    """
    globvar.ncalls += 1
    return 1 / (np.sqrt(x))


def f3(x):
    """
    Test function: ln(x) / sqrt(x)
    """
    globvar.ncalls += 1
    return np.log(x) / np.sqrt(x)


def f4(x):
    """
    Test function: 4  sqrt(1 - (1 - x)^2)
    """
    globvar.ncalls += 1
    return 4 * np.sqrt(1 - (1 - x) ** 2)


def f5(x):
    """
    Test function: exp(-x)
    """
    globvar.ncalls += 1
    return np.exp(-(x ** 2))


def amain():
    """
    Test of the routines for adaptive integration
    """
    # Initialization
    acc = 1e-7
    eps = 1e-7
    a = 0
    b = 1
    exact1 = 2 / 3
    exact2 = 2
    exact3 = -4
    exact4 = np.pi

    print('Integrating sqrt(x) from (x) =\n',a)
    print('to (x) =\n', b)
    res1, err1, recmax1 = integ.integ_recursive(f1, a, b, acc, eps) 
    print('The integral is', res1)
    print('The error on the integral is', err1)
    print('The actual error is', abs(res1 - exact1))
    print('The number of function calls was', globvar.ncalls)
    print('The depth of the recursion was', recmax1)
    print('\n\n')

    globvar.ncalls = 0
    print('Integrating 1 / sqrt(x) from (x) =\n',a)
    print('to (x) =\n', b)
    res2, err2, recmax2 = integ.integ_recursive(f2, a, b, acc, eps) 
    print('The integral is', res2)
    print('The error on the integral is', err2)
    print('The actual error is', abs(res2 - exact2))
    print('The number of function calls was', globvar.ncalls)
    print('The depth of the recursion was', recmax2)
    print('\n\n')

    globvar.ncalls = 0
    print('Integrating ln(x) / sqrt(x) from (x) =\n',a)
    print('to (x) =\n', b)
    res3, err3, recmax3 = integ.integ_recursive(f3, a, b, acc, eps) 
    print('The integral is', res3)
    print('The error on the integral is', err3)
    print('The actual error is', abs(res3 - exact3))
    print('The number of recursions was', globvar.ncalls)
    print('The depth of the recursion was', recmax3)
    print('\n\n')

    globvar.ncalls = 0
    print('Integrating 4 sqrt(1 - (1 -x)^2) from (x) =\n',a)
    print('to (x) =\n', b)
    res4, err4, recmax4 = integ.integ_recursive(f4, a, b, acc, eps) 
    print('The integral is', res4)
    print('The error on the integral is', err4)
    print('The actual error is', abs(res4 - exact4))
    print('The number of recursions was', globvar.ncalls)
    print('The depth of the recursion was', recmax4)
    print('\n\n')
    
    # Initialization
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
    print('\n\n')
amain()
