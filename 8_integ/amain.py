import numpy as np
import integ
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
    return 4 * np.sqrt(1 - (1 - x) ** 2)


def amain():
    """
    Test of the routines for adaptive integration
    """
    # Initialization
    acc = 1e-3
    eps = 1e-3
    a1 = 0
    b1 = 1
    exact1 = 2 / 3

    print('Integrating sqrt(x) from (x) =\n',a1)
    print('to (x) =\n', b1)
    res1, err1 = integ.integ_recursive(f1, a1, b1, acc, eps) 
    print('The integral is ', res1)
    print('The error on the integral is ', err1)
    print('The actual error is ',abs(res1 - exact1))

amain()
