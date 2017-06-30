import numpy as np
import mcquasi
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../9_mcinteg'))
from mcinteg import mc_plain


def xsiny(q):
    """
    Test function: x * sin(y)
    """
    return q[0] * np.sin(q[1])


def xyz(q):
    """
    Test function: 8 * x * y * z
    """
    return 8 * q[0] * q[1] * q[2]


def main():
    """
    Test of integrators routines
    """
    # Initialization
    a = np.array([0, np.pi])
    b = np.array([np.pi ** 2, 2 * np.pi])
    exact = - (np.pi ** 4)

    ns = np.arange(100, 2001, 100)

    # Compute the integrals
    print('# n\tactual error on plain\tactual error on quasi\t' +
          'estimated error on plain\testimated error on quasi')
    for n in ns:
        res_plain, err_plain = mc_plain(xsiny, a, b, n)
        res_quasi, err_quasi = mcquasi.mc_quasi_err(xsiny, a, b, n)
        acterr_plain = abs(res_plain - exact)
        acterr_quasi = abs(res_quasi - exact)
        print('%s\t%s\t%s\t%s\t%s' %
              (n, acterr_plain, acterr_quasi, err_plain, err_quasi))


    print('\n\n')
    # Initialization
    a = np.array([1, 2, 0])
    b = np.array([2, 3, 1])
    exact = 15

    # Compute the integrals
    print('# n\tactual error on plain\tactual error on quasi\t' +
          'estimated error on plain\testimated error on quasi')
    for n in ns:
        res_plain, err_plain = mc_plain(xyz, a, b, n)
        res_quasi, err_quasi = mcquasi.mc_quasi_err(xyz, a, b, n)
        acterr_plain = abs(res_plain - exact)
        acterr_quasi = abs(res_quasi - exact)
        print('%s\t%s\t%s\t%s\t%s' %
              (n, acterr_plain, acterr_quasi, err_plain, err_quasi))
main()
