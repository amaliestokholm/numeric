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


def main():
    """
    Test of integrators routines
    """
    # Initialization
    a = np.array([0, np.pi])
    b = np.array([np.pi ** 2, 2 * np.pi])
    exact = - (np.pi ** 4)

    Ns = np.arange(100, 2001, 100)

    # Compute the integrals
    print('# N\tActual error on plain\tActual error on quasi\t' +
          'Estimated error on plain\tEstimated error on quasi')
    for N in Ns:
        res_plain, err_plain = mc_plain(xsiny, a, b, N)
        res_quasi, err_quasi = mcquasi.mc_quasi_err(xsiny, a, b, N)
        acterr_plain = abs(res_plain - exact)
        acterr_quasi = abs(res_quasi - exact)
        print('%s\t%s\t%s\t%s\t%s' %
              (N, acterr_plain, acterr_quasi, err_plain, err_quasi))
main()

