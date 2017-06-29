import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../5_roots/'))
assert os.path.exists(sys.path[-1]), sys.path[-1]
import root_finding as roots
import scipy.optimize


def update(D, u, p):
    """
    This routine computes the eigenvalues of the matrix A,
    which is given as A = D + e(p*)u^T + u*e(p)^T.
    A is a size-n real symmetric matrix.
    D and u are modified if u[p] != 0.
    Arguments:
        - 'D': Diagonal matrix of size n x n
        - 'u': Vector
        - 'e(p)': Vector containing the unit vector in direction of p (1<p<n)
    Returns:
        - 'l': Vector containing the eigenvalues
        - 'V': Matrix containing the eigenvectors

    """
    # Initialization
    D = np.asarray(D)
    n, m = D.shape
    assert n == m, 'D is not symmetric!'
    assert np.logical_and(1 <= p, p <= n) , 'p is not within the range (1 < p < n)!'
    assert np.allclose(D, D.real), 'D is not real!'

    x0 = np.zeros(n, dtype='float64')
    dx = np.zeros(n, dtype='float64')
    
    # Assumption
    D[p, p] += 2 * u[p]
    u[p] = 0 

    # Fill
    for i in range(n):
        x0[i] = D[i, i]
    dx = np.copy(u)

    def eveq(l):
        """
        The secular equation for a symmetric row/column update
        """
        f = (- D[p, p] + l)
        for i in range(n):
            f *= (D[i, i] - l)
        for k in range(n):
            if k != p:
                t = u[k] * u[k]
                for i in range(n):
                    if i != k:
                        t *= (D[i, i] - l)
                f += t
        return f
    l = scipy.optimize.fsolve(eveq, x0)
    # l = roots.newton(eveq, x0, dx, eps=1e-10)
    return l, eveq
