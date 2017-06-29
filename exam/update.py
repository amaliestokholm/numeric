import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../5_roots/'))
assert os.path.exists(sys.path[-1]), sys.path[-1]
import root_finding as roots



def update(D, u, p):
    """
    This routine computes the eigenvalues of the matrix A,
    which is given as A = D + e(p*)u^T + u*e(p)^T.
    A is a size-n real symmetric matrix.
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

    A = np.zeros((n, n), dtype='float64')
    x0 = np.zeros(n, dtype='float64')
    dx = np.zeros(n, dtype='float64')

    # Fill
    for i in range(n):
        x0[i] = D[i, i]
        dx[i] = u[i]

    def eveq(l):
        """
        The secular equation for a symmetric row/column update
        """
        f =(D[p] - l)
        for k in range(n):
            if k != p:
                f += (u[k] * u[k]) / (D[k] - l)
        return f
    l = roots.newton(eveq, x0, dx)
