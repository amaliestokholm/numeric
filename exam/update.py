import numpy as np

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

    for i in range(n):
        A[p] = 

