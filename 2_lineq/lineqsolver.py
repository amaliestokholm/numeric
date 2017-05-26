import numpy as np


def backsubstitution(R, b):
    """
    This function performs an in-place back-substitution
    """
    n = len(R[:, 0])
    list = np.arange(n)
    for i in list[::-1]:
        # b is substituted for the solution to the system
        b[i] /= R[i, i]
        for j in np.arange(i+1, n):
            b[i] -= b[j] * R[i, j] / R[i, i]


def qr_gs_decomp(A, R):
    """
    This function performs an in-place modified Gram-Schmidt orthogonalization of an nxm matrix A.
    A turns into Q and the square mxm matrix R is computed.
    """
    n, m = A.shape
    p, q = R.shape
    assert n >= m
    assert p == q
    assert m == p

    # Gram-Schmidt orthogonalization
    for i in np.arange(m):
        R[i, i] = np.sqrt(np.dot(A[:, i], A[:, i]))
        # A is substituted for Q with orthogonalized vectors.
        A[:, i] /= R[i, i]
        for j in np.arange(i+1, m):
            R[i, j] = np.dot(A[:, i], A[:, j])
            A[:, j] -= A[:, i] * R[i, j]
            R[j, i] = 0


def qr_gs_solve(Q, R, b):
    """
    This function solves the triangular system QR*x = b
    """
    