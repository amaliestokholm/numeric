import numpy as np


def backsubstitution(R, b):
    """
    This function performs an in-place back-substitution
    """
    n, m = R.shape
    list = range(m)
    for i in list[::-1]:
        # b is substituted for the solution to the system
        b[i] /= R[i, i]
        for j in range(i+1, m):
            b[i] -= b[j] * R[i, j] / R[i, i]


def qr_gs_decomp(A, R):
    """
    This function performs an in-place modified Gram-Schmidt
    orthogonalization of an nxm matrix A.
    A turns into Q and the square mxm matrix R is computed.
    """
    n, m = A.shape
    p, q = R.shape
    assert n >= m
    assert p == q
    assert m == p

    # Gram-Schmidt orthogonalization
    for i in range(m):
        R[i, i] = np.sqrt(np.dot(A[:, i], A[:, i]))
        # A is substituted for Q with orthogonalized vectors.
        A[:, i] /= R[i, i]
        for j in range(i+1, m):
            R[i, j] = np.dot(A[:, i], A[:, j])
            A[:, j] -= A[:, i] * R[i, j]
            R[j, i] = 0


def qr_gs_solve(Q, R, b):
    """
    This function solves the triangular system QR*x = b by substituting b
    with Q^Tb and doing back-substitution. b is substituted with
    the solution x.
    """
    b[:] = np.dot(Q.T, b)
    backsubstitution(R, b)


def qr_gs_inverse(Q, R, b):
    """
    This function calculates the inverse of the matrix Q into the matrix b
    """
    n, m = Q.shape
    assert b.shape == (m, m)
    eye = np.identity(m)
    for i in range(m):
        b[:, i] = eye[:, i]
        qr_gs_solve(Q, R, b[:, i])


def qr_gv_decomp(A):
    """
    This function decomposes a matrix A using the Givens rotations
    """
    n, m = A.shape
    assert n >= m
    for p in range(m):
        for q in range(p+1, n):
            theta = np.arctan2(A[q, p], A[p, p])
            for i in range(p, m):
                xp = A[p, i]
                xq = A[q, i]
                A[p, i] = xp * np.cos(theta) + xq * np.sin(theta)
                A[q, i] = -xp * np.sin(theta) + xq * np.cos(theta)
            # Store the rotation angles in the corresponding place in matrix A
            A[q, p] = theta


def qr_gv_solve(A, b):
    """
    This function solves Ax = b if A has been decomposed
    """
    n, m = A.shape
    for p in range(m):
        for q in range(p+1, n):
            theta = A[q, p]
            xp = np.copy(b[p])
            xq = np.copy(b[q])
            b[p] = xp * np.cos(theta) + xq * np.sin(theta)
            b[q] = - xp * np.sin(theta) + xq * np.cos(theta)
    backsubstitution(A, b)


def qr_gv_inverse(A, b):
    """
    This function calculates the inverse of A into b. A is a symmetric matrix.
    See also qr_gv_rinverse.
    """
    n, m = A.shape
    assert n == m
    b[:] = np.identity(m)
    for i in range(m):
        qr_gv_solve(A, b[:, i])


def build_r(A):
    """
    This function calculates R from A = QR
    """
    n, m = A.shape
    R = np.zeros((m, m), dtype='float64')

    for i in range(m):
        for j in range(i + 1):
            R[j, i] = A[j, i]
    return R


def qr_gv_rinverse(A, Rinv):
    """
    This routine calculates the inverse of R from A = QR.
    """
    n, m = A.shape
    assert Rinv.shape == (m, m)
    Rinv = np.identity(m)
    for i in range(m):
        backsubstitution(A, Rinv[:, i])


def qr_gv_solve_return(A, b):
    """
    This function is almost identical to qr_gv_solve
    """
    v = b.copy()
    qr_gv_solve(A, v)
    x = np.zeros(A.shape[1])
    for i in range(x.size):
        x[i] = v[i]
    return x
