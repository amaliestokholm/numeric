import numpy as np

# Define dimensions of matrices
n = 3
m = 3
assert n >= m

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

    Q = np.zeros((m, m),)

    # Gram-Schmidt orthogonalization
    for i in np.arange(m):
        R[i, i] = np.sqrt(np.dot(A[:, i], A[:, i]))
        Q[:, i] = A[:, i] / R[i, i]
        for j in np.arange(i+1, m):
            R[i, j] = np.dot(Q[:, i], A[:, j])
            A[:, j] = A[:, j] - Q[:, i] * R[i, j]


A = np.random.rand(n, m)
R = np.random.rand(m, m)

qr_gs_decomp(A, R)
