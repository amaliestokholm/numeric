import numpy as np


# Define dimensions of matrices
n = 3
m = 3
assert n >= m

def qr_gs_decomp(A, R):
    """
    This function performs an in-place modified GRam-Schmidt orthogonalization of an nxm matrix A.
    A turns into Q and the square mxm matrix R is computed.
    """
    n, m = A.shape
    p, q = R.shape
    assert n >= m
    assert p == q
    assert m == p

    for i in np.arange(m):



