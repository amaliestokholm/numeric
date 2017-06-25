import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../2_lineq/'))
assert os.path.exists(sys.path[-1]), sys.path[-1]
from lineqsolver import qr_gv_decomp as decomp
from lineqsolver import qr_gv_solve_return as solve
from lineqsolver import qr_gv_inverse as inverse
from lineqsolver import build_r
sys.path.append(os.path.join(os.path.dirname(__file__), '../3_eigen/'))
import eigen


def QR_lsfit(flist, x, y, dy):
    """
    This routine calculates the fit of a linear combinations of a series of
    functions \sum{c_i * f_i(x)} to the data (x, y) with error dy on y, using
    least mean squares and QR-decomposition with Given's rotation. 
    Arguments:
        - 'flist': List of functions to fit the data
        - 'x': Vector with x data
        - 'y': Vector with y data
        - 'dy': Vector with error on y data
    Returns:
        - 'c': Matrix containing the fitting coefficients
        - 'dc': Uncertainties on the coefficients
        - 'S': The covariance matrix
    """
    # Initialization
    n = len(x)
    m = len(flist)
    A = np.zeros((n, m), dtype='float64')
    b = np.zeros(n, dtype='float64')
    c = np.zeros(m, dtype='float64')
    dc = np.zeros(m, dtype='float64')
    Rinv = np.zeros((m, m), dtype='float64')

    # Fill A and c
    for i in range(n):
        # Weight data by error
        b[i] = y[i] / dy[i]

        for j in range(m):
            A[i, j] = flist[j](x[i]) / dy[i]

    # Decompose using Given's rotation and solve by in-place backsub
    decomp(A)
    x = solve(A, b)

    # Save it in c
    for i in range(m):
        c[i] = x[i]

    # Calculate the inverse
    inverse(build_r(A), Rinv)

    # Calculate the covariance matrix S
    S = np.dot(Rinv, np.transpose(Rinv))

    # Calculate the uncertainties on the coefficients from S
    for i in range(m):
        dc[i] = np.sqrt(S[i, i])

    return c, dc


def evalfunc(c, flist, x):
    """
    Evaluates the fit of a linear combination \sum{c_i * f_i(x)} at point x.
    """
    return sum([c[i] * flist[i](x) for i in range(len(flist))])


def singular_lsfit(flist, x, y, dy):
    """
    This routine takes the same input and returns the same as QR_lsfit, but the
    calculation is done using singular value decomposition
    Arguments:
        - 'flist': List of functions to fit the data
        - 'x': Vector with x data
        - 'y': Vector with y data
        - 'dy': Vector with error on y data
    Returns:
        - 'c': Matrix containing the fitting coefficients
        - 'dc': Uncertainties on the coefficients
        - 'S': The covariance matrix
    """
    # Initialization
    n = len(x)
    m = len(flist)
    A = np.zeros((n, m), dtype='float64')
    b = np.zeros(n, dtype='float64')
    c = np.zeros(m, dtype='float64')
    dc = np.zeros(m, dtype='float64')
    Rinv = np.zeros((m, m), dtype='float64')

    # Fill A and c
    for i in range(n):
        # Weight data by error
        b[i] = y[i] / dy[i]

        for j in range(m):
            A[i, j] = flist[j](x[i]) / dy[i]

    # Decompose using singular value decomposition 
    U, e, V = singular_decomp(A)
    c = singular_solve(U, e, V, b)

    # Calculate the covariance matrix S
    VDinv = np.zeros((m, m), dtype='float64')
    for i in range(m):
        dinv_i = 1 / (e[i] * e[i])
        for i in range(m):
            VDinv[j, i] = V[j, i] * dinv_i
    S = np.dot(VDinv, np.transpose(V))


    # Calculate the uncertainties on the coefficients from S
    for i in range(m):
        dc[i] = np.sqrt(S[i, i])

    return c, dc


def singular_decomp(A):
    """
    This routine computes the singular value decomposition using Jacobi's
    algorithm for dialonalization.
    The matrix A is decomposed into A = U * S * V^T, where 
    U = A * V * D^{-1/2} and S = D^{1/2}. D is a diagonal matrix containing
    the eigenvalues for A and V contains the corresponding eigenvectors.
    Arguments:
        - 'A': Input matrix
    Returns:
        - 'U': U = A * V * e^{-1/2}
        - 'e': A vector containing the eigenvalues
        - 'V': Matrix of the corresponding eigenvectors
    """
    # Initialization
    n, m = A.shape
    U = np.zeros((n, m), dtype='float64')

    # Diagonalization of A^T * A
    rot, e, V = eigen.diag(np.dot(np.transpose(A), A))

    # Calculate U
    U = np.dot(A, V)
    for i in range(m):
        e[i] = np.sqrt(e[i])
        U[:, i] /= e[i]

    return U, e, V


def singular_solve(U, e, V, b):
    """
    This function find the least-squares solution to (U * S * V^T) * x = b
    from singular value decomposition, where the diagonal of S is saved in e.
    New arguments:
        - 'b': Right-hand side of equation
    Returns:
        - 'x': Vector containing the solution
    """
    # Calculate S * V^T * x = U^T * b
    y = np.dot(np.transpose(U), b)

    for i in range(len(y)):
        y[i] /= e[i]

    # Solve
    x = np.dot(V, y)

    return x
