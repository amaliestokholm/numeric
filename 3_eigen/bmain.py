import numpy as np
from eigen import eliminate as jacobi


def bmain():
    """
    Test of the jacobi diagonalization using the eigenvalue-by-eigenvalue
    """
    # Initialize
    np.random.seed(314)  # Initialize the random generator
    N = 3
    a = np.random.rand(N, N)
    A = a + a.T
    A_copy = np.copy(A)

    # Run test
    print('Testing Jacobi diagonalization eigenvalue-by-eigenvalue')
    print('A = \n', A)
    rot, e, V = jacobi(A)

    print('\nNumber of rotations used =', rot)
    print('\nEigenvalues of A are:\ne =\n', e)
    print('\nEigenvectors of A are:\nV =\n', V)
    print('\nA after the diagonalization is:\nA =\n', A)
    print('\nAre V.T A V = e?')
    e_check = np.dot(np.dot(np.transpose(V), A_copy), V)

    # Sort and flatten arrays, so the check is made easier
    e_check = np.sort(np.ravel(np.diagonal(e_check)))
    e = np.sort(np.ravel(e))
    print(np.allclose(e_check, e))

    print('\nAre the eigenvalues identical to the ones found using np.linalg?')
    w, v = np.linalg.eig(A_copy)
    w = np.ravel(np.sort(w))
    print(np.allclose(w, e))
bmain()
