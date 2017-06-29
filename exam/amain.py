import numpy as np
import update

# Chose a specific generator in order to be consistent
np.random.seed(8)

def amain():
    """
    Test function for rotuine
    """
    # Initialization
    n = 4
    diag = np.random.rand(n)
    D = np.diag(diag)
    u = np.random.rand(n)
    D_copy = np.copy(D)
    u_copy = np.copy(u)
    p = 2
    print(D)
    print(u)
    l, eveq = update.update(D, u, p)

    print(l)
    print(eveq(l))

    A = np.copy(D_copy)
    A[p] += u_copy
    A[:, p] += u_copy
    test_lambda, test_vectors = np.linalg.eig(A)

    print('Print from numpy')
    print(test_lambda)
    print(eveq(test_lambda))

    print('Difference')
    print(np.sort(test_lambda) - np.sort(l))

amain()
