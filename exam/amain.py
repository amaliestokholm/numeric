import numpy as np
import update

# Chose a specific generator in order to be consistent
np.random.seed(10)

def amain():
    """
    Test function for rotuine
    """
    # Initialization
    n = 4
    p = 2
    diag = np.random.rand(n)
    D = np.diag(diag)
    u = np.random.rand(n)
    D_copy = np.copy(D)
    u_copy = np.copy(u)
    
    print('Test of symmetric row/column update of a size-n symmetric')
    print('eigenvalue problem')
    print('The matrix to diagonalize is given in the form')
    print('A = D + e(p)*u^T + u*e(p)^T')
    print('where D is a diagonal matrix, which in this test is\n', D)
    print('where u is a vector, which in this test is\n', u)
    print('where e(p) is a unit vector in the direction p, where 1<=p<n.') 
    l, eveq = update.update(D, u, p)
    print('\n')
    print('The eigenvalues found are\n', l)
    print('And the secular equation evaluated in the eigenvalues are\n',
          eveq(l))

    A = np.copy(D_copy)
    A[p] += u_copy
    A[:, p] += u_copy
    test_lambda, test_vectors = np.linalg.eig(A)

    print('\n')
    print('The eigenvalues found by numpy are', test_lambda)
    print('The secular equation evaluated in the numpy eigenvalues are',
          eveq(test_lambda))
    print('The difference between the two solutions are',
          np.sort(test_lambda) - np.sort(l))

amain()
