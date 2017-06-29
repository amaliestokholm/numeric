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
    p = 2
    print(D)
    print(u)
    update.update(D, u, p)

    A = np.copy(D)
    A[p] = u
    A[:, p] = u
    A[p, p] = D[p, p] + u[p] ** 2
    print(np.linalg.eig(A))
amain()
