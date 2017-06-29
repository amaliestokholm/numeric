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
    update.update(D, u, p)
amain()
