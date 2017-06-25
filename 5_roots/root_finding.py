import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../2_lineq/'))
assert os.path.exists(sys.path[-1]), sys.path[-1]
from lineqsolver import qr_gv_decomp as decomp
from lineqsolver import qr_gv_solve as solve
sys.path.append(os.path.join(os.path.dirname(__file__), '../3_eigen/'))
assert os.path.exists(sys.path[-1]), sys.path[-1]
import eigen

def newton(f, x0, dx, eps=1e-10):
    """
    This routine finds the root of a function using Newton's method.
    It uses Given's rotation for QR_decomposition.
    Arguments:
        - 'f': Function f(x) to root-find, which takes x as a vector.
        - 'x0': Vector containing starting point.
        - 'dx': Vector containing the slope used in evalution of the Jacobian
        - 'eps': Desired tolerance.
    Returns:
        - 'x': Approximated root
    """
    # Initialization
    x = np.copy(x0)
    n = len(x)
    J =  np.zeros((n, n), dtype='float64')
    fx = f(x)

    # Begin root search
    while True:

        # Fill the Jacobian matrix
        for j in range(n):
            x[j] += dx[j]
            df = f(x) - fx

            for i in range(n):
                J[i, j] = df[i] / dx[j]

            x[j] -= dx[j]

        # Decompose and solve using Given's rotations
        decomp(J)
        Dx = -fx
        solve(J, Dx)

        #