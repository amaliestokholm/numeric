import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../2_lineq/'))
assert os.path.exists(sys.path[-1]), sys.path[-1]
from lineqsolver import qr_gv_decomp as decomp
from lineqsolver import qr_gv_solve as solve
import globvar


def newton_minimize(f, grad, hessian, x0, alpha, eps=1e-10):
    """
    This routine finds the minimum of a function using Newton's method.
    The derivatives (gradiant and hessian) are supplied by the user.
    It uses Given's rotation for QR_decomposition.
    Arguments:
        - 'f': Function f(x) to root-find, which takes x as a vector.
        - 'grad': Function which calculates the gradient of f
        - 'hessian': Function which calculates the hessian matrix of f
        - 'x0': Vector containing starting point.
        - 'dx': Vector containing the slope used in evalution of the Jacobian
        - 'alpha': Scaling factor
        - 'eps': Desired tolerance.
    Returns:
        - 'x': Approximated root
    """
    # Initialization
    x = np.copy(x0)
    n = len(x)
    fx = f(x)
    df = grad(x)
    stepmax = 5000

    # Begin root search
    while True:
        globvar.steps += 1

        # Calculate Hessian matrix
        H = hessian(x)

        # Decompose and solve using Given's rotations
        decomp(H)
        Dx = -df
        solve(H, Dx)

        # Begin backtracking linesearch
        lamb = 2.0
        dfDx = np.dot(df, Dx)

        while True: 
            lamb /= 2
            y = x + Dx * lamb
            fy = f(y)

            # The Armijo condition
            if (fy < fx + alpha * lamb * dfDx) or (lamb < (1 / 128.0)): 
                break

        # Save latest approximation
        x = y
        fx = fy
        df = grad(x)

        dfnorm = np.linalg.norm(df)
        if (dfnorm < eps) or (globvar.steps > stepmax):
            break

    if globvar.steps > stepmax:
        print('\nToo many steps used!')
        exit()
    else:
        return x


def qnewton(f, grad, x0, alpha, eps=1e-10):
    """
    jf½
    """
