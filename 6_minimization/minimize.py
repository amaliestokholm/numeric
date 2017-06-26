import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../2_lineq/'))
assert os.path.exists(sys.path[-1]), sys.path[-1]
from lineqsolver import qr_gv_decomp as decomp
from lineqsolver import qr_gv_solve as solve
import globvar


def newton_minimize(f, grad, hessian, x0, alpha=1e-4, eps=1e-10):
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
    globvar.ncalls = 0
    x = np.copy(x0)
    n = len(x)
    fx = f(x)
    df = grad(x)
    stepmax = 5000

    # Begin root search
    while True:
        globvar.ncalls += 1

        # Calculate Hessian matrix
        H = hessian(x)

        # Decompose and solve using Given's rotations
        decomp(H)
        Dx = -df
        solve(H, Dx)

        # Begin linear backtracking linesearch
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
        if (dfnorm < eps) or (globvar.ncalls > stepmax):
            break

    if globvar.ncalls > stepmax:
        print('\nToo many steps used!')
        exit()
    else:
        return x


def qnewton_minimize(f, grad, x0, alpha=1e-4, eps=1e-10):
    """
    This routine finds the minimum of a function using Quasi-Newton's method.
    The gradient is supplied by the user.
    Arguments:
        - 'f': Function f(x) to root-find, which takes x as a vector.
        - 'grad': Function which calculates the gradient of f
        - 'x0': Vector containing starting point.
        - 'dx': Vector containing the slope used in evalution of the Jacobian
        - 'alpha': Scaling factor
        - 'eps': Desired tolerance.
    Returns:
        - 'x': Approximated root
    """
    # Initialization
    globvar.ncalls = 0
    x = np.copy(x0)
    n = len(x)
    fx = f(x)
    df = grad(x)
    stepmax = 5000

    # Calculate the inverse of the Hessian matrix as I
    Hinv = np.identity(n)

    # Begin root search
    while True:
        globvar.ncalls += 1

        # Put derivatives into the inverse Hessian matrix
        Dx = np.dot(Hinv, -df)

        # Begin linear backtracking linesearch
        lamb = 2.0
        dfDx = np.dot(df, Dx)

        while True: 
            lamb /= 2
            y = x + Dx * lamb
            fy = f(y)

            # The Armijo condition
            if (fy < fx + alpha * lamb * dfDx):
                break

            # Reset if the update diverges
            if (lamb < (1 / 128.0)):
                Hinv = np.identity(n)
                break

        # Apply the update
        dfy = grad(y)
        z = dfy - df
        u = Dx * lamb - np.dot(Hinv, z)

        # SR1-update
        Hinv += np.outer(u, u) / np.dot(u, z)

        # Save latest approximation
        x = y
        fx = fy
        df = grad(x)

        dfnorm = np.linalg.norm(df)
        if (dfnorm < eps) or (globvar.ncalls > stepmax):
            break

    if globvar.ncalls > stepmax:
        print('\nToo many steps used!')
        exit()
    else:
        return x
