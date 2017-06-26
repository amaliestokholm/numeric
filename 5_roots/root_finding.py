import numpy as np
import os
import sys
import globvar
sys.path.append(os.path.join(os.path.dirname(__file__), '../2_lineq/'))
assert os.path.exists(sys.path[-1]), sys.path[-1]
from lineqsolver import qr_gv_decomp as decomp
from lineqsolver import qr_gv_solve as solve


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
    globvar.ncalls = 0
    x = np.copy(x0)
    n = len(x)
    J = np.zeros((n, n), dtype='float64')
    fx = f(x)

    # Begin root search
    while True:
        globvar.ncalls += 1

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

        # Begin backtracking linesearch
        lamb = 2.0
        while True: 
            lamb /= 2
            y = x + Dx * lamb
            fy = f(y)

            fynorm = np.linalg.norm(fy)
            fxnorm = np.linalg.norm(fx)

            if (fynorm < (1 - lamb / 2) * fxnorm) or (lamb < (1 / 128.0)):
                break

        # Save latest approximation
        x = y
        fx = fy

        Dxnorm = np.linalg.norm(Dx)
        fxnorm = np.linalg.norm(fx)
        dxnorm = np.linalg.norm(dx)
        if Dxnorm < dxnorm or fxnorm < eps:
            break

    return x


def newton_jacobian(f, x0, Jf, eps=1e-10):
    """
    This routine finds the root of a function using Newton's method
    using a user-supplied Jacobian.
    It uses Given's rotation for QR_decomposition.
    Arguments:
        - 'f': Function f(x) to root-find, which takes x as a vector.
        - 'x0': Vector containing starting point.
        - 'Jf': Function to calculate the Jacobian of f. 
        - 'eps': Desired tolerance.
    Returns:
        - 'x': Approximated root
    """
    # Initialization
    globvar.ncalls = 0
    x = np.copy(x0)
    n = len(x)
    J = np.zeros((n, n), dtype='float64')
    fx = f(x)

    # Begin root search
    while True:
        globvar.ncalls += 1

        # Calculate Jacobian
        J = Jf(x)

        # Decompose and solve using Given's rotations
        decomp(J)
        Dx = -fx
        solve(J, Dx)

        # Begin backtracking linesearch
        lamb = 2.0
        while True: 
            lamb /= 2
            y = x + Dx * lamb
            fy = f(y)

            fynorm = np.linalg.norm(fy)
            fxnorm = np.linalg.norm(fx)

            if (fynorm < (1 - lamb / 2) * fxnorm) or (lamb < (1 / 128.0)):
                break

        # Save latest approximation
        x = y
        fx = fy

        fxnorm = np.linalg.norm(fx)
        if fxnorm < eps:
            break

    return x


def newton_quad(f, x0, dx, eps=1e-10):
    """
    This routine finds the root of a function using Newton's method.
    It uses quadratic interpolation and a simple backtracking.
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
    globvar.ncalls = 0
    x = np.copy(x0)
    n = len(x)
    J = np.zeros((n, n), dtype='float64')
    fx = f(x)

    # Begin root search
    while True:
        globvar.ncalls += 1

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

        # Begin quadratic linesearch 
        lamb = 1.0
        y = x + Dx * lamb
        fy = f(y)

        fxnorm = np.linalg.norm(fx)
        fynorm = np.linalg.norm(fy)

        # Define the known values of the minimization function (Eq. 9)
        g0 = 0.5 * fxnorm ** 2
        dg0 = - fxnorm ** 2

        while (fynorm > (1 - lamb / 2) * fxnorm) and (lamb > (1 / 128.0)):
            glamb = 0.5 * fynorm ** 2
            c = (glamb - g0 - dg0 * lamb) / (lamb ** 2)

            # Update step
            lamb = - dg0 / (2 * c)
            y = x + Dx * lamb
            fy = f(y)
            fynorm = np.linalg.norm(fy)

        # Save latest approximation
        x = y
        fx = fy

        Dxnorm = np.linalg.norm(Dx)
        dxnorm = np.linalg.norm(dx)
        if Dxnorm < dxnorm or fxnorm < eps:
            break

    return x
