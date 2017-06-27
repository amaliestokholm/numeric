import numpy as np
import globvar


def rkstep23(F, x, y, h):
    """
    Embedded Runge-Kutta stepper of orders 3 and 2 usiung Bogacki-Shampine
    method
    Arguments:
        - 'F': Function containing the right-hand side
        - 'x': Position
        - 'y': Value of function
        - 'h': Step-size
    Returns:
        - 'yh': Estimate of function
        - 'errnorm': Norm of error on the step
    """
    # Arrange data in Butchers tableau
    c1 = 0
    c2 = 1 / 2.0
    c3 = 3.0 / 4.0
    c4 = 1

    # Make Runge-Kutta matrix
    a21 = 1 / 2.0
    a31 = 0
    a32 = 3 / 4.0
    a41 = 2 / 9.0
    a42 = 1 / 3.0
    a43 = 4 / 9.0

    # Determine the weights
    b1 = 2 / 9.0
    b2 = 1 / 3.0
    b3 = 4 / 9.0
    b4 = 0

    bs1 = 7 / 24.0
    bs2 = 1 / 4.0
    bs3 = 1 / 3.0
    bs4 = 1 / 8.0

    # Calculate ks
    k1 = h * F(x + c1 * h, y)
    k2 = h * F(x + c2 * h, y + a21 * k1)
    k3 = h * F(x + c3 * h, y + a31 * k1 + a32 * k2)
    k4 = h * F(x + c4 * h, y + a41 * k1 + a42 * k2 + a43 * k3)

    # Approximate next step and error
    yh = y + b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4
    yhs = y + bs1 * k1 + bs2 * k2 + bs3 * k3 + bs4 * k4
    err = yh - yhs
    errnorm = np.linalg.norm(err)

    return yh, errnorm


def rkstep3(F, x, y, h):
    """
    Non-embedded Runge-Kutta stepper of order 3, using Runge's error
    estimate.
    Arguments:
        - 'F': Function containing the right-hand side
        - 'x': Position
        - 'y': Value of function
        - 'h': Step-size
    Returns:
        - 'yh': Estimate of function
        - 'errnorm': Norm of error on the step
    """
    # Arrange data in Butchers tableau
    c1 = 0
    c2 = 1 / 2.0
    c3 = 3.0 / 4.0
    c4 = 1

    # Make Runge-Kutta matrix
    a21 = 1 / 2.0
    a31 = 0
    a32 = 3 / 4.0
    a41 = 2 / 9.0
    a42 = 1 / 3.0
    a43 = 4 / 9.0

    # Determine the weights
    b1 = 2 / 9.0
    b2 = 1 / 3.0
    b3 = 4 / 9.0
    b4 = 0

    # Initialization
    yl = np.copy(y)

    # Estimate the error from full_step and two_half_steps
    for l in range(3):
        if l == 1:
            yfull = np.copy(yh)
            h /= 2
        if l == 2:
            yl = np.copy(yh)
            x += h

        # Calculate ks
        k1 = h * F(x + c1 * h, yl)
        k2 = h * F(x + c2 * h, yl + a21 * k1)
        k3 = h * F(x + c3 * h, yl + a31 * k1 + a32 * k2)
        k4 = h * F(x + c4 * h, yl + a41 * k1 + a42 * k2 + a43 * k3)

        # Approximate next step and error
        yh = yl + b1 * k1 + b2 * k2 + b3 * k3 + b4 * k4

    err = (yfull - yh) / 7  # 7 = 2^p - 1, p = 3 Eq. [11]
    errnorm = np.linalg.norm(err)

    return yfull, errnorm


def rkdriver(F, a, b, ya, h, method, acc=1e-9, eps=1e-9, calls=False):
    """
    This function is an adaptive step-size driver routine, which advances
    the solution from a given value.
    Arguments:
        - 'F': Function to evolve
        - 'a': Starting point
        - 'b': End point
        - 'ya': Function value at a
        - 'h': Initial step-size
        - 'method': Name of the stepper function to be used
        - 'acc': Absolute precision
        - 'eps': Relative precision
    Returns:
        - 'xs': Stored data
        - 'ys': Stored data
    """
    # Identify which stepper to use
    if method.lower() in ['rkstep23', 'rk23']:
        stepper = rkstep23
    elif method.lower() in ['rkstep3', 'rk3']:
        stepper = rkstep3
    else:
        print('Unknown stepper chosen!')
        return

    # Initialize
    xs = np.array([a], dtype='float64')
    ys = np.array([ya], dtype='float64')
    cs = np.array([globvar.ncalls], dtype='float64')
    power = 0.25
    safety = 0.95

    # Start evolving
    while True:
        x = xs[-1]
        y = ys[-1]

        # Stopping criteria
        if x >= b:
            break

        # If the step ends outside the interval, step to the edge
        if (x + h) > b:
            h = b - x

        # Perform the step
        yh, err = stepper(F, x, y, h)

        # Calculate tolerance
        tol = (eps * np.linalg.norm(yh) + acc) * np.sqrt(h / (b - a))

        # Accept step if error is less than tolerance
        if err < tol:
            xs = np.append(xs, x + h)
            ys = np.vstack([ys, yh])
            cs = np.append(cs, globvar.ncalls)

        # Decrease the step if error is non-zero, else double it.
        if err > 0:
            h *= (tol / err) ** (power) * safety
        else:
            h *= 2

    if calls is True:
        return xs, ys, cs

    else:
        return xs, ys
