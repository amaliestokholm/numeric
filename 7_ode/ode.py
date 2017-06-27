import numpy as np

def rkstep23():
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
    b1 = 1 / 3.0
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
    err = yh- yhs
    errnorm = np.linalg.norm(err)

    return yh, errnorm


