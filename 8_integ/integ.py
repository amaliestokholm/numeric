import numpy as np
import globvar


def integ_recursive(F, a, b, acc=1e-4, eps=1e-4):
    """
    This routine computes recursive adaptive integration using closed
    QUADratures. It uses a 4th order trapezium evaluation rule with
    2nd order rectangular rule for error estimation.
    Arguments:
        - 'F': Function to be integrated
        - 'a': Starting point
        - 'b': End point
        - 'acc': Absolute error
        - 'eps': Relative error
    Returns:
        - 'Q': The integral of F from a to b
        - 'err': An estimate of the error on Q
    """
    assert a < b

    # Perform integration
    h = b - a
    x1 = 0
    x4 = 1
    f1 = F(a + h * x1)
    f4 = F(a + h * x4)

    Q, err = quad_integrator(F, a, b, f1, f4, nrec=0, acc=acc, eps=eps)
    return Q, err


def quad_integrator(F, a, b, f1, f4, nrec, acc, eps):
    """
    This function integrates the function from a to b using previously
    determined points. If the error is too big, the interval is
    sub-divided and the integrator calls itself.
    New arguments:
        - 'f1': Estimated value of F(a)
        - 'f4': Estimated value of F(b)
        - 'nrec': The current level of recursion
    """
    assert nrec < globvar.nrecmax

    # Estimate to two points in-between
    h = b - a
    x2 = 1 / 3.0
    x3 = 2 / 3.0
    f2 = F(a + h * x2)
    f3 = F(a + h * x3)

    # Weights of 4th order trapezium rule
    w1 = 1 / 8.0
    w2 = 3 / 8.0
    w3 = 3 / 8.0
    w4 = 1 / 8.0

    # Weights of 4th order trapezium rule
    v1 = 1 / 4.0
    v2 = 1 / 4.0
    v3 = 1 / 4.0
    v4 = 1 / 4.0

    # Calculate
    Q = h * (w1 * f1 + w2 * f2 + w3 * f3 + w4 * f4)
    q = h * (v1 * f1 + v2 * f2 + v3 * f3 + v4 * f4)

    # Estimate error and determine tolerance
    err = abs(Q - q)
    tol = acc + eps * abs(Q)

    # Accept integration is the error is small
    if err < tol:
        return Q, err

    # If error is small, then divide the interval in three pieces
    # and integrate each section.
    else:
        divide = np.sqrt(3)
        Q1, err1 = quad_integrator(F, a, (a + h)/3., f1, f2, nrec+1,
                                   (acc / divide), eps)
        Qm, errm = quad_integrator(F, (a + h)/3., (a + 2 * h)/3.0, f2, f3, nrec+1,
                                   (acc / divide), eps)
        Qr, errr = quad_integrator(F, (a + 2 * h)/3.0, b, f3, f4, nrec+1,
                                   (acc / divide), eps)
        Qtot = Q1 + Qm + Qr
        errtot = np.sqrt(err1 * err1 + errm * errm + errr * errr)
        return Qtot, errtot
