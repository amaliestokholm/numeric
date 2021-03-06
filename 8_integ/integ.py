import numpy as np
import globvar


def nodes(a, b, x):
    """
    Calculation that finds the nodes
    """
    return a + x * (b - a)


def integ_recursive(F, a_orig, b_orig, acc=1e-4, eps=1e-4):
    """
    This routine computes recursive adaptive integration using open
    QUADratures. It uses a 4th order trapezium evaluation rule with
    2nd order rectangular rule for error estimation.
    Arguments:
        - 'F': Function to be integrated
        - 'a_orig': Starting point
        - 'b_orig': End point
        - 'acc': Absolute error
        - 'eps': Relative error
    Returns:
        - 'Q': The integral of F from a to b
        - 'err': An estimate of the error on Q
    """
    assert a_orig < b_orig, 'Check your limits!'
    # An infinite limit integral is converted by a variable transformation
    # to a finite limit integral
    f = F
    a = a_orig
    b = b_orig
    if np.any(np.isinf([a_orig, b_orig])):
        if a_orig is np.NINF and b_orig is not np.PINF:
            F = lambda t: f(b_orig + (t / (1 + t))) * (1 / ((1 + t) ** 2))
            a = -1
            b = 0
        elif a_orig is not np.NINF and b_orig is np.PINF:
            F = lambda t: f(a_orig + (t / (1 - t))) * (1 / ((1 - t) ** 2))
            a = 0
            b = 1
        else:
            F = lambda t: f(t / (1 - t ** 2)) * ((1 + t ** 2) / (1 - t ** 2) ** 2)
            a = -1
            b = 1

    # Perform integration
    x = np.array([1 / 6, 2 / 6, 4 / 6, 5 / 6])
    f1 = F(nodes(a, b, x[1]))
    f2 = F(nodes(a, b, x[2]))

    Q, err, recmax = quad_integrator(F, a, b, x, f1, f2, nrec=0, acc=acc, eps=eps)
    return Q, err, recmax


def quad_integrator(F, a, b, x, f1, f2, nrec, acc, eps):
    """
    This function integrates the function from a to b using previously
    determined points using an open set of equidistant nodes.
    If the error is too big, the interval is sub-divided
    and the integrator calls itself.
    New arguments:
        - 'f1': Estimated value of F(a + (b - a)/3)
        - 'f2': Estimated value of F(a + 2(b - a)/3)
        - 'nrec': The current level of recursion
    """
    # Estimate to two points end points
    h = b - a
    f0 = F(nodes(a, b, x[0]))
    f3 = F(nodes(a, b, x[3]))

    # Weights of 4th order trapezium rule
    w = np.array([2 / 6.0, 1 / 6.0, 1 / 6.0, 2 / 6.0])

    # Weights of 4th order trapezium rule
    v = np.ones(4) * (1 / 4.0)

    # Calculate
    Q = h * (w[0] * f0 + w[1] * f1 + w[2] * f2 + w[3] * f3)
    q = h * (v[0] * f0 + v[1] * f1 + v[2] * f2 + v[3] * f3)

    # Estimate error and determine tolerance
    err = abs(Q - q)
    tol = acc + eps * abs(Q)

    # Accept integration is the error is small
    if err < tol:
        return Q, err, nrec

    # If error is small, then divide the interval in three pieces
    # and integrate each section.
    else:
        accsec = acc / np.sqrt(2)
        sec = (a + b) / 2
        Q1, err1, rec1 = quad_integrator(F, a, sec, x, f0, f1, nrec+1,
                                         acc=accsec, eps=eps)
        Q2, err2, rec2 = quad_integrator(F, sec, b, x, f2, f3, nrec+1,
                                         acc=accsec, eps=eps)
        Qtot = Q1 + Q2
        errtot = np.sqrt(err1 * err1 + err2 * err2)
        recmax = max(rec1, rec2)
        return Qtot, errtot, recmax
