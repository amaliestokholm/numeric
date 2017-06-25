import numpy as np


def diag(A):
    """
    This routine performs matrix diagonalization on a real and symmetric
    matrix A using the Jacobi eigenvalue method with cyclic sweeps.
    Arguments:
        - 'A': The real, symmetric input matrix to be diagonalized.
        The upper triangle is destroyed.
    Returns:
        - 'rotations_counter': The number of rotations used.
        - 'e': Vector containing the eigenvalues.
        - 'V': Matrix containing the eigenvectors.
    """
    # Check if A is quadratic, symmetric and real
    A = np.asarray(A)
    n, m = A.shape
    assert n == m
    assert np.allclose(A.T, A)
    assert np.allclose(A, A.real)

    # Initialization
    e = np.zeros((n, 1))
    V = np.identity(n)
    rotations_counter = 0
    change = True

    # Store all diagonal elements in e
    for i in range(n):
        e[i] = A[i, i]

    while change:
        rotations_counter += 1
        change = False
        for p in range(n):
            for q in range(p + 1, n):
                change, rotations_counter = rotation(A, e, V, p, q, n,
                                                     change,
                                                     rotations_counter, False)
    return rotations_counter, e, V



def eliminate(A):
    """
    This routine performs matrix diagonalization on a real and symmetric
    matrix A using the Jacobi eigenvalue-by-eigenvalue method..
    Arguments:
        - 'A': The real, symmetric input matrix to be diagonalized.
               The upper triangle is destroyed.
    Returns:
        - 'rotations_counter': The number of rotations used.
        - 'e': Vector containing the eigenvalues.
        - 'V': Matrix containing the eigenvectors.
    """
    # Check if A is quadratic, symmetric and real
    A = np.asarray(A)
    n, m = A.shape
    assert n == m
    assert np.allclose(A.T, A)
    assert np.allclose(A, A.real)

    # Initialization
    e = np.zeros((n, 1))
    V = np.identity(n)
    rotations_counter = 0

    # Store all diagonal elements in e
    for i in range(n):
        e[i] = A[i, i]

    # Sweep row-by-row
    for p in range(n - 1):
        change = True
        while change:
            change = False
            for q in range(p + 1, n):
                change, rotations_counter = rotation(A, e, V, p, q, n,
                                                     change, rotations_counter,
                                                     True)
    return rotations_counter, e, V


def rotation(A, e, V, p, q, n, change, rotations_counter, pre_rows):
    """
    This routine performs a Jacobi rotation.
    New arguments:
        - 'pre_rows': Whether previous rows are eliminated or not.
                      It can reduce the number of iterations in the
                      non-cyclic method.
    """

    # Get the different entries
    app = e[p]
    aqq = e[q]
    apq = A[p, q]

    # Calculate different terms
    """
    The smallest eigenvalue is found first in this implementation as the
    it is the one we compare to the others. If the first element given to
    arctan2 is 0, then it will return 0 (pi), if the next element
    is positive(negative) and the routine will make no (many) rotations.
    We can find the largest eigenvalue first by adding pi/2.
    """
    # phi = 0.5 * np.arctan2((2 * apq), (aqq - app))  # smallest first
    phi = 0.5 * (np.pi + np.arctan2((2 * apq), (aqq - app)))  # greatest first
    c = np.cos(phi)
    s = np.sin(phi)

    # Calculate new diagonal elements
    app_n = c * c * app - 2 * s * c * apq + s * s * aqq
    aqq_n = s * s * app + 2 * s * c * apq + c * c * aqq

    # Compare
    if app_n != app or aqq_n != aqq:
        change = True
        rotations_counter += 1

        # Update the diagonal elements and apq
        e[p] = app_n
        e[q] = aqq_n
        A[p, q] = 0

        # Update remaining elements. The first loop does not affect the result
        # if it eliminates eigenvalue by eigenvalue.
        if not pre_rows:
            for i in range(p):
                aip = A[i, p]
                aiq = A[i, q]
                A[i, p] = c * aip - s * aiq
                A[i, q] = c * aiq + s * aip

        for i in range(p + 1, q):
            api = A[p, i]
            aiq = A[i, q]
            A[p, i] = c * api - s * aiq
            A[i, q] = c * aiq + s * api

        for i in range(q + 1, n):
            api = A[p, i]
            aqi = A[q, i]
            A[p, i] = c * api - s * aqi
            A[q, i] = c * aqi + s * api

        for i in range(n):
            vip = V[i, p]
            viq = V[i, q]
            V[i, p] = c * vip - s * viq
            V[i, q] = c * viq + s * vip

    return change, rotations_counter
