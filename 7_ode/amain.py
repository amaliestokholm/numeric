import numpy as np
import ode


def f(x, y):
    """
    ODE where y'' = -y, so trigometric function
    """
    return np.array([y[1], -y[0]])

def amain():
    """
    Test of the ODE routines
    """
    # Initialization
    a = 0
    b = np.pi ** 2
    acc = 1e-3
    eps = 1e-3
    step = 0.1
    yinit = np.array([0, 1], dtype='float64')

    # Evolve the system
    xs, ys = ode.rkdriver(f, a, b, yinit, step, 'rkstep23', acc, eps)

    # Print output
    for i in range(len(xs)):
        print(xs[i], ys[i, 0], ys[i, 1], sep='\t')
amain()
