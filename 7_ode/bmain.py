import numpy as np
import ode
import globvar


def f(x, y):
    """
    ODE where y'' = -y, so trigometric function
    """
    globvar.ncalls += 1
    return np.array([y[1], -y[0]])


def bmain():
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
    globvar.ncalls = 0

    # Evolve the system
    xs, ys, cs = ode.rkdriver(f, a, b, yinit, step, 'rkstep23', acc, eps,
                              calls=True)

    print('Output from RK23: y0')
    # Print output
    for i in range(len(xs)):
        print(xs[i], ys[i, 0], cs[i], sep='\t')
    print('Output from RK23: y1')
    for i in range(len(xs)):
        print(xs[i], ys[i, 1], cs[i], sep='\t')

    globvar.ncalls = 0
    # Evolve the system
    xs, ys, cs = ode.rkdriver(f, a, b, yinit, step, 'rkstep3', acc, eps,
                              calls=True)

    print('Output from RK3: y0')
    # Print output
    for i in range(len(xs)):
        print(xs[i], ys[i, 0], cs[i], sep='\t')
    print('Output from RK3: y1')
    for i in range(len(xs)):
        print(xs[i], ys[i, 1], cs[i], sep='\t')

bmain()
