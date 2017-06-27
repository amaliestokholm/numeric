import numpy as np
import mcinteg as mc


def f1(q):
    """
    Test function to integrate: y * sin(x)
    """
    return np.sin(q[0]) * q[1]


def f2(q):
    """
    Test functino to integrate: x * (x - y^2)
    """
    return q[0] * (q[0] - (q[1] * q[1]))


def amain():
    """
    Test of the Monte Carlo routine
    """
    a1 = np.array([0, 0])
    b1 = np.array([np.pi / 2., np.pi])
    N1 = 1000

    print('Integrating y * sin(x) from (x,y) =\n', a1)
    print('to (x,y) = \n', b1)
    print('The sampling in points is', N1)
    res1, err1 = mc.mc_plain(f1, a1, b1, N1)
    print('The integral is', res1)
    print('The error on the integral is', err1)

amain()

