import numpy as np
import mcinteg as mc


def f1(q):
    """
    Test function to integrate: y * sin(x)
    """
    return np.sin(q[0]) * q[1]


def f2(q):
    """
    Test function to integrate: (1 - cos(x)cos(y)cos(z))^(-1)
    """
    func = (1 - np.cos(q[0]) * np.cos(q[1]) * np.cos(q[2])) ** (-1) 
    norm = np.pi ** 3
    return func / norm


def amain():
    """
    Test of the Monte Carlo routine
    """
    # Initialization
    a1 = np.array([0, 0])
    b1 = np.array([np.pi / 2., np.pi])
    a2 = np.array([0, 0, 0])
    b2 = np.array([np.pi, np.pi, np.pi])
    N1 = 1000
    N2 = 100000
    exact2 = 1.3932039296856768591842462603255

    print('Integrating y * sin(x) from (x,y) =\n', a1)
    print('to (x,y) = \n', b1)
    print('The sampling in points is', N1)
    res1, err1 = mc.mc_plain(f1, a1, b1, N1)
    print('The integral is', res1)
    print('The error on the integral is', err1)

    print('Integrating (1 - cos(x) * cos(y) * cos(z))^{-1} from (x,y) =\n',a2)
    print('to (x,y) = \n', b2)
    print('The sampling in points is', N2)
    res2, err2 = mc.mc_plain(f2, a2, b2, N2)
    print('The integral is', res2)
    print('The error on the integral is', err2)
    print('The actual error is',abs(res2 - exact2))

amain()

