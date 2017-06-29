import numpy as np
import matplotlib.pyplot as plt
from mcquasi import halton, lattice


def main():
    N = 1000
    p1 = [halton(i, 2) for i in range(N)]
    x1, y1 = np.transpose(p1)

    p2 = [lattice(i, 2) for i in range(N)]
    x2, y2 = np.transpose(p2)

    plt.figure()
    plt.plot(x1, y1, 'o')
    plt.savefig('plot_halton.png')

    plt.figure()
    plt.plot(x2, y2, 'o')
    plt.savefig('plot_lattice.png')

if __name__ == '__main__':
    main()
