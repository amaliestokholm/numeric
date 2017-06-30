Numerical Methods 2017 - Examination Assignment
===============================================

**Author:** Amalie Stokholm

**Project:** 26. Multidimensional pseudo-random (plain Monte Carlo) vs quasi-random (Halton and/or lattice sequence) integrators.


Exercise
---------

* Implement Halton and lattice sequence integrator
* Investigate the convergence rates (of some interesting integrals in different dimensions) as a function of the number of sample points.


Implementation
--------------

This project is implemented in Python3 with Gnuplot and Matplotlib used for plotting. NumPy is used extensively for handling of arrays.

**Overview of the files**
* *Makefile* builds the project. The entire project can be built by running 'make', while 'make clean' will remove all output.
* *main.py* contains the test of the implementation and it calls the different parts of the exercise. The implementation is tested using two different d-dimensional integrals: f(x, y) = x sin(y) and f(x, y, z) = 8xyz.
* *mcquasi.py* contains the implementation of numerical integration using quasi-random integrators.
* *plot_sequences.py* contains the implementation of plotting the different quasi-random sequences. These plots are made using Matplotlib instead of gnuplot just for fun.
* *../9_mcinteg/mcinteg.py* contains the implementation of numerical integration using a pseudo-random integrator.


**Output**
* *nerror_xsiny.pdf* and *nerror_xyz* contain tests of convergence (error as a function of sample size N) and show that the quasi-random integrators converge faster than the pseudo-random integrator. 
* *check.txt* contains the data plotted in *nerror_xsiny.pdf* and *nerror_xyz.pdf*.
* *plot_halton.pdf* and *plot_lattice.pdf* plot the distributions of the Halton and Lattice sequences respectively. These plots reproduce Figure 9.2 in the Lecture Notes quite nicely.
