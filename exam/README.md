Numerical Methods 2017 - Examination Assignment
===============================================

**Author:** Amalie Stokholm

**Project:** 26. Multidimensional pseudo-random (plain Monte Carlo) vs quasi-random (Halton and/or lattice sequence) integrators


Exercise:
---------

* Implement Halton and lattice sequence integrator
* Investigate the convergence rates (of some interesting integrals in different dimensions) as function of the number of sample points.


Implementation
--------------

This project is implemented in Python3 with Gnuplot and Matplotlib used for plotting. NumPy is used extensively for handling of arrays.

**Overview of the files**
* *Makefile* builds the project. The entire project can be built by running 'make', while 'make clean' will remove all output.
* *main.py* contains the test of the implementation and it calls the different parts of the exercise.
* *mcquasi.py* contains the implementation of numerical integration using quasi-random integrators.
* *plot_sequences.py* contains the implementation of plotting the different quasi-random sequences.
* *../9_mcinteg/mcinteg.py* contains the implementation of numerical integration using a pseudo-random integrator.


**Output**

