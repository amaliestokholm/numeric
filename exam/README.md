Numerical Methods 2017 - Examination Assignment
===============================================

**Author:** Amalie Stokholm

**Project:** 26. Multidimensional pseudo-random (plain Monte Carlo) vs quasi-random (Halton and/or lattice sequence) integrators.


Exercise:
---------

* Implement Halton and lattice sequence integrator
* Investigate the convergence rates (of some interesting integrals in different dimensions) as a function of the number of sample points.


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
* *nerror.pdf* contains a test of convergence (error as a function of sample size N) and shows that the quasi-random integrators converge faster than the pseudo-random integrator. 
* *check.txt* contains the data plotted in nerror.pdf.
* *plot_halton.png* and *plot_lattice.png* plot the distributions of the Halton and Lattice sequences respectively. These plots reproduce Figure 9.2 in the Lecture Notes quite nicely.
