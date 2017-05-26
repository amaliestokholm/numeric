import numpy as np
from lineqsolver import qr_gs_decomp, qr_gs_solve

"""
CHECKS
"""

# Define dimensions of matrices - for tall matrix n > m
n = 5
m = 3

# Relative tolerance
rtol = 1e-04

A = np.random.rand(n, m)
A_check = A.copy() 
R = np.random.rand(m, m)

print('Given the matrix A with dimensions {}x{}, where A in this run is\n{}'.format(n, m, A))
print('The first test is whether qr_qs_decomp can decompose the matrix A into Q and R')
qr_gs_decomp(A=A, R=R)
print('After the run of qr_qs_decomp, Q is \n{}'.format(A))
print('We check if Q is orthogonal by checking whether Q^T Q = I')
print('Q^T * Q =\n {}*\n{} \n= {}'.format(A.T, A, np.dot(A.T, A)))
print('Due to numerical errors, the off-diagonal elements are probably not exactly 0, but close')
print('We check if R is upper triangular:\n{}'.format(R))
print('We check if the product QR is equal to A:')
QR = np.dot(A, R)
print('Q * R \n= {}*\n{} \n= {},\nis it equal to A within a tolerance of {}? \n{}'.format(A, R, QR, rtol, np.allclose(QR, A_check, rtol)))

