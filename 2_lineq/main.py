import numpy as np
from lineqsolver import qr_gs_decomp, qr_gs_solve, qr_gs_inverse

"""
CHECKS
"""

print('Check part A.1')
# Define dimensions of matrices - for tall matrix n > m
n = 5
m = 3

# Relative tolerance
rtol = 1e-16

A = np.random.rand(n, m)
A_check = A.copy() 
R = np.random.rand(m, m)

print('Given the matrix A with dimensions {}x{}, where A in this run is\n{}'.format(*A.shape, A))
print('The first test is whether qr_qs_decomp can decompose the matrix A into Q and R')
qr_gs_decomp(A=A, R=R)
print('After the run of qr_qs_decomp, Q is \n{}'.format(A))
print('We check if Q is orthogonal by checking whether Q^T Q = I')
print('Q^T * Q =\n {}*\n{} \n= {}'.format(A.T, A, np.dot(A.T, A)))
print('Due to numerical errors, the off-diagonal elements are probably not exactly 0, but close')
print('We check if R is upper triangular:\n{}'.format(R))
QR = np.dot(A, R)
print('Are QR equal to A within a tolerance of {}? \n{}'.format(A, R, QR, rtol, np.allclose(QR, A_check, rtol)))

n = 4
A = np.random.rand(n, n)
A_check = A.copy()
R = np.random.rand(n, n)
b = np.random.rand(n)
b_check = b.copy()

print('\nCheck part A.2')
print('Given the square matrix A with dimensions {}x{}, where A in this run is\n{}'.format(*A.shape, A))
print('and given a random vector b of size {}, which in this run is\n{}'.format(*b.shape, b))
qr_gs_decomp(A=A, R=R)
qr_gs_solve(Q=A, R=R, b=b)
print('Using qr_gs_decomp on A and then using back-substitution, yields the solution x=\n{}'.format(b))
Ax = np.dot(A_check, b)
print('Is Ax = b within a tolerance of {}?\n {}'.format(rtol, np.allclose(Ax, b_check, rtol)))

print('\nCheck part B')
A = np.random.rand(m, m)
A_check = A.copy()
R = np.random.rand(m, m)
b = np.random.rand(m, m)
print('Given the square matrix A of size {}x{}\n {}'.format(*A.shape, A))
print('We calculate the inverse of A by GS-decomposing it')
qr_gs_decomp(A=A, R=R)
qr_gs_inverse(Q=A, R=R, b=b)
print(b)
AAi = np.dot(A_check, b) 
print(AAi)
print('Is the product of A and A^(-1) equal to I within a tolerance of {}?\n{}'.format(rtol, np.allclose(AAi, np.identity(m))))
