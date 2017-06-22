import numpy as np
from lineqsolver import qr_gs_decomp, qr_gs_solve, qr_gs_inverse, qr_gv_decomp, qr_gv_solve, qr_gv_inverse

"""
CHECKS
"""
# Use the same random seed everytime
np.random.seed(11)

print('Check part A.1')
# Define dimensions of matrices - for tall matrix n > m
n = 5
m = 3

# Relative tolerance
rtol = 1e-16

A = np.random.rand(n, m)
A_check = np.copy(A) 
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
print('Are QR equal to A within a tolerance of {}? \n{}'.format(rtol, np.allclose(QR, A_check, rtol)))


print('\nCheck part A.2')
n = 4
A = np.random.rand(n, n)
A_check = np.copy(A)
R = np.random.rand(n, n)
b = np.random.rand(n)
b_check = np.copy(b)

print('Given the square matrix A with dimensions {}x{}, where A in this run is\n{}'.format(*A.shape, A))
print('and given a random vector b of size {}, which in this run is\n{}'.format(*b.shape, b))
qr_gs_decomp(A=A, R=R)
qr_gs_solve(Q=A, R=R, b=b)
print('Using qr_gs_decomp on A and then using back-substitution, yields the solution x=\n{}'.format(b))
Ax = np.dot(A_check, b)
print('Is Ax = b within a tolerance of {}?\n {}'.format(rtol, np.allclose(Ax, b_check, rtol)))


print('\nCheck part B')
A = np.random.rand(m, m)
A_check = np.copy(A)
R = np.random.rand(m, m)
b = np.random.rand(m, m)
print('Given the square matrix A of size {}x{}\n {}'.format(*A.shape, A))
print('We calculate the inverse of A by GS-decomposing it')
qr_gs_decomp(A=A, R=R)
qr_gs_inverse(Q=A, R=R, b=b)
AAi = np.dot(A_check, b) 
print('Is the product of A and A^(-1) equal to I within a tolerance of {}?\n{}'.format(rtol, np.allclose(AAi, np.identity(m))))


print('\nCheck part C')
# Define the dimensions
n = 5

A = np.random.rand(n, n)
Ai = np.random.rand(n, n)
A_gv = np.copy(A) 
A_check = np.copy(A) 
R = np.random.rand(n, n)
b = np.random.rand(n)
b_check = np.copy(b)

print('Given the matrix A with dimensions {}x{}, where A in this run is\n{}'.format(*A.shape, A))
print('We try to solve the equation using Gram-Schmidt as earlier')
qr_gs_decomp(A=A, R=R)
qr_gs_solve(Q=A, R=R, b=b)
print('After the run of qr_qs_decomp, b is \n{}'.format(b))
qr_gv_decomp(A=A_gv)
qr_gv_solve(A=A_gv, b=b_check)
print('If we try to solve the equation using Givens-rotations, we get \n{}'.format(b_check)) 
print('Are the two solutions equal within a tolerance of {}?\n{}'.format(rtol, np.allclose(b, b_check, rtol))) 
qr_gv_inverse(A_gv, Ai)
AAi = np.dot(A_check, Ai)
eye = np.identity(n)
print(AAi)
print('We calculate the inverse and checks if AA^(-1) = I. Are AA^(-1)=I within the tolerance {}?\n {}'.format(rtol, np.allclose(AAi, eye, rtol)))

