import numpy as np
from interp import interpolation as interp

# Define start and end point and the number of points in the function (n) and in the interpolation (N)
n = 50
N = 100
start = 0
end = 4 * np.pi

# Make input -- as an example, the cosine function is used.
x = np.linspace(start, end, n)
y = np.cos(x)
z = np.linspace(start, end, N)
y_dev = - np.sin(x)
y_int = np.sin(x)

# Initialize the class
i_cos = interp(x, y)

# Interpolate
s_cspl, s_csplint, s_cspldev = i_cos.cspline(z)

# Print the values in order to save the stdout
for i in range(N):
    if i < n:
        print('%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s'
              % (z[i], s_cspl[i], s_csplint[i], s_cspldev[i], x[i], y[i], y_int[i], y_dev[i]))
    else:
        print('%s\t%s\t%s\t%s' % (z[i], s_cspl[i], s_csplint[i], s_cspldev[i]))
