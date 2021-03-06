import numpy as np
from interp import interpolation as interp

# Define start and end point and the number of points in the function (n) and in the interpolation (N)
n = 50
N = 200
start = 0
end = 4 * np.pi

# Make input -- as an example, the cosine function is used.
x = np.linspace(start, end, n)
y = np.cos(x)
z = np.linspace(start, end, N)
y_int = np.sin(x)


# Initialize the class
i_cos = interp(x, y)

# Interpolate
s_lin, s_linint = i_cos.linterp(z)

# Print the values in order to save the stdout
for i in range(0, N):
    if i < n:
        print('%s\t%s\t%s\t%s\t%s\t%s' % (z[i], s_lin[i], s_linint[i], x[i], y[i], y_int[i]))
    else:
        print('%s\t%s\t%s' % (z[i], s_lin[i], s_linint[i]))
