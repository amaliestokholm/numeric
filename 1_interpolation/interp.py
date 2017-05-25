import numpy as np


class interpolation:
    """
    Class for linear and quadratic interpolation. If the input has more than one dimension, the input will be flatted into a one dimensional shape.
    """    

    def __init__(self, x, y): 
        """
        Initialize and create an empty object and fills it with x and y, when class is called. If x and y is not one dimensional, they will be contiguostly flattened.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        x = np.ravel(x)
        y = np.ravel(y)
        if x.shape != y.shape:
            raise ValueError("x and y must have the same length")
        self.x = x
        self.y = y
        
        self.n = int(len(self.x))


    def __binarysearch(self, z):
        """
        Binary search algorithm
        """
        i = 0
        j = self.n-1
        while j - i > 1:
            m = int(np.floor((i+j) / 2))
            # Binary search
            if z > self.x[m]:
                i = m
            else:
                j = m
        # Calculate slope
        p = (self.y[i+1] - self.y[i]) / (self.x[i+1] - self.x[i])
        return p, i


    def __qspline_params(self):
        """
        Calculate the coefficients b and c
        """
        b = np.zeros(self.n-1)
        c = np.zeros(self.n-1)
        dx = np.zeros(self.n-1)
        p = np.zeros(self.n-1)

        # Calculate x-interval and slope
        for j in np.arange(self.n-1):
            dx[j] = self.x[j+1] - self.x[j]
            p[j] = (self.y[j+1] - self.y[j]) / dx[j]
        
        # Find c forward-recursively
        list = np.arange(self.n-2)
        for i in list:
            c[i+1] = (p[i+1] - p[i] - c[i] * dx[i]) / dx[i+1]
        
        # Find c backward-recursively from 1/2c_n-1
        c[-1] = c[-1] / 2
        for i in list[::-1]:
            c[i] = (p[i+1] - p[i] - c[i+1] * dx[i+1]) / dx[i]

        # Find b
        for i in np.arange(self.n-1):
            b[i] = p[i] - c[i] * dx[i]
        return b, c


    def __cspline_params(self):
        """
        Calculate the coefficients b, c, and d for cubic spline
        """
        b = np.zeros(self.n)
        c = np.zeros(self.n-1)
        d = np.zeros(self.n-1)
        B = np.zeros(self.n)
        Q = np.ones(self.n-1)
        D = 2 * np.ones(self.n)
        dx = np.zeros(self.n-1)
        p = np.zeros(self.n-1)

        # Calculate x-interval and slope
        for j in np.arange(self.n-1):
            dx[j] = self.x[j+1] - self.x[j]
            p[j] = (self.y[j+1] - self.y[j]) / dx[j]

        # Fill B
        B[0] = 3 * p[0]
        for i in np.arange(self.n-2):
            B[i+1] = 3 * (p[i] + p[i+1] * dx[i] / dx[i+1])
        B[-1] = 3 * p[-2]
        
        # Fill D
        for i in np.arange(self.n-2):
            D[i+1] = 2 * dx[i] / dx[i+1] + 2

        # Fill Q
        for i in np.arange(self.n-2):
            Q[i+1] = dx[i] / dx[i+1]

        # Gauss elimination
        for i in np.arange(1, self.n):
            D[i] = D[i] - Q[i-1] / D[i-1]
            B[i] = B[i] - B[i-1] / D[i-1]

        # Back-substitution
        b[-1] = B[-1] / D[-1]
        list = np.arange(self.n-1)
        for i in list[::-1]:
            b[i] = (B[i] - Q[i] * b[i+1]) / D[i]

        # Calculate c and d
        for i in np.arange(self.n-1):
            c[i] = (3 * p[i] - 2 * b[i] - b[i+1]) / dx[i]
            d[i] = (b[i] + b[i+1] - 2 * p[i]) / dx[i]
        c[-1] = -3 * d[-1] * dx[-1]

        return b, c, d



    def __linterp_integ(self, z):
        """
        Linear interpolation integral
        """
        result = 0
        p, i = self.__binarysearch(z)
        for j in np.arange(i):
            dx = self.x[j+1] - self.x[j]
            result = result + self.y[j] * dx + 0.5 * (self.y[j+1] - self.y[j]) * dx ** 2
        result = (result +
                  self.y[i] * (z - self.x[i]) + 0.5 * (self.y[i+1] - self.y[i]) * (z - self.x[i]) ** 2)  
        return result
    

    def __qspline_integ(self, z):
        """
        Quadratic interpolation integral
        """
        result = 0
        p, i = self.__binarysearch(z)
        b, c = self.__qspline_params()
        for j in np.arange(i):
            dx = self.x[j+1] - self.x[j]
            result += (self.y[j] * dx + 0.5 * b[j] * dx ** 2 + (1 / 3) * c[j] * dx ** 2)
        zi = z - self.x[i]
        result +=  self.y[i] * zi + 0.5 * b[i] * zi ** 2 + (1 / 3) * c[i] * zi ** 2
        return result


    def __cspline_integ(self, x):
        """
        Cubic spline integral
        """
        result = 0
        p, i = self.__binarysearch(z)
        b, c = self.__qspline_params()
        for j in np.arange(i):
            dx = self.x[j+1] - self.x[j]
            result += (self.y[j] * dx + 0.5 * b[j] * dx ** 2 + (1 / 3) * c[j] * dx ** 2 
                       + (1 / 4) * d[j] * dx ** 3)
        zi = z - self.x[i]
        result +=  self.y[i] * zi + 0.5 * b[i] * zi ** 2 + (1 / 3) * c[i] * zi ** 2 + (1 / 4) * d[i] * zi ** 3
        return result

    def linterp(self, z):
        """
        Linear interpolation routine
        """
        z = np.asarray(z)
        s = np.zeros(z.shape)
        si = np.zeros(z.shape)
        for j in np.arange(z.size):
            p, i = self.__binarysearch(z[j])
            s[j] = self.y[i] + p * (z[j] - self.x[i])
            si[j] = self.__linterp_integ(z[j])
        return s, si


    def qspline(self, z, deriv_flag=1, int_flag=1, func_flag=1):
        """
        Quadratic spline interpolation routine
        """
        z = np.asarray(z)
        s = np.zeros(z.shape)
        si = np.zeros(z.shape)
        sd = np.zeros(z.shape)
        b, c = self.__qspline_params()
        for j in np.arange(z.size):
            p, i = self.__binarysearch(z[j])
            if func_flag is not None:
                s[j] = self.y[i] + b[i] * (z[j] - self.x[i]) + c[i] * (z[j] - self.x[i]) ** 2 
            if int_flag is not None:
                si[j] = self.__qspline_integ(z[j])
            if deriv_flag is not None:
                sd[j] = b[i] + 2 * c[i] * (z[j] - self.x[i])
        return s, si, sd


    def cspline(self, z, deriv_flag=1, int_flag=1, func_flag=1):
        """
        Cubic spline interpolation routine
        """
        z = np.asarray(z)
        s = np.zeros(z.shape)
        si = np.zeros(z.shape)
        sd = np.zeros(z.shape)
        b, c, d = self.__cspline_params()
        for j in np.arange(z.size):
            p, i = self.__binarysearch(z[j])
            if func_flag is not None:
                s[j] = (self.y[i] + b[i] * (z[j] - self.x[i]) + c[i] * (z[j] - self.x[i]) ** 2 
                        + d[i] * (z[j] - self.x[i]) ** 3) 
            if int_flag is not None:
                si[j] = self.__qspline_integ(z[j])
            if deriv_flag is not None:
                sd[j] = b[i] + 2 * c[i] * (z[j] - self.x[i]) + 3 * d[i] * (z[j] - self.x[i]) ** 2
        return s, si, sd
