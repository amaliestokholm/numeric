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


    def __binarysearch(self, z):
        """
        Binary search algorithm
        """
        n = int(len(self.x))
        i = 0
        j = n-1
        while j - i > 1:
            m = int(np.floor((i+j) / 2))
            # Binary search
            if z > self.x[m]:
                i = m
            else:
                j = m
        # Calculate slope
        p = (self.y[i+1] - self.y[i]) / (self.x [i+1] - self.x[i])
        return p, i

    
    def __linterp_integ(self, z):
        """
        Linear interpolation integral
        """
        result = 0
        p, i = self.__binarysearch(z)
        for j in np.arange(i):
            deltax = self.x[j+1] - self.x[j]
            result = result + self.y[j] * deltax + 0.5 * (self.y[j+1] - self.y[j]) * deltax ** 2
        result = (result +
                  self.y[i] * (z - self.x[i]) + 0.5 * (self.y[i+1] - self.y[i]) * (z - self.x[i]) ** 2)  
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





