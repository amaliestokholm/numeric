import numpy as np
import minimize
import systems

# Initialize
alpha = 1e-4
initpar = np.array([1, 1, 1], dtype='float64')
t, y, s = np.loadtxt('data.txt').T
xs = np.linspace(0, 10, 50)

# Optimize
sl = lambda p: systems.master(t, y, s, p)
grad_sl = lambda p: systems.grad_master(t, y, s, p)
par = minimize.qnewton_minimize(sl, grad_sl, initpar, alpha)
for i in range(len(xs)):
    if i < len(t):
        print('%s\t%s\t%s\t&s\t%s'
              % (xs[i], systems.decay(par, xs[i]), t[i], y[i], s[i]))
    else:
        print('%s\t%s' % (xs[i], systems.decay(par, xs[i])))
