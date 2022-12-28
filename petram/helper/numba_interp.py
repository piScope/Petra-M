import numpy as np
from numba import njit, float64, int32, types
from numba.types import Array


@njit(float64(Array(float64, 1, 'A', readonly=True), Array(float64, 1, 'A', readonly=True), float64))
def interp1d_nearest(x, p, x0):
    i = np.argmin(np.abs(x - x0))
    return p[i]


@njit(float64(Array(float64, 1, 'A', readonly=True), Array(float64, 1, 'A', readonly=True), float64))
def interp1d_linear(x, p, x0):
    size = len(x)
    if x0 < x[0]:
        return p[0]
    if x0 > x[size-1]:
        return p[size-1]

    i = np.argmin(np.abs(x - x0))
    if i == 0:
        pass
    elif i == size - 1:
        i = size - 2
    else:
        if abs(x[i-1] - x0) < abs(x[i+1] - x0):
            i = i-1
        else:
            i = i

    d1 = p[i+1] * (x0 - x[i])/(x[i+1]-x[i]) + \
        p[i] * (x[i+1] - x0)/(x[i+1]-x[i])
    return d1


@njit(float64(Array(float64, 1, 'A', readonly=True), Array(float64, 1, 'A', readonly=True), float64))
def interp1d_cubic(x, p, x0):
    '''
    cubic herimit.  function and gradients are constrained. 
    '''
    size = len(x)
    if x0 < x[1]:
        return interp1d_linear(x, p, x0)
    if x0 > x[size-2]:
        return interp1d_linear(x, p, x0)

    i = np.argmin(np.abs(x - x0))
    if i == 1:
        pass
    elif i == size - 2:
        i = size - 3
    else:
        if abs(x[i-1] - x0) < abs(x[i+1] - x0):
            i = i-1
        else:
            i = i

    x = (x0 - x[i])/(x[i+1] - x[i])
    return p[i] + 0.5 * x*(p[i+1] - p[i-1] + x*(2.0*p[i-1] - 5.0*p[i] + 4.0*p[i+1] - p[i+2] + x*(3.0*(p[i] - p[i+1]) + p[i+2] - p[i-1])))

x = np.linspace(0, np.pi*6, 10)
y = np.sin(x/10*np.pi)

@njit
def test1d():
    new_x = np.linspace(0, np.pi*6, 1000)
    #new_y = np.zeros(len(new_x), dtype=float64)
    #for i in range(len(new_x)):
    #    new_y[i] = interp1d_nearest(x, y, new_x[i])
    #
    
    new_y0 = np.array([interp1d_nearest(x, y, xx) for xx in new_x])

    new_y1 = np.array([interp1d_linear(x, y, xx) for xx in new_x])

    new_y3 = np.array([interp1d_cubic(x, y, xx) for xx in new_x])

    xo = np.linspace(0, np.pi*6, 1000)
    yo = np.sin(xo/10*np.pi)
    return x, y, xo, yo, new_x, new_y0, new_y1, new_y3


print()
