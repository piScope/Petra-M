import numpy as np
from numba import njit, float64, int32, types
from numba.types import Array

arr1D = Array(float64, 1, 'A', readonly=True)
arr2D = Array(float64, 2, 'C', readonly=True)


@njit(float64(arr1D, arr1D, float64))
def interp1d_nearest(x, p, x0):
    i = np.argmin(np.abs(x - x0))
    return p[i]


@njit(float64(arr1D, arr1D, float64))
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


@njit(float64(arr1D, arr1D, float64))
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


@njit(float64(arr1D, arr1D, arr2D, float64, float64))
def interp2d_nearest(x, y, p, x0, y0):
    i = np.argmin(np.abs(x - x0))
    j = np.argmin(np.abs(y - y0))
    return p[j, i]


@njit(float64(arr1D, arr1D, arr2D, float64, float64))
def interp2d_linear(x, y, p, x0, y0):
    size = len(y)
    if y0 < y[0]:
        return interp1d_linear(x, p[0, :], x0)
    if y0 > y[-1]:
        return interp1d_linear(x, p[-1, :], x0)

    i = np.argmin(np.abs(y - y0))
    if i == 0:
        pass
    elif i == size - 1:
        i = size - 2
    else:
        if abs(y[i-1] - y0) < abs(y[i+1] - y0):
            i = i-1
        else:
            i = i

    a0 = interp1d_linear(x, p[i, :], x0)
    a1 = interp1d_linear(x, p[i+1, :], x0)

    return a0 * (y[i+1] - y0)/(y[i+1] - y[i]) + a1 * (y0 - y[i])/(y[i+1]-y[i])


@njit(float64(arr1D, arr1D, arr2D, float64, float64))
def interp2d_cubic(x, y, p, x0, y0):
    size = len(y)
    if y0 < y[1]:
        a0 = interp1d_cubic(x, p[0, :], x0)
        a1 = interp1d_cubic(x, p[1, :], x0)
        return a0 * (y[1] - y0)/(y[1] - y[0]) + a1 * (y0 - y[0])/(y[1]-y[0])
    if y0 > y[-2]:
        a0 = interp1d_cubic(x, p[-2, :], x0)
        a1 = interp1d_cubic(x, p[-1, :], x0)
        return a0 * (y[-1] - y0)/(y[-1] - y[-2]) + a1 * (y0 - y[-2])/(y[-1]-y[-2])

    i = np.argmin(np.abs(y - y0))
    if i == 1:
        pass
    elif i == size - 2:
        i = size - 3
    else:
        if abs(y[i-1] - y0) < abs(y[i+1] - y0):
            i = i-1
        else:
            i = i

    a0 = interp1d_cubic(x, p[i-1, :], x0)
    a1 = interp1d_cubic(x, p[i, :], x0)
    a2 = interp1d_cubic(x, p[i+1, :], x0)
    a3 = interp1d_cubic(x, p[i+2, :], x0)

    yy = (y0 - y[i])/(y[i+1] - y[i])
    return a1 + 0.5 * yy*(a2 - a0 + yy*(2.0*a0 - 5.0*a1 + 4.0*a2 - a3 + yy*(3.0*(a1 - a2) + a3 - a0)))


x = np.linspace(0, np.pi*6, 10)
y = np.sin(x/10*np.pi)


@njit
def test1d():
    new_x = np.linspace(0, np.pi*6, 1000)
    #new_y = np.zeros(len(new_x), dtype=float64)
    # for i in range(len(new_x)):
    #    new_y[i] = interp1d_nearest(x, y, new_x[i])
    #

    new_y0 = np.array([interp1d_nearest(x, y, xx) for xx in new_x])

    new_y1 = np.array([interp1d_linear(x, y, xx) for xx in new_x])

    new_y3 = np.array([interp1d_cubic(x, y, xx) for xx in new_x])

    xo = np.linspace(0, np.pi*6, 1000)
    yo = np.sin(xo/10*np.pi)
    return x, y, xo, yo, new_x, new_y0, new_y1, new_y3


@njit
def test2d():
    size = 20

    x = np.linspace(-3, 3, size)
    y = np.linspace(-3, 3, size)

    p = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            p[i, j] = np.exp(-(x[i]**2 + y[j]**2)/3)

    new_x = np.linspace(-3, 3, 100)

    size = 100
    out0 = np.zeros((size, size))
    out1 = np.zeros((size, size))
    out2 = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            out0[j, i] = interp2d_nearest(x, y, p, new_x[i], new_x[j])
            out1[j, i] = interp2d_linear(x, y, p, new_x[i], new_x[j])
            out2[j, i] = interp2d_cubic(x, y, p, new_x[i], new_x[j])

    print("here")
    return x, y, p, new_x, out0, out1, out2


print()
