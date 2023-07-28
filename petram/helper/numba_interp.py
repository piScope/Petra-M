import numpy as np
from numba import njit, float64, int32, types
from numba.types import Array

arr1D = Array(float64, 1, 'A', readonly=True)
arr2D = Array(float64, 2, 'C', readonly=True)
arr3D = Array(float64, 3, 'C', readonly=True)


@njit(float64(arr1D, arr1D, float64))
def interp1d_nearest(x, p, x0):
    '''
    1D nearest neighbor interpolation
       x 1D array. This must be monotocically increasing. Note the routine does not check this.
       p 1D array
       x0 point to interpolate
    '''

    i = np.argmin(np.abs(x - x0))
    return p[i]


@njit(float64(arr1D, arr1D, float64))
def interp1d_linear(x, p, x0):
    '''
    1D linear interpolation
       x 1D array. This must be monotocically increasing. Note the routine does not check this.
       p 1D array
       x0 point to interpolate
    '''

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
    1D cubic herimit interpolation
       x 1D array. This must be monotocically increasing. Note the routine does not check this.
       p 1D array
       x0 point to interpolate
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
    '''
    2D nearest neighbor interpolation
       x 1D array. This must be monotocically increasing. Note the routine does not check this.
       y 1D array. This must be monotocically increasing. Note the routine does not check this.
       p 2D array
       x0 point to interpolate
       y0 point to interpolate
    '''

    i = np.argmin(np.abs(x - x0))
    j = np.argmin(np.abs(y - y0))
    return p[j, i]


@njit(float64(arr1D, arr1D, arr2D, float64, float64))
def interp2d_linear(x, y, p, x0, y0):
    '''
    2D linear interpolation
       x 1D array. This must be monotocically increasing. Note the routine does not check this.
       y 1D array. This must be monotocically increasing. Note the routine does not check this.
       p 2D array
       x0 point to interpolate
       y0 point to interpolate
    '''

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
    '''
    2D cubic interpolation
       x 1D array. This must be monotocically increasing. Note the routine does not check this.
       y 1D array. This must be monotocically increasing. Note the routine does not check this.
       p 2D array
       x0 point to interpolate
       y0 point to interpolate
    '''

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


@njit(float64(arr1D, arr1D, arr1D, arr3D, float64, float64, float64))
def interp3d_nearest(x, y, z, p, x0, y0,  z0):
    '''
    3D nearest neighbor interpolation
       x 1D array. This must be monotocically increasing. Note the routine does not check this.
       y 1D array. This must be monotocically increasing. Note the routine does not check this.
       z 1D array. This must be monotocically increasing. Note the routine does not check this.
       p 3D array
       x0 point to interpolate
       y0 point to interpolate
       z0 point to interpolate
    '''

    i = np.argmin(np.abs(x - x0))
    j = np.argmin(np.abs(y - y0))
    k = np.argmin(np.abs(z - z0))
    return p[k, j, i]


@njit(float64(arr1D, arr1D, arr1D, arr3D, float64, float64, float64))
def interp3d_linear(x, y, z, p, x0, y0,  z0):
    '''
    3D linear interpolation
       x 1D array. This must be monotocically increasing. Note the routine does not check this.
       y 1D array. This must be monotocically increasing. Note the routine does not check this.
       z 1D array. This must be monotocically increasing. Note the routine does not check this.
       p 3D array
       x0 point to interpolate
       y0 point to interpolate
       z0 point to interpolate
    '''

    if z0 < z[0]:
        return interp2d_linear(x, y, p[0, :, :], x0, y0)
    if z0 > z[-1]:
        return interp2d_linear(x, y, p[-1, :, :], x0, y0)

    size3 = len(z)
    k = np.argmin(np.abs(z - z0))
    if k == 0:
        pass
    elif k == size3 - 1:
        k = size3 - 2
    else:
        if abs(z[k-1] - z0) < abs(z[k+1] - z0):
            k = k-1
        else:
            k = k

    a0 = interp2d_linear(x, y, p[k, :, :], x0, y0)
    a1 = interp2d_linear(x, y, p[k, :, :], x0, y0)

    return a0 * (z[k+1] - z0)/(z[k+1] - z[k]) + a1 * (z0 - z[k])/(z[k+1]-z[k])


@njit(float64(arr1D, arr1D, arr1D, arr3D, float64, float64, float64))
def interp3d_cubic(x, y, z, p, x0, y0,  z0):
    '''
    3D cubic interpolation
       x 1D array. This must be monotocically increasing. Note the routine does not check this.
       y 1D array. This must be monotocically increasing. Note the routine does not check this.
       z 1D array. This must be monotocically increasing. Note the routine does not check this.
       p 3D array
       x0 point to interpolate
       y0 point to interpolate
       z0 point to interpolate
    '''

    if z0 < z[1]:
        a0 = interp2d_cubic(x, p[0, :, :], x0, y0)
        a1 = interp2d_cubic(x, p[1, :, :], x0, y0)
        return a0 * (z[1] - z0)/(z[1] - z[0]) + a1 * (z0 - z[0])/(z[1]-z[0])
    if z0 > z[-2]:
        a0 = interp2d_cubic(x, y, p[-2, :, :], x0, y0)
        a1 = interp2d_cubic(x, y, p[-1, :, :], x0, y0)
        return a0 * (z[-1] - z0)/(z[-1] - z[-2]) + a1 * (z0 - z[-2])/(z[-1]-z[-2])

    size3 = len(z)
    k = np.argmin(np.abs(z - z0))
    if k == 1:
        pass
    elif k == size3 - 2:
        i = size3 - 3
    else:
        if abs(z[k-1] - z0) < abs(z[k+1] - z0):
            k = k-1
        else:
            k = k

    a0 = interp2d_cubic(x, y, p[k-1, :, :], x0, y0)
    a1 = interp2d_cubic(x, y, p[k, :, :], x0, y0)
    a2 = interp2d_cubic(x, y, p[k+1, :, :], x0, y0)
    a3 = interp2d_cubic(x, y, p[k+2, :, :], x0, y0)

    zz = (z0 - z[k])/(z[k+1] - z[k])
    return a1 + 0.5 * zz*(a2 - a0 + zz*(2.0*a0 - 5.0*a1 + 4.0*a2 - a3 + zz*(3.0*(a1 - a2) + a3 - a0)))


@njit
def test1d():
    x = np.linspace(0, np.pi*6, 10)
    y = np.sin(x/10*np.pi)

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
            p[j, i] = np.exp(-(x[i]**2 + y[j]**2)/3)

    new_x = np.linspace(-3, 3, 100)

    size = 100
    out0 = np.zeros((size, size))
    out1 = np.zeros((size, size))
    out2 = np.zeros((size, size))

    for i in range(size):
        for j in range(size):
            out0[j, i] = interp2d_nearest(x, y, p, new_x[i], new_y[j])
            out1[j, i] = interp2d_linear(x, y, p, new_x[i], new_y[j])
            out2[j, i] = interp2d_cubic(x, y, p, new_x[i], new_y[j])

    print("here")
    return x, y, p, new_x, out0, out1, out2


@njit
def test3d():
    size1, size2, size3 = 10, 15, 20

    x = np.linspace(-3, 3, size1)
    y = np.linspace(-3, 3, size2)
    z = np.linspace(-3, 3, size3)

    p = np.zeros((size3, size2, size1))
    for k in range(size3):
        for j in range(size2):
            for i in range(size1):
                p[k, j, i] = np.exp(-(x[i]**2 + y[j]**2 + z[k]**2)/3)

    new_x = np.linspace(-3, 3, 20)

    size = 100
    out0 = np.zeros((size, size, size))
    out1 = np.zeros((size, size, size))
    out2 = np.zeros((size, size, size))

    for i in range(size):
        for j in range(size):
            for k in range(size):
                out0[k, j, i] = interp3d_nearest(
                    x, y, z, p, new_x[i], new_y[j], new_z[k])
                out1[k, j, i] = interp3d_linear(
                    x, y, z, p, new_x[i], new_y[j], new_z[k])
                out2[k, j, i] = interp3d_cubic(x, y, , z p, new_x[i], new_y[j], new_z[k])

    print("here")
    return x, y, p, new_x, out0, out1, out2


print()
