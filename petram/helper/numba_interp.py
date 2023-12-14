import numpy as np
from numba import njit, float64, int32, types
from numba.types import Array

arr1D = Array(float64, 1, 'A', readonly=True)
arr2D = Array(float64, 2, 'A', readonly=True)
arr3D = Array(float64, 3, 'A', readonly=True)


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
        if x[i] > x0:
            i = i-1
        else:
            i = i

    assert x[i+1]-x[i] > 0, "input array has to be monotonic"

    d1 = p[i+1] * (x0 - x[i])/(x[i+1]-x[i]) + \
        p[i] * (x[i+1] - x0)/(x[i+1]-x[i])
    return d1


@njit(float64(arr1D, arr1D, float64))
def interp1d_cubic(x, p, x0):
    '''
    1D cubic herimit interpolation
       x 1D array. This must be monotonically increasing. Note the routine does not check this.
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
        if x[i] > x0:
            i = i-1
        else:
            i = i

    xx = (x0 - x[i])/(x[i+1] - x[i])

    d1 = (x[i] - x[i-1])
    d2 = (x[i+1] - x[i])
    d3 = (x[i+2] - x[i+1])

    assert d1 > 0, "input array has to be monotonic"
    assert d2 > 0, "input array has to be monotonic"
    assert d3 > 0, "input array has to be monotonic"

    f0 = p[i]
    f1 = p[i+1]
    f0p = (-p[i-1]*d2 - p[i]*(d1 - d2) + p[i+1]*d1)/2/d1/d2*(x[i+1] - x[i])
    f1p = (-p[i]*d3 - p[i+1]*(d2 - d3) + p[i+2]*d2)/2/d2/d3*(x[i+1] - x[i])

    a = 2*f0 - 2*f1 + f0p + f1p
    b = -3*f0 + 3*f1 - 2*f0p-f1p
    c = f0p
    d = f0

    return a*xx**3 + b*xx**2 + c*xx**1 + d

    '''
    return p[i] + 0.5 * x*(p[i+1] - p[i-1] + x*(2.0*p[i-1] - 5.0*p[i] + 4.0*p[i+1] - p[i+2] + x*(3.0*(p[i] - p[i+1]) + p[i+2] - p[i-1])))
    '''


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
        if y[i] > y0:
            i = i-1
        else:
            i = i

    a0 = interp1d_linear(x, p[i, :], x0)
    a1 = interp1d_linear(x, p[i+1, :], x0)

    assert y[i+1]-y[i] > 0, "input array has to be monotonic"

    return a0 * (y[i+1] - y0)/(y[i+1] - y[i]) + a1 * (y0 - y[i])/(y[i+1]-y[i])


@njit(float64(arr1D, arr1D, arr2D, float64, float64))
def interp2d_cubic(x, y, p, x0, y0):
    '''
    2D cubic interpolation
       x 1D array. This must be monotoci. Note the routine does not check this.
       y 1D array. This must be monotoci. Note the routine does not check this.
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
        if y[i] > y0:
            i = i-1
        else:
            i = i

    a0 = interp1d_cubic(x, p[i-1, :], x0)
    a1 = interp1d_cubic(x, p[i, :], x0)
    a2 = interp1d_cubic(x, p[i+1, :], x0)
    a3 = interp1d_cubic(x, p[i+2, :], x0)

    yy = (y0 - y[i])/(y[i+1] - y[i])

    d1 = (y[i] - y[i-1])
    d2 = (y[i+1] - y[i])
    d3 = (y[i+2] - y[i+1])

    assert d1 > 0, "input array has to be monotonic"
    assert d2 > 0, "input array has to be monotonic"
    assert d3 > 0, "input array has to be monotonic"

    f0 = a1
    f1 = a2
    f0p = (-a0*d2 - a1*(d1 - d2) + a2*d1)/2/d1/d2*(y[i+1] - y[i])
    f1p = (-a1*d3 - a2*(d2 - d3) + a3*d2)/2/d2/d3*(y[i+1] - y[i])

    a = 2*f0 - 2*f1 + f0p + f1p
    b = -3*f0 + 3*f1 - 2*f0p-f1p
    c = f0p
    d = f0

    return a*yy**3 + b*yy**2 + c*yy**1 + d

    '''
    return a1 + 0.5 * yy*(a2 - a0 + yy*(2.0*a0 - 5.0*a1 + 4.0*a2 - a3 + yy*(3.0*(a1 - a2) + a3 - a0)))
    '''


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
        if z[k] > z0:
            k = k-1
        else:
            k = k

    a0 = interp2d_linear(x, y, p[k, :, :], x0, y0)
    a1 = interp2d_linear(x, y, p[k+1, :, :], x0, y0)

    assert z[k+1]-z[k] > 0, "input array has to be monotonic"

    return a0 * (z[k+1] - z0)/(z[k+1] - z[k]) + a1 * (z0 - z[k])/(z[k+1]-z[k])


@njit(float64(arr1D, arr1D, arr1D, arr3D, float64, float64, float64))
def interp3d_cubic(x, y, z, p, x0, y0,  z0):
    '''
    3D cubic interpolation
       x 1D array. This must be monotoci. Note the routine does not check this.
       y 1D array. This must be monotoci. Note the routine does not check this.
       z 1D array. This must be monotoci. Note the routine does not check this.
       p 3D array
       x0 point to interpolate
       y0 point to interpolate
       z0 point to interpolate
    '''

    if z0 < z[1]:
        a0 = interp2d_cubic(x, y, p[0, :, :], x0, y0)
        a1 = interp2d_cubic(x, y, p[1, :, :], x0, y0)
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
        if z[k] > z0:
            k = k-1
        else:
            k = k

    a0 = interp2d_cubic(x, y, p[k-1, :, :], x0, y0)
    a1 = interp2d_cubic(x, y, p[k, :, :], x0, y0)
    a2 = interp2d_cubic(x, y, p[k+1, :, :], x0, y0)
    a3 = interp2d_cubic(x, y, p[k+2, :, :], x0, y0)

    zz = (z0 - z[k])/(z[k+1] - z[k])

    d1 = (z[k] - z[k-1])
    d2 = (z[k+1] - z[k])
    d3 = (z[k+2] - z[k+1])

    assert d1 > 0, "input array has to be monotonic"
    assert d2 > 0, "input array has to be monotonic"
    assert d3 > 0, "input array has to be monotonic"

    f0 = a1
    f1 = a2
    f0p = (-a0*d2 - a1*(d1 - d2) + a2*d1)/2/d1/d2*(z[k+1] - z[k])
    f1p = (-a1*d3 - a2*(d2 - d3) + a3*d2)/2/d2/d3*(z[k+1] - z[k])

    a = 2*f0 - 2*f1 + f0p + f1p
    b = -3*f0 + 3*f1 - 2*f0p-f1p
    c = f0p
    d = f0

    return a*zz**3 + b*zz**2 + c*zz**1 + d


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
    size1 = 15
    size2 = 21

    x = np.linspace(-3, 3, size1)
    y = np.linspace(-3, 3, size2)
    p = np.zeros((len(y), len(x)))
    for j in range(len(y)):
        for i in range(len(x)):
            p[j, i] = np.exp(-(x[i]**2 + y[j]**2)/3)

    x2 = np.hstack((np.linspace(-3, 1, size1*3)
                   [:-1], np.linspace(1, 3, size1*1000)))
    y2 = np.hstack((np.linspace(-3, 1, size2*3)
                   [:-1], np.linspace(1, 3, size2*1000)))
    p2 = np.zeros((len(y2), len(x2)))
    for j in range(len(y2)):
        for i in range(len(x2)):
            p2[j, i] = np.exp(-(x2[i]**2 + y2[j]**2)/3)

    new_x = np.linspace(-3, 3, 2500)
    new_y = np.linspace(-3, 3, 30)

    shape = (len(new_y), len(new_x))
    out0 = np.zeros(shape)
    out1 = np.zeros(shape)
    out2 = np.zeros(shape)

    for j in range(shape[0]):
        for i in range(shape[1]):
            out0[j, i] = interp2d_nearest(x2, y2, p2, new_x[i], new_y[j])
            out1[j, i] = interp2d_linear(x2, y2, p2, new_x[i], new_y[j])
            out2[j, i] = interp2d_cubic(x2, y2, p2, new_x[i], new_y[j])

    return x, y, p, new_x, new_y, out0, out1, out2


@njit
def test3d():
    size1, size2, size3 = 9, 7, 11

    x = np.linspace(-3, 3, size1)
    y = np.linspace(-3, 3, size2)
    z = np.linspace(-3, 3, size3)

    p = np.zeros((size3, size2, size1))
    for k in range(size3):
        for j in range(size2):
            for i in range(size1):
                p[k, j, i] = np.exp(-(x[i]**2 + y[j]**2 + z[k]**2)/3)

    new_x = np.linspace(-3, 3, 25)
    new_y = np.linspace(-3, 3, 15)
    new_z = np.linspace(-3, 3, 31)

    shape = (len(new_z), len(new_y), len(new_x))
    out0 = np.zeros(shape)
    out1 = np.zeros(shape)
    out2 = np.zeros(shape)

    for k in range(shape[0]):
        for j in range(shape[1]):
            for i in range(shape[2]):
                out0[k, j, i] = interp3d_nearest(
                    x, y, z, p, new_x[i], new_y[j], new_z[k])
                out1[k, j, i] = interp3d_linear(
                    x, y, z, p, new_x[i], new_y[j], new_z[k])
                out2[k, j, i] = interp3d_cubic(
                    x, y, z, p, new_x[i], new_y[j], new_z[k])

    return x, y, z, p, new_x, new_y, new_z, out0, out1, out2


print()
