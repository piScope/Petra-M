#
#   define metric tensor and christoffel for cylindrical coordinate
#

import numpy as np
from numba import njit, void, int32, int64, float64, complex128, types

from petram.mfem_config import use_parallel, get_numba_debug
if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

from petram.phys.numba_coefficient import NumbaCoefficient

#
# christoffel
#


def christoffel():
    if mode == 'polar':
        def func(r, z):
            data2 = np.zeros((3, 3, 3), dtype=np.float64)
            data2[0, 1, 1] = -r
            data2[1, 0, 1] = 1/r
            return data2.flatten()
        func = njit(float64[:](float64, float64))(func)
    #
    #  transform function to coefficient
    #

    jitter = mfem.jit.vector(complex=False, shape=(27, ))

    def christoffel(ptx):
        return func(ptx[0], ptx[1])

    return NumbaCoefficient(jitter(christoffel))

#
# metric g_ij (covariant compnent)
#


def metric_diag():
    def func(r, z):
        data2 = np.zeros((3, ), dtype=np.float64)
        data2[0] = 1
        data2[1] = r**2
        data2[2] = 1
        return data2.flatten()
    func = njit(float64[:](float64, float64))(func)

    def metric(ptx):
        return func(ptx[0], ptx[1])

    #
    #  transform function to coefficient
    #
    jitter = mfem.jit.vector(complex=False, shape=(3, ))

    return NumbaCoefficient(jitter(metric))
