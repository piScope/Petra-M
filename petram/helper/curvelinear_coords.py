#
#   define metric tensor and christoffel for cylindrical coordinate
#
from abc import ABC, abstractclassmethod

import numpy as np
from numba import njit, void, int32, int64, float64, complex128, types

from petram.mfem_config import use_parallel, get_numba_debug
if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

from petram.phys.numba_coefficient import NumbaCoefficient


class coordinate_system(ABC):

    @abstractclassmethod
    def is_diag_metric(cls):
        pass

    @abstractclassmethod
    def christoffel(cls):
        #
        #  christoffel symbol
        #

    @abstractclassmethod
    def metric(cls):
        #
        # metric g_ij (covariant compnent)
        #

        # this method should return vector if is_diag_metric=True
        # otherwise, it returns matrix

        #
        # cylindrical
        #


class polar_coords(coordinate_system):
    @classmethod
    def is_diag_metric(cls):
        return True

    @classmethod
    def christoffel(cls):
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

    @classmethod
    def metric(cls):
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
