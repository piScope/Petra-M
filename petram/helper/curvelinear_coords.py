#
#   define metric tensor and christoffel for cylindrical coordinate
#
from abc import ABC, abstractclassmethod, abstractmethod

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
        ...

    @abstractclassmethod
    def christoffel(cls):
        #
        #  christoffel symbol
        #
        ...

    @abstractclassmethod
    def dchristoffel(cls):
        #
        #  derivative of christoffel symbol [i,j,k, l] = d/dx^l gamma^i_jk
        #
        ...

    @abstractclassmethod
    def metric(cls):
        #
        # metric g_ij (covariant compnent)
        #
        # this method should return vector if is_diag_metric=True
        # otherwise, it returns matrix
        ...

#
# cylindrical
#


def cyl_chris(r):
    data2 = np.zeros((3, 3, 3), dtype=np.float64)
    data2[0, 1, 1] = -r
    data2[1, 0, 1] = 1/r
    data2[1, 1, 0] = 1/r
    return data2.flatten()


def cyl_dchris(r):
    data2 = np.zeros((3, 3, 3, 3), dtype=np.float64)
    data2[0, 1, 1, 0] = -1
    data2[1, 0, 1, 0] = -1/r/r
    data2[1, 1, 0, 0] = -1/r/r
    return data2.flatten()


def cyl_metric(r):
    #
    # g_ij
    #
    data2 = np.zeros((3, ), dtype=np.float64)
    data2[0] = 1
    data2[1] = r**2
    data2[2] = 1
    return data2.flatten()


class cylindrical1d(coordinate_system):
    @classmethod
    def is_diag_metric(self):
        return True

    @classmethod
    def christoffel(self):
        func = njit(float64[:](float64))(cyl_chris)

        jitter = mfem.jit.vector(complex=False, shape=(27, ))

        def christoffel(ptx):
            return func(ptx[0])

        return jitter(christoffel)

    @classmethod
    def dchristoffel(self):
        func = njit(float64[:](float64))(cyl_dchris)

        jitter = mfem.jit.vector(complex=False, shape=(81, ))

        def dchristoffel(ptx):
            return func(ptx[0])

        return jitter(dchristoffel)

    @classmethod
    def metric(self):
        func = njit(float64[:](float64))(cyl_metric)

        def metric(ptx):
            return func(ptx[0])
        jitter = mfem.jit.vector(complex=False, shape=(3, ))

        return jitter(metric)


cylindrical2d = cylindrical1d
