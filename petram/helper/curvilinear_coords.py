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


def eval_metric_txt(txt, g, l, return_txt=False):
    name = txt.split(',')[0]
    params = ','.join(txt.split(',')[1:])

    try:
        params = eval(params, g, l)
    except BaseException:
        print("can not process metric input: " + txt)
        raise

    if return_txt:
        return name, params

    if name in globals():
        return globals()[name](params)
    else:
        try:
            metric = eval(name, g, l)
        except BaseException:
            print("can not process metric input: " + txt)
            raise
    return metric(params)


class coordinate_system(ABC):
    @abstractclassmethod
    def __init__(self, params):
        ...

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
# planer
#


class planer1d(coordinate_system):
    def __init__(self, params, use_covariant_vec=False):
        self.vdim1 = 3  # test space size
        self.vdim2 = 3  # trial space size

        self.esindex = (0, params[0]*1j, params[1]*1j)
        self.use_covariant_vec = use_covariant_vec

    @classmethod
    def metric(cls):
        return None


class planer2d(coordinate_system):
    def __init__(self, params, use_covariant_vec=False):
        self.vdim1 = 3  # test space size
        self.vdim2 = 3  # trial space size

        self.esindex = (0, 1, params[0]*1j)
        self.use_covariant_vec = use_covariant_vec

    @classmethod
    def metric(cls):
        return None

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


def cyl_cometric(r):
    #
    # g_ij
    #
    data2 = np.zeros((3, ), dtype=np.float64)
    data2[0] = 1
    data2[1] = r**2
    data2[2] = 1
    return data2.flatten()


def cyl_ctmetric(r):
    #
    # g^ij
    #
    data2 = np.zeros((3, ), dtype=np.float64)
    data2[0] = 1
    data2[1] = 1/r**2
    data2[2] = 1
    return data2.flatten()


class cylindrical1d(coordinate_system):
    def __init__(self, params, use_covariant_vec=False):
        self.vdim1 = 3  # test space size
        self.vdim2 = 3  # trial space size

        self.esindex = (0, params[0]*1j, params[1]*1j)
        self.use_covariant_vec = use_covariant_vec

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
    def cometric(self):
        func = njit(float64[:](float64))(cyl_cometric)

        def metric(ptx):
            return func(ptx[0])
        jitter = mfem.jit.vector(complex=False, shape=(3, ))

        return jitter(metric)

    @classmethod
    def ctmetric(self):
        func = njit(float64[:](float64))(cyl_ctmetric)

        def metric(ptx):
            return func(ptx[0])
        jitter = mfem.jit.vector(complex=False, shape=(3, ))

        return jitter(metric)

    @classmethod
    def metric(cls):
        mm1 = cls.cometric()
        mm2 = cls.ctmetric()

        return (mm1, mm2)


def cylindrical1dct(params):
    return cylindrical1d(params, use_covariant_vec=False)


def cylindrical1dco(params):
    return cylindrical1d(params, use_covariant_vec=True)


class cylindrical2d(cylindrical1d):
    def __init__(self, params, use_covariant_vec=False):
        self.vdim1 = 3  # test space size
        self.vdim2 = 3  # trial space size
        self.esindex = (0, params[0]*1j, 1)
        self.use_covariant_vec = use_covariant_vec


def cylindrical2dct(params):
    return cylindrical2d(params, use_covariant_vec=False)


def cylindrical2dco(params):
    return cylindrical2d(params, use_covariant_vec=True)
