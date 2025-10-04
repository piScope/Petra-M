from itertools import product as prod
import numpy as np
from numpy.linalg import det, norm, inv
from scipy.spatial.distance import pdist

from petram.mfem_config import use_parallel
if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints(
    'PyVectorFEIntegratorBase')


class PyVectorFEIntegratorBase(mfem.PyBilinearFormIntegrator):
    support_metric = False

    def __init__(self, *args, **kwargs):
        mfem.PyBilinearFormIntegrator.__init__(self, *args, **kwargs)
        self._q_order = 0
        self._metric = None
        self._christoffel = None
        self._realimag = False
        self._dmats = [mfem.DenseMatrix() for _i in range(5)]
        self._hmats = None
        self._v = mfem.Vector()

    @property
    def q_order(self):
        return self._q_order

    @q_order.setter
    def q_order(self, value):
        self._q_order = value

    @classmethod
    def coeff_shape(cls, itg_param):
        raise NotImplementedError("subclass must implement coeff_shape")

    @classmethod
    def set_ir(self, trial_fe,  test_fe, trans, delta=0):
        # return integration order
        raise NotImplementedError("subclass must implement coeff_shape")

    @staticmethod
    def get_ds(ir, trans):
        # return step size for finite difference
        raise NotImplementedError("subclass must implement coeff_shape")
