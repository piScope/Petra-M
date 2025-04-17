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
dprint1, dprint2, dprint3 = petram.debug.init_dprints('PyVectorFEIntegratorBase')


class PyVectorFEIntegratorBase(mfem.PyBilinearFormIntegrator):
    support_metric = False

    def __init__(self, *args, **kwargs):
        mfem.PyBilinearFormIntegrator.__init__(self, *args, **kwargs)
        self._q_order = 0
        self._metric = None
        self._christoffel = None
        self._realimag = False

    @property
    def q_order(self):
        return self._q_order

    @q_order.setter
    def q_order(self, value):
        self._q_order = value

    def set_ir(self, trial_fe,  test_fe, trans, delta=0):
        order = (trial_fe.GetOrder() + test_fe.GetOrder() +
                 trans.OrderW() + self.q_order + delta)

        if trial_fe.Space() == mfem.FunctionSpace.rQk:
            ir = mfem.RefinedIntRules.Get(trial_fe.GetGeomType(), order)
        else:
            ir = mfem.IntRules.Get(trial_fe.GetGeomType(), order)

        self.ir = ir

    @staticmethod
    def get_ds(ir, trans):
        ptx = [trans.Transform(ir.IntPoint(i)) for i in range(ir.GetNPoints())]
        ds = np.sqrt(2e-16)*np.max(pdist(ptx))

        return ds
    
    @staticmethod
    def get_dshape(fe, ip, trans, itrans, ds):
        sdim = trans.GetSpaceDim()    
        v = mfem.Vector(sdim)
        v.Assign(0.0)
        
        trans.Transform(ip, v)
        vv = v.GetDataArray()        
        v0  = vv.copy()

        ip2 = mfem.IntegrationPoint()

        mats = [mfem.DenseMatrix(fe.GetDof(), sdim) for _i in range(5)]

        ret = []
        for si in range(sdim):
            for ii, j in enumerate([-2, -1, 0, 1, 2]):
               vv[:]= v0
               vv[si] += h
            itrans.Transform(v, ip2)
            fe.CalcVShape(ip2, mats[ii])
            dd1 = (mats[0].GetDataArray()
                  -8*mats[1].GetDataArray()
                  +8*mats[3].GetDataArray()
                  -mats[4].GetDataArray())/12/ds
            ret.appendd(dd1)

        return np.stack(ret) # shape sdim (dir. grad), Dof, sdim

