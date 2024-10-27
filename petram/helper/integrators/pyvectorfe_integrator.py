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

    @classmethod
    def coeff_shape(cls, itg_param):
        raise NotImplementedError("subclass must implement coeff_shape")

    def set_ir(self, trial_fe,  test_fe, trans, delta=0):
        delta=0
        order = trial_fe.GetOrder() + test_fe.GetOrder() - 1

 
        #if trial_fe.Space() == mfem.FunctionSpace.rQk:
        #    ir = mfem.RefinedIntRules.Get(trial_fe.GetGeomType(), order)
        #else:
        ir = mfem.IntRules.Get(trial_fe.GetGeomType(), order)

        print("order here", order)
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
        
        ptx = trans.Transform(ip)
        vv = v.GetDataArray()        

        ip2 = mfem.IntegrationPoint()

        mats = [mfem.DenseMatrix(fe.GetDof(), sdim) for _i in range(5)]

        ret = []
        for si in range(sdim):
            for ii, j in enumerate([-2, -1, 0, 1, 2]):
               vv[:]= ptx
               vv[si] += ds*j
               itrans.Transform(v, ip2)
               trans.SetIntPoint(ip2)
               fe.CalcVShape(trans, mats[ii])
               
            dd1 = (mats[0].GetDataArray()
                  -8*mats[1].GetDataArray()
                  +8*mats[3].GetDataArray()
                  -mats[4].GetDataArray())/12/ds
            ret.append(dd1)
            #ret.append(mats[3].GetDataArray())

        return np.stack(ret) # shape sdim (dir. grad), Dof, sdim

class PyVectorFEPartialIntegrator(PyVectorFEIntegratorBase):
    '''
  
    vec(v)_i lamblda(i, j, k)  \partial_j vec(u)_k

    V and U are ND

    '''
    def __init__(self, lam, vdim1, vdim2, ir=None):
        PyVectorFEIntegratorBase.__init__(self, ir)

        self._ir = self.GetIntegrationRule()

        self.te_vshape = mfem.DenseMatrix()

        self.partelmat = mfem.DenseMatrix()
        self.val = mfem.Vector()

        self.vdim1=vdim1
        self.vdim2=vdim2
        self.lam = lam

    @classmethod
    def coeff_shape(cls, vdim1, vdim2=None, ir=None):
        assert vdim1 == vdim2, "vdim1 must be the same as vdim2."
        return (vdim1, vdim1, vdim1)

    def AssembleElementMatrix(self, el, trans, elmat):
        self.AssembleElementMatrix2(el, el, trans, elmat)

    def AssembleElementMatrix2(self, trial_fe, test_fe, trans, elmat):
        if self._ir is None:
            self.set_ir(trial_fe, test_fe, trans)

        sdim = trans.GetSpaceDim()
        tr_vdim = max(sdim, trial_fe.GetRangeDim())
        te_vdim = max(sdim, test_fe.GetRangeDim())

        tr_nd = trial_fe.GetDof()
        te_nd = test_fe.GetDof()
        
        self.partelmat.SetSize(te_nd, tr_nd)
        self.partelmat.Assign(0.0)
        partelmat_arr = self.partelmat.GetDataArray()
        
        self.te_vshape.SetSize(te_nd, te_vdim)
        self.val.SetSize(tr_vdim**3)
        
        itrans = mfem.InverseElementTransformation(trans)

        elmat.SetSize(te_nd, tr_nd)
        elmat.Assign(0.0)

        
        ds = self.get_ds(self.ir, trans)
        rr = range(tr_vdim)
        print(self.ir.GetNPoints())
        for i in range(self.ir.GetNPoints()):
            
            ip = self.ir.IntPoint(i)
            trans.SetIntPoint(ip)
            test_fe.CalcVShape(trans, self.te_vshape)

            # test_i lamblda(i, j, k)  \partial_j trial_k
            self.lam.Eval(self.val, trans, ip)
            lam = self.val.GetDataArray().reshape(tr_vdim, tr_vdim, tr_vdim)
            
            dshape = self.get_dshape(trial_fe, ip, trans, itrans, ds)

            # DoF sdim * sdim(grad) DoF sdim
            udvdx = np.tensordot(self.te_vshape.GetDataArray(),
                                 dshape, 0)* ip.weight * trans.Weight()
            
            for ii, jj, kk in prod(rr, rr, rr):
               partelmat_arr += lam[ii, kk, jj]*udvdx[:, ii, kk, :, jj]
               
        elmat.AddMatrix(self.partelmat, 0, 0)
