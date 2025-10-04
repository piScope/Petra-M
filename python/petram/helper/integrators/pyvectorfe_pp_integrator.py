#
# partial-partial integrator
#

from petram.helper.integrators.pyvectorfe_base import PyVectorFEIntegratorBase
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


class PyVectorFEPartialPartialIntegrator(PyVectorFEIntegratorBase):
    #
    #
    #    vec(v)_i lamblda(i, j, l, k)  \partial_l \partial_j vec(u)_k
    #
    #
    def __init__(self, lam, vdim1, vdim2, ir=None):
        PyVectorFEIntegratorBase.__init__(self, ir)

        self._ir = self.GetIntegrationRule()

        self.te_vshape = mfem.DenseMatrix()

        self.partelmat = mfem.DenseMatrix()
        self.val = mfem.Vector()

        self.vdim1 = vdim1
        self.vdim2 = vdim2
        self.lam = lam

    @classmethod
    def coeff_shape(cls, vdim1, vdim2=None, ir=None):
        assert vdim1 == vdim2, "vdim1 must be the same as vdim2."
        return (vdim1, vdim1, vdim1, vdim1)

    @classmethod
    def set_ir(self, trial_fe,  test_fe, trans, delta=0):

        if trial_fe.Space() == mfem.FunctionSpace.rQk:
            ir = mfem.RefinedIntRules.Get(trial_fe.GetGeomType(), order)
            order = trial_fe.GetOrder() + test_fe.GetOrder() - 2
        else:
            order = trial_fe.GetOrder() + test_fe.GetOrder() - 1

        ir = mfem.IntRules.Get(trial_fe.GetGeomType(), order)

        self.ir = ir

    @staticmethod
    def get_ds(ir, trans):
        ptx = [trans.Transform(ir.IntPoint(i)) for i in range(ir.GetNPoints())]
        ds = np.sqrt(np.sqrt(2e-16))*np.max(pdist(ptx))
        #ds2 = np.sqrt(2e-16)*np.max(pdist(ptx))
        return ds*100

    def get_hessian(self, fe, ip, trans, itrans, ds):

        sdim = trans.GetSpaceDim()

        self._v.SetSize(sdim)

        if self._hmats is None:
            self._hmats = [mfem.DenseMatrix(
                fe.GetDof(), sdim) for _i in range(5**sdim)]
        else:
            for m in self._hmats:
                m.SetSize(fe.GetDof(), sdim)
        for m in self._hmats:
            m.Assign(0.0)

        vv = self._v.GetDataArray()
        ptx = trans.Transform(ip)
        ip2 = mfem.IntegrationPoint()

        idx = [0, 1, 2, 3, 4]
        sp = [-2, -1, 0, 1, 2]

        itrans.SetInitialGuessType(3)
        itrans.SetInitialGuess(ip)

        if sdim == 2:
            hmats = np.array(self._hmats).reshape(5, 5)
            for ii, jj in prod(idx, idx):
                vv[:] = ptx
                vv[0] += ds*sp[ii]
                vv[1] += ds*sp[jj]
                itrans.Transform(self._v, ip2)
                trans.SetIntPoint(ip2)
                fe.CalcVShape(trans, hmats[ii, jj])

            xx = (-hmats[0, 2].GetDataArray()
                  + 16*hmats[1, 2].GetDataArray()
                  - 30*hmats[2, 2].GetDataArray()
                  + 16*hmats[3, 2].GetDataArray()
                  - hmats[4, 2].GetDataArray())/12
            yy = (-hmats[2, 0].GetDataArray()
                  + 16*hmats[2, 1].GetDataArray()
                  - 30*hmats[2, 2].GetDataArray()
                  + 16*hmats[2, 3].GetDataArray()
                  - hmats[2, 4].GetDataArray())/12

            def dx(i):
                x0 = (hmats[i, 0].GetDataArray()
                      - 8*hmats[i, 1].GetDataArray()
                      + 8*hmats[i, 3].GetDataArray()
                      - hmats[i, 4].GetDataArray())/12
                return x0

            def dy(i):
                x0 = (hmats[0, i].GetDataArray()
                      - 8*hmats[1, i].GetDataArray()
                      + 8*hmats[3, i].GetDataArray()
                      - hmats[4, i].GetDataArray())/12
                return x0
            xy = (dx(0) - 8*dx(1) + 8*dx(3) - dx(4))/12
            yx = (dy(0) - 8*dy(1) + 8*dy(3) - dy(4))/12

            ret = [xx, (xy+yx)/2, (xy+yx)/2., yy]

        elif sdim == 3:
            hmats = np.array(self._hmats).reshape(5, 5, 5)
            for ii, jj, kk in prod(idx, idx, idx):
                vv[:] = ptx
                vv[0] += ds*sp[ii]
                vv[1] += ds*sp[jj]
                vv[2] += ds*sp[kk]
                itrans.Transform(self._v, ip2)
                trans.SetIntPoint(ip2)
                fe.CalcVShape(trans, hmats[ii, jj, kk])

            xx = (-hmats[0, 2, 2].GetDataArray()
                  + 16*hmats[1, 2, 2].GetDataArray()
                  - 30*hmats[2, 2, 2].GetDataArray()
                  + 16*hmats[3, 2, 2].GetDataArray()
                  - hmats[4, 2, 2].GetDataArray())/12
            yy = (-hmats[2, 0, 2].GetDataArray()
                  + 16*hmats[2, 1, 2].GetDataArray()
                  - 30*hmats[2, 2, 2].GetDataArray()
                  + 16*hmats[2, 3, 2].GetDataArray()
                  - hmats[2, 4, 2].GetDataArray())/12
            zz = (-hmats[2, 2, 0].GetDataArray()
                  + 16*hmats[2, 2, 1].GetDataArray()
                  - 30*hmats[2, 2, 2].GetDataArray()
                  + 16*hmats[2, 2, 3].GetDataArray()
                  - hmats[2, 2, 4].GetDataArray())/12

            def dx(i, j):
                x0 = (hmats[0, i, j].GetDataArray()
                      - 8*hmats[1, i, j].GetDataArray()
                      + 8*hmats[3, i, j].GetDataArray()
                      - hmats[4, i, j].GetDataArray())/12
                return x0

            def dy(i, j):
                x0 = (hmats[i, 0, j].GetDataArray()
                      - 8*hmats[i, 1, j].GetDataArray()
                      + 8*hmats[i, 3, j].GetDataArray()
                      - hmats[i, 4, j].GetDataArray())/12
                return x0

            def dz(i, j):
                x0 = (hmats[i, j, 0].GetDataArray()
                      - 8*hmats[i, j, 1].GetDataArray()
                      + 8*hmats[i, j, 3].GetDataArray()
                      - hmats[i, j, 4].GetDataArray())/12
                return x0

            xy = (dx(0, 2) - 8*dx(1, 2) + 8*dx(3, 2) - dx(4, 2))/12
            xz = (dx(2, 0) - 8*dx(2, 1) + 8*dx(2, 3) - dx(2, 4))/12
            yz = (dy(2, 0) - 8*dy(2, 1) + 8*dy(2, 3) - dy(2, 4))/12

            ret = [xx, xy, xz, xy, yy, yz, xz, yz, zz]

        # shape sdim (dir. grad), Dof, sdim
        w = ds**2
        ret = np.stack(ret).reshape(sdim, sdim, fe.GetDof(), sdim)/w
        return ret

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

        for i in range(self.ir.GetNPoints()):

            ip = self.ir.IntPoint(i)
            trans.SetIntPoint(ip)
            test_fe.CalcVShape(trans, self.te_vshape)

            # test_i lamblda(i, j, k)  \partial_j trial_k
            self.lam.Eval(self.val, trans, ip)
            lam = self.val.GetDataArray().reshape(tr_vdim, tr_vdim, tr_vdim, tr_vdim)

            hshape = self.get_hessian(trial_fe, ip, trans, itrans, ds)

            # DoF sdim * sdim(grad) sdim(grad),  DoF sdim
            udvdx = np.tensordot(self.te_vshape.GetDataArray(),
                                 hshape, 0) * ip.weight * trans.Weight()
            # if i == 0:
            #    print("hshape xx", hshape[0,1,:,:])

            for ii, jj, kk, ll in prod(rr, rr, rr, rr):
                partelmat_arr += lam[ii, kk, ll, jj] * \
                    udvdx[:, ii, kk, ll, :, jj]

        elmat.AddMatrix(self.partelmat, 0, 0)
