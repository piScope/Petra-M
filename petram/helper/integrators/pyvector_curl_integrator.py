#
#  curl integrator
#  directional curl integrator
#

from petram.helper.integrators.pyvector_integrator_base import PyVectorIntegratorBase
from petram.phys.phys_const import levi_civita3
from itertools import product as prod
import numpy as np
from numpy.linalg import det, norm, inv

from petram.mfem_config import use_parallel
if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints(
    'PyVectorCurlIntegrator')


class PyVectorCurlIntegratorBase(PyVectorIntegratorBase):
    use_complex_coefficient = True
    support_metric = True

    def __init__(self, lam, vdim1=None, vdim2=None, esindex=None, metric=None,
                 use_covariant_vec=False, *, ir=None):

        PyVectorIntegratorBase.__init__(self, use_covariant_vec, ir)
        self.init_step2(lam, vdim1, vdim2, esindex, metric)

    def alloc_workspace(self):
        #
        #  allocate array for assembly
        #
        self.tr_shape = mfem.Vector()
        self.tr_shape = mfem.Vector()
        self.te_shape = mfem.Vector()
        self.tr_dshape = mfem.DenseMatrix()
        self.tr_dshapedxt = mfem.DenseMatrix()

        self.partelmat = mfem.DenseMatrix()
        self.valr = mfem.Vector()
        self.vali = mfem.Vector()

    @classmethod
    def coeff_shape(cls, vdim1, vdim2=None, esindex=None, ir=None):

        metric_obj = cls._proc_vdim1vdim2(vdim1, vdim2)

        if metric_obj:
            vdim1 = metric_obj.vdim1
            vdim2 = metric_obj.vdim2
            esindex = metric_obj.esindex
        else:
            if vdim2 is None:
                vdim2 = vdim1

        if esindex is None:
            esdim = vdim2
        else:
            esdim = len(esindex)

        return (vdim1, vdim2,)

    def AssembleElementMatrix(self, el, trans, elmat):
        self.AssembleElementMatrix2(el, el, trans, elmat)

    def AssembleElementMatrix2(self, trial_fe, test_fe, trans, elmat):
        assert False, "subclass has to implement this method"


class PyVectorCurlIntegrator(PyVectorCurlIntegratorBase):
    def AssembleElementMatrix2(self, trial_fe, test_fe, trans, elmat):
        # if self.ir is None:
        #    self.ir = mfem.DiffusionIntegrator.GetRule(trial_fe, test_fe)
        if self._ir is None:
            self.set_ir(trial_fe, test_fe, trans)

        tr_nd = trial_fe.GetDof()
        te_nd = test_fe.GetDof()

        elmat.SetSize(te_nd*self.vdim_te, tr_nd*self.vdim_tr)
        elmat.Assign(0.0)
        self.partelmat.SetSize(te_nd, tr_nd)

        partelmat_arr = self.partelmat.GetDataArray()

        dim = trial_fe.GetDim()
        sdim = trans.GetSpaceDim()
        square = (dim == sdim)

        self.tr_shape.SetSize(tr_nd)
        self.te_shape.SetSize(te_nd)
        self.tr_dshape.SetSize(tr_nd, dim)
        self.tr_dshapedxt.SetSize(tr_nd, sdim)

        tr_shape_arr = self.tr_shape.GetDataArray()
        te_shape_arr = self.te_shape.GetDataArray()
        tr_dshapedxt_arr = self.tr_dshapedxt.GetDataArray()

        tr_merged_arr = np.zeros((tr_nd, self.esdim), dtype=np.complex128)

        #scalar_coeff = isinstance(self.lam_real, mfem.Coefficient)
        # if scalar_coeff:
        #    assert self.vdim_te == self.vdim_tr, "scalar coefficeint allows only for square matrix"

        #print(self.es_weight, self.esflag2, self.esflag)
        for i in range(self.ir.GetNPoints()):

            ip = self.ir.IntPoint(i)
            trans.SetIntPoint(ip)
            w = trans.Weight()

            trial_fe.CalcShape(ip, self.tr_shape)
            test_fe.CalcShape(ip, self.te_shape)

            trial_fe.CalcDShape(ip, self.tr_dshape)

            mfem.Mult(self.tr_dshape, trans.AdjugateJacobian(),
                      self.tr_dshapedxt)

            w1 = np.sqrt(1./w) if square else np.sqrt(1/w/w/w)
            w2 = np.sqrt(w)

            # construct merged test/trial shape
            tr_merged_arr[:, self.esflag] = tr_dshapedxt_arr*w1

            for i, k in enumerate(self.esflag2):
                tr_merged_arr[:, k] = tr_shape_arr*w2 * \
                    self.es_weight[i]  # nd vdim(d/dx)

            shape = (self.vdim_te, self.vdim_tr)
            lam = self.eval_complex_lam(trans, ip, shape)

            if self._metric is not None:
                vu = np.tensordot(
                    te_shape_arr*w2, tr_shape_arr*w2, 0)*ip.weight

                g_xx = self.eval_cometric(trans, ip)  # g_xx
                chris = self.eval_christoffel(trans, ip, self.esdim)

                if self.use_covariant_vec:
                    tmp = np.tensordot(g_xx, levi_civita3,
                                       axes=(1, 0))  # lp + pkj = lkj
                else:
                    tmp = np.tensordot(levi_civita3, g_xx,
                                       axes=(2, 0))  # lkq + qj = lkj

                # il lkj  = ikj
                L = np.tensordot(lam, tmp, (1, 0))

                if self.use_covariant_vec:
                    # ikj mjk  = im
                    M = -np.tensordot(L, chris, ((1, 2), (2, 1)))
                else:
                    # ikj jmk  = im
                    M = np.tensordot(L, chris, ((1, 2), (2, 0)))

            else:
                L = np.tensordot(lam, levi_civita3, (1, 0))
                M = None

            vdudx = np.tensordot(
                te_shape_arr*w2, tr_merged_arr, 0)*ip.weight  # nd, nd, vdim(d/dx)

            if self._realimag:
                for i, j in prod(range(self.vdim_te), range(self.vdim_tr)):
                    self.partelmat.Assign(0.0)

                    for k in range(self.esdim):
                        partelmat_arr[:, :] += (L[i, k, j]
                                                * vdudx[:, :, k]).real
                    if M is not None:
                        partelmat_arr[:, :] += (M[i, j]
                                                * vu[:, :]).real

                    elmat.AddMatrix(self.partelmat, te_nd*i, tr_nd*j)

            else:
                for i, j in prod(range(self.vdim_te), range(self.vdim_tr)):
                    self.partelmat.Assign(0.0)
                    for k in range(self.esdim):
                        partelmat_arr[:, :] += (L[i, k, j]
                                                * vdudx[:, :, k]).imag
                    if M is not None:
                        partelmat_arr[:, :] += (M[i, j]
                                                * vu[:, :]).imag
                    elmat.AddMatrix(self.partelmat, te_nd*i, tr_nd*j)


class PyVectorDirectionalCurlIntegrator(PyVectorCurlIntegratorBase):
    def AssembleElementMatrix2(self, trial_fe, test_fe, trans, elmat):
        # if self.ir is None:
        #    self.ir = mfem.DiffusionIntegrator.GetRule(trial_fe, test_fe)
        if self._ir is None:
            self.set_ir(trial_fe, test_fe, trans)

        tr_nd = trial_fe.GetDof()
        te_nd = test_fe.GetDof()

        elmat.SetSize(te_nd*self.vdim_te, tr_nd*self.vdim_tr)
        elmat.Assign(0.0)
        self.partelmat.SetSize(te_nd, tr_nd)

        partelmat_arr = self.partelmat.GetDataArray()

        dim = trial_fe.GetDim()
        sdim = trans.GetSpaceDim()
        square = (dim == sdim)

        self.tr_shape.SetSize(tr_nd)
        self.te_shape.SetSize(te_nd)
        self.tr_dshape.SetSize(tr_nd, dim)
        self.tr_dshapedxt.SetSize(tr_nd, sdim)

        tr_shape_arr = self.tr_shape.GetDataArray()
        te_shape_arr = self.te_shape.GetDataArray()
        tr_dshapedxt_arr = self.tr_dshapedxt.GetDataArray()

        tr_merged_arr = np.zeros((tr_nd, self.esdim), dtype=np.complex128)

        #scalar_coeff = isinstance(self.lam_real, mfem.Coefficient)
        # if scalar_coeff:
        #    assert self.vdim_te == self.vdim_tr, "scalar coefficeint allows only for square matrix"

        #print(self.es_weight, self.esflag2, self.esflag)
        for i in range(self.ir.GetNPoints()):

            ip = self.ir.IntPoint(i)
            trans.SetIntPoint(ip)
            w = trans.Weight()

            trial_fe.CalcShape(ip, self.tr_shape)
            test_fe.CalcShape(ip, self.te_shape)

            trial_fe.CalcDShape(ip, self.tr_dshape)

            mfem.Mult(self.tr_dshape, trans.AdjugateJacobian(),
                      self.tr_dshapedxt)

            w1 = np.sqrt(1./w) if square else np.sqrt(1/w/w/w)
            w2 = np.sqrt(w)

            # construct merged test/trial shape
            tr_merged_arr[:, self.esflag] = tr_dshapedxt_arr*w1

            for i, k in enumerate(self.esflag2):
                tr_merged_arr[:, k] = tr_shape_arr*w2 * \
                    self.es_weight[i]  # nd vdim(d/dx)

            vdudx = np.tensordot(
                te_shape_arr*w2, tr_merged_arr, 0)*ip.weight  # nd, nd, vdim(d/dx)

            shape = (self.vdim_te, self.vdim_tr)
            lam = self.eval_complex_lam(trans, ip, shape)

            if self._metric is not None:
                vu = np.tensordot(
                    te_shape_arr*w2, tr_shape_arr*w2, 0)*ip.weight

                g_xx = self.eval_cometric(trans, ip)  # g_xx
                chris = self.eval_christoffel(trans, ip, self.esdim)

                if self.use_covariant_vec:
                    tmp = np.tensordot(g_xx, levi_civita3,
                                       axes=(1, 0))  # pi ilj -> plj
                else:
                    tmp = np.tensordot(levi_civita3, g_xx,
                                       axes=(2, 0))  # ilm mj -> ilj

            else:
                tmp = levi_civita3

            # plj (or ilj) + lk   -> pjk or ijk
            tmp = np.tensordot(tmp, lam, (1, 0))
            L = np.swapaxes(tmp, 1, 2)  # pkj or ikj

            if self._metric is not None:
                if self.use_covariant_vec:
                    # pkj + nkj -> pn
                    M = -np.tensordot(L, chris, ((1, 2), (1, 2)))
                else:
                    # ikj jkn  -> in
                    M = np.tensordot(L, chris, ((2, 1), (0, 1)))
            else:
                M = None

            if self._realimag:
                for i, j in prod(range(self.vdim_te), range(self.vdim_tr)):
                    self.partelmat.Assign(0.0)

                    for k in range(self.esdim):
                        partelmat_arr[:, :] += (L[i, k, j]
                                                * vdudx[:, :, k]).real
                    if M is not None:
                        partelmat_arr[:, :] += (M[i, j]
                                                * vu[:, :]).real

                    elmat.AddMatrix(self.partelmat, te_nd*i, tr_nd*j)

            else:
                for i, j in prod(range(self.vdim_te), range(self.vdim_tr)):
                    self.partelmat.Assign(0.0)
                    for k in range(self.esdim):
                        partelmat_arr[:, :] += (L[i, k, j]
                                                * vdudx[:, :, k]).imag
                    if M is not None:
                        partelmat_arr[:, :] += (M[i, j]
                                                * vu[:, :]).imag

                    elmat.AddMatrix(self.partelmat, te_nd*i, tr_nd*j)
