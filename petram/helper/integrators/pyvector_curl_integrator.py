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

        #
        # Generalization of curl-curl integrator to support metric
        #

        PyVectorIntegratorBase.__init__(self, use_covariant_vec, ir)

        if not hasattr(lam, "get_real_coefficient"):
            self.lam_real = lam
            self.lam_imag = None
        else:
            self.lam_real = lam.get_real_coefficient()
            self.lam_imag = lam.get_imag_coefficient()

        if metric is None:
            metric_obj = self.__class__._proc_vdim1vdim2(vdim1, vdim2)
        else:
            metric_obj = metric

        self.config_metric_vdim_esindex(metric_obj, vdim1, vdim2, esindex)

        self._ir = self.GetIntegrationRule()
        self.alloc_workspace()

        # print('esdim flag', self.esdim, self.esflag, self.esflag2)

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

            '''
                lam = self.lam_real.Eval(trans, ip)
                if self.lam_imag is not None:
                    lam = lam + 1j*self.lam_imag.Eval(trans, ip)
                lam = np.diag([lam]*self.vdim_te)
            else:
                self.lam_real.Eval(self.valr, trans, ip)
                lam = self.valr.GetDataArray()
                if self.lam_imag is not None:
                    self.lam_imag.Eval(self.vali, trans, ip)
                lam = lam + 1j*self.vali.GetDataArray()
            lam = lam.reshape
            '''

            # il + lkj (or lkq)-> ikj (or ikq)
            tmp = np.tensordot(lam, levi_civita3, (1, 0))

            if self._metric is not None and not self.use_covariant_vec:
                g_qj = self.eval_cometric(trans, ip)  # g_{qj}
                tmp = np.tensordot(tmp, g_qj,  axes=(2, 0))  # ikq + qj = ikj
            else:
                pass
            L = tmp

            vdudx = np.tensordot(
                te_shape_arr*w2, tr_merged_arr, 0)*ip.weight  # nd, nd, vdim(d/dx)

            if self._realimag:
                for i, j in prod(range(self.vdim_te), range(self.vdim_tr)):
                    self.partelmat.Assign(0.0)

                    for k in range(self.esdim):
                        partelmat_arr[:, :] += (L[i, k, j]
                                                * vdudx[:, :, k]).real

                    elmat.AddMatrix(self.partelmat, te_nd*i, tr_nd*j)

            else:
                for i, j in prod(range(self.vdim_te), range(self.vdim_tr)):
                    self.partelmat.Assign(0.0)
                    for k in range(self.esdim):
                        partelmat_arr[:, :] += (L[i, k, j]
                                                * vdudx[:, :, k]).imag
                    elmat.AddMatrix(self.partelmat, te_nd*i, tr_nd*j)

            if self._metric is not None and not self._use_covariant_vec:
                vu = np.tensordot(
                    te_shape_arr*w2, tr_shape_arr*w2, 0)*ip.weight

                chris = self.eval_christoffel(trans, ip, self.esdim)
                # if self._use_covariant_vec:
                #    M = -np.tensordot(L, chris, ([2, 1], [1, 2]))
                #    # this is zero due to symmetry
                # else:
                M = np.tensordot(L, chris, ([2, 1], [0, 1]))

                if self._realimag:
                    for i, j in prod(range(self.vdim_te), range(self.vdim_tr)):
                        self.partelmat.Assign(0.0)
                        partelmat_arr[:, :] += (M[i, j]
                                                * vu[:, :]).real
                        elmat.AddMatrix(self.partelmat, te_nd*i, tr_nd*j)

                else:
                    for i, j in prod(range(self.vdim_te), range(self.vdim_tr)):
                        self.partelmat.Assign(0.0)
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

            shape = (self.vdim_te, self.vdim_tr)
            lam = self.eval_complex_lam(trans, ip, shape)

            '''
            if scalar_coeff:
                lam = self.lam_real.Eval(trans, ip)
                if self.lam_imag is not None:
                    lam = lam + 1j*self.lam_imag.Eval(trans, ip)
                lam = np.diag([lam]*self.vdim_te)
            else:
                self.lam_real.Eval(self.valr, trans, ip)
                lam = self.valr.GetDataArray()
                if self.lam_imag is not None:
                    self.lam_imag.Eval(self.vali, trans, ip)
                lam = lam + 1j*self.vali.GetDataArray()

            lam = lam.reshape(self.vdim_te, self.vdim_tr)
            '''

            # ilj (or ilm)+ lk -> ijk (or imk)
            tmp = np.tensordot(levi_civita3, lam, (1, 0))

            # -> ikj ikm
            tmp = np.swapaxes(tmp, 1, 2)

            if self._metric is not None and not self.use_covariant_vec:
                g_mj = self.eval_cometric(trans, ip)  # g_{mj}
                tmp = np.tensordot(tmp, g_mj,  axes=(2, 0))  # ikm + mj = ikj
            else:
                pass
            L = tmp

            vdudx = np.tensordot(
                te_shape_arr*w2, tr_merged_arr, 0)*ip.weight  # nd, nd, vdim(d/dx)

            if self._realimag:
                for i, j in prod(range(self.vdim_te), range(self.vdim_tr)):
                    self.partelmat.Assign(0.0)

                    for k in range(self.esdim):
                        partelmat_arr[:, :] += (L[i, k, j]
                                                * vdudx[:, :, k]).real

                    elmat.AddMatrix(self.partelmat, te_nd*i, tr_nd*j)

            else:
                for i, j in prod(range(self.vdim_te), range(self.vdim_tr)):
                    self.partelmat.Assign(0.0)
                    for k in range(self.esdim):
                        partelmat_arr[:, :] += (L[i, k, j]
                                                * vdudx[:, :, k]).imag
                    elmat.AddMatrix(self.partelmat, te_nd*i, tr_nd*j)

            if self._metric is not None:
                vu = np.tensordot(
                    te_shape_arr*w2, tr_shape_arr*w2, 0)*ip.weight

                chris = self.eval_christoffel(trans, ip, self.esdim)
                if self._use_covariant_vec:
                    M = -np.tensordot(L, chris, ([2, 1], [1, 2]))
                else:
                    M = np.tensordot(L, chris, ([2, 1], [0, 1]))

                if self._realimag:
                    for i, j in prod(range(self.vdim_te), range(self.vdim_tr)):
                        self.partelmat.Assign(0.0)
                        partelmat_arr[:, :] += (M[i, j]
                                                * vu[:, :]).real
                        elmat.AddMatrix(self.partelmat, te_nd*i, tr_nd*j)

                else:
                    for i, j in prod(range(self.vdim_te), range(self.vdim_tr)):
                        self.partelmat.Assign(0.0)
                        partelmat_arr[:, :] += (M[i, j]
                                                * vu[:, :]).imag
                        elmat.AddMatrix(self.partelmat, te_nd*i, tr_nd*j)
