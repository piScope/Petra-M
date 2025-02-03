from petram.helper.integrators.pyvector_integrator_base import PyVectorIntegratorBase

from itertools import product as prod
import numpy as np
from numpy.linalg import det, norm, inv

from petram.mfem_config import use_parallel
if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('PyVectorMassIntegrator')


class PyVectorMassIntegrator(PyVectorIntegratorBase):
    support_metric = True

    def __init__(self, lam, vdim1=None, vdim2=None, metric=None, use_covariant_vec=False,
                 *, ir=None):
        '''
           integrator for

              lmabda(i,k) gte(i) * gtr(k)

               gte : test function (v_i)
               gtr : trial function (u_k)

           vdim : size of i and k. vector dim of FE space.

           (note) If both test and trail are scalar, the same as VectorMassIntegrator.
                  Either test or trial can be VectorFE and coefficient can be rectangular

        '''
        PyVectorIntegratorBase.__init__(self, use_covariant_vec, ir)

        self.lam = None if lam is None else lam
        if self.lam is None:
            return

        self.lam = lam

        if metric is None:
            metric_obj = self.__class__._proc_vdim1vdim2(vdim1, vdim2)
        else:
            metric_obj = metric

        self.config_metric_vdim_esindex(metric_obj, vdim1, vdim2, None)

        self._ir = self.GetIntegrationRule()
        self.alloc_workspace()

    def alloc_workspace(self):
        self.tr_shape = None
        self.te_shape = None
        self.partelmat = mfem.DenseMatrix()
        self.val = mfem.Vector()

    @classmethod
    def coeff_shape(cls, vdim1, vdim2=None, ir=None):
        if vdim2 is not None:
            return (vdim1, vdim2)
        return (vdim1, vdim1)

    def AssembleElementMatrix(self, el, trans, elmat):
        self.AssembleElementMatrix2(el, el, trans, elmat)

    def AssembleElementMatrix2(self, trial_fe, test_fe, trans, elmat):
        if self.lam is None:
            return

        if self._ir is None:
            self.set_ir(trial_fe, test_fe, trans)

        if self.te_shape is None:
            if test_fe.GetRangeType() == mfem.FiniteElement.VECTOR:
                self.te_shape = mfem.DenseMatrix()
            elif test_fe.GetRangeType() == mfem.FiniteElement.SCALAR:
                self.te_shape = mfem.Vector()
            else:
                assert False, "should not come here"

        if self.tr_shape is None:
            if trial_fe.GetRangeType() == mfem.FiniteElement.VECTOR:
                self.tr_shape = mfem.DenseMatrix()
            elif trial_fe.GetRangeType() == mfem.FiniteElement.SCALAR:
                self.tr_shape = mfem.Vector()
            else:
                assert False, "should not come here"

        tr_nd = trial_fe.GetDof()
        te_nd = test_fe.GetDof()
        tr_shape = [tr_nd]
        te_shape = [te_nd]

        shape = [te_nd, tr_nd]
        if test_fe.GetRangeType() == mfem.FiniteElement.SCALAR:
            shape[0] *= self.vdim_te
        else:
            te_shape.append(self.vdim_te)

        if trial_fe.GetRangeType() == mfem.FiniteElement.SCALAR:
            shape[1] *= self.vdim_tr
        else:
            tr_shape.append(self.vdim_tr)

        elmat.SetSize(*shape)
        elmat.Assign(0.0)
        self.partelmat.SetSize(te_nd, tr_nd)
        partelmat_arr = self.partelmat.GetDataArray()

        self.tr_shape.SetSize(*tr_shape)
        self.te_shape.SetSize(*te_shape)

        self.tr_shape_arr = self.tr_shape.GetDataArray()
        self.te_shape_arr = self.te_shape.GetDataArray()

        scalar_coeff = isinstance(self.lam, mfem.Coefficient)
        if scalar_coeff:
            assert self.vdim_te == self.vdim_tr, "scalar coefficeint allows only for square matrix"

        if (test_fe.GetRangeType() == mfem.FiniteElement.SCALAR and
                trial_fe.GetRangeType() == mfem.FiniteElement.SCALAR):

            # tr_shape = (tr_nd)
            # te_shape = (te_nd)
            # elmat = (te_nd*vdim_te, tr_nd*vdim_tr)

            for ii in range(self.ir.GetNPoints()):
                ip = self.ir.IntPoint(ii)
                trans.SetIntPoint(ip)
                w = trans.Weight()

                trial_fe.CalcShape(ip, self.tr_shape)
                test_fe.CalcShape(ip, self.te_shape)

                w2 = np.sqrt(w)
                dudxdvdx = np.tensordot(
                    self.te_shape_arr*w2, self.tr_shape_arr*w2, 0)*ip.weight

                if scalar_coeff:
                    lam = self.lam.Eval(trans, ip)
                    lam = np.diag([lam]*self.vdim_te)
                else:
                    self.lam.Eval(self.val, trans, ip)
                    lam = self.val.GetDataArray()
                    if len(lam) == self.vdim_te*self.vdim_tr:
                        lam = lam.reshape(self.vdim_te, self.vdim_tr)
                    else:
                        lam = np.diag(lam)

                if self._metric is not None:
                    detm = self.eval_sqrtg(trans, ip)
                    lam *= detm

                for i, k in prod(range(self.vdim_te), range(self.vdim_tr)):
                    self.partelmat.Assign(0.0)
                    partelmat_arr[:, :] += lam[i, k]*dudxdvdx[:, :]
                    elmat.AddMatrix(self.partelmat, te_nd*i, tr_nd*k)

        elif (test_fe.GetRangeType() == mfem.FiniteElement.SCALAR and
              trial_fe.GetRangeType() == mfem.FiniteElement.VECTOR):

            # tr_shape = (tr_nd, sdim)
            # te_shape = (te_nd)
            # elmat = (te_nd*vdim_te, tr_nd)

            for ii in range(self.ir.GetNPoints()):
                ip = self.ir.IntPoint(ii)
                trans.SetIntPoint(ip)
                w = trans.Weight()

                trial_fe.CalcVShape(trans, self.tr_shape)
                test_fe.CalcShape(ip, self.te_shape)

                w2 = np.sqrt(w)
                dudxdvdx = np.tensordot(
                    self.te_shape_arr*w2, self.tr_shape_arr*w2, 0)*ip.weight

                if scalar_coeff:
                    lam = self.lam.Eval(trans, ip)
                    lam = np.diag([lam]*self.vdim_te)
                else:
                    self.lam.Eval(self.val, trans, ip)
                    if len(lam) == self.vdim_te*self.vdim_tr:
                        lam = lam.reshape(self.vdim_te, self.vdim_tr)
                    else:
                        lam = np.diag(lam)

                if self._metric is not None:
                    detm = self.eval_sqrtg(trans, ip)
                    lam *= detm

                for i in range(self.vdim_te):  # test
                    self.partelmat.Assign(0.0)
                    for k in range(self.vdim_tr):  # trial
                        partelmat_arr[:, :] += lam[i, k]*dudxdvdx[:, :, k]

                    elmat.AddMatrix(self.partelmat, te_nd*i, 0)

        elif (test_fe.GetRangeType() == mfem.FiniteElement.VECTOR and
              trial_fe.GetRangeType() == mfem.FiniteElement.SCALAR):

            # tr_shape = (tr_nd,)
            # te_shape = (te_nd, sdim)
            # elmat = (te_nd, tr_nd*vdim_tr)

            for ii in range(self.ir.GetNPoints()):
                ip = self.ir.IntPoint(ii)
                trans.SetIntPoint(ip)
                w = trans.Weight()

                trial_fe.CalcShape(ip, self.tr_shape)
                test_fe.CalcVShape(trans, self.te_shape)

                w2 = np.sqrt(w)
                dudxdvdx = np.tensordot(
                    self.te_shape_arr*w2, self.tr_shape_arr*w2, 0)*ip.weight

                if scalar_coeff:
                    lam = self.lam.Eval(trans, ip)
                    lam = np.diag([lam]*self.vdim_te)
                else:
                    self.lam.Eval(self.val, trans, ip)
                    if len(lam) == self.vdim_te*self.vdim_tr:
                        lam = lam.reshape(self.vdim_te, self.vdim_tr)
                    else:
                        lam = np.diag(lam)

                if self._metric is not None:
                    detm = self.eval_sqrtg(trans, ip)
                    lam *= detm

                for k in range(self.vdim_tr):  # trial
                    self.partelmat.Assign(0.0)
                    for i in range(self.vdim_te):  # test
                        partelmat_arr[:, :] += lam[i, k]*dudxdvdx[:, i, :]

                    elmat.AddMatrix(self.partelmat, 0, tr_nd*k)

        else:
            assert False, "Use VectorFE Mass Integrator"
