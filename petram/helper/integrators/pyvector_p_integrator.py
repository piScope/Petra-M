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
dprint1, dprint2, dprint3 = petram.debug.init_dprints(
    'PyVectorPartialIntegrator')


class PyVectorPartialIntegrator(PyVectorIntegratorBase):
    def __init__(self, lam, vdim1, vdim2=None, esindex=None, ir=None):
        '''
           integrator for

              lmabda(i,k.l) gte(i) * gtr(k,l)

               gte : test function

               gtr : generalized gradient of trial function
                  l < sdim d u_k/dx_l
                  l >= sdim  u_k
                or
                  l not in exindex: u_k/dx_l
                  l in esindex:  u_k


           vdim : size of i and k. vector dim of FE space.
           sdim : space dimension of v
           esindex: 
              0, 1, 2... direction of gradient
              -1     ... the vector index where periodicity is assumed. (f -> ikf)
              ex) [0, 1, -1]  -> df/dx df/dy, f

           note: esdim == vdim


        '''
        PyVectorIntegratorBase.__init__(self, ir)
        self.lam = lam
        if vdim2 is not None:
            self.vdim_te = vdim1
            self.vdim_tr = vdim2
        else:
            self.vdim_te = vdim1
            self.vdim_tr = vdim1

        if esindex is None:
            esindex = list(range(self.vdim_tr))
        self.esflag = np.where(np.array(esindex) >= 0)[0]
        self.esflag2 = np.where(np.atleast_1d(esindex) == -1)[0]
        self.esdim = len(esindex)

        self._ir = self.GetIntegrationRule()

        self.tr_shape = mfem.Vector()
        self.te_shape = mfem.Vector()
        self.tr_dshape = mfem.DenseMatrix()
        self.tr_dshapedxt = mfem.DenseMatrix()
        self.tr_merged = mfem.DenseMatrix()

        self.partelmat = mfem.DenseMatrix()
        self.val = mfem.Vector()

    @classmethod
    def coeff_shape(cls, vdim1, vdim2=None, esindex=None, ir=None):
        if vdim2 is None:
            vdim2 = vdim1
        if esindex is None:
            esdim = vdim2
        else:
            esdim = len(esindex)

        return (vdim1, esdim, vdim2,)

    def AssembleElementMatrix(self, el, trans, elmat):
        self.AssembleElementMatrix2(el, el, trans, elmat)

    def AssembleElementMatrix2(self, trial_fe, test_fe, trans, elmat):
        if self.lam is None:
            return
        if self._ir is None:
            self.set_ir(trial_fe,  test_fe, trans)

        tr_nd = trial_fe.GetDof()
        te_nd = test_fe.GetDof()

        elmat.SetSize(te_nd*self.vdim_tr, tr_nd*self.vdim_te)
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

        self.tr_merged.SetSize(tr_nd, self.esdim)

        tr_shape_arr = self.tr_shape.GetDataArray()
        te_shape_arr = self.te_shape.GetDataArray()
        tr_dshapedxt_arr = self.tr_dshapedxt.GetDataArray()
        tr_merged_arr = self.tr_merged.GetDataArray()

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

            self.lam.Eval(self.val, trans, ip)
            lam = self.val.GetDataArray().reshape(self.vdim_te, self.esdim, self.vdim_tr)

            # construct merged test/trial shape
            tr_merged_arr[:, self.esflag] = tr_dshapedxt_arr*w1

            for k in self.esflag2:
                tr_merged_arr[:, k] = tr_shape_arr*w2

            dudxdvdx = np.tensordot(
                te_shape_arr*w2, tr_merged_arr, 0)*ip.weight

            for i in range(self.vdim_te):  # test
                for j in range(self.vdim_tr):  # trial
                    self.partelmat.Assign(0.0)
                    for k in range(self.esdim):
                        partelmat_arr[:, :] += lam[i, k, j]*dudxdvdx[:, :, k]

                    elmat.AddMatrix(self.partelmat, te_nd*i, tr_nd*j)


class PyVectorWeakPartialIntegrator(PyVectorIntegratorBase):
    def __init__(self, lam, vdim1, vdim2=None, esindex=None, ir=None):
        '''
           weak version of integrator

           coefficient index order M[i, k, j] is the same as strong 
           version. In order to fill a negative transpose, swap i-j. 
        '''
        PyVectorIntegratorBase.__init__(self, ir)
        self.lam = lam
        if vdim2 is not None:
            self.vdim_te = vdim1
            self.vdim_tr = vdim2
        else:
            self.vdim_te = vdim1
            self.vdim_tr = vdim1

        if esindex is None:
            esindex = list(range(self.vdim_tr))
        self.esflag = np.where(np.array(esindex) >= 0)[0]
        self.esflag2 = np.where(np.atleast_1d(esindex) == -1)[0]
        self.esdim = len(esindex)

        self._ir = self.GetIntegrationRule()

        self.tr_shape = mfem.Vector()
        self.te_shape = mfem.Vector()
        self.te_dshape = mfem.DenseMatrix()
        self.te_dshapedxt = mfem.DenseMatrix()
        self.te_merged = mfem.DenseMatrix()

        self.partelmat = mfem.DenseMatrix()
        self.val = mfem.Vector()

    @classmethod
    def coeff_shape(cls, vdim1, vdim2=None, esindex=None, ir=None):
        if vdim2 is None:
            vdim2 = vdim1
        if esindex is None:
            esdim = vdim2
        else:
            esdim = len(esindex)

        return (vdim1, esdim, vdim2,)

    def AssembleElementMatrix(self, el, trans, elmat):
        self.AssembleElementMatrix2(el, el, trans, elmat)

    def AssembleElementMatrix2(self, trial_fe, test_fe, trans, elmat):
        if self.lam is None:
            return
        if self._ir is None:
            self.set_ir(trial_fe, test_fe, trans)

        tr_nd = trial_fe.GetDof()
        te_nd = test_fe.GetDof()

        elmat.SetSize(te_nd*self.vdim_tr, tr_nd*self.vdim_te)
        elmat.Assign(0.0)
        self.partelmat.SetSize(te_nd, tr_nd)

        partelmat_arr = self.partelmat.GetDataArray()

        dim = trial_fe.GetDim()
        sdim = trans.GetSpaceDim()
        square = (dim == sdim)

        self.tr_shape.SetSize(tr_nd)
        self.te_shape.SetSize(te_nd)
        self.te_dshape.SetSize(te_nd, dim)
        self.te_dshapedxt.SetSize(te_nd, sdim)

        self.te_merged.SetSize(tr_nd, self.esdim)

        tr_shape_arr = self.tr_shape.GetDataArray()
        te_shape_arr = self.te_shape.GetDataArray()
        te_dshapedxt_arr = self.te_dshapedxt.GetDataArray()
        te_merged_arr = self.te_merged.GetDataArray()

        for i in range(self.ir.GetNPoints()):

            ip = self.ir.IntPoint(i)
            trans.SetIntPoint(ip)
            w = trans.Weight()

            trial_fe.CalcShape(ip, self.tr_shape)
            test_fe.CalcShape(ip, self.te_shape)
            test_fe.CalcDShape(ip, self.te_dshape)

            mfem.Mult(self.te_dshape, trans.AdjugateJacobian(),
                      self.te_dshapedxt)

            w1 = np.sqrt(1./w) if square else np.sqrt(1/w/w/w)
            w2 = np.sqrt(w)

            self.lam.Eval(self.val, trans, ip)
            lam = self.val.GetDataArray().reshape(self.vdim_te, self.esdim, self.vdim_tr)

            # construct merged test/trial shape
            te_merged_arr[:, self.esflag] = te_dshapedxt_arr*w1

            for k in self.esflag2:
                te_merged_arr[:, k] = te_shape_arr*w2

            dudxdvdx = np.tensordot(
                te_merged_arr, tr_shape_arr*w2, 0)*ip.weight

            for i in range(self.vdim_te):  # test
                for j in range(self.vdim_tr):  # trial
                    self.partelmat.Assign(0.0)
                    for k in range(self.esdim):
                        partelmat_arr[:, :] -= lam[i, k, j]*dudxdvdx[:, k, :]

                    elmat.AddMatrix(self.partelmat, te_nd*i, tr_nd*j)
