'''
 
   PyBilinearform:
     Additional bilinearform Integrator written in Python

     Vector field operator:
        PyVectorMassIntegrator : (u M v)   M is rank-2
        PyVectorPartialIntegrator : (du/x M v) M is rank-3
        PyVectorPartialPartialIntegrator : (du/x M dv/dx) M is rank-4

   Copyright (c) 2024-, S. Shiraiwa (PPPL)
'''
import numpy as np

from petram.mfem_config import use_parallel
if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('Variables')


class PyVectorIntegratorBase(mfem.PyBilinearFormIntegrator):
    def __init__(self, *args, **kwargs):
        mfem.PyBilinearFormIntegrator.__init__(self, *args, **kwargs)
        self._q_order = 0

    @property
    def q_order(self):
        return self._q_order

    @q_order.setter
    def q_order(self, value):
        self._q_order = value

    def set_ir(self, trial_fe,  test_fe, trans):
        order = (trial_fe.GetOrder() + test_fe.GetOrder() +
                 trans.OrderW() + self.q_order)

        if trial_fe.Space() == mfem.FunctionSpace.rQk:
            ir = mfem.RefinedIntRules.Get(trial_fe.GetGeomType(), order)
        else:
            ir = mfem.IntRules.Get(trial_fe.GetGeomType(), order)

        self.ir = ir
        #self.ir = mfem.DiffusionIntegrator.GetRule(trial_fe, test_fe)


class PyVectorMassIntegrator(PyVectorIntegratorBase):
    def __init__(self, _lam, lam, ir=None):
        '''
           integrator for

              lmabda(i,k) gte(i) * gtr(k)

               gte : test function (v_i)
               gtr : trial function (u_k)

           vdim : size of i and k. vector dim of FE space.

           (note) This is essentially the same as VectorMassIntegrator.
                  Implemented for verificaiton.
        '''
        PyVectorIntegratorBase.__init__(self, ir)

        self.lam = None if lam is None else lam
        if self.lam is None:
            return

        self.lam = lam
        self.vdim = lam.vdim
        self._ir = self.GetIntegrationRule()

        self.tr_shape = mfem.Vector()
        self.te_shape = mfem.Vector()

        self.partelmat = mfem.DenseMatrix()

    def AssembleElementMatrix(self, el, trans, elmat):
        self.AssembleElementMatrix2(el, el, trans, elmat)

    def AssembleElementMatrix2(self, trial_fe, test_fe, trans, elmat):
        if self.lam is None:
            return

        if self._ir is None:
            self.set_ir(trial_fe,  test_fe, trans)

        tr_nd = trial_fe.GetDof()
        te_nd = test_fe.GetDof()

        elmat.SetSize(te_nd*self.vdim, tr_nd*self.vdim)
        elmat.Assign(0.0)
        self.partelmat.SetSize(te_nd, tr_nd)

        partelmat_arr = self.partelmat.GetDataArray()

        self.tr_shape.SetSize(tr_nd)
        self.te_shape.SetSize(te_nd)

        self.tr_shape_arr = self.tr_shape.GetDataArray()
        self.te_shape_arr = self.te_shape.GetDataArray()

        for i in range(self.ir.GetNPoints()):
            ip = self.ir.IntPoint(i)
            trans.SetIntPoint(ip)
            w = trans.Weight()

            trial_fe.CalcShape(ip, self.tr_shape)
            test_fe.CalcShape(ip, self.te_shape)

            w2 = np.sqrt(w)
            dudxdvdx = np.tensordot(
                self.te_shape_arr*w2, self.tr_shape_arr*w2, 0)*ip.weight

            transip = trans.Transform(ip)
            lam = self.lam(transip)

            for i in range(self.vdim):  # test
                for k in range(self.vdim):  # trial
                    self.partelmat.Assign(0.0)
                    partelmat_arr[:, :] += lam[i, k]*dudxdvdx[:, :]

                    elmat.AddMatrix(self.partelmat, te_nd*i, tr_nd*k)


class PyVectorPartialIntegrator(PyVectorIntegratorBase):
    def __init__(self, _lam, lam, ir=None):
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
        self.lam = None if lam is None else lam
        if self.lam is None:
            return

        self.esflag = np.where(np.array(lam.esindex) >= 0)[0]
        self.esflag2 = np.where(np.atleast_1d(lam.esindex) == -1)[0]
        self.esdim = len(lam.esindex)
        self.vdim = lam.vdim

        assert self.vdim == self.esdim, "vector dim and extedned spacedim must be the same"
        #print('esdim flag', self.esflag, self.esflag2)

        self._ir = self.GetIntegrationRule()

        self.tr_shape = mfem.Vector()
        self.te_shape = mfem.Vector()
        self.tr_dshape = mfem.DenseMatrix()
        self.tr_dshapedxt = mfem.DenseMatrix()
        self.tr_merged = mfem.DenseMatrix()

        self.partelmat = mfem.DenseMatrix()

    def AssembleElementMatrix(self, el, trans, elmat):
        self.AssembleElementMatrix2(el, el, trans, elmat)

    def AssembleElementMatrix2(self, trial_fe, test_fe, trans, elmat):
        if self.lam is None:
            return
        if self._ir is None:
            self.set_ir(trial_fe,  test_fe, trans)

        tr_nd = trial_fe.GetDof()
        te_nd = test_fe.GetDof()

        elmat.SetSize(te_nd*self.vdim, tr_nd*self.vdim)
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

            transip = trans.Transform(ip)
            lam = self.lam(transip)

            # construct merged test/trial shape
            tr_merged_arr[:, self.esflag] = tr_dshapedxt_arr*w1
            for k in self.esflag2:
                tr_merged_arr[:, k] = tr_shape_arr*w2

            dudxdvdx = np.tensordot(
                te_shape_arr*w2, tr_merged_arr, 0)*ip.weight

            for i in range(self.vdim):  # test
                for j in range(self.vdim):  # trial
                    self.partelmat.Assign(0.0)
                    for k in range(self.esdim):
                        partelmat_arr[:, :] += lam[i, k, j]*dudxdvdx[:, :, k]

                    elmat.AddMatrix(self.partelmat, te_nd*i, tr_nd*j)


class PyVectorPartialPartialIntegrator(PyVectorIntegratorBase):
    def __init__(self, _lam, lam, ir=None):
        '''
           integrator for

              lmabda(i,j,k.l) gte(i,j) * gtr(k,l)

               gte : generalized gradient of test function
                  j < sdim d v_i/dx_j
                  j >= sdim  v_i
                or
                  j not in exindex: v_i/dx_j
                  j in esindex:  v_i

               gtr : generalized gradient of trial function
                  l < sdim d u_k/dx_l
                  l >= sdim  u_k
                or
                  l not in exindex: u_k/dx_l
                  l in esindex:  u_k


           vdim : size of i and k. vector dim of FE space.
           sdim : space dimension of v

           esdim : size of j and l. extended space dim.
           esindex: specify the index for extendend space dim.


        '''
        PyVectorIntegratorBase.__init__(self, ir)
        self.lam = None if lam is None else lam
        if self.lam is None:
            return

        self.esflag = np.where(np.array(lam.esindex) >= 0)[0]
        self.esflag2 = np.where(np.atleast_1d(lam.esindex) == -1)[0]
        self.esdim = len(lam.esindex)
        self.vdim = lam.vdim

        assert self.vdim == self.esdim, "vector dim and extedned spacedim must be the same"
        #print('esdim flag', self.esflag, self.esflag2)

        self._ir = self.GetIntegrationRule()

        self.tr_shape = mfem.Vector()
        self.te_shape = mfem.Vector()
        self.tr_dshape = mfem.DenseMatrix()
        self.te_dshape = mfem.DenseMatrix()
        self.tr_dshapedxt = mfem.DenseMatrix()
        self.te_dshapedxt = mfem.DenseMatrix()

        self.tr_merged = mfem.DenseMatrix()
        self.te_merged = mfem.DenseMatrix()

        self.partelmat = mfem.DenseMatrix()

    def AssembleElementMatrix(self, el, trans, elmat):
        self.AssembleElementMatrix2(el, el, trans, elmat)

    def AssembleElementMatrix2(self, trial_fe, test_fe, trans, elmat):
        # if self.ir is None:
        #    self.ir = mfem.DiffusionIntegrator.GetRule(trial_fe, test_fe)
        if self.lam is None:
            return
        if self._ir is None:
            self.set_ir(trial_fe, test_fe, trans)

        tr_nd = trial_fe.GetDof()
        te_nd = test_fe.GetDof()

        elmat.SetSize(te_nd*self.vdim, tr_nd*self.vdim)
        elmat.Assign(0.0)
        self.partelmat.SetSize(te_nd, tr_nd)

        partelmat_arr = self.partelmat.GetDataArray()

        dim = trial_fe.GetDim()
        sdim = trans.GetSpaceDim()
        square = (dim == sdim)

        self.tr_shape.SetSize(tr_nd)
        self.te_shape.SetSize(te_nd)
        self.tr_dshape.SetSize(tr_nd, dim)
        self.te_dshape.SetSize(te_nd, dim)
        self.tr_dshapedxt.SetSize(tr_nd, sdim)
        self.te_dshapedxt.SetSize(te_nd, sdim)

        self.tr_merged.SetSize(tr_nd, self.esdim)
        self.te_merged.SetSize(te_nd, self.esdim)

        tr_shape_arr = self.tr_shape.GetDataArray()
        te_shape_arr = self.te_shape.GetDataArray()
        tr_dshapedxt_arr = self.tr_dshapedxt.GetDataArray()
        te_dshapedxt_arr = self.te_dshapedxt.GetDataArray()

        tr_merged_arr = self.tr_merged.GetDataArray()
        te_merged_arr = self.te_merged.GetDataArray()

        for i in range(self.ir.GetNPoints()):

            ip = self.ir.IntPoint(i)
            trans.SetIntPoint(ip)
            w = trans.Weight()

            trial_fe.CalcShape(ip, self.tr_shape)
            test_fe.CalcShape(ip, self.te_shape)

            trial_fe.CalcDShape(ip, self.tr_dshape)
            test_fe.CalcDShape(ip, self.te_dshape)

            mfem.Mult(self.tr_dshape, trans.AdjugateJacobian(),
                      self.tr_dshapedxt)
            mfem.Mult(self.te_dshape, trans.AdjugateJacobian(),
                      self.te_dshapedxt)

            w1 = np.sqrt(1./w) if square else np.sqrt(1/w/w/w)
            w2 = np.sqrt(w)

            # construct merged test/trial shape
            tr_merged_arr[:, self.esflag] = tr_dshapedxt_arr*w1
            te_merged_arr[:, self.esflag] = te_dshapedxt_arr*w1
            for k in self.esflag2:
                tr_merged_arr[:, k] = tr_shape_arr*w2
                te_merged_arr[:, k] = te_shape_arr*w2

            dudxdvdx = np.tensordot(te_merged_arr, tr_merged_arr, 0)*ip.weight

            transip = trans.Transform(ip)
            lam = self.lam(transip)

            for i in range(self.vdim):  # test
                for j in range(self.vdim):  # trial

                    self.partelmat.Assign(0.0)
                    for k in range(self.esdim):
                        for l in range(self.esdim):
                            partelmat_arr[:, :] += lam[l, i,
                                                       k, j]*dudxdvdx[:, l, :, k]

                    elmat.AddMatrix(self.partelmat, te_nd*i, tr_nd*j)
