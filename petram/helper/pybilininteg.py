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

    def set_ir(self, trial_fe,  test_fe, trans, delta=0):
        order = (trial_fe.GetOrder() + test_fe.GetOrder() +
                 trans.OrderW() + self.q_order + delta)

        if trial_fe.Space() == mfem.FunctionSpace.rQk:
            ir = mfem.RefinedIntRules.Get(trial_fe.GetGeomType(), order)
        else:
            ir = mfem.IntRules.Get(trial_fe.GetGeomType(), order)

        self.ir = ir
        #self.ir = mfem.DiffusionIntegrator.GetRule(trial_fe, test_fe)


class PyVectorMassIntegrator(PyVectorIntegratorBase):
    def __init__(self, lam, vdim1, vdim2=None, ir=None):
        '''
           integrator for

              lmabda(i,k) gte(i) * gtr(k)

               gte : test function (v_i)
               gtr : trial function (u_k)

           vdim : size of i and k. vector dim of FE space.

           (note) If both test and trail are scalar, the same as VectorMassIntegrator.
                  Either test or trial can be VectorFE and coefficient can be rectangular

        '''
        PyVectorIntegratorBase.__init__(self, ir)

        self.lam = None if lam is None else lam
        if self.lam is None:
            return

        self.lam = lam
        self.lam = lam
        if vdim2 is not None:
            self.vdim_te = vdim1
            self.vdim_tr = vdim2
        else:
            self.vdim_te = vdim1
            self.vdim_tr = vdim1

        self._ir = self.GetIntegrationRule()

        self.tr_shape = None
        self.te_shape = None

        self.partelmat = mfem.DenseMatrix()

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
        sdim = trans.GetSpaceDim()
        tr_shape = [tr_nd]
        te_shape = [te_nd]

        shape = [te_nd, tr_nd]
        if test_fe.GetRangeType() == mfem.FiniteElement.SCALAR:
            shape[0] *= self.vdim_te
        else:
            te_shape.append(sdim)

        if trial_fe.GetRangeType() == mfem.FiniteElement.SCALAR:
            shape[1] *= self.vdim_tr
        else:
            tr_shape.append(sdim)

        elmat.SetSize(*shape)
        elmat.Assign(0.0)
        self.partelmat.SetSize(te_nd, tr_nd)
        partelmat_arr = self.partelmat.GetDataArray()

        self.tr_shape.SetSize(*tr_shape)
        self.te_shape.SetSize(*te_shape)

        self.tr_shape_arr = self.tr_shape.GetDataArray()
        self.te_shape_arr = self.te_shape.GetDataArray()

        #print("DoF", tr_nd, te_nd)

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

                transip = trans.Transform(ip)
                lam = self.lam(transip)

                for i in range(self.vdim_te):  # test
                    for k in range(self.vdim_tr):  # trial
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

                transip = trans.Transform(ip)
                lam = self.lam(transip)

                #print("lam shape (1)", lam.shape, dudxdvdx.shape)

                for i in range(self.vdim_te):  # test
                    self.partelmat.Assign(0.0)
                    for k in range(sdim):  # trial
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

                transip = trans.Transform(ip)
                lam = self.lam(transip)

                #print("lam shape(2)", lam.shape, dudxdvdx.shape)

                for k in range(self.vdim_tr):  # trial
                    self.partelmat.Assign(0.0)
                    for i in range(sdim):  # test
                        partelmat_arr[:, :] += lam[i, k]*dudxdvdx[:, i, :]

                    elmat.AddMatrix(self.partelmat, 0, tr_nd*k)

        else:
            assert False, "Use VectorFE Mass Integrator"


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

            transip = trans.Transform(ip)
            lam = self.lam(transip)

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


class PyVectorWeakPartialPartialIntegrator(PyVectorIntegratorBase):
    def __init__(self, lam, vdim1, vdim2=None, ir=None):
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

            for i in range(self.vdim_te):  # test
                for j in range(self.vdim_tr):  # trial

                    self.partelmat.Assign(0.0)
                    for k in range(self.esdim):
                        for l in range(self.esdim):
                            partelmat_arr[:, :] += lam[l, i,
                                                       k, j]*dudxdvdx[:, l, :, k]

                    elmat.AddMatrix(self.partelmat, te_nd*i, tr_nd*j)


class PyVectorPartialPartialIntegrator(PyVectorIntegratorBase):
    def __init__(self, lam, vdim1, vdim2=None, ir=None):
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

        assert self.vdim_tr == self.esdim, "vector dim and extedned spacedim must be the same"
        #print('esdim flag', self.esflag, self.esflag2)

        self._ir = self.GetIntegrationRule()

        self.tr_shape = mfem.Vector()
        self.te_shape = mfem.Vector()
        self.tr_dshape = mfem.DenseMatrix()
        self.tr_dshapedxt = mfem.DenseMatrix()
        self.tr_hshape = mfem.DenseMatrix()

        self.tr_merged = mfem.DenseMatrix()

        self.partelmat = mfem.DenseMatrix()

    def AssembleElementMatrix(self, el, trans, elmat):
        self.AssembleElementMatrix2(el, el, trans, elmat)

    def AssembleElementMatrix2(self, trial_fe, test_fe, trans, elmat):
        # if self.ir is None:
        #    self.ir = mfem.DiffusionIntegrator.GetRule(trial_fe, test_fe)
        if self.lam is None:
            return
        if self._ir is None:
            self.set_ir(trial_fe, test_fe, trans, -2)

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
        self.tr_hshape.SetSize(tr_nd, dim*(dim+1)//2)

        tr_shape_arr = self.tr_shape.GetDataArray()
        te_shape_arr = self.te_shape.GetDataArray()
        tr_dshapedxt_arr = self.tr_dshapedxt.GetDataArray()
        tr_hshape_arr = self.tr_hshape.GetDataArray()

        tr_merged_arr = np.zeros((tr_nd, self.esdim, self.esdim))

        for i in range(self.ir.GetNPoints()):

            ip = self.ir.IntPoint(i)
            trans.SetIntPoint(ip)

            test_fe.CalcPhysShape(trans, self.te_shape)
            trial_fe.CalcPhysShape(trans, self.tr_shape)
            trial_fe.CalcPhysDShape(trans, self.tr_dshape)
            trial_fe.CalcPhysHessian(trans, self.tr_hshape)

            if dim == 3:
                hess = tr_hshape_arr[:, [0, 1, 2, 1, 5,
                                         3, 2, 3, 4]].reshape(tr_nd, 3, 3)
            elif dim == 2:
                hess = tr_hshape_arr[:, [0, 1, 1, 2]].reshape(tr_nd, 2, 2)
            elif dim == 1:
                hess = tr_hshape_arr[:, [0, ]].reshape(tr_nd, 1, 1)

            for i in self.esflag:
                for j in self.esflag:
                    tr_merged_arr[:, i, j] = hess[:, i, j]
            for i in self.esflag:
                for j in self.esflag2:
                    tr_merged_arr[:, i, j] = tr_dshapedxt_arr[:, i]
            for i in self.esflag2:
                for j in self.esflag:
                    tr_merged_arr[:, i, j] = tr_dshapedxt_arr[:, j]
            for i in self.esflag2:
                for j in self.esflag2:
                    tr_merged_arr[:, i, j] = tr_shape_arr

            detJ = trans.Weight()
            weight = ip.weight
            dudxdvdx = np.tensordot(te_shape_arr, tr_merged_arr, 0)*weight*detJ

            transip = trans.Transform(ip)
            lam = self.lam(transip)

            for i in range(self.vdim_te):  # test
                for j in range(self.vdim_tr):  # trial

                    self.partelmat.Assign(0.0)
                    for k in range(self.esdim):
                        for l in range(self.esdim):
                            partelmat_arr[:, :] += lam[i, k,
                                                       l, j]*dudxdvdx[:, :, k, l]

                    elmat.AddMatrix(self.partelmat, te_nd*i, tr_nd*j)
