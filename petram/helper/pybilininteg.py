'''
 
   PyBilinearform:
     Additional bilinearform Integrator written in Python

     Vector field operator:
        PyVectorMassIntegrator : 
            (Mu_j, v_i)   
             M_ij is rank-2

        PyVectorPartialIntegrator : 
            (Mdu_j/x_k, v_i) 
             M_ikj is rank-3
        PyVectorWeakPartialIntegrator : 
            (Mu_j, -dv_i/dx_k) 
             M_ikj is rank-3. Note index order is the same as strong version

        PyVectorDiffusionIntegrator : 
           (du_j/x_k M dv_i/dx_l) 
            M_likj is rank-4
        PyVectorPartialPartialIntegrator : 
            (M du_j^2/x_kl^2, v_i)
            M_iklj is rank-4
        PyVectorWeakPartialPartialIntegrator : (
            Mu_j, d^2v_i/dx^2) 
            M_iklj is rank-4. Note index order is the same as strong version

   Copyright (c) 2024-, S. Shiraiwa (PPPL)
'''
import numpy as np
from numpy.linalg import det, norm

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
        self._support_metric = False
        self._metric = None
        self._christoffel = None

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

    @classmethod
    def coeff_shape(cls, itg_param):
        raise NotImplementedError("subclass must implement coeff_shape")

    def set_metric(self, metric, christoffel, use_covariant_vec=True):
        #
        #  g_ij (metric tensor) is set

        #  sqrt(g_ij) dxdydz:
        #     integration uses sqrt(g_ij) for volume integraiton Jacobian :
        #
        #  inter product is computed by
        #      v_i *u^j   (use_covariant_vec=True)
        #
        #     or
        #
        #      v^i *u_j   (use_covariant_vec= False)
        #
        #
        #   nabla is treated as covariant derivatives thus
        #
        #      d_k v^i = dv^i/dx^k + {i/ j, k} v_^i
        #      d_k v_i = dv^i/dx^k _ {i/ j, k} v_^i

        if not self._support_metric:
            raise NotImplementedError(
                "the integrator does not support metric tensor")

        self._metric = metric
        self._christoffel = christoffel
        self.metric = mfem.Vector()

    def eval_metric(self, trans, ip):
        self._metric.Eval(self.metric, trans, ip)
        m = self.metric.GetDataArray()
        mm = int(np.sqrt(m.shape))
        m = m.reshape(mm, mm)
        return m


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
        self._support_metric = True

        self.lam = None if lam is None else lam
        if self.lam is None:
            return

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

                self.lam.Eval(self.val, trans, ip)
                lam = self.val.GetDataArray().reshape(self.vdim_te, self.vdim_tr)

                if self._metric is not None:
                    m = self.eval_metric(trans, ip):
                    lam /= det(m)

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

                self.lam.Eval(self.val, trans, ip)
                lam = self.val.GetDataArray().reshape(self.vdim_te, self.vdim_tr)

                if self._metric is not None:
                    m = self.eval_metric(trans, ip):
                    lam /= det(m)

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

                self.lam.Eval(self.val, trans, ip)
                lam = self.val.GetDataArray().reshape(self.vdim_te, self.vdim_tr)

                if self._metric is not None:
                    m = self.eval_metric(trans, ip):
                    lam /= det(m)

                for k in range(self.vdim_tr):  # trial
                    self.partelmat.Assign(0.0)
                    for i in range(self.vdim_te):  # test
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

        assert sdim == len(
            self.esflag), "mesh SpaceDim is not same as esflag length"

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
        self.te_dshape.SetSize(te_nd, dim)
        self.te_dshapedxt.SetSize(te_nd, sdim)

        assert sdim == len(
            self.esflag), "mesh SpaceDim is not same as esflag length"

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


class PyVectorDiffusionIntegrator(PyVectorIntegratorBase):
    def __init__(self, lam, vdim1, vdim2=None, esindex=None, ir=None):
        '''
           integrator for

              lmabda(l, i, k. j) gte(i,l) * gtr(j, k)

               gte : generalized gradient of test function
                  j not in exindex: v_i/dx_l
                  j in esindex:  v_i

               gtr : generalized gradient of trial function
                  l < sdim d u_j/dx_k
                  l >= sdim  u_j
                or
                  l not in exindex: u_j/dx_k
                  l in esindex:  u_j


           vdim1 : size of trial space
           vdim2 : size of test space
           esindex: specify the index for extendend space dim for trial

        '''
        PyVectorIntegratorBase.__init__(self, ir)
        self._support_metric = True

        self.lam = lam
        if vdim2 is not None:
            self.vdim_te = vdim1
            self.vdim_tr = vdim2
        else:
            self.vdim_te = vdim1
            self.vdim_tr = vdim1

        if esindex is None:
            esindex = list(range(self.vdim_tr))

        esindex = np.array(esindex)
        if np.iscomplexobj(esindex):
            self.esflag = np.where(np.iscoplex(esindex) == False)[0]
            self.esflag2 = np.where(np.iscoplex(esindex))[0]
            self.es_weight = esindex[self.esflag2]
        else:
            self.esflag = np.where(esindex >= 0)[0]
            self.esflag2 = np.where(esindex == -1)[0]
            self.es_weight = np.ones(len(self.esflag2))
        self.esdim = len(esindex)

        #print('esdim flag', self.esdim, self.esflag, self.esflag2)

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
        self.val = mfem.Vector()

    @classmethod
    def coeff_shape(cls, vdim1, vdim2=None, esindex=None, ir=None):
        if vdim2 is None:
            vdim2 = vdim1
        if esindex is None:
            esdim = vdim2
        else:
            esdim = len(esindex)

        return (esdim, vdim1, esdim, vdim2)

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
                tr_merged_arr[:, k] = tr_shape_arr*w2*self.es_weight[k]
                te_merged_arr[:, k] = te_shape_arr*w2*self.es_weight[k]

            dudxdvdx = np.tensordot(te_merged_arr, tr_merged_arr, 0)*ip.weight

            self.lam.Eval(self.val, trans, ip)
            lam = self.val.GetDataArray().reshape(self.esdim, self.vdim_te,
                                                  self.esdim, self.vdim_tr)

            if self._metric is not None:
                m = self.eval_metric(trans, ip):
                lam /= det(m)

            for i in range(self.vdim_te):  # test
                for j in range(self.vdim_tr):  # trial

                    self.partelmat.Assign(0.0)
                    for k in range(self.esdim):
                        for l in range(self.esdim):
                            partelmat_arr[:, :] += lam[l, i,
                                                       k, j]*dudxdvdx[:, l, :, k]

                    elmat.AddMatrix(self.partelmat, te_nd*i, tr_nd*j)


class PyVectorPartialPartialIntegrator(PyVectorIntegratorBase):
    def __init__(self, lam, vdim1, vdim2=None, esindex=None, ir=None):
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
        self.val = mfem.Vector()

    @classmethod
    def coeff_shape(cls, vdim1, vdim2=None, esindex=None, ir=None):
        if vdim2 is None:
            vdim2 = vdim1
        if esindex is None:
            esdim = vdim2
        else:
            esdim = len(esindex)

        return (vdim1, esdim, esdim, vdim2)

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

            self.lam.Eval(self.val, trans, ip)
            lam = self.val.GetDataArray().reshape(self.vdim_te, self.esdim,
                                                  self.esdim, self.vdim_tr)

            for i in range(self.vdim_te):  # test
                for j in range(self.vdim_tr):  # trial

                    self.partelmat.Assign(0.0)
                    for k in range(self.esdim):
                        for l in range(self.esdim):
                            partelmat_arr[:, :] += lam[i, k,
                                                       l, j]*dudxdvdx[:, :, k, l]

                    elmat.AddMatrix(self.partelmat, te_nd*i, tr_nd*j)


class PyVectorWeakPartialPartialIntegrator(PyVectorIntegratorBase):
    def __init__(self, lam, vdim1, vdim2=None, esindex=None, ir=None):
        '''
           Weak version

           coefficient index order M[i, k, l, j] is the same as strong 
           version. In order to fill a transpose, swap i-j and k-l together.
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
        self.te_dshape = mfem.DenseMatrix()
        self.te_dshapedxt = mfem.DenseMatrix()
        self.te_hshape = mfem.DenseMatrix()

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

        return (vdim1, esdim, esdim, vdim2)

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
        self.te_dshape.SetSize(te_nd, dim)
        self.te_dshapedxt.SetSize(te_nd, sdim)
        self.te_hshape.SetSize(te_nd, dim*(dim+1)//2)

        tr_shape_arr = self.tr_shape.GetDataArray()
        te_shape_arr = self.te_shape.GetDataArray()
        te_dshapedxt_arr = self.te_dshapedxt.GetDataArray()
        te_hshape_arr = self.te_hshape.GetDataArray()

        te_merged_arr = np.zeros((te_nd, self.esdim, self.esdim))

        for i in range(self.ir.GetNPoints()):

            ip = self.ir.IntPoint(i)
            trans.SetIntPoint(ip)

            trial_fe.CalcPhysShape(trans, self.tr_shape)
            test_fe.CalcPhysShape(trans, self.te_shape)
            test_fe.CalcPhysDShape(trans, self.te_dshape)
            test_fe.CalcPhysHessian(trans, self.te_hshape)

            if dim == 3:
                hess = te_hshape_arr[:, [0, 1, 2, 1, 5,
                                         3, 2, 3, 4]].reshape(te_nd, 3, 3)
            elif dim == 2:
                hess = te_hshape_arr[:, [0, 1, 1, 2]].reshape(te_nd, 2, 2)
            elif dim == 1:
                hess = te_hshape_arr[:, [0, ]].reshape(te_nd, 1, 1)

            for i in self.esflag:
                for j in self.esflag:
                    te_merged_arr[:, i, j] = hess[:, i, j]
            for i in self.esflag:
                for j in self.esflag2:
                    te_merged_arr[:, i, j] = te_dshapedxt_arr[:, i]
            for i in self.esflag2:
                for j in self.esflag:
                    te_merged_arr[:, i, j] = te_dshapedxt_arr[:, j]
            for i in self.esflag2:
                for j in self.esflag2:
                    te_merged_arr[:, i, j] = te_shape_arr

            detJ = trans.Weight()
            weight = ip.weight
            dudxdvdx = np.tensordot(te_merged_arr, tr_shape_arr, 0)*weight*detJ

            self.lam.Eval(self.val, trans, ip)
            lam = self.val.GetDataArray().reshape(self.vdim_te, self.esdim,
                                                  self.esdim, self.vdim_tr)

            for i in range(self.vdim_te):  # test
                for j in range(self.vdim_tr):  # trial

                    self.partelmat.Assign(0.0)
                    for k in range(self.esdim):
                        for l in range(self.esdim):
                            partelmat_arr[:, :] += lam[i, k,
                                                       l, j]*dudxdvdx[:, k, l, :]

                    elmat.AddMatrix(self.partelmat, te_nd*i, tr_nd*j)
