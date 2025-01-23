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
    'PyVectorPartialPartialIntegrator')


class PyVectorDiffusionIntegrator(PyVectorIntegratorBase):
    use_complex_coefficient = True
    support_metric = True

    def __init__(self, lam, vdim1, vdim2=None, esindex=None, metric=None,
                 use_covariant_vec=False, *, ir=None):

        #
        #   integrator for
        #
        #      lmabda(l, i, k. j) gte(i,l) * gtr(j, k)
        #
        #       gte : generalized gradient of test function
        #          j not in exindex: v_i/dx_l
        #          j in esindex:  v_i
        #
        #       gtr : generalized gradient of trial function
        #          l < sdim d u_j/dx_k
        #          l >= sdim  u_j
        #        or
        #          l not in exindex: u_j/dx_k
        #          l in esindex:  u_j
        #
        #   vdim1 : size of test space
        #   vdim2 : size of trial space
        #   esindex: specify the index for extendend space dim for trial
        #
        #   when christoffel {i/j, k} is given, dx_k is replaced by
        #   covariant delivative
        #
        #    d_k is covariant delivative
        #      d_k v^i = dv^i/dx^k + {i/ j, k} v_^i
        #      d_k v_i = dv^i/dx^k - {i/ j, k} v_^i
        #
        #    then we compute lam_ij^kl d_l v^i  d_k u^j  (sqrt(det(g_nn))) dxdydz
        #    where lam_ij^kl is rank-2,2 tensor
        #
        #    for contravariant u and v
        #
        #    one can use lam_ij^kl = g_ij * coeff^kl for
        #    diffusion coefficient in curvelinear coodidnates.
        #

        PyVectorIntegratorBase.__init__(self, ir)

        if not hasattr(lam, "get_real_coefficient"):
            self.lam_real = lam
            self.lam_imag = None
        else:
            self.lam_real = lam.get_real_coefficient()
            self.lam_imag = lam.get_imag_coefficient()

        flag, params = self.__class__._proc_vdim1vdim2(vdim1, vdim2)
        if flag:
            vdim1, vdim2, esindex, metric, use_covariant_vec = params

        if metric is not None:
            self.set_metric(metric, use_covariant_vec=use_covariant_vec)

        if vdim2 is not None:
            self.vdim_te = vdim1
            self.vdim_tr = vdim2
        else:
            self.vdim_te = vdim1
            self.vdim_tr = vdim1

        if esindex is None:
            esindex = list(range(self.vdim_tr))

        self._proc_esindex(esindex)

        # print('esdim flag', self.esdim, self.esflag, self.esflag2)

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
        self.valr = mfem.Vector()
        self.vali = mfem.Vector()

    @classmethod
    def coeff_shape(cls, vdim1, vdim2=None, esindex=None, ir=None):

        flag, params = cls._proc_vdim1vdim2(vdim1, vdim2)

        if flag:
            vdim1, vdim2, esindex, _metric, use_covarient_vec = params
        else:
            if vdim2 is None:
                vdim2 = vdim1

        if esindex is None:
            esdim = vdim2
        else:
            esdim = len(esindex)

        return (esdim, vdim1, esdim, vdim2,)

    def AssembleElementMatrix(self, el, trans, elmat):
        self.AssembleElementMatrix2(el, el, trans, elmat)

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
        self.te_dshape.SetSize(te_nd, dim)
        self.tr_dshapedxt.SetSize(tr_nd, sdim)
        self.te_dshapedxt.SetSize(te_nd, sdim)

        self.tr_merged.SetSize(tr_nd, self.esdim)
        self.te_merged.SetSize(te_nd, self.esdim)

        tr_shape_arr = self.tr_shape.GetDataArray()
        te_shape_arr = self.te_shape.GetDataArray()
        tr_dshapedxt_arr = self.tr_dshapedxt.GetDataArray()
        te_dshapedxt_arr = self.te_dshapedxt.GetDataArray()

        tr_merged_arr = np.zeros((self.esdim, tr_nd), dtype=np.complex128)
        te_merged_arr = np.zeros((self.esdim, te_nd), dtype=np.complex128)

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
            tr_merged_arr[self.esflag, :] = (tr_dshapedxt_arr*w1).transpose()
            te_merged_arr[self.esflag, :] = (te_dshapedxt_arr*w1).transpose()

            print(self.esflag, self.esflag2, self.es_weight)
            for i, k in enumerate(self.esflag2):
                tr_merged_arr[k, :] = (
                    tr_shape_arr*w2*self.es_weight[i]).transpose()
                te_merged_arr[k, :] = (
                    te_shape_arr*w2*self.es_weight[i].conj()).transpose()

            if self._metric:
                # shape = sdim, nd, sdim
                # index : v_p, d/dx^q nd
                tr_merged_arr_t = np.stack([tr_merged_arr]*self.vdim_tr)
                te_merged_arr_t = np.stack([tr_merged_arr]*self.vdim_te)

                chris = self.eval_christoffel(trans, ip, self.esdim)

                if self._use_covariant_vec:
                    for k in range(self.esdim):
                        print("here", chris[k, :, :])
                        te_merged_arr_t -= np.tensordot(
                            chris[k, :, :], te_shape_arr*w2, 0)
                        #tr_merged_arr_t -= np.tensordot(
                        #    chris[k, :, :], tr_shape_arr*w2, 0)
                        tr_merged_arr_t += np.tensordot(
                            chris[:, k, :], tr_shape_arr*w2, 0)

                else:
                    for k in range(self.esdim):
                        print("here", chris[:, k, :])
                        te_merged_arr_t += np.tensordot(
                            chris[:, k, :], te_shape_arr*w2, 0)
                        #tr_merged_arr_t += np.tensordot(
                        #    chris[:, k, :], tr_shape_arr*w2, 0)
                        tr_merged_arr_t -= np.tensordot(
                            chris[k, :, :], tr_shape_arr*w2, 0)

                dudxdvdx = np.tensordot(
                    te_merged_arr_t, tr_merged_arr_t, 0)*ip.weight

            else:
                dudxdvdx = np.tensordot(
                    te_merged_arr, tr_merged_arr, 0)*ip.weight

            self.lam_real.Eval(self.valr, trans, ip)
            lam = self.valr.GetDataArray()
            if self.lam_imag is not None:
                self.lam_imag.Eval(self.vali, trans, ip)
                lam = lam + 1j*self.vali.GetDataArray()
            lam = lam.reshape(self.esdim, self.vdim_te,
                              self.esdim, self.vdim_tr)
            print(lam)
            # if self._metric is not None:
            #    detm = self.eval_metric(trans, ip)
            #    lam *= detm
            #    # m_co = 1/m   # inverse of diagnal matrix

            if self._realimag:
                for i, j in prod(range(self.vdim_te), range(self.vdim_tr)):
                    self.partelmat.Assign(0.0)

                    if not self._metric:
                        for k, l in prod(range(self.esdim), range(self.esdim)):
                            partelmat_arr[:, :] += (lam[l, i,
                                                        k, j]*dudxdvdx[l, :, k, :]).real
                    else:
                        for k, l in prod(range(self.esdim), range(self.esdim)):
                            partelmat_arr[:, :] += (lam[l, i,
                                                        k, j]*dudxdvdx[i, l, :, j, k, :]).real

                    elmat.AddMatrix(self.partelmat, te_nd*i, tr_nd*j)

            else:
                for i, j in prod(range(self.vdim_te), range(self.vdim_tr)):
                    self.partelmat.Assign(0.0)

                    if not self._metric:
                        for k, l in prod(range(self.esdim), range(self.esdim)):
                            partelmat_arr[:, :] += (lam[l, i,
                                                        k, j]*dudxdvdx[l, :, k, :]).imag
                    else:
                        for k, l in prod(range(self.esdim), range(self.esdim)):
                            partelmat_arr[:, :] += (lam[l, i,
                                                        k, j]*dudxdvdx[i, l, :, j, k, :]).imag

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

        self._proc_esindex(esindex)

        # assert self.vdim_tr == self.esdim, "vector dim and extedned spacedim must be the same"
        # print('esdim flag', self.esflag, self.esflag2)

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
                #u_xx, u_xy, u_xz, u_yz, u_zz, u_yy
                # hess = tr_hshape_arr[:, [0, 1, 2, 1, 5,
                #                         3, 2, 3, 4]].reshape(tr_nd, 3, 3)
                #u_xx, u_xy, u_xz, u_yy, u_yz, u_zz
                hess = tr_hshape_arr[:, [0, 1, 2, 1, 3,
                                         4, 2, 4, 5]].reshape(tr_nd, 3, 3)
            elif dim == 2:
                hess = tr_hshape_arr[:, [0, 1, 1, 2]].reshape(tr_nd, 2, 2)
            elif dim == 1:
                hess = tr_hshape_arr[:, [0, ]].reshape(tr_nd, 1, 1)

            for i in self.esflag:
                for j in self.esflag:
                    tr_merged_arr[:, i, j] = hess[:, i, j]
            for i in self.esflag:
                for kk, j in enumerate(self.esflag2):
                    tr_merged_arr[:, i, j] = tr_dshapedxt_arr[:,
                                                              i]*self.es_weight[kk]
            for kk, i in enumerate(self.esflag2):
                for j in self.esflag:
                    tr_merged_arr[:, i, j] = tr_dshapedxt_arr[:,
                                                              j]*self.es_weight[kk]
            for kk, i in enumerate(self.esflag2):
                for ll, j in enumerate(self.esflag2):
                    tr_merged_arr[:, i, j] = tr_shape_arr * \
                        self.es_weight[kk]*self.es_weight[ll]

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

        #assert self.vdim_tr == self.esdim, "vector dim and extedned spacedim must be the same"
        # print('esdim flag', self.esflag, self.esflag2)

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
                #u_xx, u_xy, u_xz, u_yz, u_zz, u_yy
                # hess = tr_hshape_arr[:, [0, 1, 2, 1, 5,
                #                         3, 2, 3, 4]].reshape(tr_nd, 3, 3)
                #u_xx, u_xy, u_xz, u_yy, u_yz, u_zz
                hess = tr_hshape_arr[:, [0, 1, 2, 1, 3,
                                         4, 2, 4, 5]].reshape(tr_nd, 3, 3)

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
