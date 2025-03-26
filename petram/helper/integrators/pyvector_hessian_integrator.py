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


class PyVectorHessianIntegrator(PyVectorIntegratorBase):
    use_complex_coefficient = True
    support_metric = True

    def __init__(self, lam, vdim1=None, vdim2=None, esindex=None, metric=None,
                 use_covariant_vec=False, *, ir=None):
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
        PyVectorIntegratorBase.__init__(self, use_covariant_vec, ir)
        self.init_step2(lam, vdim1, vdim2, esindex, metric)

        self.enforce_p_none = False

    def alloc_workspace(self):

        self.tr_shape = mfem.Vector()
        self.te_shape = mfem.Vector()
        self.tr_dshape = mfem.DenseMatrix()
        self.tr_dshapedxt = mfem.DenseMatrix()
        self.tr_hshape = mfem.DenseMatrix()

        self.tr_merged = mfem.DenseMatrix()

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

        return (vdim1, esdim, esdim, vdim2)

    def AssembleElementMatrix(self, el, trans, elmat):
        self.AssembleElementMatrix2(el, el, trans, elmat)

    def AssembleElementMatrix2(self, trial_fe, test_fe, trans, elmat):
        # if self.ir is None:
        #    self.ir = mfem.DiffusionIntegrator.GetRule(trial_fe, test_fe)
        if self.lam_real is None:
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

        assert dim == sdim, "dim must be sdim"

        self.tr_shape.SetSize(tr_nd)
        self.te_shape.SetSize(te_nd)
        self.tr_dshape.SetSize(tr_nd, dim)
        self.tr_hshape.SetSize(tr_nd, dim*(dim+1)//2)

        tr_shape_arr = self.tr_shape.GetDataArray()
        te_shape_arr = self.te_shape.GetDataArray()
        tr_dshape_arr = self.tr_dshape.GetDataArray()
        tr_hshape_arr = self.tr_hshape.GetDataArray()

        trh_merged_arr = np.zeros(
            (tr_nd, self.esdim, self.esdim), dtype=np.complex128)

        if self._metric is not None:
            trd_merged_arr = np.zeros((tr_nd, self.esdim), dtype=np.complex128)

        #print(self.es_weight, self.esflag2)

        for i in range(self.ir.GetNPoints()):

            ip = self.ir.IntPoint(i)
            trans.SetIntPoint(ip)

            shape = (self.vdim_te, self.esdim, self.esdim, self.vdim_tr)
            lam = self.eval_complex_lam(trans, ip, shape)

            # construct test/trial space array

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

            for kk, i in enumerate(self.esflag):
                for ll, j in enumerate(self.esflag):
                    trh_merged_arr[:, i, j] = hess[:, kk, ll]
            for ll, i in enumerate(self.esflag):
                for kk, j in enumerate(self.esflag2):
                    trh_merged_arr[:, i, j] = tr_dshape_arr[:,
                                                            ll]*self.es_weight[kk]
            for kk, i in enumerate(self.esflag2):
                for ll, j in enumerate(self.esflag):
                    trh_merged_arr[:, i, j] = tr_dshape_arr[:,
                                                            ll]*self.es_weight[kk]
            for kk, i in enumerate(self.esflag2):
                for ll, j in enumerate(self.esflag2):
                    trh_merged_arr[:, i, j] = tr_shape_arr * \
                        self.es_weight[kk]*self.es_weight[ll]

            detJ = trans.Weight()
            weight = ip.weight

            dudxdvdx = np.tensordot(
                te_shape_arr, trh_merged_arr, 0)*weight*detJ

            if self._metric is not None:
                detm = self.eval_sqrtg(trans, ip)
                lam *= detm

                # construct merged trial du/dx
                trd_merged_arr[:, self.esflag] = tr_dshape_arr
                for i, k in enumerate(self.esflag2):
                    trd_merged_arr[:, k] = tr_shape_arr * \
                        self.es_weight[i]  # nd vdim(d/dx)

                vdudx = np.tensordot(
                    te_shape_arr, trd_merged_arr, 0)*weight*detJ  # nd, nd, vdim(d/dx)
                vu = np.tensordot(
                    te_shape_arr, tr_shape_arr, 0)*weight*detJ  # nd, nd

                chris = self.eval_christoffel(trans, ip, self.esdim)
                dchris = self.eval_dchristoffel(trans, ip, self.esdim)

                if self.use_covariant_vec:
                    # B^i P[i,k,j] \partial_k A_j
                    # (i, k, l,j ) (m, l, j) -> i, k, j (m becomes j)
                    P1 = - np.tensordot(lam, chris, ([2, 3], [1, 2]))
                    # (i, k, l,j ) (m, l, k) -> i, j, k (m becomes k)
                    P2 = - np.tensordot(lam, chris, ([1, 2], [2, 1]))
                    P2 = np.swapaxes(P2, 1, 2)
                    # (i, k, l,j ) (m, j, k) -> i, k, j (m becomes j, l becomes k)
                    P3 = - np.tensordot(lam, chris, ([1, 3], [2, 1]))

                    P = P1 + P2 + P3

                    # B^i Q[i,j] A_j
                    tmp = np.tensordot(chris, chris, (0, 1))
                    #  this is either
                    # (m, l, k) (n, m, j) -> l, k, n, j
                    # (m, j, k) (n, l, m) -> j, k, n, l

                    # (i, k, l, j) + (l, k, n, j) -> i, n
                    Q1 = np.tensordot(lam, tmp, ((1, 2, 3), (1, 0, 3)))
                    # (i, k, l, j) + (j, k, n, l) -> i, n
                    Q2 = np.tensordot(lam, tmp, ((1, 2, 3), (1, 3, 0)))
                    # (i, k, l, j) + (m, l, j, k) -> i, m
                    Q3 = -np.tensordot(lam, dchris, ((1, 2, 3), (3, 1, 2)))

                    Q = Q1 + Q2 + Q3

                else:
                    # B^i P[i,k,j] \partial_k A_j
                    # (i, k, l,j ) (j, l, m) -> i, k, j (m becomes j)
                    P1 = np.tensordot(lam, chris, ([2, 3], [1, 0]))
                    # (i, k, l,j ) (m, l, k) -> i, j, k (m becomes k)
                    P2 = - np.tensordot(lam, chris, ([1, 2], [2, 1]))
                    P2 = np.swapaxes(P2, 1, 2)
                    # (i, k, l,j ) (j, m, k) -> i, k, j (m becomes j, l becomes k)
                    P3 = np.tensordot(lam, chris, ([1, 3], [2, 0]))
                    P = P1 + P2 + P3

                    # B^i Q[i,j] A_j
                    tmp = np.tensordot(chris, chris, (0, 1))
                    #  this is either
                    # (m, l, k) (j, m, n) -> l, k, j, n
                    # (m, l, n) (j, m, k) -> l, n, j, k

                    # (i, k, l, j) + (l, k, j, n) -> i, n
                    Q1 = -np.tensordot(lam, tmp, ((1, 2, 3), (1, 0, 2)))
                    # (i, k, l, j) + (l, n, j, k) -> i, n
                    Q2 = np.tensordot(lam, tmp, ((1, 2, 3), (3, 0, 2)))
                    # (i, k, l, j) + (j, l, m, k) -> i, m
                    Q3 = np.tensordot(lam, dchris, ((1, 2, 3), (3, 1, 0)))

                    Q = Q1 + Q2 + Q3
            else:
                P = None

            if self.enforce_p_none:
                P = None

            if self._realimag:
                for i, j in prod(range(self.vdim_te), range(self.vdim_tr)):
                    self.partelmat.Assign(0.0)
                    for k, l in prod(range(self.esdim), range(self.esdim)):
                        partelmat_arr[:, :] += (lam[i, k,
                                                    l, j]*dudxdvdx[:, :, k, l]).real

                    if P is not None:
                        for k in range(self.esdim):
                            partelmat_arr[:,
                                          :] += (P[i, k, j] * vdudx[:, :, k]).real
                        partelmat_arr[:, :] += (Q[i, j] * vu[:, :]).real

                    elmat.AddMatrix(self.partelmat, te_nd*i, tr_nd*j)

            else:
                for i, j in prod(range(self.vdim_te), range(self.vdim_tr)):
                    self.partelmat.Assign(0.0)
                    for k, l in prod(range(self.esdim), range(self.esdim)):
                        partelmat_arr[:, :] += (lam[i, k,
                                                    l, j]*dudxdvdx[:, :, k, l]).imag
                    if P is not None:
                        for k in range(self.esdim):
                            partelmat_arr[:,
                                          :] += (P[i, k, j] * vdudx[:, :, k]).imag
                        partelmat_arr[:, :] += (Q[i, j] * vu[:, :]).imag

                    elmat.AddMatrix(self.partelmat, te_nd*i, tr_nd*j)


# alias for backword compatibility
PyVectorPartialPartialIntegrator = PyVectorHessianIntegrator


class PyVectorStrongCurlCurlIntegrator(PyVectorHessianIntegrator):
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

    def eval_complex_lam(self, trans, ip, shape):
        '''
        generate lambda in (3,3,3,3) shape on the fly
        as for indexing, see the implementation note.
        '''
        shape = (3, 3)
        lam = PyVectorHessianIntegrator.eval_complex_lam(
            self, trans, ip, shape)
        scale = self.eval_lam_scale(trans, ip)

        if self._metric is not None:
            g_xx = self.eval_cometric(trans, ip)  # x g_{lp}

            tmp = lam/self.eval_sqrtg(trans, ip)   # x /sqrt(g)
            tmp /= self.eval_sqrtg(trans, ip)   # x /sqrt(g)

            if not self.use_covariant_vec:
                tmp = np.tensordot(tmp, g_xx, (1, 0))  # km mn -> kn
            else:
                pass  # km == kn
        else:
            tmp = lam

        tmp = np.tensordot(lam, tmp, 0)  # jrkn

        # pjk, jrkn -> prn
        tmp = np.tensordot(levi_civita3, tmp, ((1, 2), (0, 2)))

        if self._metric is not None:
            tmp = np.tensordot(g_xx, tmp, (1, 0))   # ip prn -> irn

        tmp = np.tensordot(lam, tmp, 0)  # sq irn

        # tsi sqirn -> tqrn
        tmp = np.tensordot(levi_civita3, tmp, ((1, 2), (0, 2)))

        if self._metric is not None and self.use_covariant_vec:
            tmp = np.tensordot(g_xx, tmp, (1, 0))  # lt tqrn -> lqrn
        else:
            pass  # sqrn == lqrn

        # tmp follows iklj index rule  (i = test j = trial, k and l derivative)

        # self.enforce_p_none = True

        return tmp*scale
