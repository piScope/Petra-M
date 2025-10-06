from petram.phys.phys_const import levi_civita3
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
    'PyVectorWHIntegrator')


class PyVectorWHIntegrator(PyVectorIntegratorBase):
    use_complex_coefficient = True
    support_metric = True

    def __init__(self, lam, vdim1=None, vdim2=None, esindex=None, metric=None,
                 use_covariant_vec=False, *, ir=None):
        #
        #   weak integrator for hessian matrix
        #
        #   base integrator for weakform which has  (\partial test, lambda \partial trial)
        #
        #      derivded class includes
        #         - diffusion
        #         - curl-curl
        #
        #   lmabda(l, i, k. j) gte(i,l) * gtr(j, k)
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
        #    then we compute  lam[l,i,k,j] d_l v_i  d_k u^j  (sqrt(g)) dxdydz
        #    where lam[l,i,k,j] is a coefficient.
        #
        #    if use_covarient_vec is True, u is treated as covarient, correspondingly
        #    v is treated contravariant.
        #

        PyVectorIntegratorBase.__init__(self, use_covariant_vec, ir)
        self.init_step2(lam, vdim1, vdim2, esindex, metric)

    def alloc_workspace(self):
        #
        #  allocate array for assembly
        #
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

            shape = (self.esdim, self.vdim_te, self.esdim, self.vdim_tr)
            lam = self.eval_complex_lam(trans, ip, shape)

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

            for i, k in enumerate(self.esflag2):
                tr_merged_arr[k, :] = (
                    tr_shape_arr*w2*self.es_weight[i]).transpose()
                te_merged_arr[k, :] = (
                    te_shape_arr*w2*self.es_weight[i].conj()).transpose()

            # vdim(dv/dx), nd, vdim(du/dx), nd
            dudxdvdx = np.tensordot(te_merged_arr, tr_merged_arr, 0)*ip.weight

            if self._metric is not None:
                vdudx = np.tensordot(
                    te_shape_arr*w2, tr_merged_arr, 0)*ip.weight  # nd, vdim(du/dx), nd
                dvdxu = np.tensordot(
                    te_merged_arr, tr_shape_arr*w2, 0)*ip.weight  # vdim(dv/dx), nd, nd
                vu = np.tensordot(
                    te_shape_arr*w2, tr_shape_arr*w2, 0)*ip.weight  # nd, nd

                # computing additional coefficients for curvilinear coords.
                #
                # lam is [l, i, k, j]
                # note: in the comment below, index is notes as (n, i, k, j)
                # in order to match the discription in the implementation note.

                lam *= self.eval_sqrtg(trans, ip)   # x sqrt(g)

                chris = self.eval_christoffel(trans, ip, self.esdim)
                if self.use_covariant_vec:
                    # ipn, nikj -> pkj
                    M = np.tensordot(chris, lam, ((0, 2), (1, 0)))
                    # nikj, qjk -> niq
                    N = np.tensordot(lam, chris, ((2, 3), (2, 1)))
                    # pkj, qjk -> pq
                    P = np.tensordot(M, chris, ((1, 2), (2, 1)))

                    N = -N
                    P = -P
                else:
                    # pin, nikj -> pkj (p->i)
                    M = np.tensordot(chris, lam, ((1, 2), (1, 0)))
                    # nikj, jqk -> niq (q->j)
                    N = np.tensordot(lam, chris, ((2, 3), (2, 0)))
                    # pkj, jqk  -> pq (ij)
                    P = np.tensordot(M, chris, ((1, 2), (2, 0)))

                    M = -M
                    P = -P

            else:
                M = None

            if self._realimag:
                for i, j in prod(range(self.vdim_te), range(self.vdim_tr)):
                    self.partelmat.Assign(0.0)

                    for k, l in prod(range(self.esdim), range(self.esdim)):
                        partelmat_arr[:, :] += (lam[l, i,
                                                    k, j]*dudxdvdx[l, :, k, :]).real
                    if M is not None:
                        for k in range(self.esdim):
                            partelmat_arr[:,
                                          :] += (M[i, k, j]*vdudx[:, k, :]).real
                        for l in range(self.esdim):
                            partelmat_arr[:,
                                          :] += (N[l, i, j]*dvdxu[l, :, :]).real
                        partelmat_arr[:, :] += (P[i, j]*vu[:, :]).real
                    elmat.AddMatrix(self.partelmat, te_nd*i, tr_nd*j)

            else:
                for i, j in prod(range(self.vdim_te), range(self.vdim_tr)):
                    self.partelmat.Assign(0.0)

                    for k, l in prod(range(self.esdim), range(self.esdim)):
                        partelmat_arr[:, :] += (lam[l, i,
                                                    k, j]*dudxdvdx[l, :, k, :]).imag
                    if M is not None:
                        for k in range(self.esdim):
                            partelmat_arr[:,
                                          :] += (M[i, k, j]*vdudx[:, k, :]).imag
                        for l in range(self.esdim):
                            partelmat_arr[:,
                                          :] += (N[l, i, j]*dvdxu[l, :, :]).imag
                        partelmat_arr[:, :] += (P[i, j]*vu[:, :]).imag

                    elmat.AddMatrix(self.partelmat, te_nd*i, tr_nd*j)


class PyVectorDiffusionIntegrator(PyVectorWHIntegrator):
    #
    #    implementation of l[l, u, k, j] g^nl \nabla_n \nabla_k u^j
    #
    #
    def eval_complex_lam(self, trans, ip, shape):
        lam = PyVectorWHIntegrator.eval_complex_lam(self, trans, ip, shape)
        if self._metric is not None:
            gij = self.eval_ctmetric(trans, ip)  # x g^{ij}

            # (l, n) (l, i, k, j) ->  (n, i, k, j) (n becomes l)
            lam = np.tensordot(gij, lam, axes=(0, 0))

        return lam


class PyVectorCurlCurlIntegrator(PyVectorWHIntegrator):
    @classmethod
    def coeff_shape(cls, vdim1, vdim2=None, esindex=None, ir=None):
        ret = PyVectorWHIntegrator.coeff_shape(vdim1, vdim2, esindex, ir)
        # ret = (esdim, vdim1, esdim, vdim2,)
        return ret[1], ret[3]

    def eval_complex_lam(self, trans, ip, shape):
        # shape is given as (self.esdim, self.vdim_te, self.esdim, self.vdim_tr)
        shape0 = (shape[1], shape[3])
        lam = PyVectorWHIntegrator.eval_complex_lam(self, trans, ip, shape0)

        lev = levi_civita3

        if self._metric is not None:
            # / sqrt(g) needed for E (epsiolon/sqrt(g))
            lam /= self.eval_sqrtg(trans, ip)
            # / sqrt(g) needed for E (epsiolon/sqrt(g))
            lam /= self.eval_sqrtg(trans, ip)

            if self.use_covariant_vec:
                g = self.eval_cometric(trans, ip)  # x g_{pm}
                L3 = np.tensordot(g, lev, (0, 0))  # pm piq -> miq
                L3 = np.tensordot(L3, g, (2, 0))  # miq ql -> mil
                L4 = lev   # nkj
            else:
                g = self.eval_cometric(trans, ip)  # x g_{mp}
                L4 = np.tensordot(g, lev, (1, 0))  # np nkq -> nkq
                L4 = np.tensordot(L4, g, (2, 0))  # nkq qj -> nkj
                L3 = lev   # mil
        else:
            L3 = lev
            L4 = lev

        lam = np.tensordot(L3, lam, (0, 0))  # mil, mn = iln
        lam = np.tensordot(lam, L4, (2, 0))  # iln, nkj = ilkj
        lam = np.swapaxes(lam, 0, 1)  # ilkj -> likj

        return lam
