from itertools import product as prod
import numpy as np
from numpy.linalg import det, norm, inv

from petram.mfem_config import use_parallel
if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('PyVectorIntegratorBase')


class PyVectorIntegratorBase(mfem.PyBilinearFormIntegrator):
    support_metric = False

    def __init__(self, use_covariant_vec, *args, **kwargs):
        mfem.PyBilinearFormIntegrator.__init__(self, *args, **kwargs)
        self._q_order = 0
        self._metric = None
        self._christoffel = None
        self._realimag = False
        self._use_covariant_vec = use_covariant_vec

        # allocate workspace will assigne object to these names.
        self.valr = None
        self.vali = None

        self.scale_real = None
        self.scale_imag = None

    def init_step2(self, lam, vdim1, vdim2, esindex, metric):
        if lam is None:
            assert False, "lam is None"
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

    def alloc_workspace(self):
        assert False, "subclass shouuld implement alloc_workspace"

    @property
    def q_order(self):
        return self._q_order

    @q_order.setter
    def q_order(self, value):
        self._q_order = value

    @property
    def use_covariant_vec(self):
        return self._use_covariant_vec

    def set_ir(self, trial_fe,  test_fe, trans, delta=0):
        order = (trial_fe.GetOrder() + test_fe.GetOrder() +
                 trans.OrderW() + self.q_order + delta)

        if trial_fe.Space() == mfem.FunctionSpace.rQk:
            ir = mfem.RefinedIntRules.Get(trial_fe.GetGeomType(), order)
        else:
            ir = mfem.IntRules.Get(trial_fe.GetGeomType(), order)

        self.ir = ir
        # self.ir = mfem.DiffusionIntegrator.GetRule(trial_fe, test_fe)

    @classmethod
    def coeff_shape(cls, itg_param):
        raise NotImplementedError("subclass must implement coeff_shape")

    def config_metric_vdim_esindex(self, metric_obj, vdim1, vdim2, esindex):
        if metric_obj is not None:
            self.set_metric(metric_obj)
        else:
            if vdim1 is None:
                # skipping this. in this case set_metric should be called separately.
                return
            if vdim2 is not None:
                self.vdim_te = vdim1
                self.vdim_tr = vdim2
            else:
                self.vdim_te = vdim1
                self.vdim_tr = vdim1

            if esindex is None:
                esindex = list(range(self.vdim_tr))
            self._proc_esindex(esindex)

    def use_conjugate_periodicity(self):
        self.es_weight = np.conj(self.es_weight)

    def set_metric(self, metric_obj, vdim1=None, vdim2=None):
        #
        #  g_ij (metric tensor) is set

        #  sqrt(g_ij) dxdydz:
        #     integration uses sqrt(g_ij) for volume integraiton Jacobian :
        #
        #  inter product is computed by
        #      v^j *u^j   (use_covariant_vec=False)
        #
        #     or
        #
        #      v_i *u_j   (use_covariant_vec= True)
        #
        #
        #   nabla is treated as covariant derivatives thus
        #
        #      d_k v^i = dv^i/dx^k + {i/ j, k} v_^i
        #      d_k v_i = dv^i/dx^k - {i/ j, k} v_^i

        if not self.__class__.support_metric:
            raise NotImplementedError(
                "the integrator does not support metric tensor")

        cc = metric_obj.christoffel()
        dcc = metric_obj.dchristoffel()
        flag = metric_obj.is_diag_metric()
        metric = metric_obj.metric()

        self._metric = metric  # co anc ct metric
        self._christoffel = cc
        self._dchristoffel = dcc
        self._metric_diag = flag

        self._use_covariant_vec = metric_obj.use_covariant_vec

        self.metric = mfem.Vector()
        self.chris = mfem.Vector()
        self.dchris = mfem.Vector()

        if vdim1 is None:
            self.vdim_te = metric_obj.vdim1
        else:
            self.vdim_te = vdim1
        if vdim2 is None:
            self.vdim_tr = metric_obj.vdim2
        else:
            self.vdim_tr = vdim2

        self._proc_esindex(metric_obj.esindex)

    def eval_g(self, trans, ip):
        # determinant of contravariant metrix'
        self._metric[0].Eval(self.metric, trans, ip)
        m = self.metric.GetDataArray()
        if self._metric_diag:
            # m is vector
            detm = np.prod(m)
        else:
            # m is matrix
            detm = det(m)
        return detm

    def eval_sqrtg(self, trans, ip):
        # determinant of contravariant metrix'
        self._metric[0].Eval(self.metric, trans, ip)
        m = self.metric.GetDataArray()
        if self._metric_diag:
            # m is vector
            detm = np.prod(m)
        else:
            # m is matrix
            detm = det(m)
        return np.sqrt(detm)

    def eval_cometric(self, trans, ip):
        self._metric[0].Eval(self.metric, trans, ip)
        m = self.metric.GetDataArray()
        if self._metric_diag:
            # m is vector
            m = np.diag(m)
        return m

    def eval_ctmetric(self, trans, ip):
        self._metric[1].Eval(self.metric, trans, ip)
        m = self.metric.GetDataArray()
        if self._metric_diag:
            # m is vector
            m = np.diag(m)
        return m

    def eval_christoffel(self, trans, ip, esdim):
        self._christoffel.Eval(self.chris, trans, ip)
        chris = self.chris.GetDataArray().reshape(esdim, esdim, esdim)
        return chris

    def eval_dchristoffel(self, trans, ip, esdim):
        self._dchristoffel.Eval(self.dchris, trans, ip)
        dchris = self.dchris.GetDataArray().reshape(esdim, esdim, esdim, esdim)
        return dchris

    def set_realimag_mode(self, mode):
        self._realimag = (mode == 'real')

    @classmethod
    def _proc_vdim1vdim2(cls, vdim1, vdim2):

        import petram.helper.curvilinear_coords
        known_metric = ("planer1d", "planer2d",
                        "cylindrical1d", "cylindrical1dco", "cylindrical1dct",
                        "cylindrical2d", "cylindrical2dco", "cylindrical2dct",)

        if vdim1 in known_metric:
            if not cls.support_metric:
                assert False, "metric is specified, but the integrator does not support it"

            cls = getattr(petram.helper.curvilinear_coords, vdim1)
            metric_obj = cls(vdim2)
            return metric_obj
        elif isinstance(vdim1, str):
            assert False, "unknonw metric specifier:" + vdim1
        else:
            return None

    def _proc_esindex(self, esindex):
        def iscomplex(x):
            return isinstance(x, (complex, np.complex128, np.complex64))

        flag1 = [not iscomplex(x) for x in esindex]
        flag2 = [iscomplex(x) for x in esindex]

        esindex = np.array(esindex)
        if any(flag2):
            self.esflag = np.where(flag1)[0]
            self.esflag2 = np.where(flag2)[0]
            self.es_weight = esindex[flag2]
        else:
            self.esflag = np.where(esindex >= 0)[0]
            self.esflag2 = np.where(esindex == -1)[0]
            self.es_weight = np.ones(len(self.esflag2))
        self.esdim = len(esindex)

        #print("weight", self.esflag, self.esflag2, self.es_weight)

    def eval_real_lam(self, trans, ip, shape):
        if len(shape) == 2:
            # in this case, coefficient can be scalar
            scalar_coeff = isinstance(self.lam_real, mfem.Coefficient)
            if scalar_coeff:
                assert shape[0] == shape[1], "scalar coefficeint allows only for square matrix"

        else:
            scalar_coeff = False

        if scalar_coeff:
            lam = self.lam_real.Eval(trans, ip)
            lam = np.diag([lam]*shape[0])
        else:
            self.lam_real.Eval(self.valr, trans, ip)
            lam = self.valr.GetDataArray()

            if len(lam) == np.prod(shape):
                lam = lam.reshape(shape)
            elif len(lam) == 1 and len(shape) == 2 and shape[0] == shape[1]:
                lam = np.diag([lam[0]]*shape[0])
            elif len(lam) == shape[0] and len(shape) == 2 and shape[0] == shape[1]:
                lam = np.diag(lam)
            else:
                assert False, "wrong coeffi. shape: " + \
                    str(shape) + "is needed." + str(lam.shape) + " was found"

        return lam

    def eval_complex_lam(self, trans, ip, shape):
        if len(shape) == 2:
            # in this case, coefficient can be scalar
            scalar_coeff = isinstance(self.lam_real, mfem.Coefficient)
            if scalar_coeff:
                assert shape[0] == shape[1], "scalar coefficeint allows only for square matrix"
        else:
            scalar_coeff = False

        if scalar_coeff:
            lam = self.lam_real.Eval(trans, ip)
            if self.lam_imag is not None:
                lam = lam + 1j*self.lam_imag.Eval(trans, ip)

            lam = np.diag([lam]*shape[0])
        else:
            self.lam_real.Eval(self.valr, trans, ip)
            lam = self.valr.GetDataArray()

            if self.lam_imag is not None:
                self.lam_imag.Eval(self.vali, trans, ip)
                lam = lam + 1j*self.vali.GetDataArray()

            if len(lam) == np.prod(shape):
                lam = lam.reshape(shape)
            elif len(lam) == 1 and len(shape) == 2 and shape[0] == shape[1]:
                lam = np.diag([lam[0]]*shape[0])
            elif len(lam) == shape[0] and len(shape) == 2 and shape[0] == shape[1]:
                lam = np.diag(lam)
            else:
                assert False, "wrong coeffi. shape: " + \
                    str(shape) + " is needed." + str(lam.shape) + " was found"
        return lam

    def set_lam_scale(self, scale):
        if not hasattr(scale, "get_real_coefficient"):
            self.scale_real = scale
            self.scale_imag = None
        else:
            self.scale_real = scale.get_real_coefficient()
            self.scale_imag = scale.get_imag_coefficient()

    def eval_lam_scale(self, trans, ip):
        if self.scale_real is None:
            return 1.0
        s = self.scale_real.Eval(trans, ip)
        if self.scale_imag is not None:
            s = s + 1j*self.scale_imag.Eval(trans, ip)
        return s
