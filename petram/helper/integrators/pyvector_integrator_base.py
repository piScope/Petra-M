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

    def __init__(self, *args, **kwargs):
        mfem.PyBilinearFormIntegrator.__init__(self, *args, **kwargs)
        self._q_order = 0
        self._metric = None
        self._christoffel = None
        self._realimag = False

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
        # self.ir = mfem.DiffusionIntegrator.GetRule(trial_fe, test_fe)

    @classmethod
    def coeff_shape(cls, itg_param):
        raise NotImplementedError("subclass must implement coeff_shape")

    def set_metric(self, metric, use_covariant_vec=False):
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

        mm = metric.metric()
        cc = metric.christoffel()
        flag = metric.is_diag_metric()

        self._metric = mm
        self._christoffel = cc
        self._metric_diag = flag

        self._use_covariant_vec = use_covariant_vec

        self.metric = mfem.Vector()
        self.chris = mfem.Vector()

    def eval_metric(self, trans, ip):
        self._metric.Eval(self.metric, trans, ip)
        m = self.metric.GetDataArray()
        if self._metric_diag:
            # m is vector
            detm = np.prod(m)
            #m = np.diag(m)
        else:
            # m is matrix
            detm = det(m)
        return detm

    def eval_christoffel(self, trans, ip, esdim):
        self._christoffel.Eval(self.chris, trans, ip)
        chris = self.chris.GetDataArray().reshape(esdim, esdim, esdim)
        return chris

    def set_realimag_mode(self, mode):
        self._realimag = (mode == 'real')

    @classmethod
    def _proc_vdim1vdim2(cls, vdim1, vdim2):

        if vdim1.startswith('cyclindrical2d'):
            if vdim1 == 'cyclindrical2dco':
                 use_covariant_vec = True
            elif vdim1 == 'cyclindrical2dct':
                 use_covariant_vec = False
            else:
                assert False, "unsupported option"

            vdim1 = 3
            esindex = (0, vdim2*1j, 1)
            vdim2 = 3

            from petram.helper.curvelinear_coords import cylindrical2d

            return True, (vdim1, vdim2, esindex, cylindrical2d, use_covariant_vec)

        elif vdim1.startswith('cyclindrical1d'):
            if vdim1 == 'cyclindrical1dco':
                 use_covariant_vec = True
            elif vdim1 == 'cyclindrical1dct':
                 use_covariant_vec = False
            else:
                assert False, "unsupported option"

            vdim1 = 3
            esindex = [0]
            esindex.append(vdim2[0]*1j)
            esindex.append(vdim2[1]*1j)

            vdim2 = 3

            from petram.helper.curvelinear_coords import cylindrical1d

            return True, (vdim1, vdim2, esindex, cylindrical1d, use_covariant_vec)

        elif vdim1 == 'planer2d':
            vdim1 = 3
            esindex = (0, 1, vdim2*1j)
            vdim2 = 3
            metric = None
            return True, (vdim1, vdim2, esindex, metric)

        elif vdim1 == 'planer1d':
            vdim1 = 3
            esindex = [0]
            esindex.append(vdim2[0]*1j)
            esindex.append(vdim2[1]*1j)
            vdim2 = 3
            metric = None
            return True, (vdim1, vdim2, esindex, metric)
        else:
            pass

        return False, None

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
