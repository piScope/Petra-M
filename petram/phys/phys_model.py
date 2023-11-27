from __future__ import print_function
from petram.debug import handle_allow_python_function_coefficient
from petram.phys.vtable import VtableElement, Vtable
from petram.helper.variables import Variable, eval_code
from petram.phys.vtable import VtableElement, Vtable, Vtable_mixin
from petram.helper.variables import NativeCoefficientGenBase, CoefficientVariable
from petram.mfem_config import use_parallel

import numpy as np
from os.path import dirname, basename, isfile, join
import warnings
import glob
import types
import numbers
from abc import abstractmethod

import petram
from petram.model import Model, Bdry, Domain
from petram.namespace_mixin import NS_mixin

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('Phys')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem


# not that PyCoefficient return only real number array


class PhysConstant(mfem.ConstantCoefficient):
    def __init__(self, value):
        self.value = value
        mfem.ConstantCoefficient.__init__(self, value)

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.value) + ")"


class PhysVectorConstant(mfem.VectorConstantCoefficient):
    def __init__(self, value):
        self._value = mfem.Vector(value)
        self.value = self._value.GetDataArray()
        mfem.VectorConstantCoefficient.__init__(self, value)

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self._value) + ")"


class PhysMatrixConstant(mfem.MatrixConstantCoefficient):
    def __init__(self, value):
        v = mfem.Vector(np.transpose(value).flatten())
        m = mfem.DenseMatrix(v.GetData(), value.shape[0], value.shape[1])
        self._value = (v, m)
        self.value = m.GetDataArray()
        mfem.MatrixConstantCoefficient.__init__(self, m)

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self._value) + ")"


def try_eval(exprs, l, g):
    '''
    first evaulate as it is
    if the result is list.. return
    if not evaluate w/o knowing any Variables
    '''
    try:
        value = eval(exprs, l, g)
        if isinstance(value, list):
            return True, [value]
        ll = [x for x in l if not isinstance(x, Variable)]
        gg = [x for x in g if not isinstance(x, Variable)]
        value = eval(exprs, ll, gg)
        return True, [value]
    except BaseException:
        return False, exprs


class Coefficient_Evaluator(object):
    def __init__(self, exprs, ind_vars, l, g, real=True):
        '''
        this is complicated....
           elemental (or array) form
            [1,a,3]  (a is namespace variable) is given as [[1, a (valued), 3]]

           single box
              1+1j   is given as '(1+1j)' (string)

           matrix
              given as string like '[0, 1, 2, 3, 4]'
              if variable is used it is become string element '['=variable', 1, 2, 3, 4]'
              if =Varialbe in matrix form, it is passed as [['Variable']]
        '''
        flag, exprs = try_eval(exprs, l, g)
        #print("after try_eval", flag, exprs)
        if not flag:
            if isinstance(exprs, str):
                exprs = [exprs]
            # elif isinstance(exprs, float):
            #    exprs = [exprs]
            # elif isinstance(exprs, long):
            #    exprs = [exprs]
            elif isinstance(exprs, numbers.Number):
                exprs = [exprs]
            else:
                pass
        if isinstance(exprs, list) and isinstance(exprs[0], list):
            exprs = exprs[0]
        #dprint1("final exprs", exprs)
        self.l = {}
        # TODO g must be copied since some may passs _global_ns.
        # (I don't do it now since it requires a substantial testing)
        self.g = g
        for key in l.keys():
            self.g[key] = l[key]
        self.real = real
        self.variables = []

        self.co = []

        for expr in exprs:
            if isinstance(expr, str):
                code = compile(expr.strip(), '<string>', 'eval')
                names = code.co_names
                for n in names:
                    if (n in g and isinstance(g[n], NativeCoefficientGenBase)):
                        coeff_var = CoefficientVariable(g[n], l, g)
                        self.variables.append((n, coeff_var))
                    elif n in g and isinstance(g[n], Variable):
                        self.variables.append((n, g[n]))
                        for nn in g[n].dependency:
                            self.variables.append((nn, g[nn]))
                    else:
                        pass
                self.co.append(code)
            else:
                self.co.append(expr)

        # 'x, y, z' -> 'x', 'y', 'z'
        self.ind_vars = [x.strip() for x in ind_vars.split(',')]
        self.exprs = exprs
        self.flags = [isinstance(co, types.CodeType) for co in self.co]
        self.variables_dd = dict(self.variables)

    def EvalValue(self, x):
        for k, name in enumerate(self.ind_vars):
            self.l[name] = x[k]
        for n, v in self.variables:
            kwargs = {}
            for nn in v.dependency:
                kwargs[nn] = self.variables_dd[nn]()
            for nn in v.grad:
                kwargs['grad'+nn] = self.variables_dd[nn].eval_grad()
            for nn in v.curl:
                kwargs['curl'+nn] = self.variables_dd[nn].eval_curl()
            for nn in v.div:
                kwargs['div'+nn] = self.variables_dd[nn].eval_div()
            self.l[n] = v(**kwargs)

        val = [eval_code(co, self.g, self.l, flag=flag)
               for co, flag in zip(self.co, self.flags)]
        return np.array(val, copy=False).flatten()
        # return np.array(val, copy=False).ravel()  ## this may be okay


class PhysCoefficient(mfem.PyCoefficient, Coefficient_Evaluator):
    def __init__(self, exprs, ind_vars, l, g, real=True, isArray=False):
        # if not isArray:
        #    exprs = [exprs]
        Coefficient_Evaluator.__init__(self, exprs, ind_vars, l, g, real=real)
        mfem.PyCoefficient.__init__(self)

        handle_allow_python_function_coefficient(
            "Python function coefficient is created")

    def __repr__(self):
        return self.__class__.__name__ + "(PhysCoefficeint)"

    def Eval(self, T, ip):
        for n, v in self.variables:
            v.set_point(T, ip, self.g, self.l)
        return super(PhysCoefficient, self).Eval(T, ip)

    def EvalValue(self, x):
        # set x, y, z to local variable so that we can use in
        # the expression.

        # note that this class could return array, since
        # a user may want to define multiple variables
        # as an array. In such case, subclass should pick
        # one element.
        val = Coefficient_Evaluator.EvalValue(self, x)
        if len(self.co) == 1 and len(val) == 1:
            return val[0]
        return val


class VectorPhysCoefficient(mfem.VectorPyCoefficient, Coefficient_Evaluator):
    def __init__(self, sdim, exprs, ind_vars, l, g, real=True):
        Coefficient_Evaluator.__init__(self, exprs, ind_vars, l, g, real=real)
        mfem.VectorPyCoefficient.__init__(self, sdim)
        self.sdim = sdim

        handle_allow_python_function_coefficient(
            "Python function coefficient is created")

    def __repr__(self):
        return self.__class__.__name__ + "(VectorPhysCoefficeint)"

    def Eval(self, V, T, ip):
        # if V is Matrix, we need to make our own loop to
        # properly set spacial points in variables.
        if isinstance(ip, mfem.IntegrationPoint):
            for n, v in self.variables:
                v.set_point(T, ip, self.g, self.l)
            return super(VectorPhysCoefficient, self).Eval(V, T, ip)
        elif isinstance(ip, mfem.IntegrationRule):
            M = V
            ir = ip
            Mi = mfem.Vector()
            M.SetSize(self.sdim, ir.GetNPoints())
            for k in range(ir.GetNPoints()):
                M.GetColumnReference(k, Mi)
                ip = ir.IntPoint(k)
                for n, v in self.variables:
                    v.set_point(T, ip, self.g, self.l)
                super(VectorPhysCoefficient, self).Eval(Mi, T, ip)

    def EvalValue(self, x):
        return Coefficient_Evaluator.EvalValue(self, x)


class MatrixPhysCoefficient(mfem.MatrixPyCoefficient, Coefficient_Evaluator):
    def __init__(self, sdim, exprs, ind_vars, l, g, real=True):
        self.sdim = sdim
        Coefficient_Evaluator.__init__(self, exprs, ind_vars, l, g, real=real)
        mfem.MatrixPyCoefficient.__init__(self, sdim)

        handle_allow_python_function_coefficient(
            "Python function coefficient is created")

    def __repr__(self):
        return self.__class__.__name__ + "(MatrixPhysCoefficeint)"

    def Eval(self, K, T, ip):
        for n, v in self.variables:
            v.set_point(T, ip, self.g, self.l)
        return super(MatrixPhysCoefficient, self).Eval(K, T, ip)

    def EvalValue(self, x):
        val = Coefficient_Evaluator.EvalValue(self, x)
        # reshape tosquare matrix (not necessariliy = sdim x sdim)
        # if elment is just one, it formats to diagonal matrix
        s = val.size
        if s == 1:
            return np.eye(self.sdim) * val[0]
        else:
            dim = int(np.sqrt(s))
            return val.reshape(dim, dim)


class Phys(Model, Vtable_mixin, NS_mixin):
    hide_ns_menu = True
    hide_nl_panel = False
    dep_vars_base = []
    der_vars_base = []
    der_vars_vec = []

    has_essential = False
    is_secondary_condition = False   # if true, there should be other
    # condtion assigned to the same
    # edge/face/domain

    _has_4th_panel = True
    can_rename = True
    allow_custom_intorder = False

    # matrix element can be time-dependent
    can_timedpendent = False

    vt = Vtable(tuple())
    vt2 = Vtable(tuple())
    vt3 = Vtable(tuple())
    nlterms = []

    def __init__(self, *args, **kwargs):
        super(Phys, self).__init__(*args, **kwargs)
        NS_mixin.__init__(self, *args, **kwargs)

    def get_info_str(self):
        return NS_mixin.get_info_str(self)

    @property
    def update_flag(self):
        if not hasattr(self, '_update_flag'):
            self._update_flag = False
        return self._update_flag

    @update_flag.setter
    def update_flag(self, prop):
        self._update_flag = prop

    def attribute_set(self, v):
        v = super(Phys, self).attribute_set(v)
        self.vt.attribute_set(v)
        self.vt3.attribute_set(v)

        nl_config = dict()
        for k in self.nlterms:
            nl_config[k] = []
        v['nl_config'] = (False, nl_config)
        v['timestep_config'] = [True, False, False]
        v['timestep_weight'] = ["1", "0", "0"]
        v['isTimeDependent'] = False
        v['isTimeDependent_RHS'] = False
        v['isJacobian'] = False
        v['add_intorder'] = 0
        return v

    def get_possible_bdry(self):
        return []

    def get_possible_domain(self):
        return []

    def get_possible_edge(self):
        return []

    def get_possible_pair(self):
        return []

    def get_possible_point(self):
        return []

    def get_probe(self):
        '''
        return probe name
        '''
        return ''

    def get_independent_variables(self):
        p = self.get_root_phys()
        ind_vars = p.ind_vars
        return [x.strip() for x in ind_vars.split(',')]

    def is_complex(self):
        return False

    def get_restriction_array(self, engine, idx=None):
        mesh = engine.emeshes[self.get_root_phys().emesh_idx]
        intArray = mfem.intArray

        if isinstance(self, Domain):
            size = max(np.max(mesh.attributes.ToList()), engine.max_attr)
        else:
            size = max(
                np.max(
                    mesh.bdr_attributes.ToList()),
                engine.max_bdrattr)

        arr = [0] * size
        if idx is None:
            idx = self._sel_index
        for k in idx:
            arr[k - 1] = 1

        return intArray(arr)

    def restrict_coeff(self, coeff, engine, vec=False, matrix=False,
                       idx=None):
        if len(self._sel_index) == 1 and self._sel_index[0] == -1:
            return coeff
        arr = self.get_restriction_array(engine, idx)
        ret = []
        flag = False

        if not isinstance(coeff, tuple):
            flag = True
            coeff = [coeff]

        for c in coeff:
            if (isinstance(c, mfem.Coefficient) or
                isinstance(c, mfem.VectorCoefficient) or
                    isinstance(c, mfem.MatrixCoefficient)):
                if vec:
                    ret.append(mfem.VectorRestrictedCoefficient(c, arr))
                elif matrix:
                    ret.append(mfem.MatrixRestrictedCoefficient(c, arr))
                else:
                    ret.append(mfem.RestrictedCoefficient(c, arr))
            else:
                ret.append(c)

        if not flag:
            return tuple(ret)
        else:
            return ret[0]

    def get_essential_idx(self, idx):
        '''
        return index of essentail bdr for idx's fespace
        '''
        raise NotImplementedError(
            "you must specify this method in subclass")

    def get_root_phys(self):
        p = self
        while isinstance(p.parent, Phys):
            p = p.parent
        return p

    def get_projection(self):
        return 1

    def update_param(self):
        '''
        called everytime it assembles either matrix or rhs
        '''
        if hasattr(self, 'vt'):
            self.vt.preprocess_params(self)

    def postprocess_extra(self, sol, flag, sol_extra):
        '''
        postprocess extra (Lagrange multiplier) segment
        of solutions.

        sol_extra is a dictionary in which this method
        will add processed data (dataname , value).
        '''
        raise NotImplementedError(
            "you must specify this method in subclass")

    def get_extra_NDoF(self, kfes=0):
        return 0

    def has_extra_DoF(self, kfes=0):
        return False

    def is_extra_RHSonly(self):
        return False

    def has_extra_DoF2(self, kfes, phys, jmatrix):
        '''
        subclass has to overwrite this if extra DoF can couple
        with other FES.
        '''
        if not self.check_jmatrix(jmatrix):
            return False

        if (phys == self.get_root_phys()):
            # if module set only has_extra_DoF, the extra variable
            # couples only in the same Phys module and jmatrix == 0

            # ToDo. Should add deprecated message
            return self.has_extra_DoF(kfes=kfes)
        else:
            return False

    def extra_DoF_name(self):
        '''
        default DoF name
        '''
        return self.get_root_phys().dep_vars[0] + '_' + self.name()

    def extra_DoF_name2(self, kfes=0):
        '''
        default DoF name
        '''
        return self.extra_DoF_name()

    def check_extra_update(self, mode):
        '''
        mode = 'B' or 'M'
        'M' return True, if M needs to be updated
        'B' return True, if B needs to be updated
        default, return true for B not M
        '''
        if self._update_flag:
            if mode == 'B':
                return True
        return False

    def has_bf_contribution(self, kfes):
        return False

    def has_bf_contribution2(self, kfes, jmatrix):
        '''
        subclass has to overwrite this if extra DoF can couple
        with other FES.
        '''
        if not self.check_jmatrix(jmatrix):
            return False
        return self.has_bf_contribution(kfes)

    def has_lf_contribution(self, kfes):
        return False

    def has_lf_contribution2(self, kfes, jmatrix):
        '''
        subclass has to overwrite this if extra DoF can couple
        with other FES.
        '''
        if not self.check_jmatrix(jmatrix):
            return False
        return self.has_lf_contribution(kfes)

    def has_interpolation_contribution(self, kfes):
        return False

    def has_mixed_contribution(self):
        return False

    def has_mixed_contribution2(self, jmatrix):
        '''
        subclass has to overwrite this if extra DoF can couple
        with other FES.
        '''
        if not self.check_jmatrix(jmatrix):
            return False
        return self.has_mixed_contribution()

    def has_aux_op(self, phys1, kfes1, phys2, kfes2):
        return False

    def has_aux_op2(self, phys1, kfes1, phys2, kfes2, jmatrix):
        if not self.check_jmatrix(jmatrix):
            return False
        return self.has_aux_op(phys1, kfes1, phys2, kfes2)

    '''
    def set_matrix_weight(self, get_matrix_weight):
        # self._mat_weight = get_matrix_weight(self.timestep_config,
        #                                      self.timestep_weight)
        w = get_matrix_weight(self.timestep_config)
        if sum(w) > 0:
           self._mat_weight = w
    '''

    def set_matrix_weight(self, w):
        '''
        matrix weight = [y, dy/dt, dy/dt^2, Jac(y), Jac(dy/dt), Jac(dy/dt^2)
        '''
        from petram.engine import max_matrix_num
        ww = [0]*max_matrix_num

        for i, val in enumerate(w):
            ww[i] = val
        if self.isJacobian:
            ww[:(max_matrix_num//2)] = [False]*(max_matrix_num//2)
        else:
            ww[(max_matrix_num//2):] = [False]*(max_matrix_num//2)
        self._mat_weight = ww

    def get_matrix_weight(self):
        return self._mat_weight

    def check_jmatrix(self, jmatrix):
        if not hasattr(self, "_mat_weight"):
            # _mat_weight is "usually" set by solver before
            # if it is not set, we are not solving this physics
            return False

        return self._mat_weight[jmatrix] != 0

    def get_mixedbf_loc(self):
        '''
        r, c, and flag of MixedBilinearForm
        flag -1 :  use conjugate when building block matrix

        if r, c is int, it is offset in the physics module, a couple between
        variables in the same module can use this mode. Otherwise, r and c
        are the name of variables
        '''
        return []

    def add_bf_contribution(self, engine, a, real=True, **kwargs):
        raise NotImplementedError(
            "you must specify this method in subclass")

    def add_lf_contribution(self, engine, b, real=True, **kwargs):
        raise NotImplementedError(
            "you must specify this method in subclass")

    def add_extra_contribution(self, engine, kfes, **kwargs):
        '''
        return four elements
        M12, M21, M22, rhs
        '''
        raise NotImplementedError(
            "you must specify this method in subclass")

    def add_interplation_contribution(self, engine, **kwargs):
        '''
        P^t A P y = P^t f, x = Py
        return P, nonzero, zero diagonals.
        '''
        raise NotImplementedError(
            "you must specify this method in subclass")

    def apply_essential(self, engine, gf, **kwargs):
        raise NotImplementedError(
            "you must specify this method in subclass")

    def get_init_coeff(self, engine, **kwargs):
        return None

    def apply_init(self, engine, gf, **kwargs):
        import warnings
        warnings.warn("apply init is not implemented to " + str(self))

    def add_mix_contribution(self, engine, a, r, c, is_trans, real=True):
        '''
        return array of crossterms
        [[vertical block elements], [horizontal block elements]]
        array length must be the number of fespaces
        for the physics

        r, c : r,c of block matrix
        is_trans: indicate if transposed matrix is filled
        '''
        raise NotImplementedError(
            "you must specify this method in subclass")

    def add_mix_contribution2(self, engine, a, r, c,
                              is_trans, is_conj, real=True):
        return self.add_mix_contribution(engine, a, r, c, is_trans, real=real)

    def add_variables(self, solvar, n, solr, soli=None):
        '''
        add model variable so that a user can interept simulation
        results. It is also used in cross-physics interaction.

        '''
        pass

    def add_domain_variables(self, v, n, suffix, ind_vars):
        pass

    def add_bdr_variables(self, v, n, suffix, ind_vars):
        pass

    def panel1_param(self):
        return self.vt.panel_param(self)

    def get_panel1_value(self):
        return self.vt.get_panel_value(self)

    def preprocess_params(self, engine):
        self.vt.preprocess_params(self)
        self.vt3.preprocess_params(self)
        return

    def import_panel1_value(self, v):
        return self.vt.import_panel_value(self, v)

    def panel1_tip(self):
        return self.vt.panel_tip()

    def panel3_param(self):
        from petram.pi.widget_nl import NonlinearTermPanel
        l = self.vt3.panel_param(self)
        if self.hide_nl_panel or len(self.nlterms) == 0:
            return l
        setting = {'UI': NonlinearTermPanel, 'names': self.nlterms}
        l.append([None, None, 99, setting])
        return l

    def get_panel3_value(self):
        if self.hide_nl_panel or len(self.nlterms) == 0:
            return self.vt3.get_panel_value(self)
        else:
            return self.vt3.get_panel_value(self) + [self.nl_config]

    def import_panel3_value(self, v):
        if self.hide_nl_panel or len(self.nlterms) == 0:
            self.vt3.import_panel_value(self, v)
        else:
            self.vt3.import_panel_value(self, v[:-1])
            self.nl_config = v[-1]

    def panel3_tip(self):
        if self.hide_nl_panel or len(self.nlterms) == 0:
            return self.vt3.panel_tip()
        else:
            return self.vt3.panel_tip() + [None]

    def panel4_param(self):
        setting = {"text": ' '}
        if self.has_essential:
            ll = [['', False, 3, {"text": "Time dependent"}], ]

        else:
            ll = [['y(t)', True, 3, {"text": ""}],
                  ['dy/dt', False, 3, {"text": ""}],
                  ['d2y/dt2', False, 3, {"text": ""}],
                  ['Gradient', False, 3, {"text": ""}],
                  ['Varying (in time/for loop) Term.', False, 3, {"text": ""}], ]
        if self.allow_custom_intorder:
            ll.append(['Increase int. order', '0', 400, ''])
        return ll

    def panel4_tip(self):
        return None

    def import_panel4_value(self, value):
        if self.has_essential:
            self.isTimeDependent = value[0]

        else:
            self.timestep_config[0] = value[0]
            self.timestep_config[1] = value[1]
            self.timestep_config[2] = value[2]
            self.isJacobian = value[3]
            self.isTimeDependent = value[4]

        if self.allow_custom_intorder:
            self.add_intorder = int(value[-1])

    def get_panel4_value(self):
        if self.has_essential:
            ret = [self.isTimeDependent]
        else:
            ret = self.timestep_config[0:3] + \
                [self.isJacobian, self.isTimeDependent]
        if self.allow_custom_intorder:
            ret = ret + [self.add_intorder]
        return ret

    @property
    def geom_dim(self):
        root_phys = self.get_root_phys()
        return root_phys.geom_dim

    @property
    def dim(self):
        root_phys = self.get_root_phys()
        return root_phys.dim

    @property
    def integrator_realimag_mode(self):
        if hasattr(self, "_realimag_mode"):
            return self._realimag_mode
        else:
            return ""

    def set_integrator_realimag_mode(self, value):
        if value:
            self._realimag_mode = "real"
        else:
            self._realimag_mode = "imag"

    def process_complex_coefficient(self, coeff):
        if not hasattr(coeff[0], "get_real_coefficient"):
            return coeff

        if self.integrator_realimag_mode == "real":
            return sum(((coeff[0].get_real_coefficient(),), coeff[1:]), ())
        elif self.integrator_realimag_mode == "imag":
            return sum(((coeff[0].get_imag_coefficient(),), coeff[1:]), ())
        else:
            return coeff

    def add_integrator(self, engine, name, coeff, adder, integrator, idx=None, vt=None,
                       transpose=False, ir=None):
        if coeff is None:
            return
        if vt is None:
            vt = self.vt
        # if vt[name].ndim == 0:
        if not isinstance(coeff, tuple):
            coeff = (coeff, )

        coeff = self.process_complex_coefficient(coeff)

        if coeff[0] is None:
            return
        elif isinstance(coeff[0], mfem.Coefficient):
            coeff = self.restrict_coeff(coeff, engine, idx=idx)
        elif isinstance(coeff[0], mfem.VectorCoefficient):
            coeff = self.restrict_coeff(coeff, engine, vec=True, idx=idx)
        elif isinstance(coeff[0], mfem.MatrixCoefficient):
            coeff = self.restrict_coeff(coeff, engine, matrix=True, idx=idx)
        else:
            assert False, "Unknown coefficient type: " + str(type(coeff[0]))

        itg = integrator(*coeff)
        itg._linked_coeff = coeff  # make sure that coeff is not GCed.

        if transpose:
            itg2 = mfem.TransposeIntegrator(itg)
            itg2._link = itg
            if ir is not None:
                itg2.SetIntRule(ir)
            adder(itg2)
        else:
            if ir is not None:
                itg.SetIntRule(ir)
            adder(itg)

    def onItemSelChanged(self, evt):
        '''
        GUI response when model object is selected in
        the dlg_edit_model
        '''
        viewer = evt.GetEventObject().GetTopLevelParent().GetParent()
        viewer.set_view_mode('phys', self)

    #
    #  utilities for add_domain_variables
    #

    def do_add_scalar_expr(self, v, suffix, ind_vars, name, f_name,
                           add_diag=False):

        from petram.helper.variables import add_expression, add_constant
        from petram.helper.variables import NativeCoefficientGenBase

        kywds = {}
        if isinstance(self, Domain):
            kywds['domains'] = self._sel_index
            kywds['gdomain'] = self._global_ns
        elif isinstance(self, Bdry):
            kywds['bdrs'] = self._sel_index
            kywds['gbdr'] = self._global_ns

        if isinstance(f_name, NativeCoefficientGenBase):
            pass
        elif isinstance(f_name, str):
            add_expression(v, name, suffix, ind_vars, f_name,
                           [], **kywds)

        else:
            add_constant(v, name, suffix, f_name, **kywds)

        if add_diag:
            '''
            convension:
               scalar parameters should be s+name

               ex) in RF epsilonr is (3,3)
                  sepsilonr : scarlar version
                  epsilonr : matrix
            '''
            size = int(add_diag)
            add_expression(v, name[1:], suffix, ind_vars,
                           'diag([1]*' + str(size) + ')*' + name,
                           [name], **kywds)

    def do_add_matrix_expr(self, v, suffix, ind_vars, name, f_name):

        from petram.helper.variables import add_expression, add_constant
        from petram.helper.variables import NativeCoefficientGenBase

        kywds = {}
        if isinstance(self, Domain):
            kywds['domains'] = self._sel_index
            kywds['gdomain'] = self._global_ns
        elif isinstance(self, Bdry):
            kywds['bdrs'] = self._sel_index
            kywds['gbdr'] = self._global_ns

        if isinstance(f_name, NativeCoefficientGenBase):
            pass
        elif len(f_name) == 1:
            if not isinstance(f_name[0], str):
                expr = f_name[0].__repr__()
            else:
                expr = f_name[0]
            add_expression(v, name, suffix, ind_vars, expr,
                           [], **kywds)
        else:  # elemental format
            expr_txt = [x.__repr__() if not isinstance(x, str) else x
                        for x in f_name]
            a = '[' + ','.join(expr_txt[:3]) + ']'
            b = '[' + ','.join(expr_txt[3:6]) + ']'
            c = '[' + ','.join(expr_txt[6:]) + ']'
            expr = '[' + ','.join((a, b, c)) + ']'
            add_expression(v, name, suffix, ind_vars, expr,
                           [], **kywds)

    def do_add_matrixlike_component_expr_scalar(
            self, v, suffix, ind_vars, var, name):
        '''
        this routine add matrix-like access to scalar variable

           epsilonr1xx = epsilonr1
           epsilonr1yy = epsilonr1
           epsilonr1zz = epsilonr1
           the reset is defined as 0
        '''

        from petram.helper.variables import add_component_expression as addc_expression
        from itertools import product

        kywds = {}
        if isinstance(self, Domain):
            kywds['domains'] = self._sel_index
            kywds['gdomain'] = self._global_ns
        elif isinstance(self, Bdry):
            kywds['bdrs'] = self._sel_index
            kywds['gbdr'] = self._global_ns

        ll = len(var)
        for lll in product(var, var):
            componentname = lll[0] + lll[1]
            if lll[0] == lll[1]:
                addc_expression(v, name, suffix, ind_vars, name, [name], componentname,
                                **kywds)
            else:
                addc_expression(v, name, suffix, ind_vars, '0', [], componentname,
                                **kywds)

    def do_add_matrix_component_expr(self, v, suffix, ind_vars, var, name):
        '''
        this routine add matrix-like access to scalar variable

           epsilonr1xx = epsilonr1[0,0]
           epsilonr1yy = epsilonr1[1,1]

        var
        '''

        from petram.helper.variables import add_component_expression as addc_expression
        from itertools import product

        kywds = {}
        if isinstance(self, Domain):
            kywds['domains'] = self._sel_index
            kywds['gdomain'] = self._global_ns
        elif isinstance(self, Bdry):
            kywds['bdrs'] = self._sel_index
            kywds['gbdr'] = self._global_ns

        ll = len(var)
        for lll in product(var, var):
            componentname = lll[0] + lll[1]
            i1 = var.index(lll[0])
            i2 = var.index(lll[1])
            expr = name + '[' + str(i1) + ',' + str(i2) + ']'
            addc_expression(v, name, suffix, ind_vars, expr, [name], componentname,
                            **kywds)

    def get_coefficient_from_expression(
            self, c, cotype, use_dual=False, real=True, is_conj=False):
        from petram.phys.coefficient import SCoeff, VCoeff, DCoeff, MCoeff

        if self.get_root_phys().vdim > 1:
            dim = self.get_root_phys().vdim
        else:
            dim = self.get_root_phys().geom_dim

        if cotype == 'S':
            # for b in self.itg_choice():
            #   if b[0] == self.integrator: break
            # if not "S*2" in b[3]:
            if not use_dual:
                c_coeff = SCoeff(c, self.get_root_phys().ind_vars,
                                 self._local_ns, self._global_ns,
                                 real=real, conj=is_conj)
            else:  # so far this is only for an elastic integrator
                c_coeff = (SCoeff(c, self.get_root_phys().ind_vars,
                                  self._local_ns, self._global_ns,
                                  real=real, conj=is_conj, component=0),
                           SCoeff(c, self.get_root_phys().ind_vars,
                                  self._local_ns, self._global_ns,
                                  real=real, conj=is_conj, component=1))
        elif cotype == 'V':
            c_coeff = VCoeff(dim, c, self.get_root_phys().ind_vars,
                             self._local_ns, self._global_ns,
                             real=real, conj=is_conj)
        elif cotype == 'M':
            c_coeff = MCoeff(dim, c, self.get_root_phys().ind_vars,
                             self._local_ns, self._global_ns,
                             real=real, conj=is_conj)
        elif cotype == 'D':
            c_coeff = DCoeff(dim, c, self.get_root_phys().ind_vars,
                             self._local_ns, self._global_ns,
                             real=real, conj=is_conj)
        return c_coeff

    def compile_coeffs(self, *kargs):
        '''
        jit compile coefficient
        '''
        pass


data = [("order", VtableElement("order", type='int',
                                guilabel="order", no_func=True,
                                default=1, tip="element order",))]


class PhysModule(Phys):
    hide_ns_menu = False
    dim_fixed = True  # if ndim of physics is fixed
    can_rename = False
    vt_order = Vtable(data)

    def __setstate__(self, state):
        Phys.__setstate__(self, state)
        if self.sel_index == 'all':
            self.sel_index = ['all']

    def get_info_str(self):
        txt = self.dep_vars
        if NS_mixin.get_info_str(self) != "":
            txt.append(NS_mixin.get_info_str(self))
        return ",".join(txt)

    def preprocess_params(self, engine):
        self.vt_order.preprocess_params(self)

    @property
    def geom_dim(self):  # dim of geometry
        return len(self.ind_vars.split(','))

    @property
    def dim(self):  # dim of FESpace independent variable
        return self.ndim

    @dim.setter
    def dim(self, value):
        self.ndim = value

    @property
    # mesh_idx is dynamic value to point a mesh index used in solve.
    def emesh_idx(self):
        return self._emesh_idx

    @emesh_idx.setter
    def emesh_idx(self, value):
        self._emesh_idx = value

    def goem_signature(self):
        pass

    @abstractmethod
    def get_fec_type(self, idx):
        pass
        '''
        return FEC type
        H1
        H1v2 (vector dim)
        ND
        RT
        '''

    def fes_order(self, idx):
        self.vt_order.preprocess_params(self)
        return self.order

    def attribute_set(self, v):
        v = super(PhysModule, self).attribute_set(v)
        v["vdim"] = 1
        v["element"] = 'H1_FECollection'
        v["ndim"] = 2  # logical dimension (= ndim of mfem::Mesh)
        v["ind_vars"] = 'x, y'
        v["dep_vars_suffix"] = ''
        v["mesh_idx"] = 0
        v['sel_index'] = ['all']
        v['sel_index_txt'] = 'all'

        # subclass can use this flag to generate FESpace for time derivative
        # see WF_model
        v['generate_dt_fespace'] = False
        v = self.vt_order.attribute_set(v)
        return v

    def get_dependent_variables(self):
        from petram.helper.variables import append_suffix_to_expression
        cls = self.__class__
        suffix = self.dep_vars_suffix
        vecs = cls.der_vars_vec
        names = [append_suffix_to_expression(x, vecs, suffix)
                 for x in cls.der_vars_base]
        return names

    def onVarNameChanged(self, evt):
        evt.GetEventObject().TopLevelParent.OnRefreshTree()

    def get_var_suffix_var_name_panel(self):
        from petram.pi.widgets import TextCtrlCallBack
        a = ["dep. vars. suffix", self.dep_vars_suffix, 99, {'UI': TextCtrlCallBack,
                                                             'callback_on_enter': True,
                                                             'callback_method': self.onVarNameChanged}]
        b = ["dep. vars.", ','.join(self.dep_vars_base), 99, {'UI': TextCtrlCallBack,
                                                              'callback_on_enter': True,
                                                              'callback_method': self.onVarNameChanged}]

        return a, b

    def panel1_param(self):
        ll = self.vt_order.panel_param(self)
        return [["mesh num.", self.mesh_idx, 400, {}],
                ["element", self.element, 2, {}]] + ll

    def panel1_tip(self):
        ll = self.vt_order.panel_tip()
        return ["index of mesh", "element type"] + ll

    def get_panel1_value(self):
        ll = self.vt_order.get_panel_value(self)
        return [self.mesh_idx, self.element] + ll

    def import_panel1_value(self, v):
        self.mesh_idx = int(v[0])
        self.element = str(v[1])
        self.vt_order.import_panel_value(self, [v[2]])
        return v[3:]

    def panel2_param(self):
        import wx

        if self.geom_dim == 3:
            choice = ("Volume", "Surface", "Edge")
        elif self.geom_dim == 2:
            choice = ("Surface", "Edge")
        elif self.geom_dim == 1:
            choice = ("Edge", )

        from petram.model import validate_sel

        if self.dim_fixed:
            return [["index", 'all', 0, {'changing_event': True,
                                         'setfocus_event': True,
                                         'validator': validate_sel,
                                         'validator_param': self}, ]]

        else:
            p = ["Type", choice[0], 4,
                 {"style": wx.CB_READONLY, "choices": choice}]
            return [p, ["index", 'all', 0, {'changing_event': True,
                                            'setfocus_event': True,
                                            'validator': validate_sel,
                                            'validator_param': self}, ]]

    def get_panel2_value(self):
        choice = ["Point", "Edge", "Surface", "Volume", ]
        if self.dim_fixed:
            return (self.sel_index_txt, )
        else:
            return choice[self.dim], self.sel_index_txt

    def import_panel2_value(self, v):
        if self.dim_fixed:
            self.sel_index_txt = str(v[0])
        else:
            if str(v[0]) == "Volume":
                self.dim = 3
            elif str(v[0]) == "Surface":
                self.dim = 2
            elif str(v[0]) == "Edge":
                self.dim = 1
            else:
                self.dim = 1
            self.sel_index_txt = str(v[1])

        self.update_dom_selection()
        return True

    @property
    def dep_vars(self):
        raise NotImplementedError(
            "you must specify this method in subclass")

    @property
    def dep_vars_base(self, name):
        raise NotImplementedError(
            "you must specify this method in subclass")

    @property
    def dep_vars0(self):
        # solver uses this method to find dep_vars to construct linear system
        # used (for example) to omit d/dt term from FESspace variables
        raise NotImplementedError(
            "you must specify this method in subclass")

    def dep_var_index(self, name):
        return self.dep_vars.index(name)

    def check_new_fespace(fespaces, meshes):
        mesh = meshes[self.mesh_name]
        fespacs = fespeces[self]

    def get_fec(self):
        name = self.dep_vars
        return [(name[0], self.element)]

    def is_complex(self):
        return False

    _possible_constraints = None

    @classmethod
    def _set_possible_constraints(cls, name):
        '''
        utility to define a possible constraints dynamically expandable.
        '''
        from petram.helper.phys_module_util import get_phys_constraints
        constraints = get_phys_constraints(name)
        cls._possible_constraints = constraints

    def get_possible_domain(self):
        from petram.phys.wf.wf_constraints import WF_WeakDomainBilinConstraint, WF_WeakDomainLinConstraint
        return [WF_WeakDomainBilinConstraint, WF_WeakDomainLinConstraint]

    def get_possible_bdry(self):
        from petram.phys.wf.wf_constraints import WF_WeakBdryBilinConstraint, WF_WeakBdryLinConstraint
        return [WF_WeakBdryBilinConstraint, WF_WeakBdryLinConstraint]

    def get_possible_point(self):
        from petram.phys.wf.wf_constraints import (WF_WeakPointBilinConstraint,
                                                   WF_WeakPointLinConstraint)

        # (note) BilinearForm does not suppoert delta coefficent
        return [WF_WeakPointBilinConstraint, WF_WeakPointLinConstraint]

    def get_possible_child(self):
        from petram.phys.aux_variable import AUX_Variable
        from petram.phys.aux_operator import AUX_Operator
        return [AUX_Variable, AUX_Operator]

    def get_possible_pair(self):
        from petram.phys.projection import BdrProjection, DomainProjection
        return [DomainProjection, BdrProjection, ]

    def soldict_to_solvars(self, soldict, variables):
        keys = list(soldict)
        depvars = self.dep_vars
        suffix = self.dep_vars_suffix
        ind_vars = [x.strip() for x in self.ind_vars.split(',')]

        for k in keys:
            n = k.split('_')[0]
            if n in depvars:
                sol = soldict[k]
                solr = sol[0]
                soli = sol[1] if len(sol) > 1 else None
                self.add_variables(variables, n, solr, soli)

        # collect all definition (domain specific expressions) from children
        for mm in self.walk():
            if not mm.enabled:
                continue
            if mm is self:
                continue

            mm.add_domain_variables(variables, n, suffix, ind_vars)
            mm.add_bdr_variables(variables, n, suffix, ind_vars)


    def onItemSelChanged(self, evt):
        '''
        GUI response when model object is selected in
        the dlg_edit_model
        '''
        viewer = evt.GetEventObject().GetTopLevelParent().GetParent()
        viewer.set_view_mode('phys', self)

    def is_viewmode_grouphead(self):
        return True

    def get_mesh_ext_info(self, m):
        from petram.mesh.mesh_extension import MeshExtInfo

        info = MeshExtInfo(dim=self.dim, base=self.mesh_idx)

        if len(self.sel_index) == 0:
            return None
        if self.sel_index[0] != 'all':
            info.set_selection(self.sel_index)
        else:
            if self.dim == 3:
                info.set_selection(list(m.extended_connectivity['vol2surf']))
            elif self.dim == 2:
                info.set_selection(list(m.extended_connectivity['surf2line']))
            else:
                info.set_selection(list(m.extended_connectivity['vert2vert']))

        return info

    def update_dom_selection(self, all_sel=None):
        from petram.model import convert_sel_txt
        try:
            arr = convert_sel_txt(self.sel_index_txt, self._global_ns)
            self.sel_index = arr
        except BaseException:
            assert False, "failed to convert " + self.sel_index_txt

        if all_sel is None:
            # clinet GUI panel operation ends here
            return

        allv, alls, alle = all_sel
        if len(self.sel_index) != 0 and self.sel_index[0] == 'all':
            if self.dim == 3:
                self.sel_index = allv
            if self.dim == 2:
                self.sel_index = alls
            if self.dim == 1:
                self.sel_index = alle

    def collect_probes(self):
        probes = [mm.get_probe() for mm in self.walk() if mm.is_enabled()]
        probes = [x for x in probes if len(x) != 0]
        txt = ','.join(probes)
        return txt

    def get_dom_bdr_pnt_choice(self, mesh):

        from collections import defaultdict

        def get_internal_bdr(d, dom_choice):
            dd = []
            for k in d:
                if k in dom_choice:
                    dd.extend(d[k])
            seen = defaultdict(int)
            for k in dd:
                seen[k] += 1
            return [k for k in seen if seen[k] > 1]

        if self.dim == 3:
            kk = 'vol2surf'
        elif self.dim == 2:
            kk = 'surf2line'
        elif self.dim == 1:
            kk = 'line2vert'
        else:
            assert False, "not supported"
        d = mesh.extended_connectivity[kk]

        if d is None:
            return [], [], [], []

        dom_choice = list(d)
        bdr_choice = list(set(np.hstack([list(d[x]) for x in d])))
        pnt_choice = list(mesh.extended_connectivity["vert2vert"])

        if self.sel_index[0] != 'all':
            dom_choice = [int(x) for x in self.sel_index]
            bdrs = [d[int(x)] if int(x) in d else [] for x in self.sel_index]
            bdr_choice = list(np.unique(np.hstack(bdrs)).astype(int))
        internal_bdr = get_internal_bdr(d, dom_choice)

        from petram.mfem_config import use_parallel
        if use_parallel:
            from mfem.common.mpi_debug import nicePrint
            from petram.helper.mpi_recipes import allgather
            dom_choice = list(set(sum(allgather(dom_choice), [])))
            bdr_choice = list(set(sum(allgather(bdr_choice), [])))
            internal_bdr = list(set(sum(allgather(internal_bdr), [])))

        # return unique list
        return list(set(dom_choice)), list(
            set(bdr_choice)), list(set(pnt_choice)), list(set(internal_bdr)),
        # return dom_choice, bdr_choice
