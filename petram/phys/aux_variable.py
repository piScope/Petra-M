from petram.helper.matrix_file import write_coo_matrix, write_vector
import sys
import os
import numpy as np
import scipy.sparse
from collections import OrderedDict
from warnings import warn

from petram.mfem_config import use_parallel
if use_parallel:
    from petram.helper.mpi_recipes import *
    import mfem.par as mfem
else:
    import mfem.ser as mfem
import mfem.common.chypre as chypre

# these are only for debuging
from mfem.common.parcsr_extra import ToScipyCoo
from mfem.common.mpi_debug import nicePrint

from petram.phys.phys_model import Phys
from petram.model import Domain, Bdry, ModelDict
from petram.phys.vtable import VtableElement, Vtable


import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('AUX_Variable')

#groups = ['Domain', 'Boundary', 'Edge', 'Point', 'Pair']
groups = ['Domain', 'Boundary', 'Pair']

data0 = [("oprt_diag", VtableElement("oprt_diag", type='array',
                                     guilabel="diag", default="0.0",
                                     tip="oprator (diag)",)),
         ("rhs_vec", VtableElement("rhs_vec", type='array',
                                   guilabel="rhs", default="0.0",
                                   tip="rhs vector",))]


class AUX_Variable(Phys):
    vt_diag_rhs = Vtable(data0)
    has_3rd_panel = True
    _has_4th_panel = True
    extra_diagnostic_print = True

    def attribute_set(self, v):
        v = super(AUX_Variable, self).attribute_set(v)
        v["variable_name"] = ""
        v["aux_connection"] = OrderedDict({})
        v["jmatrix_config"] = None
        v = self.vt_diag_rhs.attribute_set(v)

        if not hasattr(self, '_vt_array'):
            self._vt_array = []
        for vt in self.vt_array:
            v = vt.attribute_set(v)
            #vv = vt.attribute_set({})
            # for key in vv:
            #    if hasattr(self, key): vv[key] = getattr(self, key)
            #    v[key] = vv[key]
        return v

    def save_attribute_set(self, skip_def_check):
        attrs = super(AUX_Variable, self).save_attribute_set(skip_def_check)
        for vt in self.vt_array:
            vv = vt.attribute_set({})
            for key in vv:
                if not key in attrs:
                    attrs.append(key)
        return attrs

    def extra_DoF_name(self):
        return self.variable_name

    @property
    def vt_array(self):
        if not hasattr(self, 'aux_connection'):
            self._vt_array = []
            return self._vt_array

        if not hasattr(self, '_vt_array'):
            self._vt_array = []
        for key in self.aux_connection:
            if len(self._vt_array) > key:
                continue
            sidx = str(key)
            data = [("oprt1_"+sidx, VtableElement("oprt1_"+sidx, type='any',
                                                  guilabel="operator1", default="",
                                                  tip="oprator (horizontal)",)),
                    ("oprt2_"+sidx, VtableElement("oprt2_"+sidx, type='any',
                                                  guilabel="operator2", default="",
                                                  tip="oprator (vertical)",)), ]
            vt = Vtable(data)
            self._vt_array.append(vt)
        return self._vt_array

    def panel1_param(self):
        import wx
        from wx import BU_EXACTFIT

        b1 = {"label": "+", "func": self.onAddConnection,
              "noexpand": True, "style": BU_EXACTFIT}  # , "sendevent":True}
        b2 = {"label": "-", "func": self.onRmConnection,
              "noexpand": True, "style": BU_EXACTFIT}  # , "sendevent":True}

        ll = [["name"+" "*16,   self.variable_name, 0, {}], ]

        ll.extend(self.vt_diag_rhs.panel_param(self))
        ll.append([None, None, 241, {'buttons': [b1, b2],
                                     'alignright':True,
                                     'noexpand': True}])

        mfem_physroot = self.get_root_phys().parent
        names, pnames, pindex = mfem_physroot.dependent_values()
        names = [n+" ("+p + ")" for n, p in zip(names, pnames)]

        '''
        if not hasattr(self, '_vt_array'): self._vt_array = []        
        for key in self.aux_connection:
            if len(self._vt_array) > key: continue
            sidx = str(key)
            data = [("oprt1_"+sidx, VtableElement("oprt1_"+sidx, type='any',
                                           guilabel = "operator1", default = "",
                                           tip = "oprator (horizontal)",)),
                    ("oprt2_"+sidx, VtableElement("oprt2_"+sidx, type='any',
                                           guilabel = "operator2", default = "",
                                           tip = "oprator (vertical)",)),]
            vt = Vtable(data)
            self._vt_array.append(vt)
        '''
        self.update_attribute_set()
        for j, key in enumerate(self.aux_connection):
            ll1 = [["paired variable", "S", 4,
                    {"style": wx.CB_READONLY, "choices": names}]]
            ll2 = self._vt_array[j].panel_param(self)
            ll.extend(ll1 + ll2)

        return ll

    def import_panel1_value(self, v):
        mfem_physroot = self.get_root_phys().parent
        names, pnames, pindex = mfem_physroot.dependent_values()

        if len(str(v[0])) == 0:
            dprint1("Name of variable must be given")
        self.variable_name = str(v[0])
        self.vt_diag_rhs.import_panel_value(self, v[1:3])
        i_st = 4
        for i, key in enumerate(self.aux_connection):

            idx = names.index(str(v[i_st]).split("(")[0].strip())
            self.aux_connection[key] = (pnames[idx], pindex[idx])
            # if len(self._vt_array) >= i: continue
            self._vt_array[i].import_panel_value(self,
                                                 v[(i_st+1):(i_st+3)])
            i_st = i_st + 3

    def get_panel1_value(self):
        def get_label(pair):
            if pair is None:
                n = self.get_root_phys().dep_vars[0]
                p = self.get_root_phys().name()
            else:
                mfem_physroot = self.get_root_phys().parent
                var_s = mfem_physroot[pair[0]].dep_vars
                n = var_s[pair[1]]
                p = pair[0]
            var = n + " ("+p + ")"
            return var

        v = [self.variable_name, ]
        v.extend(self.vt_diag_rhs.get_panel_value(self))
        v.append(None)
        for i, key in enumerate(self.aux_connection):
            v.append(get_label(self.aux_connection[key]))
            v.extend(self._vt_array[i].get_panel_value(self))

        return v

    def panel2_param(self):
        return [[None, "Auxiriary varialbe is global",  2,   {}], ]

    def import_panel2_value(self, v):
        pass

    def get_panel2_value(self):
        return [None]

    def panel3_param(self):
        return [[None, "Auxiriary varialbe is linear/no init.",  2,   {}], ]

    def import_panel3_value(self, v):
        pass

    def get_panel3_value(self):
        return [None]

    def panel4_param(self):
        ll = super(AUX_Variable, self).panel4_param()
        ll.append(['Varying (in time/for loop) RHS', False, 3, {"text": ""}])
        return ll

    def panel4_tip(self):
        return None

    def import_panel4_value(self, value):
        super(AUX_Variable, self).import_panel4_value(value[:-1])
        self.isTimeDependent_RHS = value[-1]

    def get_panel4_value(self):
        value = super(AUX_Variable, self).get_panel4_value()
        value.append(self.isTimeDependent_RHS)
        return value

    def check_extra_update(self, mode):
        '''
        mode = 'B' or 'M'
        'M' return True, if M needs to be updated
        'B' return True, if B needs to be updated
        '''
        if self._update_flag:
            if mode == 'B':
                return self.isTimeDependent_RHS
            if mode == 'M':
                return self.isTimeDependent
        return False

    def onAddConnection(self, evt):
        mfem_physroot = self.get_root_phys().parent
        names, pnames, pindex = mfem_physroot.dependent_values()
        names = [n+" ("+p + ")" for n, p in zip(names, pnames)]

        keys = self.aux_connection.keys()
        if len(keys) == 0:
            self.aux_connection[0] = (pnames[0], 0)
        else:
            self.aux_connection[max(keys)+1] = (pnames[0], 0)
        evt.GetEventObject().TopLevelParent.OnItemSelChanged()

    def onRmConnection(self, evt):
        if len(self._vt_array) < 1:
            return
        keys = self.aux_connection.keys()
        del self.aux_connection[keys[-1]]
        self._vt_array = self._vt_array[:-1]
        evt.GetEventObject().TopLevelParent.OnItemSelChanged()

    def has_extra_DoF2(self, kfes, phys, jmatrix):
        if not self.timestep_config[jmatrix]:
            return False

        flag = False
        for key in self.aux_connection:
            phys_name, kkfes = self.aux_connection[key]
            if (phys.name() == phys_name) and (kfes == kkfes):
                flag = True
            if flag:
                return True
        return False

    def postprocess_extra(self, sol, flag, sol_extra):
        name = self.variable_name
        sol_extra[name] = sol.toarray()

    def preprocess_params(self, engine):
        self.vt_diag_rhs.preprocess_params(self)
        for vt in self._vt_array:
            vt.preprocess_params(self)
        super(AUX_Variable, self).preprocess_params(engine)

    def _add_vt_array(self, sidx):
        data = [("oprt1_"+sidx, VtableElement("oprt1_"+sidx, type='any',
                                              guilabel="operator1", default="",
                                              tip="oprator (horizontal)",)),
                ("oprt2_"+sidx, VtableElement("oprt2_"+sidx, type='any',
                                              guilabel="operator2", default="",
                                              tip="oprator (vertical)",)), ]
        vt = Vtable(data)
        self._vt_array.append(vt)
        vt.preprocess_params(self)

    def add_extra_contribution(self, engine, ess_tdof=None,
                               kfes=0, phys=None):
        dprint1("Add Extra contribution : " + self.name())
        range = self.get_root_phys().dep_vars[kfes]

        diag, rhs_vec = self.vt_diag_rhs.make_value_or_expression(self)

        if isinstance(rhs_vec, str):
            g = self._global_ns.copy()
            l = self._local_ns.copy()
            rhs_vec = np.atleast_1d(eval(rhs_vec, g, l)).flatten()
            dprint1("rhs after evaluation", rhs_vec)

        if isinstance(diag, str):
            g = self._global_ns.copy()
            l = self._local_ns.copy()
            diag = np.atleast_1d(eval(diag, g, l)).flatten()
            dprint1("diag after evaluation", diag)

        for i, key in enumerate(self.aux_connection):
            if len(self._vt_array) < i+1:
                self._add_vt_array(str(key))
            opr1, opr2 = self._vt_array[i].make_value_or_expression(self)

        t1 = None
        t2 = None
        t3 = None
        t4 = None

        from petram.helper.expression import Expression

        name = phys.dep_vars[kfes]
        fes = engine.fespaces[name]

        ind_vars = self.get_root_phys().ind_vars
        is_complex = self.get_root_phys().is_complex()

        diag_size = -1
        if opr1 is not None or opr2 is not None:
            if opr1 is not None:
                assert isinstance(opr1, str), "operator1 must be an expression"
                expr = Expression(opr1, engine=engine, trial=fes,
                                  ind_vars=ind_vars, is_complex=is_complex)
                t1 = expr.assemble(g=self._global_ns)
                diag_size = t1.shape[0]
            if opr2 is not None:
                assert isinstance(opr2, str), "operator2 must be an expression"
                expr = Expression(opr2, engine=engine, trial=fes, transpose=True,
                                  ind_vars=ind_vars, is_complex=is_complex)
                t2 = expr.assemble(g=self._global_ns)
                if diag_size > -1:
                    # print t1.shape, t2.shape
                    assert diag_size == t2.shape[1], "t1 and t2 shapes are inconsistent"
                diag_size = t2.shape[1]

        from mfem.common.chypre import IdentityPyMat, Array2PyVec
        if diag is not None:
            if not self.get_root_phys().is_complex():
                diag = diag.real
            t3 = IdentityPyMat(diag_size, diag=diag)

        # all node does do the same job, but Array2PyVec will use the data from
        # root node.

        rhs = np.atleast_1d(rhs_vec).astype(float)
        if diag_size != 1 and len(rhs) == 1:
            rhs = np.hstack([rhs[0]]*diag_size).flatten()

        if self.get_root_phys().is_complex():
            rhs = rhs.astype(complex, copy=False)
        if not self.get_root_phys().is_complex():
            if np.iscomplexobj(rhs):
                rhs = rhs.real

        t4 = Array2PyVec(rhs)

        '''
        Format of extra.
        [M,  t2]   [  ]
        [      ] = [  ]
        [t1, t3]   [t4]

        and it returns if Lagurangian will be saved.

        '''
        return (t2, t1, t3, t4, True)
