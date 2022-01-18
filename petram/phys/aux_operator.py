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
dprint1, dprint2, dprint3 = petram.debug.init_dprints('AUX_Operator')

#groups = ['Domain', 'Boundary', 'Edge', 'Point', 'Pair']
groups = ['Domain', 'Boundary', 'Pair']

data = [("coeff_lambda", VtableElement("coeff_lambda", type='array',
                                       guilabel="lambda",
                                       default=1.0,
                                       tip="coefficient",)),
        ("axu_oprt", VtableElement("aux_oprt", type='any',
                                   guilabel="operator",
                                   default="",
                                   no_func=True,
                                   tip="oprator (horizontal)",)), ]


class AUX_Operator(Phys):
    vt_oprt = Vtable(data)
    has_3rd_panel = True
    _has_4th_panel = True

    def attribute_set(self, v):
        v = super(AUX_Operator, self).attribute_set(v)
        v['paired_var'] = None  # (phys_name, index)
        v['src_var'] = 0  # (index)
        v['use_symmetric'] = False
        v['use_anti_symmetric'] = False
        v['no_elimination'] = False
        v['use_conj'] = False
        v['coeff_type'] = 'Scalar'
        v = self.vt_oprt.attribute_set(v)
        #vv = self.vt_oprt.attribute_set({})
        # for key in vv:
        #    if hasattr(self, key): vv[key] = getattr(self, key)
        #    v[key] = vv[key]
        return v

    def save_attribute_set(self, skip_def_check):
        attrs = super(AUX_Operator, self).save_attribute_set(skip_def_check)
        vv = self.vt_oprt.attribute_set({})
        for key in vv:
            if not key in attrs:
                attrs.append(key)
        return attrs

    def panel1_param(self):
        import wx

        mfem_physroot = self.get_root_phys().parent
        names, pnames, pindex = mfem_physroot.dependent_values()
        names = [n+" ("+p + ")" for n, p in zip(names, pnames)]

        dep_vars = self.get_root_phys().dep_vars

        ll1 = [["trial space (Cols)", names[0], 4,
                {"style": wx.CB_READONLY, "choices": names}],
               ["test space (Rows)", dep_vars[0], 4,
                {"style": wx.CB_READONLY, "choices": dep_vars}],
               ["coeff. type", "Scalar", 4,
                {"style": wx.CB_READONLY,
                 "choices": ["Scalar", "Vector", "Diagonal", "Matrix"]}, ], ]
        ll2 = self.vt_oprt.panel_param(self)
        ll3 = [["symmetric",  self.use_symmetric,        3, {"text": ""}],
               ["anti-symmetric",  self.use_anti_symmetric,   3, {"text": ""}],
               ["use  conjugate",  self.use_conj,             3, {"text": ""}],
               ["use as ess. Dof",  self.no_elimination,   3, {"text": ""}], ]

        return ll1 + ll2 + ll3

    def import_panel1_value(self, v):
        mfem_physroot = self.get_root_phys().parent
        names, pnames, pindex = mfem_physroot.dependent_values()

        idx = names.index(str(v[0]).split("(")[0].strip())
        self.paired_var = (pnames[idx], pindex[idx])

        self.src_var = self.get_root_phys().dep_vars.index(str(v[1]))
        self.coeff_type = str(v[2])
        self.vt_oprt.import_panel_value(self, v[3:-4])
        self.use_symmetric = v[-4]
        self.use_anti_symmetric = v[-3]
        self.use_conj = v[-2]
        self.no_elimination = v[-1]

    def get_panel1_value(self):
        if self.paired_var is None:
            n = self.get_root_phys().dep_vars[0]
            p = self.get_root_phys().name()
        else:
            mfem_physroot = self.get_root_phys().parent
            if self.paired_var[0] in mfem_physroot:
                var_s = mfem_physroot[self.paired_var[0]].dep_vars
                n = var_s[self.paired_var[1]]
                p = self.paired_var[0]
            else:
                # if paird var does not exist in the model
                n = self.get_root_phys().dep_vars[0]
                p = self.get_root_phys().name()

        var = n + " ("+p + ")"

        svar = self.get_root_phys().dep_vars[self.src_var]

        v1 = [var, svar, self.coeff_type]
        val, expr = self.vt_oprt.get_panel_value(self)
        if expr.strip().startswith('='):
            expr = expr.strip()[1:]
        v1.extend([val, expr])
        v3 = [self.use_symmetric, self.use_anti_symmetric,
              self.use_conj, self.no_elimination]
        return v1 + v3

    def panel2_param(self):
        return [[None, "Auxiriary varialbe is global",  2,   {}], ]

    def import_panel2_value(self, v):
        pass

    def get_panel2_value(self):
        return [None]

    def has_extra_DoF(self, kfes):
        return False

    def get_extra_NDoF(self):
        return 0

    def verify_setting(self):
        policy = self.root()['General'].diagpolicy
        if policy == 'keep' and self.no_elimination:
            return (False, "DiagKeep with AUX Ebssential",
                    "Diag Policy is not one but used as essential")
        else:
            return True, "", ""

    def preprocess_params(self, engine):
        self.vt_oprt.preprocess_params(self)
        super(AUX_Operator, self).preprocess_params(engine)

    def has_aux_op(self, phys1, kfes, phys2, kfes2):
        # check
        trialname2 = phys2.dep_vars[kfes2]
        testname2 = phys1.dep_vars[kfes]

        mfem_physroot = self.get_root_phys().parent
        var_s = mfem_physroot[self.paired_var[0]].dep_vars
        trialname = var_s[self.paired_var[1]]
        testname = self.get_root_phys().dep_vars[self.src_var]

        if (trialname == trialname2 and
                testname == testname2):
            return True

        if (trialname == testname2 and
            testname == trialname2 and
                (self.use_symmetric or self.use_anti_symmetric)):
            return True

        return False

    def get_aux_op(self, engine, phys1, kfes1, phys2, kfes2,
                   trial_ess_tdof=None,
                   test_ess_tdof=None):

        mfem_physroot = self.get_root_phys().parent

        var_s = mfem_physroot[self.paired_var[0]].dep_vars
        trialname = var_s[self.paired_var[1]]
        testname = self.get_root_phys().dep_vars[self.src_var]

        oprt = self.aux_oprt_txt.strip()
        if oprt.startswith('='):  # old setting
            oprt = oprt[1:]
        coeff = self.vt_oprt['coeff_lambda'].make_value_or_expression(self)

        from petram.helper.expression import Expression

        fes1 = engine.fespaces[trialname]
        fes2 = engine.fespaces[testname]
        ind_vars = self.get_root_phys().ind_vars
        is_complex = self.get_root_phys().is_complex()

        diag_size = -1

        if oprt is not None:
            dprint1(self.name() + " Assembling Operator: ", oprt)
            assert isinstance(oprt, str), "operator1 must be an expression"

            cotype = self.coeff_type[0]
            c_coeff1 = self.get_coefficient_from_expression(coeff, cotype,
                                                            use_dual=False,
                                                            real=True,
                                                            is_conj=False)
            if is_complex:
                c_coeff2 = self.get_coefficient_from_expression(coeff, cotype,
                                                                use_dual=False,
                                                                real=False,
                                                                is_conj=False)
            else:
                c_coeff2 = None
            c_coeff = (c_coeff1, c_coeff2)

            expr = Expression(oprt, engine=engine, trial=fes1, test=fes2,
                              trial_ess_tdof=trial_ess_tdof,
                              test_ess_tdof=test_ess_tdof,
                              ind_vars=ind_vars,
                              is_complex=is_complex,
                              c_coeff=c_coeff)
            op = expr.assemble(g=self._global_ns)

        trialname2 = phys2.dep_vars[kfes2]

        if testname == trialname2 and (self.use_symmetric or self.use_anti_symmetric):
            op = op.transpose()
            if self.use_anti_symmetric:
                op *= -1.0
            if self.use_conj:
                op = op.conj()

        return op
