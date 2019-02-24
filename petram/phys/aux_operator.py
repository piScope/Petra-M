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

#these are only for debuging
from mfem.common.parcsr_extra import ToScipyCoo
from mfem.common.mpi_debug import nicePrint

from petram.phys.phys_model import Phys
from petram.model import Domain, Bdry, ModelDict
from petram.phys.vtable import VtableElement, Vtable


import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('AUX_Operator')
from petram.helper.matrix_file import write_coo_matrix, write_vector

#groups = ['Domain', 'Boundary', 'Edge', 'Point', 'Pair']
groups = ['Domain', 'Boundary', 'Pair']

data = [("axu_oprt", VtableElement("aux_oprt", type='any',
                                    guilabel = "operator", default = "",
                                    tip = "oprator (horizontal)",)),]

class AUX_Operator(Phys):
    vt_oprt = Vtable(data)
    has_3rd_panel = True
    _has_4th_panel = True            
        
    def attribute_set(self, v):
        v = super(AUX_Operator, self).attribute_set(v)
        v['paired_var'] = None #(phys_name, index)
        v['src_var'] = 0     #(index)
        v['use_symmetric'] = False
        v['use_conj'] = False
        v = self.vt_oprt.attribute_set(v)
        #vv = self.vt_oprt.attribute_set({})
        #for key in vv:
        #    if hasattr(self, key): vv[key] = getattr(self, key)
        #    v[key] = vv[key]
        return v
    def save_attribute_set(self, skip_def_check):
        attrs = super(AUX_Operator, self).save_attribute_set(skip_def_check)
        vv = self.vt_oprt.attribute_set({})
        for key in vv:
           if not key in attrs: attrs.append(key)
        return attrs
        
    def panel1_param(self):
        import wx
      
        mfem_physroot = self.get_root_phys().parent
        names, pnames, pindex = mfem_physroot.dependent_values()
        names = [n+" ("+p + ")" for n, p in zip(names, pnames)]

        dep_vars = self.get_root_phys().dep_vars

        
        ll1 = [["trial space (Cols)", names[0], 4,
                {"style":wx.CB_READONLY, "choices": names}],
               ["test space (Rows)", dep_vars[0], 4,
                {"style":wx.CB_READONLY, "choices": dep_vars}]]

        ll2 = self.vt_oprt.panel_param(self)
        ll3 = [["make symmetric",  self.use_symmetric,   3, {"text":""}],  
               ["use  conjugate",  self.use_conj,   3, {"text":""}],  ]
        
        return ll1+ ll2 + ll3

    def import_panel1_value(self, v):
        mfem_physroot = self.get_root_phys().parent
        names, pnames, pindex = mfem_physroot.dependent_values()

        idx = names.index(str(v[0]).split("(")[0].strip())
        self.paired_var = (pnames[idx], pindex[idx])

        self.src_var = self.get_root_phys().dep_vars.index(str(v[1]))
        self.vt_oprt.import_panel_value(self, v[2:-2])
        self.use_symmetric = v[-2]
        self.use_conj = v[-1]        

    def get_panel1_value(self):
        if self.paired_var is None:
            n = self.get_root_phys().dep_vars[0]
            p = self.get_root_phys().name()
        else:
            mfem_physroot = self.get_root_phys().parent
            var_s = mfem_physroot[self.paired_var[0]].dep_vars
            n  = var_s[self.paired_var[1]]
            p  = self.paired_var[0]

        var = n + " ("+p + ")"

        svar = self.get_root_phys().dep_vars[self.src_var]
        
        v1 = [var, svar]
        v1.extend(self.vt_oprt.get_panel_value(self))
        v3 = [self.use_symmetric, self.use_conj]        
        return v1 + v3
        
    def panel2_param(self):
        return [[None, "Auxiriary varialbe is global",  2,   {}],]
    
    def import_panel2_value(self, v):
        pass
    
    def get_panel2_value(self):
        return [None]
        
    def has_extra_DoF(self, kfes):
        return False
     
    def get_exter_NDoF(self):
        return 0
     
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
            self.use_symmetric):
            return True           

        return False
    
    def get_aux_op(self, engine, phys1, kfes1, phys2, kfes2,
                   trial_ess_tdof=None,
                   test_ess_tdof=None):

        mfem_physroot = self.get_root_phys().parent
        
        var_s = mfem_physroot[self.paired_var[0]].dep_vars
        trialname = var_s[self.paired_var[1]]
        testname = self.get_root_phys().dep_vars[self.src_var]

        oprt = self.vt_oprt.make_value_or_expression(self)
        oprt = oprt[0]
        
        from petram.helper.expression import Expression

        fes1 = engine.fespaces[trialname]
        fes2 = engine.fespaces[testname]        
        ind_vars = self.get_root_phys().ind_vars
        is_complex = self.get_root_phys().is_complex()

        diag_size = -1
        if oprt is not None:
           dprint1(self.name() + " Assembling Operator: ", oprt)
           assert isinstance(oprt, str), "operator1 must be an expression"               
           expr = Expression(oprt, engine=engine, trial=fes1, test=fes2,
                             trial_ess_tdof = trial_ess_tdof,
                             test_ess_tdof = test_ess_tdof,
                             ind_vars = ind_vars,
                             is_complex = is_complex)           
           op = expr.assemble(g=self._global_ns)


        trialname2 = phys2.dep_vars[kfes2]
        
        if testname == trialname2 and self.use_symmetric:
           op = op.transpose()
           if self.use_conj:
              op = op.conj()

        return op
