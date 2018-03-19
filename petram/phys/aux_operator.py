import sys
import os
import numpy as np
import scipy.sparse
from collections import OrderedDict
from warnings import warn
import   ifigure.widgets.dialog as dialog            

import wx

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
dprint1, dprint2, dprint3 = petram.debug.init_dprints('AUXVariable')
from petram.helper.matrix_file import write_coo_matrix, write_vector

#groups = ['Domain', 'Boundary', 'Edge', 'Point', 'Pair']
groups = ['Domain', 'Boundary', 'Pair']

data = [("axu_oprt", VtableElement("aux_oprt", type='any',
                                    guilabel = "operator", default = "",
                                    tip = "oprator (horizontal)",)),]

class AUX_Operator(Phys):
    vt_oprt = Vtable(data)
        
    def attribute_set(self, v):
        v = super(AUX_Operator, self).attribute_set(v)
        v['paired_var'] = None #(phys_name, index)
        v['src_var'] = 0     #(index)
        vv = self.vt_oprt.attribute_set({})
        for key in vv:
            if hasattr(self, key): vv[key] = getattr(self, key)
            v[key] = vv[key]
        return v
    
    def panel1_param(self):
        mfem_physroot = self.get_root_phys().parent
        names, pnames, pindex = mfem_physroot.dependent_values()
        names = [n+" ("+p + ")" for n, p in zip(names, pnames)]

        dep_vars = self.get_root_phys().dep_vars

        
        ll1 = [["paired variable", names[0], 4,
                {"style":wx.CB_READONLY, "choices": names}],
               ["source variable", dep_vars[0], 4,
                {"style":wx.CB_READONLY, "choices": dep_vars}]]

        ll2 = self.vt_oprt.panel_param(self)
        return ll1+ ll2

    def import_panel1_value(self, v):
        print v
        mfem_physroot = self.get_root_phys().parent
        names, pnames, pindex = mfem_physroot.dependent_values()

        idx = names.index(str(v[0]).split("(")[0].strip())
        self.paired_var = (pnames[idx], pindex[idx])

        self.src_var = self.get_root_phys().dep_vars.index(str(v[1]))
        self.vt_oprt.import_panel_value(self, v[2:])

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
        print v1
        return v1
        
    def panel2_param(self):
        return [[None, "Auxiriary varialbe is global",  2,   {}],]
    
    def import_panel2_value(self, v):
        pass
    
    def get_panel2_value(self):
        return [None]
        
    def has_extra_DoF(self, kfes):
        return False
     
    def get_exter_NDoF(self):
        return 1
     
    def postprocess_extra(self, sol, flag, sol_extra):
        name = self.name()+'_' + str(self.port_idx)
        sol_extra[name] = sol.toarray()
