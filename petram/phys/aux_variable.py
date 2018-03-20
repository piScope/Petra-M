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

data0 = [("oprt_diag", VtableElement("oprt_diag", type='array',
                                      guilabel = "diag", default = "",
                                      tip = "oprator (diag)",)),
         ("rhs_vec", VtableElement("rhs_vec", type='array',
                                   guilabel = "rhs", default = "",
                                   tip = "rhs vector",))]

class AUX_Variable(Phys):
    vt_diag_rhs = Vtable(data0)
    has_3rd_panel = True
    _has_4th_panel = True            
    def attribute_set(self, v):
        v = super(AUX_Variable, self).attribute_set(v)
        v["variable_name"] = ""
        v["aux_connection"] = OrderedDict({0: None})
        v["jmatrix_config"] = None
        v = self.vt_diag_rhs.attribute_set(v)
        
        if not hasattr(self, '_vt_array'): self._vt_array = []
        for vt in self._vt_array:
            vv = vt.attribute_set({})
            for key in vv:
                if hasattr(self, key): vv[key] = getattr(self, key)
                v[key] = vv[key]
        return v

    def extra_DoF_name(self):
        return self.variable_name
     
    def panel1_param(self):
        from wx import BU_EXACTFIT
        b1 = {"label": "+", "func": self.onAddConnection,
              "noexpand": True, "style": BU_EXACTFIT}#, "sendevent":True}
        b2 = {"label": "-", "func": self.onRmConnection,
              "noexpand": True, "style": BU_EXACTFIT}#, "sendevent":True}
        
        ll = [["name",   self.variable_name, 0, {}],]

        ll.extend(self.vt_diag_rhs.panel_param(self))
        ll.append([None, None, 241, {'buttons':[b1,b2],
                                     'alignright':True,
                                     'noexpand': True}])

        mfem_physroot = self.get_root_phys().parent
        names, pnames, pindex = mfem_physroot.dependent_values()
        names = [n+" ("+p + ")" for n, p in zip(names, pnames)]
        
        if not hasattr(self, '_vt_array'): self._vt_array = []        
        for key in self.aux_connection:
            if len(self._vt_array) > key: continue
            sidx = str(key)
            data = [("oprt1_"+sidx, VtableElement("oprt1_"+sidx, type='array',
                                           guilabel = "operator1", default = "",
                                           tip = "oprator (horizontal)",)),
                    ("oprt2_"+sidx, VtableElement("oprt2_"+sidx, type='array',
                                           guilabel = "operator2", default = "",
                                           tip = "oprator (vertical)",)),]
            vt = Vtable(data)
            self._vt_array.append(vt)
                    
        self.update_attribute_set()
        for j, key in enumerate(self.aux_connection):
            ll1 = [["paired variable", "S", 4,
                {"style":wx.CB_READONLY, "choices": names}]]
            ll2 = self._vt_array[j].panel_param(self)
            ll.extend(ll1+ ll2)

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
            #if len(self._vt_array) >= i: continue
            self._vt_array[i].import_panel_value(self,
                                                 v[(i_st+1):(i_st+3)])
            print  v[i_st+1:i_st+3]
            i_st = i_st + 3

    def get_panel1_value(self):
        def get_label(pair):
            if pair is None:
                n = self.get_root_phys().dep_vars[0]
                p = self.get_root_phys().name()
            else:
                mfem_physroot = self.get_root_phys().parent
                var_s = mfem_physroot[pair[0]].dep_vars
                n  = var_s[pair[1]]
                p  = pair[0]
            var = n + " ("+p + ")"
            return var
       
        v = [self.variable_name,]
        v.extend(self.vt_diag_rhs.get_panel_value(self))
        v.append(None)
        for i, key in enumerate(self.aux_connection):
            v.append(get_label(self.aux_connection[key]))
            v.extend(self._vt_array[i].get_panel_value(self))

        return v
    
    def panel2_param(self):
        return [[None, "Auxiriary varialbe is global",  2,   {}],]
    
    def import_panel2_value(self, v):
        pass
    
    def get_panel2_value(self):
        return [None]

    def panel3_param(self):
        return [[None, "Auxiriary varialbe is linear/no init.",  2,   {}],]
    
    def import_panel3_value(self, v):
        pass
    
    def get_panel3_value(self):
        return [None]
     
    def onAddConnection(self, evt):
        mfem_physroot = self.get_root_phys().parent
        names, pnames, pindex = mfem_physroot.dependent_values()
        names = [n+" ("+p + ")" for n, p in zip(names, pnames)]
        
        keys = self.aux_connection.keys()
        self.aux_connection[max(keys)+1] = (pnames[0], 0)
        evt.GetEventObject().TopLevelParent.OnItemSelChanged()
    
    def onRmConnection(self, evt):
        if len(self._vt_array) < 2: return
        keys = self.aux_connection.keys()
        del self.aux_connection[keys[-1]]
        self._vt_array = self._vt_array[:-1]
        evt.GetEventObject().TopLevelParent.OnItemSelChanged()
    
    def has_extra_DoF2(self, phys, kfes, jmatrix):
        if self.jmatrix_config is None:
            if jmatrix != 0: return False

        flag = False            
        for key in self.aux_connection:
            phys_name, kkfes = self.aux_connection[key]
            if (phys.name() == phys_name) and (kfes == kkfes): flag = True
            if self.jmatrix_config is not None:
               pass

            if flag: return True
        return False
     
     
    def postprocess_extra(self, sol, flag, sol_extra):
        name = self.variable_name
        sol_extra[name] = sol.toarray()

    def add_extra_contribution(self, engine, **kwargs):
        pass
