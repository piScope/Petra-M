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

class AUX_Variable(Phys):

    def attribute_set(self, v):
        v = super(AUX_Variable, self).attribute_set(v)
        v["variable_name"] = ""
        v["aux_connection"] = OrderedDict({0: None})
        if not hasattr(self, '_vt_array'): self._vt_array = []
        for vt in self._vt_array:
            vv = vt.attribute_set({})
            for key in vv:
                if hasattr(self, key): vv[key] = getattr(self, key)
                v[key] = vv[key]
        return v
    
    def panel1_param(self):
        from wx import BU_EXACTFIT
        b1 = {"label": "+", "func": self.onAddConnection,
              "noexpand": True, "style": BU_EXACTFIT}#, "sendevent":True}
        b2 = {"label": "-", "func": self.onRmConnection,
              "noexpand": True, "style": BU_EXACTFIT}#, "sendevent":True}
        
        ll = [["name",   self.variable_name, 0, {}],
              [None, None, 241, {'buttons':[b1,b2],
                                 'alignright':True,
                                 'noexpand': True},],]

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
                                           tip = "oprator (vertical)",)),
                    ("oprtd_"+sidx, VtableElement("oprtd_"+sidx, type='array',
                                              guilabel = "diag", default = "",
                                              tip = "oprator (diag)",)),
                    ("rhs_"+sidx, VtableElement("rhs_"+sidx, type='array',
                                              guilabel = "rhs", default = "",
                                              tip = "rhs",))]
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
        print v
        i_st = 2
        for i, key in enumerate(self.aux_connection):

            idx = names.index(str(v[i_st]).split("(")[0].strip())           
            self.aux_connection[key] = (pnames[idx], pindex[idx])
            #if len(self._vt_array) >= i: continue
            self._vt_array[i].import_panel_value(self,
                                                 v[(i_st+1):(i_st+5)])
            print  v[i_st+1:i_st+5]
            i_st = i_st + 5

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
       
        v = [self.variable_name, None]
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
        
    def onAddConnection(self, evt):
        mfem_physroot = self.get_root_phys().parent
        names, pnames, pindex = mfem_physroot.dependent_values()
        names = [n+" ("+p + ")" for n, p in zip(names, pnames)]
        
        keys = self.aux_connection.keys()
        self.aux_connection[max(keys)+1] = names[0]
        evt.GetEventObject().TopLevelParent.OnItemSelChanged()
    
    def onRmConnection(self, evt):
        if len(self._vt_array) < 2: return
        keys = self.aux_connection.keys()
        del self.aux_connection[keys[-1]]
        self._vt_array = self._vt_array[:-1]
        evt.GetEventObject().TopLevelParent.OnItemSelChanged()
    
    def has_extra_DoF(self, kfes):
        if self.mode == 'TE' and kfes == 1: return True
        elif kfes == 0:
            if self.mode == 'Ephi':
                 return True
        else:
            return False
     
    def get_exter_NDoF(self):
        return 1
     
    def postprocess_extra(self, sol, flag, sol_extra):
        name = self.name()+'_' + str(self.port_idx)
        sol_extra[name] = sol.toarray()
