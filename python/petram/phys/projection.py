'''

   Weakform : interface to use MFEM integrator

'''
import os
import numpy as np

from petram.phys.phys_model import Phys
from petram.phys.phys_model  import PhysCoefficient
from petram.phys.phys_model  import VectorPhysCoefficient
from petram.phys.phys_model  import MatrixPhysCoefficient
from petram.model import Domain, Bdry, Edge, Point, Pair

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('WeakForm')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem
   
from petram.phys.vtable import VtableElement, Vtable   


data =  (('prj_name', VtableElement('prj_name', type='string',
                                     guilabel = 'name',
                                     default = '',
                                     tip = "name of projected value")),
         ('src_trans', VtableElement('stc_trans', type='string',
                                     guilabel = 'transform(src)',
                                     default = '',
                                     tip = "src coordinate transformation")),
         ('dst_trans', VtableElement('dst_trans', type='string',
                                     guilabel = 'transform(dst)',
                                     default = '',
                                     tip = "src coordinate transformation")),
         ('prj_factor', VtableElement('prj_factor', type='any',
                                      guilabel = 'factor',
                                      default = '',
                                      tip = "src coordinate transformation")),
         ('mapping_tol', VtableElement('mapping_tol', type='any',
                                       guilabel = 'mapping tolerance',
                                       default = '',
                                       tip = "tolerance of DoF mapping")))

class DoFProjection(Pair, Phys):
    vt  = Vtable(data)
    def __init__(self, **kwargs):    
        super(DoFProjection, self).__init__(**kwargs)
        Phys.__init__(self)
    
    def attribute_set(self, v):
        v = super(DoFProjection, self).attribute_set(v)
        v['prj_fes_idx'] = 0
        return v
    
    def panel1_param(self):
        import wx
        names = self.get_root_phys().dep_vars
        ll = [["FESpace", "S", 4,
               {"style":wx.CB_READONLY, "choices": names}]]
        param = self.vt.panel_param(self)
        ll.extend(param)
        return ll

    def import_panel1_value(self, v):
        idx = self.get_root_phys().dep_vars.index(v[0])
        self.prj_fes_idx = idx
        return self.vt.import_panel_value(self, v[1:])
    
    def get_panel1_value(self):
        val = [self.get_root_phys().dep_vars[self.prj_fes_idx]]
        val.extend(self.vt.get_panel_value(self))
        return val
    
    def panel1_tip(self):
        pass

    def assemble_matrix(self):
        raise NotImplementedError("subclass must implement this")
    
class BdrProjection(DoFProjection, Bdry):
    pass
class DomainProjection(DoFProjection, Domain):
    pass

