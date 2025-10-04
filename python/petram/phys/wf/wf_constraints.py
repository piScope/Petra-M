import traceback
import numpy as np

from petram.model import Domain, Bdry, Edge, Point, Pair
from petram.phys.phys_model import VectorPhysCoefficient, PhysCoefficient
from petram.phys.weakform import WeakLinIntegration, WeakBilinIntegration

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('WF_Constraints')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem

# define variable for this BC.
from petram.phys.vtable import VtableElement, Vtable

'''
class InitValue(PhysCoefficient):
   def EvalValue(self, x):
       v = super(InitValue, self).EvalValue(x)
       if self.real:  val = v.real
       else: val =  v.imag
       return val
'''
'''    
   
class InitValueV(VectorPhysCoefficient):
   def EvalValue(self, x):
       v = super(InitValueV, self).EvalValue(x)
       if self.real:  val = v.real
       else: val =  v.imag
       return val
'''
class WF_common(object):
    can_timedpendent = True
    @property
    def vt3(self):
        names = self.get_root_phys().dep_vars_base
        names2 = [n + '_init' for n in names]
        data = []
        if hasattr(self, '_vt3'):
            vt3 = self._vt3
            if vt3.keys() == names2: return vt3
            
        data = [(n+'_init', VtableElement(n+'_init',
                                          type='array', guilabel = n+'(init)',
                                          default ="0.0", tip = "initial value",
                                          chkbox = True)) for n in names]
        self._vt3 = Vtable(data)
        v = {}
        self._vt3.attribute_set(v)
        self.do_update_attribute_set(v)
        return self._vt3
     
    def get_init_coeff(self, engine, real=True, kfes=0):
        names = self.get_root_phys().dep_vars_base
        
        if not getattr(self, 'use_'+names[kfes]+'_init'): return
        
        f_name = self.vt3.make_value_or_expression(self)

        el = self.get_root_phys().element
        if el.startswith('H1') or el.startswith('L2'):
            ll = self.get_root_phys().vdim
        else:
            ll = self.get_root_phys().ndim

        kwargs = {}

        from petram.phys.coefficient import SCoeff, VCoeff
        if ll == 1:
            coeff = SCoeff(f_name[0],
                       self.get_root_phys().ind_vars,
                       self._local_ns, self._global_ns,
                       real = real)
            
        else:
            coeff = VCoeff(ll, f_name[0],  self.get_root_phys().ind_vars,
                             self._local_ns, self._global_ns,
                             real = real)
            kwargs['vec'] = True
        return self.restrict_coeff(coeff, engine, **kwargs)

    
class WF_WeakDomainBilinConstraint(WF_common, Domain, WeakBilinIntegration):
    has_3rd_panel = True
    _has_4th_panel = True                
    def __init__(self, **kwargs):
        super(WF_WeakDomainBilinConstraint, self).__init__(**kwargs)
        Domain.__init__(self, **kwargs)
        WeakBilinIntegration.__init__(self)
        
    def attribute_set(self, v):
        Domain.attribute_set(self, v)
        WeakBilinIntegration.attribute_set(self, v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v

    @classmethod
    def fancy_tree_name(self):
        return 'WeakContribution'

    @classmethod
    def fancy_menu_name(self):
        return 'WeakContribution'
     
class WF_WeakDomainLinConstraint(WF_common, Domain, WeakLinIntegration):
    has_3rd_panel = True
    def __init__(self, **kwargs):
        super(WF_WeakDomainLinConstraint, self).__init__(**kwargs)
        Domain.__init__(self, **kwargs)
        WeakLinIntegration.__init__(self)
        
    def attribute_set(self, v):
        Domain.attribute_set(self, v)
        WeakLinIntegration.attribute_set(self, v)         
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v

    @classmethod
    def fancy_tree_name(self):
        return 'WeakLinearContribution'
     
    @classmethod
    def fancy_menu_name(self):
        return 'WeakLinearContribution'
        
class WF_WeakBdryBilinConstraint(WF_common, Bdry, WeakBilinIntegration):
    has_3rd_panel = True        
    def __init__(self, **kwargs):
        super(WF_WeakBdryBilinConstraint, self).__init__(**kwargs)        
        Bdry.__init__(self, **kwargs)
        WeakBilinIntegration.__init__(self)
        
    def attribute_set(self, v):
        Bdry.attribute_set(self, v)
        WeakBilinIntegration.attribute_set(self, v)                
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v

    @classmethod
    def fancy_tree_name(self):
        return 'WeakContribution'

    @classmethod
    def fancy_menu_name(self):
        return 'WeakContribution'
     
class WF_WeakBdryLinConstraint(WF_common, Bdry, WeakLinIntegration):
    has_3rd_panel = True        
    def __init__(self, **kwargs):
        super(WF_WeakBdryLinConstraint, self).__init__(**kwargs)        
        Bdry.__init__(self, **kwargs)
        WeakLinIntegration.__init__(self)
        
    def attribute_set(self, v):
        Bdry.attribute_set(self, v)
        WeakLinIntegration.attribute_set(self, v)                
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v
     
    @classmethod
    def fancy_tree_name(self):
        return 'WeakLinearContribution'
     
    @classmethod
    def fancy_menu_name(self):
        return 'WeakLinearContribution'
     
class WF_WeakEdgeBilinConstraint(WF_common, Edge, WeakBilinIntegration):
    has_3rd_panel = True        
    def __init__(self, **kwargs):
        super(WF_WeakEdgeBilinConstraint, self).__init__(**kwargs)        
        Edge.__init__(self, **kwargs)
        WeakBilinIntegration.__init__(self)

    def attribute_set(self, v):
        Edge.attribute_set(self, v)
        WeakBilinIntegration.attribute_set(self, v)        
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v

class WF_WeakEdgeLinConstraint(WF_common, Edge, WeakLinIntegration):
    has_3rd_panel = True        
    def __init__(self, **kwargs):
        super(WF_WeakEdgeLinConstraint, self).__init__(**kwargs)        
        Edge.__init__(self, **kwargs)
        WeakLinIntegration.__init__(self)

    def attribute_set(self, v):
        Edge.attribute_set(self, v)
        WeakLinIntegration.attribute_set(self, v)        
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v
     
from petram.phys.weakform import WeakLinDeltaIntegration, WeakBilinDeltaIntegration

class WF_WeakPointBilinConstraint(WF_common, Point, WeakBilinDeltaIntegration):
    has_3rd_panel = True        
    def __init__(self, **kwargs):
        super(WF_WeakPointBilinConstraint, self).__init__(**kwargs)        
        Point.__init__(self, **kwargs)
        WeakBilinDeltaIntegration.__init__(self)

    def attribute_set(self, v):
        Point.attribute_set(self, v)
        WeakBilinDeltaIntegration.attribute_set(self, v)
        v['sel_readonly'] = True
        v['sel_index'] = ['all']
        v['sel_index_txt'] = '["all"]'        
        return v
     
class WF_WeakPointLinConstraint(WF_common, Point, WeakLinDeltaIntegration):
    has_3rd_panel = True        
    def __init__(self, **kwargs):
        super(WF_WeakPointLinConstraint, self).__init__(**kwargs)        
        Point.__init__(self, **kwargs)
        WeakLinDeltaIntegration.__init__(self)

    def attribute_set(self, v):
        Point.attribute_set(self, v)
        WeakLinDeltaIntegration.attribute_set(self, v)
        v['sel_readonly'] = True
        v['sel_index'] = ['all']
        v['sel_index_txt'] = '["all"]'                        
        return v

     
