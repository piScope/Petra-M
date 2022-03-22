from __future__ import print_function
'''

   Points for Coeff2D

    m u'' + d u' + div( -c grad u - alpha u + gamma) 
              + beta (grad u) + a u - f = 0

  On domain boundary
     n ( c grad u + alpha u - gamma) + q u = g - h^t mu
       or 
     u = u0  

    m, d, a, f, g and h: scalar
    alpha, beta and gamma : vector
    c  : matrix (dim (space) ^2)

'''
import numpy as np

from petram.phys.phys_model  import PhysCoefficient, VectorPhysCoefficient
from petram.phys.phys_model  import MatrixPhysCoefficient, Coefficient_Evaluator
from petram.phys.coeff2d.coeff2d_base import Coeff2D_Point

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('Coeff2D_Domain')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem
   
from petram.phys.vtable import VtableElement, Vtable   


data =  (('x_delta', VtableElement('x_delta', type='array',
                                   guilabel = 'x',
                                   default = 0.0,
                                   tip = "x")),
         ('y_delta', VtableElement('y_delta', type='array',
                                   guilabel = 'y',
                                   default = 0.0,
                                   tip = "y")),
         ('s_delta', VtableElement('s_delta', type='array',
                                   guilabel = 'scale',
                                   default = 1.0,
                                   tip = "scale")),)

class Coeff2D_PointSource(Coeff2D_Point):
    vt  = Vtable(data)
    _sel_index = [-1]
    def panel2_param(self):
        return []
    def import_panel2_value(self, v):
        self.sel_index = 'all'
        
    def verify_setting(self):
        x, y, s = self.vt.make_value_or_expression(self)    
        if len(x) != len(y):
           return False, "Invalid x, y, s", "number of x and y must be the same"
        if len(x) != len(s):
           return False, "Invalid x, y, s", "number of x and s must be the same"           
        return True, "", ""
     
    def has_lf_contribution(self, kfes):
        print('point source')
        return True

    def add_lf_contribution(self, engine, b, real = True, kfes=0):
        x, y, s = self.vt.make_value_or_expression(self)    
        print('!!!!!', x, y, s)
        if len(x) != len(y):
           assert False, "number of x and y must be the same"
        if len(x) != len(s):
           assert False, "number of x and s must be the same"
        for x0, y0, s0 in zip(x, y, s):
           d = mfem.DeltaCoefficient(x0, y0, s0)
           self.add_integrator(engine, 'delta', d,
                               b.AddDomainIntegrator,
                               mfem.DomainLFIntegrator)
           
    def add_domain_variables(self, v, n, suffix, ind_vars, solr, soli = None):
        from petram.helper.variables import add_expression, add_constant
        pass

data =  (('x_delta', VtableElement('x_delta', type='array',
                                   guilabel = 'x',
                                   default = 0.0,
                                   tip = "x")),
         ('y_delta', VtableElement('y_delta', type='array',
                                   guilabel = 'y',
                                   default = 0.0,
                                   tip = "y")),
         ('value', VtableElement('value', type='array',
                                   guilabel = 'value',
                                   default = 1.0,
                                   tip = "value at (x, y)")),)
     
     
class Coeff2D_PointValue(Coeff2D_Point):
    vt  = Vtable(data)
    _sel_index = [-1]
    def panel2_param(self):
        return []
    def import_panel2_value(self, v):
        self.sel_index = 'all'
    
    def verify_setting(self):
        x, y, s = self.vt.make_value_or_expression(self)
        if len(x) != len(y):
           return False, "Invalid x, y, s", "number of x and y must be the same"
        if len(x) != len(s):
           return False, "Invalid x, y, s", "number of x and s must be the same"           
        return True, "", ""
     
    def has_extra_DoF(self, kfes):
        if kfes != 0: return False       
        return True
     
    def get_extra_NDoF(self):
        x, y, s = self.vt.make_value_or_expression(self)    
        return len(x)
     
    def postprocess_extra(self, sol, flag, sol_extra):
        name = self.name()
        sol_extra[name] = sol.toarray()
        
    def add_extra_contribution(self, engine, **kwargs):
        dprint1("Add Extra contribution" + str(self._sel_index))
        fes = engine.get_fes(self.get_root_phys(), 0)
        
        x, y, s = self.vt.make_value_or_expression(self)    
        
        from mfem.common.chypre import LF2PyVec, EmptySquarePyMat, HStackPyVec
        from mfem.common.chypre import Array2PyVec
        vecs = []
        for x0, y0, s0 in zip(x, y, s):        
           lf1 = engine.new_lf(fes)
           d = mfem.DeltaCoefficient(x0, y0, 1.0)
           itg = mfem.DomainLFIntegrator(d)
           lf1.AddDomainIntegrator(itg)
           lf1.Assemble()
           vecs.append(LF2PyVec(lf1))
        
        t3 =  EmptySquarePyMat(len(x))
        v1 = HStackPyVec(vecs)
        v2 = v1
        t4 = Array2PyVec(np.array(s))
        return (v1, v2, t3, t4, True)
