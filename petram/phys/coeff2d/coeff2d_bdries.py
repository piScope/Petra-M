'''

   Boundaries for Coeff2D

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
from petram.phys.coeff2d.coeff2d_base import Coeff2D_Bdry

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('Coeff2D_ZeroFlux')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem
   
from petram.phys.vtable import VtableElement, Vtable

data =  (('label1', VtableElement(None, 
                                  guilabel = 'ZeroFlux',
                                  default =   "n ( c grad u + alpha u - gamma) = 0",
                                  tip = "zero flux natural BC" )),)

class Coeff2D_ZeroFlux(Coeff2D_Bdry):
    is_essential = False
    nlterms = []
    vt  = Vtable(data)          

data =  (('u0', VtableElement('u0', guilabel = 'u0',
                               default =   "u = u0",
                               tip = "values at the boundary" )),)

class U0(PhysCoefficient):
   def __init__(self, *args, **kwargs):
       PhysCoefficient.__init__(self, *args, **kwargs)
   def EvalValue(self, x):
       v = super(rEt, self).EvalValue(x)
       if self.real:  return v.real
       else: return v.imag

class Coeff2D_Essential(Coeff2D_Bdry):
    has_essential = True
    nlterms = []
    vt  = Vtable(data)          
    def __init__(self, **kwargs):
        super(Coeff2D_Essential, self).__init__( **kwargs)

    def get_essential_idx(self, kfes):
        if kfes == 0:
            return self._sel_index
        else:
            return []
        
    def apply_essential(self, engine, gf, real = False, kfes = 0):
        if kfes > 0: return
        if real:       
            dprint1("Apply Ess.(real)" + str(self._sel_index))
        else:
            dprint1("Apply Ess.(imag)" + str(self._sel_index))
            
        u0 = self.vt.make_value_or_expression(self)              
        mesh = engine.get_mesh(mm = self)        
        ibdr = mesh.bdr_attributes.ToList()
        bdr_attr = [0]*mesh.bdr_attributes.Max()
        for idx in self._sel_index:
            bdr_attr[idx-1] = 1

        coeff1 = U0(u0, self.get_root_phys().ind_vars,
                        self._local_ns, self._global_ns,
                        real = real)
        gf.ProjectBdrCoefficient(coeff1,
                                     mfem.intArray(bdr_attr))
     
data =  (('u0', VtableElement(None, 
                                  guilabel = 'Zero',
                                  default =   "u = 0",
                                  tip = "zero BC" )),)
        
class Coeff2D_Zero(Coeff2D_Essential):
    vt  = Vtable(data)          
    def apply_essential(self, engine, gf, real = False, kfes = 0):
        if kfes > 0: return
        if real:       
            dprint1("Apply Ess.(real)" + str(self._sel_index))
        else:
            dprint1("Apply Ess.(imag)" + str(self._sel_index))
            
        mesh = engine.get_mesh(mm = self)        
        ibdr = mesh.bdr_attributes.ToList()
        bdr_attr = [0]*mesh.bdr_attributes.Max()
        for idx in self._sel_index:
            bdr_attr[idx-1] = 1

        coeff1 = mfem.ConstantCoefficient(0.0)
        gf.ProjectBdrCoefficient(coeff1,
                                     mfem.intArray(bdr_attr))


