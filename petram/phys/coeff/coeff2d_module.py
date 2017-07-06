'''
COEFF2D : 2D coefficient form of PDE

  m u'' + d u' + div( -c grad u - alpha u + gamma) 
              + beta (grad u) + a u - f = 0

  On domain boundary
     n ( c grad u + alpha u - gamma) + q u = g - h^t mu
       or 
     u = u0  

    m, d, a, f, g and h: scalar
    alpha, beta and gamma : vector
    c  : matrix (dim (space) ^2)

    If all coefficients are independent from u, ux,
    the system is linear.

    BC
     Zero Flux : 
        n ( c grad u + alpha u - gamma) = 0
     Flux: 
        n ( c grad u + alpha u - gamma) = - g + q u
     Dirichlet Boundary Condition
        u = u0

  Weakform integrators:
    domain integral
       c     -->  c (grad u, grad v)     bi
       alpha -->  alpha * (u, grad v)    bi
       gamma -->  gamma * grad v         li
       beta  -->  beta * (grad u, v)     bi
       f     -->  (f, v)                 bi
    surface integral
       c     -->   n dot c (grad u, v)   bi
       alpha -->   n dot alpha  (u v)    bi
       gamma -->   n dot gamma   v       li
 
    surface integral can be replaced by (g - qu, v)
        
  Domain:   
     Coeff2D          : tensor dielectric

  Boundary:
     Coeff2D_Zero     : zero essential (default)
     Coeff2D_Esse     : general essential

'''
import numpy as np

from petram.model import Domain, Bdry, Pair
from petram.phys.phys_model import Phys, PhysModule

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('Coeff2D_Model')

txt_predefined = ''    

class Coeff2D_DefDomain(Domain, Phys):
    can_delete = False
    def __init__(self, **kwargs):
        super(Coeff2D_DefDomain, self).__init__(**kwargs)

    
    def get_panel1_value(self):
        return None

    def import_panel1_value(self, v):
        pass

    def get_possible_domain(self):
        return []
        
class Coeff2D_DefBdry(Bdry, Phys):
    can_delete = False
    is_essential = False    
    def __init__(self, **kwargs):
        super(Coeff2D_DefBdry, self).__init__(**kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(Coeff2D_DefBdry, self).attribute_set(v)        
        v['sel_readonly'] = False
        v['sel_index'] = ['remaining']
        return v
        
    def get_possible_bdry(self):
        return []                

class Coeff2D_DefPair(Pair, Phys):
    can_delete = False
    is_essential = False
    is_complex = True
    def __init__(self, **kwargs):
        super(Coeff2D_DefPair, self).__init__(**kwargs)
        Phys.__init__(self)
        
    def attribute_set(self, v):
        super(Coeff2D_DefPair, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v

    def get_possible_pair(self):
        return []

class Coeff2D(PhysModule):
    dep_var_base = ['u']
    der_var_base = ['ux', 'uy']
    def __init__(self, **kwargs):
        super(Coeff2D, self).__init__()
        Phys.__init__(self)
        self['Domain'] = Coeff2D_DefDomain()
        self['Boundary'] = Coeff2D_DefBdry()
        self['Pair'] = Coeff2D_DefPair()        
        
    def attribute_set(self, v):
        v = super(Coeff2D, self).attribute_set(v)
        v["element"] = 'H1_FECollection'
        v["dim"] = 2
        v["ind_vars"] = 'x, y'
        v["dep_vars_suffix"] = ''
        return v
    
    def panel1_param(self):
        return [["element",  self.element,  0, {}],
                ["order",  self.order,    400, {}],
                ["indpendent vars.", self.ind_vars, 0, {}],
                ["dep. vars. suffix", self.dep_vars_suffix, 0, {}],
                ["dep. vars.", ','.join(Coeff2D.dep_var_base), 2, {}],
                ["derived vars.", ','.join(Coeff2D.der_var_base), 2, {}],
                ["predefined ns vars.", txt_predefined , 2, {}]]     
