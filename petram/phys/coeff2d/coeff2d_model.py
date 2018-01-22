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
  Point:
     Coeff2D_PointEsse: point essential (default)

'''
model_basename = 'Coeff2D'

import numpy as np

from petram.model import Domain, Bdry, Point, Pair
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
    
class Coeff2D_DefPoint(Point, Phys):
    can_delete = False
    is_essential = False    
    def __init__(self, **kwargs):
        super(Coeff2D_DefPoint, self).__init__(**kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(Coeff2D_DefPoint, self).attribute_set(v)        
        v['sel_readonly'] = False
        v['sel_index'] = ['']
        return v
        
    def get_possible_point(self):
        return []                

class Coeff2D_DefPair(Pair, Phys):
    can_delete = False
    is_essential = False
    is_complex = False
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
    geom_dim = 2
    def __init__(self, **kwargs):
        super(Coeff2D, self).__init__()
        Phys.__init__(self)
        self['Domain'] = Coeff2D_DefDomain()
        self['Boundary'] = Coeff2D_DefBdry()
        self['Point'] = Coeff2D_DefPoint()        
        self['Pair'] = Coeff2D_DefPair()        
        
    @property
    def dep_vars(self):
        ret = self.dep_vars_base
        return [x + self.dep_vars_suffix for x in ret]
    
    @property 
    def dep_vars_base(self):
        return self.dep_vars_base_txt.split(',')

    @property 
    def der_vars(self):
        names = []
        for t in self.dep_vars:
            names.append(t+'x')
            names.append(t+'y')            
        return names
    
    def get_fec(self):
        v = self.dep_vars
        return [(v[0], self.element),]
    
    def attribute_set(self, v):
        v = super(Coeff2D, self).attribute_set(v)
        v["element"] = 'H1_FECollection'
        v["dim"] = 2
        v["ind_vars"] = 'x, y'
        v["dep_vars_suffix"] = ''
        v["dep_vars_base_txt"] = 'u'
        return v
    
    def panel1_param(self):
        panels = super(Coeff2D, self).panel1_param()
        panels.extend([
                ["indpendent vars.", self.ind_vars, 0, {}],
                ["dep. vars. suffix", self.dep_vars_suffix, 0, {}],
                ["dep. vars.", ','.join(self.dep_vars_base), 0, {}],
                ["derived vars.", ','.join(self.der_vars), 2, {}],
                ["predefined ns vars.", txt_predefined , 2, {}]])
        return panels
                      
    def get_panel1_value(self):
        names  =  ', '.join(self.dep_vars_base)
        names2  = ', '.join(self.der_vars)
        val =  super(Coeff2D, self).get_panel1_value()
                      
        val.extend([self.ind_vars, self.dep_vars_suffix,
                     names, names2, txt_predefined])
        return val
    
    def get_panel2_value(self):
        return 'all'
                      
    def import_panel1_value(self, v):
        v = super(Coeff2D, self).import_panel1_value(v)
        self.ind_vars =  str(v[0])
        self.dep_vars_suffix =  str(v[1])
        self.dep_vars_base_txt = ','.join([x.strip() for x in str(v[2]).split(',')])

    def import_panel2_value(self, v):
        self.sel_index = 'all'

    def get_possible_domain(self):
        from coeff2d_domains       import Coeff2D_Diffusion, Coeff2D_Source, Coeff2D_Convection, Coeff2D_Absorption
        return [Coeff2D_Diffusion,  Coeff2D_Convection, Coeff2D_Absorption, Coeff2D_Source]
    
    def get_possible_bdry(self):
        from coeff2d_bdries import Coeff2D_Essential, Coeff2D_Zero,Coeff2D_ZeroFlux
        return [Coeff2D_ZeroFlux, Coeff2D_Zero, Coeff2D_Essential]
    
    def get_possible_edge(self):
        return []                
    
    def get_possible_point(self):
        from coeff2d_points       import Coeff2D_PointSource, Coeff2D_PointValue
        return [Coeff2D_PointSource, Coeff2D_PointValue]
    
    def get_possible_pair(self):
        return []

    def add_variables(self, v, name, solr, soli = None):
        from petram.helper.variables import add_coordinates
        from petram.helper.variables import add_scalar
        from petram.helper.variables import add_components
        from petram.helper.variables import add_expression
        from petram.helper.variables import add_surf_normals
        from petram.helper.variables import add_constant      

        ind_vars = [x.strip() for x in self.ind_vars.split(',')]
        suffix = self.dep_vars_suffix

        #from petram.helper.variables import TestVariable
        #v['debug_test'] =  TestVariable()
        
        add_coordinates(v, ind_vars)        
        add_surf_normals(v, ind_vars)

        dep_vars = self.dep_vars
        for dep_var in dep_vars:
            if name.startswith(dep_var):
                add_scalar(v, dep_var, suffix, ind_vars, solr, soli)

        return v
    
