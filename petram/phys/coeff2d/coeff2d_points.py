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
    def has_lf_contribution(self, kfes):
        print 'point source'
        return True

    def add_lf_contribution(self, engine, b, real = True, kfes=0):
        print 'point source'       
        x, y, s = self.vt.make_value_or_expression(self)    
        if real:       
            dprint1("Add contribution(real)" + str(self._sel_index))
        else:
            dprint1("Add contribution(imag)" + str(self._sel_index))
        print '!!!!!', x, y, s
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


data =  (('x_delta', VtableElement('x_delta', type='float',
                                   guilabel = 'x',
                                   default = 0.0,
                                   tip = "x")),
         ('y_delta', VtableElement('y_delta', type='float',
                                   guilabel = 'y',
                                   default = 0.0,
                                   tip = "y")),
         ('value', VtableElement('value', type='float',
                                   guilabel = 'scale',
                                   default = 1.0,
                                   tip = "value at (x, y)")),)
     
     
class Coeff2D_PointValue(Coeff2D_Point):
    pass

