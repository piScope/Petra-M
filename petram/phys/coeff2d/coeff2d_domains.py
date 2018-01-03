'''

   Domain for Coeff2D

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
from petram.phys.phys_model  import MatrixPhysCoefficient, PhysCoefficient, Coefficient_Evaluator
from petram.phys.coeff2d.coeff2d_base import Coeff2D_Domain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('Coeff2D_Domain')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem
   
from petram.phys.vtable import VtableElement, Vtable   


data =  (('c', VtableElement('c', type='float',
                                     guilabel = 'diffusion',
                                     suffix =[('x', 'y'), ('x', 'y')],
                                     default = np.eye(2, 2),
                                     tip = "diffusion term: div(-c grad u)" )),)
class CCoeff(MatrixPhysCoefficient):
    def __init__(self, *args, **kwargs):
        super(CCoeff, self).__init__(*args, **kwargs)
    def EvalValue(self, x):
        val = super(CCoeff, self).EvalValue(x)
        return val

class FCoeff(PhysCoefficient):
    def __init__(self, *args, **kwargs):
        super(FCoeff, self).__init__(*args, **kwargs)
    def EvalValue(self, x):
        val = super(FCoeff, self).EvalValue(x)
        return val
     
class Coeff2D_Diffusion(Coeff2D_Domain):
    vt  = Vtable(data)   
    def has_bf_contribution(self, kfes):
        return True

    def add_bf_contribution(self, engine, a, real = True, kfes=0):      
        c = self.vt.make_value_or_expression(self)    
        if real:       
            dprint1("Add contribution(real)" + str(self._sel_index))
        else:
            dprint1("Add contribution(imag)" + str(self._sel_index))

        c_coeff = CCoeff(2,  c[0],  self.get_root_phys().ind_vars,
                              self._local_ns, self._global_ns,
                              real = real)
        self.add_integrator(engine, 'c', c_coeff,
                            a.AddDomainIntegrator,
                            mfem.DiffusionIntegrator)
            
    def add_domain_variables(self, v, n, suffix, ind_vars, solr, soli = None):
        from petram.helper.variables import add_expression, add_constant
        pass
     
data =  (('f', VtableElement('f', type='float',
                                  guilabel = 'source',
                                  default = 0.0,
                                  tip = "source term: f" )),)

class Coeff2D_Source(Coeff2D_Domain):
    vt  = Vtable(data)   
    def has_lf_contribution(self, kfes):
        return True

    def add_lf_contribution(self, engine, b, real = True, kfes=0):      
        f = self.vt.make_value_or_expression(self)    
        if real:       
            dprint1("Add contribution(real)" + str(self._sel_index))
        else:
            dprint1("Add contribution(imag)" + str(self._sel_index))

        f_coeff = FCoeff(f,  self.get_root_phys().ind_vars,
                         self._local_ns, self._global_ns,
                         real = real)
        self.add_integrator(engine, 'f', f_coeff,
                            b.AddDomainIntegrator,
                            mfem.DomainLFIntegrator)
            
    def add_domain_variables(self, v, n, suffix, ind_vars, solr, soli = None):
        from petram.helper.variables import add_expression, add_constant
        pass
     


