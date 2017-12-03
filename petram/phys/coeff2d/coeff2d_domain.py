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
from petram.model import Domain
from petram.phys.phys_model  import Phys, MatrixPhysCoefficient
import numpy as np

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('Coeff2D_Domain')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem

name_suffix_m = ['_xx', '_xy', '_yx', '_yy']
name_suffix   = ['_x', '_y']

from petram.utils import set_array_attribute

class Coeff2D_Domain(Domain, Phys):
    def __init__(self, **kwargs):
        super(Coeff2D_Domain, self).__init__(**kwargs)
        Phys.__init__(self)
    

    def attribute_set(self, v):
        v = set_array_attribute(v, 'c',
                                name_suffix, ['1.0', '0.0', '0.0', '1.0'])
