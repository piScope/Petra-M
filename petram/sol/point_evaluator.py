'''

   evaluator for an aribtrary point

   1) using interpolation of nodal value


'''
import numpy as np
import parser
import weakref
from weakref import WeakKeyDictionary as WKD
from weakref import WeakValueDictionary as WVD


from petram.mfem_config import use_parallel
if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

from petram.sol.evaluator_base import EvaluatorBase

class PointEvaluator(EvaluatorBase):
    '''
    evaluate expression at spatial point...

    this is basically interpolator...
    '''
    def __init__(self, pts):
        self.pts = pts
    

