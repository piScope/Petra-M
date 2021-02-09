'''
   IntegralEvaluator:
      a thing to evaluate integral on a boundary/domain
'''
import numpy as np
import parser
import weakref
import six

from weakref import WeakKeyDictionary as WKD
from weakref import WeakValueDictionary as WVD


from petram.mfem_config import use_parallel
if use_parallel:
    import mfem.par as mfem
    from mfem.par import GlobGeometryRefiner as GR    
else:
    import mfem.ser as mfem
    from mfem.ser import GlobGeometryRefiner as GR
    
from petram.sol.evaluator_agent import EvaluatorAgent
Geom = mfem.Geometry()

class IntegralEvaluator(EvaluatorAgent):
    def __init__(self, battrs, decimate=1):
        super(IntegralEvaluator, self).__init__()
        self.battrs = battrs
        self.decimate = decimate

    def eval(self, expr, solvars, phys,
             kind='domain', idx='all', order=2):

        from .bdr_nodal_evaluator import get_emsh_idx
        
        emesh_idx = get_emesh_idx(self, expr, solvars, phys)
        if len(emesh_idx) > 1:
            assert False, "expression involves multiple mesh (emesh length != 1)"


        mesh = self.mesh()[self.emesh_idx]
