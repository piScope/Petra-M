import numpy as np
import parser
import scipy
import six
import weakref
from weakref import WeakKeyDictionary as WKD
from weakref import WeakValueDictionary as WVD


from petram.mfem_config import use_parallel
if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

from petram.sol.evaluator_agent import EvaluatorAgent
from petram.sol.bdr_nodal_evaluator import process_iverts2nodals, eval_at_nodals

class EdgeNodalEvaluator(EvaluatorAgent):
    def __init__(self, attrs, plane = None):
        '''
           attrs = [[1,2,3], ax, ay, az, c]

           cut-plane is defined as
           ax * x + ay * y + ax * z + c = 0
        '''
        super(EdgeNodalEvaluator, self).__init__()
        self.attrs = attrs
        
    def preprocess_geometry(self, attrs, plane = None):
        #from petram.sol.test import pg
        #return pg(self, battrs, plane = plane)
        self.vertices = None

        self.knowns = WKD()
        mesh = self.mesh()
        self.iverts = []
        self.attrs = attrs

        if attrs[0] == 'all':
            eattrs = 'all'
        else:
            eattrs = eval(attrs[0])

        print eattrs

        from petram.mesh.find_edges import find_edges 
        edges, bb_edges = find_edges(mesh)
        
        bb_bdrs = bb_edges.keys()
        iverts = []
        for bb_bdr in bb_bdrs:
            if eattrs != 'all':
                if isinstance(eattrs, tuple):
                    if any([not x in bb_bdr for x in eattrs]): continue
                else:
                    if not eattrs in bb_bdr: continue
            iedges = bb_edges[bb_bdr]
            iverts.extend([mesh.GetEdgeVertices(ie) for ie in iedges])

        self.ibeles = None # can not use boundary variable in this evaulator  
        if len(iverts) == 0: return
      
        iverts = np.stack(iverts)
        self.iverts = iverts
        if len(self.iverts) == 0: return

        data = process_iverts2nodals(mesh, iverts)
        for k in six.iterkeys(data):
            setattr(self, k, data[k])
            
    def eval(self, expr, solvars, phys, **kwargs):
        val = eval_at_nodals(self, expr, solvars, phys)
        if val is None: return None, None, None

#        return self.locs[self.iverts_inv], val[self.iverts_inv, ...]
        return self.locs, val, self.iverts_inv


         
