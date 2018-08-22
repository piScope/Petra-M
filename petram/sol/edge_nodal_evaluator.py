import numpy as np
import parser
import scipy
import six
from collections import defaultdict
import weakref
from weakref import WeakKeyDictionary as WKD
from weakref import WeakValueDictionary as WVD


from petram.mfem_config import use_parallel
if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

from petram.sol.evaluator_agent import EvaluatorAgent
from petram.sol.bdr_nodal_evaluator import process_iverts2nodals
from petram.sol.bdr_nodal_evaluator import eval_at_nodals, get_emesh_idx

class EdgeNodalEvaluator(EvaluatorAgent):
    def __init__(self, attrs, plane = None):
        '''
           attrs = [[1,2,3], ax, ay, az, c]

           cut-plane is defined as
           ax * x + ay * y + ax * z + c = 0
        '''
        super(EdgeNodalEvaluator, self).__init__()
        self.attrs = attrs
        
    def preprocess_geometry(self, attrs, emesh_idx=0):
        self.vertices = None

        self.knowns = WKD()
        mesh = self.mesh()[emesh_idx]
        self.iverts = []
        self.attrs = attrs

        if attrs[0] == 'all':
            eattrs = 'all'
        else:
            eattrs = attrs

        from petram.mesh.find_edges import find_edges
        if mesh.Dimension() == 3:
            edges, bb_edges = find_edges(mesh)
            bb_bdrs = bb_edges.keys()
            iverts = []
            for bb_bdr in bb_bdrs:
                if eattrs != 'all':
                    check = [sorted(tuple(eattr)) ==  sorted(bb_bdr) for eattr in eattrs]
                    if not any(check): continue
                iedges = bb_edges[bb_bdr]
                iverts.extend([mesh.GetEdgeVertices(ie) for ie in iedges])
        elif mesh.Dimension() == 2:
            kbdr = mesh.GetBdrAttributeArray()
            if eattrs == 'all': eattrs = np.unique(kbdr)
            iverts = []
            #d = defaultdict(list)
            for i in range(mesh.GetNBE()):
                attr = mesh.GetBdrAttribute(i)
                if attr in eattrs:
                    iverts.append(list(mesh.GetBdrElement(i).GetVerticesArray()))
                    #d[attr].extend(mesh.GetBdrElement(i).GetVerticesArray())
        elif mesh.Dimension() == 1:
            kbdr = mesh.GetAttributeArray()
            if eattrs == 'all': eattrs = np.unique(kbdr)
            iverts = []
            #d = defaultdict(list)
            for i in range(mesh.GetNE()):
                attr = mesh.GetAttribute(i)
                if attr in eattrs:
                    iverts.append(list(mesh.GetElement(i).GetVerticesArray()))
                    #d[attr].extend(mesh.GetBdrElement(i).GetVerticesArray())
        else:
            assert False, "Unsupported dim"
            
        self.ibeles = None # can not use boundary variable in this evaulator
        self.emesh_idx = emesh_idx
        
        if len(iverts) == 0: return
      
        iverts = np.stack(iverts)
        self.iverts = iverts
        if len(self.iverts) == 0: return

        data = process_iverts2nodals(mesh, iverts)
        for k in six.iterkeys(data):
            setattr(self, k, data[k])
        
            
    def eval(self, expr, solvars, phys, **kwargs):
        
        emesh_idx = get_emesh_idx(self, expr, solvars, phys)
        if len(emesh_idx) > 1:
            assert False, "expression involves multiple mesh (emesh length != 1)"
        #if len(emesh_idx) < 1:
        #    assert False, "expression is not defined on any mesh"
        #(this could happen when expression is pure geometryical like "x+y")                
            
        if len(emesh_idx) == 1:
            if self.emesh_idx != emesh_idx[0]:
                 self.preprocess_geometry(self.attrs, emesh_idx=emesh_idx[0])
        
        val = eval_at_nodals(self, expr, solvars, phys)
        if val is None: return None, None, None

#        return self.locs[self.iverts_inv], val[self.iverts_inv, ...]
        return self.locs, val, self.iverts_inv


         
