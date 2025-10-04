import numpy as np
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


        if mesh.Dimension() == 3:
            '''
            from petram.mesh.find_edges import find_edges
            edges, bb_edges = find_edges(mesh)
            bb_bdrs = bb_edges.keys()
            iverts = []
            for bb_bdr in bb_bdrs:
                if eattrs != 'all':
                    check = [sorted(tuple(eattr)) ==  sorted(bb_bdr) for eattr in eattrs]
                    if not any(check): continue
                iedges = bb_edges[bb_bdr]
                iverts.extend([mesh.GetEdgeVertices(ie) for ie in iedges])
            print iverts
            '''
            from petram.mesh.mesh_utils import get_extended_connectivity
            if not hasattr(mesh, 'extended_connectivity'):
               get_extended_connectivity(mesh)
            
            l2e = mesh.extended_connectivity['line2edge']
            keys = l2e.keys() if eattrs == 'all' else list(np.atleast_1d(eattrs).flatten())
            iedges = list(set(sum([l2e[k] for k in keys if k in l2e],[])))
            iverts = [mesh.GetEdgeVertices(ie) for ie in iedges]
            self.ibeles = None # can not use boundary variable in this evaulator            

        elif mesh.Dimension() == 2:
            kbdr = mesh.GetBdrAttributeArray()
            if eattrs == 'all': eattrs = np.unique(kbdr)
            iverts = []
            #d = defaultdict(list)
            for i in range(mesh.GetNBE()):
                attr = mesh.GetBdrAttribute(i)
                if attr in eattrs:
                    iverts.append(list(mesh.GetBdrElement(i).GetVerticesArray()))

            x = np.unique([mesh.GetBdrArray(e) for e in eattrs]).astype(int, copy=False)
            self.ibeles = x
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
            x = np.unique([mesh.GetDomainArray(e) for e in eattrs]).astype(int, copy=False)
            self.ibeles = x

        else:
            assert False, "Unsupported dim"

        self.emesh_idx = emesh_idx
        
        if len(iverts) == 0: return
      
        iverts = np.stack(iverts)
        self.iverts = iverts
        if len(self.iverts) == 0: return

        data = process_iverts2nodals(mesh, iverts)
        for k in six.iterkeys(data):
            setattr(self, k, data[k])
        
            
    def eval(self, expr, solvars, phys, **kwargs):
        exprs =  kwargs.pop("exprs", [expr])
        emesh_idx = get_emesh_idx(self, exprs, solvars, phys)
        
        #print("emesh_idx", emesh_idx)
        if len(emesh_idx) > 1:
            assert False, "expression involves multiple mesh (emesh length != 1)"
        #if len(emesh_idx) < 1:
        #    assert False, "expression is not defined on any mesh"
        #(this could happen when expression is pure geometryical like "x+y")                
            
        if len(emesh_idx) == 1:
            if self.emesh_idx != emesh_idx[0]:
                 self.preprocess_geometry(self.attrs, emesh_idx=emesh_idx[0])
                 
        val = eval_at_nodals(self, expr, solvars, phys,
                             edge_evaluator=True)
        if val is None: return None, None, None

#        return self.locs[self.iverts_inv], val[self.iverts_inv, ...]

        refine = kwargs.pop('refine', 1)
        
        if refine == 1:
            return self.locs, val, self.iverts_inv
        else:
            from petram.sol.nodal_refinement import refine_edge_data
            try:
                mesh = self.mesh()[self.emesh_idx]
                if mesh.Dimension() == 3:
                    assert False, "edge refinement is not supported for dim=3"
                ptx, data, ridx = refine_edge_data(mesh,
                                                  self.ibeles,
                                                  val, self.iverts_inv,
                                                  refine)
            except:
                import traceback
                traceback.print_exc()
            return ptx, data, ridx



         
