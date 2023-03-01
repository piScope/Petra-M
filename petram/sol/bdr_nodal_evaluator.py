'''
   BdrNodalEvaluator:
      a thing to evaluate solution on a boundary
'''
import numpy as np
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


def process_iverts2nodals(mesh, iverts):
    ''' 
    collect data to evalutate nodal values of mesh
    at iverts
    '''
    # we dont want to process the same vert many times.
    # so first we take a unique set...
    iverts_f, iverts_inv = np.unique(iverts.flatten(),
                                     return_inverse=True)
    iverts_inv = iverts_inv.reshape(iverts.shape)

    # then get unique set of elements relating to the verts.
    vert2el = mesh.GetVertexToElementTable()
    ieles = np.hstack([vert2el.GetRowList(i) for i in iverts_f])
    ieles = np.unique(ieles)

    # map from element -> (element's vert index, ivert_f index)
    elvert2facevert = [None]*len(ieles)
    elvertloc = [None]*len(ieles)
    elattr = [None]*len(ieles)

    wverts = np.zeros(len(iverts_f))
    for kk, iel in enumerate(ieles):
        elvert2facevert[kk] = []
        elvertloc[kk] = []
        elattr[kk] = mesh.GetAttribute(iel)
        elverts = mesh.GetElement(iel).GetVerticesArray()
        for k, elvert in enumerate(elverts):
            idx = np.searchsorted(iverts_f, elvert)
            if idx == len(iverts_f):
                continue  # not found
            if iverts_f[idx] != elvert:
                continue  # not found
            elvert2facevert[kk].append((k, idx))
            elvertloc[kk].append(mesh.GetVertexArray(elvert))
            wverts[idx] = wverts[idx]+1

    # idx of element needs to be evaluated

    return {'ieles': np.array(ieles),
            'elvert2facevert': elvert2facevert,
            'locs': np.stack([mesh.GetVertexArray(k) for k in iverts_f]),
            'elvertloc': elvertloc,
            'elattr': np.array(elattr),
            'iverts_inv': iverts_inv,
            'iverts_f': iverts_f,
            'wverts': wverts}


def edge_detect(index):
    #print("edge_detect", index.shape)
    store = []

    def check_pair(store, a, b):
        a1 = min(a, b)
        b1 = max(a, b)
        p = (a1, b1)
        if p in store:
            store.remove(p)
        else:
            store.append(p)
        return store

    for iv in index:
        store = check_pair(store, iv[0], iv[1])
        store = check_pair(store, iv[0], iv[2])
        store = check_pair(store, iv[1], iv[2])
    ret = np.vstack(store)
    return ret


def _get_emesh_idx(obj, expr, solvars, phys, default):
    from petram.helper.variables import Variable, var_g, NativeCoefficientGenBase

    code = compile(expr, '<string>', 'eval')
    names = code.co_names

    g = {}
    # print solvars.keys()
    # print phys._global_ns.keys()
    for key in phys._global_ns.keys():
        g[key] = phys._global_ns[key]
    for key in solvars.keys():
        g[key] = solvars[key]

    idx = []

    for n in names:
        if ((n in g and isinstance(g[n], NativeCoefficientGenBase)) or
                (n in g and isinstance(g[n], Variable))):
            for nn in g[n].dependency:
                idx = g[nn].get_emesh_idx(idx, g=g)

            idx.extend(g[n].get_emesh_idx(idx, g=g))

    if len(idx) == 0:
        # if expression has no emehs dependence return default.
        idx = [default]
    return list(set(idx))


def get_emesh_idx(obj, exprs, solvars, phys):
    '''
    handle multiple expressions and find unique set of emesh_idx needed.
    '''
    if not isinstance(exprs, list):
        exprs = [exprs]

    default = -1
    emesh_idx_list = []
    for item in exprs:
        emesh_idx_list.extend(_get_emesh_idx(
            obj, item, solvars, phys, default))
        default = emesh_idx_list[-1]

    ret = list(set(emesh_idx_list))

    if len(ret) == 1 and ret[0] == -1:
        return [0]
    else:
        ret = [x for x in ret if x != -1]
        return ret


def eval_at_nodals(obj, expr, solvars, phys, edge_evaluator=False):
    '''
    evaluate nodal valus based on preproceessed 
    geometry data

    to be done : obj should be replaced by a dictionary
    '''

    from petram.helper.variables import (Variable,
                                         var_g,
                                         NativeCoefficientGenBase,
                                         CoefficientVariable,
                                         NumbaCoefficientVariable,)

    if len(obj.iverts) == 0:
        return None
    variables = []

    code = compile(expr, '<string>', 'eval')
    names = code.co_names

    g = {}

    for key in phys._global_ns.keys():
        g[key] = phys._global_ns[key]
    for key in solvars.keys():
        g[key] = solvars[key]

    ll_name = []
    ll_value = []
    var_g2 = var_g.copy()

    new_names = []
    name_translation = {}

    all_names = list(names[:])

    def get_names(names):
        for n in names:
            if (n in g and isinstance(g[n], Variable)):
                new_names = g[n].get_names()
                for x in new_names:
                    all_names.append(x)
                get_names(new_names)
    get_names(names)

    for n in all_names:
        if (n in g and isinstance(g[n], NativeCoefficientGenBase)):
            g[n+"_coeff"] = CoefficientVariable(g[n], g)
            new_names.append(n+"_coeff")
            name_translation[n+"_coeff"] = n

        elif (n in g and isinstance(g[n], NumbaCoefficientVariable)):
            ind_vars = [xx.strip() for xx in phys.ind_vars.split(',')]
            g[n].set_coeff(ind_vars, g)
            new_names.append(n)
            name_translation[n] = n

        elif (n in g and isinstance(g[n], Variable)):
            new_names.extend(g[n].dependency)
            new_names.append(n)
            name_translation[n] = n

        elif n in g:
            new_names.append(n)
            name_translation[n] = n

    for n in new_names:
        if (n in g and isinstance(g[n], Variable)):
            if not g[n] in obj.knowns:
                obj.knowns[g[n]] = (
                    g[n].nodal_values(iele=obj.ieles,
                                      ibele=obj.ibeles,
                                      elattr=obj.elattr,
                                      el2v=obj.elvert2facevert,
                                      locs=obj.locs,
                                      elvertloc=obj.elvertloc,
                                      wverts=obj.wverts,
                                      mesh=obj.mesh()[obj.emesh_idx],
                                      iverts_f=obj.iverts_f,
                                      g=g,
                                      knowns=obj.knowns,
                                      edge_evaluator=edge_evaluator))

            ll_name.append(name_translation[n])
            ll_value.append(obj.knowns[g[n]])
        elif (n in g):
            var_g2[n] = g[n]

    if len(ll_value) > 0:
        val = np.array([eval(code, var_g2, dict(zip(ll_name, v)))
                        for v in zip(*ll_value)])
    else:
        # if expr does not involve Varialbe, evaluate code once
        # and generate an array
        val = np.array([eval(code, var_g2)]*len(obj.locs))

    return val


class BdrNodalEvaluator(EvaluatorAgent):
    def __init__(self, battrs, decimate=1):
        super(BdrNodalEvaluator, self).__init__()
        self.battrs = battrs
        self.decimate = decimate

    def preprocess_geometry(self, battrs, emesh_idx=0, decimate=1):

        mesh = self.mesh()[emesh_idx]
        # print 'preprocess_geom',  mesh, battrs
        self.battrs = battrs

        self.decimate = decimate
        self.knowns = WKD()
        self.iverts = []

        if mesh.Dimension() == 3:
            getarray = mesh.GetBdrArray
            getelement = mesh.GetBdrElement
        elif mesh.Dimension() == 2:
            getarray = mesh.GetDomainArray
            getelement = mesh.GetElement
        else:
            assert False, "BdrNodal Evaluator is not supported for this dimension"

        x = [getarray(battr) for battr in battrs]
        if np.sum([len(xx) for xx in x]) == 0:
            return

        ibdrs = np.hstack(x).astype(int).flatten()

        if self.decimate != 1:
            ibdrs = ibdrs[::self.decimate]

        self.ibeles = np.array(ibdrs)

        def get_vertices_array(i):
            arr = getelement(i).GetVerticesArray()
            if len(arr) == 3:
                return arr
            elif len(arr) == 4:
                x = arr[:-1]
                y = np.array([arr[0], arr[2], arr[3]])
                return np.vstack([x, y])

        # we handle quad as two triangles
        iverts = np.vstack([get_vertices_array(i) for i in ibdrs])

        self.iverts = iverts
        if len(self.iverts) == 0:
            return

        data = process_iverts2nodals(mesh, iverts)
        for k in list(data):
            setattr(self, k, data[k])
        self.emesh_idx = emesh_idx

    def eval(self, expr, solvars, phys, **kwargs):
        emesh_idx = get_emesh_idx(self, expr, solvars, phys)
        if len(emesh_idx) > 1:
            assert False, "expression involves multiple mesh (emesh length != 1)"
        # if len(emesh_idx) < 1:
        #    assert False, "expression is not defined on any mesh"
        # (this could happen when expression is pure geometryical like "x+y")
        decimate = kwargs.pop('decimate', 1)

        if len(emesh_idx) == 1:
            if self.emesh_idx != emesh_idx[0]:
                #print("process geom", emesh_idx[0])
                self.preprocess_geometry(self.battrs,
                                         emesh_idx=emesh_idx[0],
                                         decimate=decimate)

        val = eval_at_nodals(self, expr, solvars, phys)

        if val is None:
            return None, None, None

        edge_only = kwargs.pop('edge_only', False)
        export_type = kwargs.pop('export_type', 1)
        if export_type == 2:
            return self.locs, val, None

        refine = kwargs.pop('refine', 1)

        if refine == 1 or edge_only:
            if not edge_only:
                return self.locs, val, self.iverts_inv
            else:
                idx = edge_detect(self.iverts_inv)
                return self.locs, val, idx
        else:
            from petram.sol.nodal_refinement import refine_surface_data

            ptx, data, ridx = refine_surface_data(self.mesh()[self.emesh_idx],
                                                  self.ibeles,
                                                  val, self.iverts_inv,
                                                  refine)
            return ptx, data, ridx
