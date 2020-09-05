'''
   utility routines
'''
import os
import numpy as np
import resource    
from petram.mfem_config import use_parallel
import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('Utils')

if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem


def file_write(fid, *args):
    txt = ' '.join([str(x) for x in args])
    print(txt)
    fid.write(txt + "\n")


def print_mem(myid = 0):
    mem=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print("memory usage (" + str(myid) +  "): " + str(mem))

def set_array_attribute(v, base, suffix, values):
    for s, vv in zip(suffix, values):
        v[base + s] = vv
    return v

def txt2indexlist(txt):
    try:
        return [int(x) for x in txt.split(',')]
    except:
        raise ValueError("can not convert text to index list")

def eval_expression(expr, mesh, battr, ind_vars, ns, use_dom = False):
    '''
    example:
        expr = 'x', or 'sin(x)'
        ibdry = 3 (boundary attribute number)
        ind_vars ['x', 'y', 'z']
    '''
    def _do_eval(code, verts):
        l = {n: verts[k] for k, n in enumerate(ind_vars)}
        return eval(code, ns, l)

    if use_dom:
        get_array = mesh.GetDomainArray
        get_element = mesh.GetElement
    else:
        get_array = mesh.GetBdrArray
        get_element = mesh.GetBdrElement        
        
    ibdr = get_array(battr)

    code = compile(expr, '<string>', 'eval')
    
    iverts = np.stack([get_element(i).GetVerticesArray() for i in ibdr])
    locs   = np.stack([np.stack([mesh.GetVertexArray(k) for k in ivert])     
                          for ivert in iverts])
    data   =  np.stack([np.stack([_do_eval(code, mesh.GetVertexArray(k))
                                for k in ivert])     
                                for ivert in iverts])
    return locs, data

def eval_expr(model, engine, expr, battrs, phys = None):
    '''
    expr = expression to evaluate
    battrs = list of boundary attribute
    '''
    from petram.model import Bdry

    if phys is None: 
        phys = model['Phys'][list(model['Phys'])[0]]
    else:
        phys = model['Phys'][phys]

    ret = {}
    mesh = engine.get_mesh()
    engine.assign_sel_index(phys)
    
    use_dom = (phys.dim == 2 or phys.dim == 1)
    ns = phys._global_ns
    battrs = list(np.atleast_1d(eval(battrs, ns, {})).flatten().astype(int))
       
    for m in phys.walk():
        for battr in battrs:
           ind_vars = m.get_independent_variables()
           ret[battr] = eval_expression(expr, mesh, battr, ind_vars, ns, use_dom=use_dom)
    return ret
 

def eval_sol(sol, battrs, dim = 0):
    mesh = sol.FESpace().GetMesh()
    ibdr = np.hstack([mesh.GetBdrArray(battr) for battr in battrs])
    if len(ibdr) == 0: return None, None
    iverts = np.stack([mesh.GetBdrElement(i).GetVerticesArray() for i in ibdr])
    locs   = np.stack([np.stack([mesh.GetVertexArray(k) for k in ivert])     
                          for ivert in iverts])

    data = sol.GetNodalValues(dim)
    data   =  data[iverts.flatten()].reshape(iverts.shape)
    
    return locs, data

def eval_loc(sol, battrs):
    mesh = sol.FESpace().GetMesh()
    ibdr = np.hstack([mesh.GetBdrArray(battr) for battr in battrs])
    if len(ibdr) == 0: return None, None
    iverts = np.stack([mesh.GetBdrElement(i).GetVerticesArray() for i in ibdr])
    locs   = np.stack([np.stack([mesh.GetVertexArray(k) for k in ivert])     
                          for ivert in iverts])
    return locs


def get_pkg_datafile(pkg, *path):
    '''
    return package data

    ex) get_pkg_datafile(petram.geom, 'icon', 'line.png')
    '''
    file = getattr(pkg, '__file__')
    root = os.path.abspath(os.path.dirname(file))
    return os.path.join(os.path.dirname(root), 'data', *path)

def get_evn_petram_root():
    petram = os.getenv("PetraM")
    return petram

def check_cluster_access():
    petram = get_evn_petram_root()
    return os.path.exists(os.path.join(petram, "etc", "cluster_access"))

'''
This is old less accurate code.
Those derivatis should be evaulated using DiscreteLinearInterpolator.

def eval_nodal_curl_values(gf, i, vdim):
    geometries = mfem.geom.Geometry()

    fes = gf.FESpace()
    fe = fes.GetFE(i)
    tr = fes.GetElementTransformation(i)
    rule = geometries.GetVertices(fes.GetFE(i).GetGeomType())
    dof = fe.GetDof()
    n = rule.GetNPoints()
    values = []
    for k in range(n):
        tr.SetIntPoint(rule.IntPoint(k))
        v = mfem.Vector()
        gf.GetCurl(tr, v)
        values.append(v.GetDataArray().copy()[vdim-1])
    return np.stack(values)


def eval_nodal_div_values(gf, i, vdim):
    raise NotImplementedError(
          "you must specify this method in subclass")

def eval_nodal_grad_values(gf, i, vdim):
    raise NotImplementedError(
          "you must specify this method in subclass")

def get_nodal_x_values(gf, vdim, x=eval_nodal_curl_values):
    fes = gf.FESpace()
    values = [None]*fes.GetNV()
    for i in range(fes.GetNE()):
        ivert = fes.GetMesh().GetElementVertices(i)
        v = x(gf, i, vdim)
        for k, vv in zip(ivert, v):
            if values[k] is None:
                values[k] = [vv]
            else:
                values[k].append(vv)
        
    for i in range(fes.GetNV()):
       values[i] = np.stack(values[i])
       values[i] = np.mean(values[i], 0)
    return np.stack(values)

def get_nodal_curl_values(gf, vdim):
    return get_nodal_x_values(gf, vdim, x=eval_nodal_curl_values)
def get_nodal_div_values(gf, vdim):
    return get_nodal_x_values(gf, vdim, x=eval_nodal_div_values)
def get_nodal_grad_values(gf, vdim):
    return get_nodal_x_values(gf, vdim, x=eval_nodal_grad_values)

def eval_curl(sol, battr, dim = 0):
    mesh = sol.FESpace().GetMesh()
    ibdr = mesh.GetBdrArray(battr)
    if len(ibdr) == 0: return None, None
    iverts = np.stack([mesh.GetBdrElement(i).GetVerticesArray() for i in ibdr])
    locs   = np.stack([np.stack([mesh.GetVertexArray(k) for k in ivert])     
                          for ivert in iverts])

    data = get_nodal_curl_values(sol, dim)
    data   =  data[iverts.flatten()].reshape(iverts.shape)
    
    return locs, data
'''

