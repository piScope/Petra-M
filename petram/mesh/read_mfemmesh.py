'''

    extract_mesh_data: 

    digest mesh object data for plotting.
    if also extend mfem:Mesh object to include the 
    extended_connectivity information
  
    extended_connectivity = {'line2vert':{},   # two edges to make an edge
                             'surf2line':{},   # line loop for surface
                             'vol2surf': {},   # surface loop for volue
                             'line2edge':{},   # line number to mfem edge element
                             'vert2vert':{}}   # vertex number to mfem vertex id


'''
import numpy as np
from collections import defaultdict

from petram.mesh.find_edges import find_edges
from petram.mesh.find_vertex import find_vertex

from petram.mfem_config import use_parallel

if use_parallel:
    from mfem.par import GlobGeometryRefiner
    import mfem.par as mfem
    from mfem.par import ParMesh
else:
    from mfem.ser import GlobGeometryRefiner
    import mfem.ser as mfem

    # dummy class
    class ParMesh(object):
        pass
    
def extract_mesh_data_1D(mesh):
    nv = mesh.GetNV()
    X = np.vstack([mesh.GetVertexArray(k) for k in range(nv)])
    if X.shape[1] == 2:
        X = np.hstack((X, np.zeros((X.shape[0],1))))
    elif X.shape[1] == 1:
        X = np.hstack((X, np.zeros((X.shape[0],2))))
        
    cells = {}
    cell_data = {}

    attrs = mesh.GetAttributeArray()
    lverts = []; lattr=[]
    for attr in np.unique(attrs):
        idx = np.where(attrs == attr)[0]
        for i in idx:
            lverts.append(mesh.GetElementVertices(i))
        lattr.extend([attr]*len(idx))
        
    cells['line'] = np.array(lverts)
    cell_data['line'] = {}
    cell_data['line']['physical'] = np.array(lattr).flatten()

    vverts = []; vattr=[]    
    for ibe in range(mesh.GetNBE()):
        battr = mesh.GetBdrAttribute(ibe)
        vattr.append(battr)
        vverts.append(mesh.GetBdrElementVertices(ibe))
    cells['vertex'] = np.array(vverts)
    cell_data['vertex'] = {}    
    cell_data['vertex']['physical'] = np.array(vattr).flatten()

    from petram.mesh.mesh_utils import get_extended_connectivity
    
    if not hasattr(mesh, 'extended_connectivity'):
        get_extended_connectivity(mesh)
    
    return X, cells, cell_data, [None, None], None

def extract_mesh_data(mesh, refine=1):
    if isinstance(mesh, ParMesh):
        assert False, "mesh data must be processed in serial"
    
    hasNodal = mesh.GetNodalFESpace() is not None    
    ndim = mesh.Dimension()

    if hasNodal and refine != 1:
       if ndim == 3:
           from petram.mesh.read_mfemmesh3 import extract_refined_mesh_data3           
           return extract_refined_mesh_data3(mesh, refine)
       elif ndim == 2:
           from petram.mesh.read_mfemmesh2 import extract_refined_mesh_data2
           return extract_refined_mesh_data2(mesh, refine)
       else:
           from petram.mesh.read_mfemmesh1 import extract_refined_mesh_data1
           return extract_refined_mesh_data1(mesh, refine)
    if ndim == 3:
        ivert0 = [mesh.GetBdrElement(i).GetVerticesArray()
                           for i in range(mesh.GetNBE())]
        attrs = mesh.GetBdrAttributeArray()
    elif ndim == 2:
        ivert0 = [mesh.GetElement(i).GetVerticesArray()
                       for i in range(mesh.GetNE())]

        attrs = mesh.GetAttributeArray()        
    else:
        return extract_mesh_data_1D(mesh)

    nvert = np.array([len(x) for x in ivert0])
    idx3 = np.where(nvert == 3)[0]
    idx4 = np.where(nvert == 4)[0]
    ivert = []; ivert3 = None; ivert4 = None
    iv3 = []; iv4 = []

    if len(idx3) != 0:
        iv3 = [ivert0[k] for k in idx3]
        ivert3 = np.vstack(iv3)
        attrs3 = attrs[idx3]
    if len(idx4) != 0:
        iv4 = [ivert0[k] for k in idx4]        
        ivert4 = np.vstack(iv4)
        attrs4 = attrs[idx4]

    tmp = np.hstack(iv3+iv4)
    u, indices = np.unique(tmp, return_inverse=True)
    
    ll3 = 3*len(idx3)
    indices3 = indices[:ll3]
    indices4 = indices[ll3:]

    # table u -> u's idx
    table = np.zeros(np.max(u)+1, dtype=int)-1
    for k, u0 in enumerate(u):
        table[u0] = k
        
    X = np.vstack([mesh.GetVertexArray(k) for k in u])
    if X.shape[1] == 2:
        X = np.hstack((X, np.zeros((X.shape[0],1))))
    elif X.shape[1] == 1:
        X = np.hstack((X, np.zeros((X.shape[0],2))))
    
    kdom = mesh.attributes.ToList()
    
    cells = {}
    cell_data = {}
    if ivert3 is not None:
        cells['triangle'] = indices3.reshape(ivert3.shape)
        cell_data['triangle'] = {}
        cell_data['triangle']['physical'] = attrs3
    if ivert4 is not None:
        cells['quad'] = indices4.reshape(ivert4.shape)
        cell_data['quad'] = {}
        cell_data['quad']['physical'] = attrs4


    from petram.mesh.mesh_utils import populate_plotdata
    l_s_loop = populate_plotdata(mesh, table, cells, cell_data)
    iedge2bb = None # is it used?    
    return X, cells, cell_data, l_s_loop, iedge2bb
    '''
    get_extended_connectivity(mesh)
    ec = mesh.extended_connectivity

    cell_data['line'] = {}
    cell_data['vertex'] = {}
    l2e = ec['line2edge']
    v2v = ec['vert2vert']

    kedge = []

    kedge = np.array(sum([[key]*len(l2e[key]) for key in l2e], [])).astype(int)
    iverts = np.vstack([mesh.GetEdgeVertices(ie)
                        for key in l2e for ie in l2e[key]])
    cells['line'] = table[iverts]
    cell_data['line']['physical'] = np.array(kedge)

    kvert = np.array([key for key in v2v]).astype(int)    
    iverts = np.array([v2v[key] for key in v2v]).astype(int)    

    cells['vertex'] = table[iverts]    
    cell_data['vertex']['physical'] = kvert
    
    if ndim == 3:
        l_s_loop = [ec['surf2line'], ec['vol2surf']]
    elif ndim == 2:
        l_s_loop = [ec['surf2line'], None]
    else:
        l_s_loop = [None, None]

    iedge2bb = None # is it used?
    '''


