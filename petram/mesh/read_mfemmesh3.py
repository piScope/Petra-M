import numpy as np
from collections import defaultdict

from petram.mesh.find_edges import find_edges
from petram.mesh.find_vertex import find_vertex
from mfem.ser import GlobGeometryRefiner as GR

def extract_refined_mesh_data3(mesh, refine = 5):
    
    ndim = mesh.Dimension()
    sdim = mesh.SpaceDimension()
    
    ivert0 = [mesh.GetBdrElement(i).GetVerticesArray()
                           for i in range(mesh.GetNBE())]
    attrs = mesh.GetBdrAttributeArray()
    nvert = np.array([len(x) for x in ivert0])
    idx3 = np.where(nvert == 3)[0]
    idx4 = np.where(nvert == 4)[0]
    ivert = []; ivert3 = None; ivert4 = None
    iv3 = []; iv4 = []

    from refined_mfem_geom import get_geom

    ptx = []; ivx3 = None
    if len(idx3) != 0:
        base = mesh.GetBdrElementBaseGeometry(idx3[0])
        gt = mesh.GetBdrElementTransformation        
        attr3, ptx3, ivx3, ivxe3, attrx3 =  get_geom(idx3, 3, base, gt, attrs, sdim)
        ptx.append(ptx3)
        iv3 = [ivert0[k] for k in idx3]      
        ivert3 = np.vstack(iv3)
    if len(idx4) != 0:
        base = mesh.GetBdrElementBaseGeometry(idx4[0])
        gt = mesh.GetBdrElementTransformation                
        attr4, ptx4, ivx4, ivxe4, attrx4 =  get_geom(idx4, 4, base, gt, attrs,sdim)
        ptx.append(ptx4)
        iv4 = [ivert0[k] for k in idx4]        
        ivert4 = np.vstack(iv4)
        if ivx3 is not None:
            ivx4 = ivx4 + len(ptx3)       
            ivxe4 = ivxe4 + len(ptx3)        
    ptx_face = np.vstack(ptx)
    
    # unique edges for node                       
    tmp = np.hstack(iv3 + iv4)
    u, indices = np.unique(tmp, return_inverse=True)
    
    ll3 = 3*len(idx3)
    indices3 = indices[:ll3]
    indices4 = indices[ll3:]
                       
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
    cell_data['X_refined_face']=ptx_face
    if ivert3 is not None:
        cells['triangle'] = indices3.reshape(ivert3.shape)
        cells['triangle_x'] = ivx3
        cells['triangle_xe'] = ivxe3                
        cell_data['triangle'] = {}
        cell_data['triangle']['physical'] = attr3
        cell_data['triangle_x'] = {}                         
        cell_data['triangle_x']['physical'] = attrx3
    if ivert4 is not None:
        cells['quad'] = indices4.reshape(ivert4.shape)
        cells['quad_x'] = ivx4
        cells['quad_xe'] = ivxe4                
        cell_data['quad'] = {}
        cell_data['quad']['physical'] = attr4
        cell_data['quad_x'] = {}                         
        cell_data['quad_x']['physical'] = attrx4                         
    
    ## fill line/surface loop info
    loop= {}
    for k in kdom:
        loop[k] = []
        
    for ibdr in range(mesh.GetNBE()):
        idx = mesh.GetBdrElementEdgeIndex(ibdr)
        elem1, elem2 = mesh.GetFaceElements(idx)
        if elem1 != -1:
            i = mesh.GetAttribute(elem1)
            loop[i].append(attrs[ibdr])
        if elem2 != -1:
            i = mesh.GetAttribute(elem2)
            loop[i].append(attrs[ibdr])
    for k in kdom:
        loop[k] = np.unique(loop[k])
        
    l_s_loop = [None, loop]
        
    ## fill line

    edges, bb_edges = find_edges(mesh) ## bb_edges : face pair -> mfem edge id
    bb_keys = bb_edges.keys()
    kedge = []
    cell_line = []
    iedge2bb = {} ##iedge2bb  edge id (petra) to the face pair tuple
    ll = defaultdict(list);  ## face id (mfem) -> edge id (petra) (line loop)

    for k, key in enumerate(bb_keys):
        iedge2bb[k+1] = key
        for kk in key:
           ll[kk].append(k+1)

    ll.default_factory = None
    l_s_loop[0] = ll

    
    idx2 = [ie for key in bb_keys for ie in bb_edges[key]]  # all mfem edge index
    attr22 = np.hstack([[k+1]*len(bb_edges[key]) for k, key in enumerate(bb_keys)])

    attr2 = {i:attr22[k]  for k, i in enumerate(idx2)}
    iv2 = [mesh.GetEdgeVertices(ie) for key in bb_keys for ie in bb_edges[key]]
    
    base = 1
    gt = mesh.GetEdgeTransformation
    attr2, ptx2, ivx2, ivxe2, attrx2 = get_geom(idx2, 2, base, gt, attr2, sdim)

    cells['line'] = table[np.vstack(iv2)]
    cell_data['line'] = {}                
    cell_data['line']['physical'] = attr2
        
    cells['line_x'] = ivx2
    cell_data['line_x'] = {}                        
    cell_data['line_x']['physical'] = attrx2
    cell_data['X_refined_edge']=ptx2
        
    cell_data['vertex'] = {}    
    corners, iverts = find_vertex(mesh, bb_edges)
    if len(iverts) != 0:        
        cells['vertex'] = table[iverts]
        cell_data['vertex']['physical'] = np.arange(len(iverts))+1
    
    ## iedge2bb : mapping from edge_id to boundary numbrer set
    ## X, cells, cell_data : the same data strucutre as pygmsh
    return X, cells, cell_data, l_s_loop, iedge2bb
