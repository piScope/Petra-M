import numpy as np
from collections import defaultdict

from petram.mesh.find_edges import find_edges
from petram.mesh.find_vertex import find_vertex
from mfem.ser import GlobGeometryRefiner


def extract_mesh_data(mesh, refine = None):
    hasNodal = mesh.GetNodalFESpace() is not None    
    ndim = mesh.Dimension()
    
    if hasNodal and refine != 1:
       if ndim == 3:
           from read_mfemmesh3 import extract_refined_mesh_data3           
           return extract_refined_mesh_data3(mesh, refine)
       elif ndim == 2:
           from read_mfemmesh2 import extract_refined_mesh_data2
           return extract_refined_mesh_data2(mesh, refine)
       else:
           assert False, "1D mesh not supported"           
    if ndim == 3:
        ivert0 = [mesh.GetBdrElement(i).GetVerticesArray()
                           for i in range(mesh.GetNBE())]
        attrs = mesh.GetBdrAttributeArray()
    elif ndim == 2:
        ivert0 = [mesh.GetElement(i).GetVerticesArray()
                       for i in range(mesh.GetNE())]

        attrs = mesh.GetAttributeArray()        
    else:
        assert False, "1D mesh not supported"
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
    if ivert3 is not None:
        cells['triangle'] = indices3.reshape(ivert3.shape)
        cell_data['triangle'] = {}
        cell_data['triangle']['physical'] = attrs3
    if ivert4 is not None:
        cells['quad'] = indices4.reshape(ivert4.shape)
        cell_data['quad'] = {}
        cell_data['quad']['physical'] = attrs4
    
    ## fill line/surface loop info
    loop= {}
    for k in kdom:
        loop[k] = []
    battr =  mesh.GetBdrAttributeArray()
    for ibdr in range(mesh.GetNBE()):
        idx = mesh.GetBdrElementEdgeIndex(ibdr)
        elem1, elem2 = mesh.GetFaceElements(idx)
        if elem1 != -1:
            i = mesh.GetAttribute(elem1)
            loop[i].append(battr[ibdr])
        if elem2 != -1:
            i = mesh.GetAttribute(elem2)
            loop[i].append(battr[ibdr])
    for k in kdom:
        loop[k] = np.unique(loop[k])
        
    if ndim == 3:
        l_s_loop = [None, loop]
    elif ndim == 2:
        l_s_loop = loop, None
    else:
        l_s_loop = None, None
        
    ## fill line
    cell_data['line'] = {}
    cell_data['vertex'] = {}    
    kbdr = mesh.GetBdrAttributeArray()
    if ndim == 3:
        edges, bb_edges = find_edges(mesh)
        kedge = []
        cell_line = []
        iedge2bb = {}
        ll = {};
        l_s_loop[0] = ll
        for k in range(np.max(kbdr)):
            ll[k+1] = []
        
        for idx, key in enumerate(bb_edges.keys()):
            kedge.extend([idx+1]*len(bb_edges[key]))
            iedge2bb[idx+1] = key
            cell_line.append(np.vstack([mesh.GetEdgeVertices(ie)
                                         for ie in bb_edges[key]]))
            for k in key:
                ll[k].append(idx+1)
        cells['line'] = table[np.vstack(cell_line)]
        cell_data['line']['physical'] = np.array(kedge)
        corners, iverts = find_vertex(mesh, bb_edges)
        if len(iverts) != 0:        
            cells['vertex'] = table[iverts]
            cell_data['vertex']['physical'] = np.arange(len(iverts))+1
    elif ndim == 2:
        ivert = np.vstack([mesh.GetBdrElement(i).GetVerticesArray()
                           for i in range(mesh.GetNBE())])
        cells['line'] = table[ivert]
        cell_data['line']['physical'] = kbdr
        iedge2bb = {}
        for k in kbdr: iedge2bb[k] = k

        d = defaultdict(list)
        for i in range(mesh.GetNBE()):
            d[kbdr[i]].extend(ivert[i, :])
        corners = {}
        for key in d:
           seen = defaultdict(int)
           for iiv in d[key]:
               seen[iiv] += 1
           corners[key] = [kk for kk in seen if seen[kk]==1]
        iverts = np.unique(np.hstack([corners[key] for key in corners]))
        if len(iverts) != 0:
            cells['vertex'] = table[iverts]
            cell_data['vertex']['physical'] = np.arange(len(iverts))+1
    else:
        pass
    
    ## iedge2bb : mapping from edge_id to boundary numbrer set
    ## X, cells, cell_data : the same data strucutre as pygmsh
    return X, cells, cell_data, l_s_loop, iedge2bb
