import numpy as np

from petram.mesh.find_edges import find_edges

def extract_mesh_data(mesh):
    iv = mesh.GetBdrElementVertices(0)
    ndim = len(iv)
    if ndim == 3:
        ivert = np.vstack([mesh.GetBdrElement(i).GetVerticesArray()
                       for i in range(mesh.GetNBE())])
        attrs = mesh.GetBdrAttributeArray()
    elif ndim == 2:
        ivert = np.vstack([mesh.GetElement(i).GetVerticesArray()
                       for i in range(mesh.GetNE())])
        attrs = mesh.GetAttributeArray()        
    else:
        assert False, "1D mesh not supported"
            
    u, indices = np.unique(ivert.flatten(), return_inverse=True)
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
    if ivert.shape[1] == 3:
        cells['triangle'] = indices.reshape(ivert.shape)
        cell_data['triangle'] = {}
        cell_data['triangle']['physical'] = attrs
    else:
        assert False, "only triangle surface is supported"

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

    elif ndim == 2:
        ivert = np.vstack([mesh.GetBdrElement(i).GetVerticesArray()
                           for i in range(mesh.GetNBE())])
        cells['line'] = table[ivert]
        cell_data['line']['physical'] = kbdr
        iedge2bb = {}
        for k in kbdr: iedge2bb[k] = k    
    else:
        pass
    
    ## iedge2bb : mapping from edge_id to boundary numbrer set
    ## X, cells, cell_data : the same data strucutre as pygmsh
    return X, cells, cell_data, l_s_loop, iedge2bb
