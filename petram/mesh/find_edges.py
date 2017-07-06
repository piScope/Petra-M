import numpy as np
from scipy.sparse import lil_matrix

def fill_table(table1, table2, i, j, value):
    if i > j:
        if table1[i, j] == 0: 
            table1[i, j] = value
        elif table1[i, j] == value:  return
        else:
            if table1[i, j] > value:
                table2[i, j] = table1[i, j]
                table1[i, j] = value                
            else:
                table2[i, j] = value            
    else:
        if table1[j, i] == 0: 
            table1[j, i] = value
        elif table1[j, i] == value:  return            
        else:
            if table1[j, i] > value:
                table2[j, i] = table1[j, i]
                table1[j, i] = value                
            else:
                table2[j, i] = value            
            
def find_edges(mesh):

    nv = mesh.GetNV()
    nb = mesh.GetNBE()
    ne = mesh.GetNEdges()

    iattr= mesh.GetBdrAttributeArray()  # min of this array is 1
    nattr = np.max(iattr)

    table1 = lil_matrix((ne, nattr), dtype = int)

    iv = mesh.GetBdrElementVertices(0)
    l = len(iv)
    for i in range(nb):
        ie, io = mesh.GetBdrElementEdges(i)
        iattr = mesh.GetBdrAttribute(i)
        for k in ie:
            table1[k, iattr-1] = 1
    csr = table1.tocsr()
    idx = np.where(np.diff(csr.indptr) > 1)[0]
    csr = table1[idx, :].tocsr()

    csc  = csr.tocsc()

    edges = [None]*csc.shape[1]
    indptr = csc.indptr; indices = csc.indices

    for i in range(csc.shape[1]):
        edges[i] = idx[indices[indptr[i]:indptr[i+1]]]
    # this is true bdr edges.
    bb_edges = {}
    indptr = csr.indptr; indices = csr.indices

    for i in range(csr.shape[0]):
        idxs = tuple(indices[indptr[i]:indptr[i+1]]+1)
        if idxs in bb_edges:
            bb_edges[idxs].append(idx[i])
        else:
            bb_edges[idxs] = [idx[i]]

    return edges, bb_edges

def plot_edges(mesh, face = 'all'):
    '''
    plot edges between boundary and boundary
       bb : 'all' : plot all edges
             35   : plot edges of battr = 35 as one solid object
    '''
    
    from ifigure.interactive import figure
    
    edges, bb_edges = find_edges(mesh)

    if face == 'all':
        battrs = np.arange(len(edges), dtype=int)
    else:
        battrs = [face-1]
        
    viewer = figure()
    viewer.threed('on')

    for battr in battrs:
        iedges = edges[battr]
        verts = np.stack([np.vstack((mesh.GetVertexArray(mesh.GetEdgeVertices(ie)[0]),
                                     mesh.GetVertexArray(mesh.GetEdgeVertices(ie)[1])))
                                     for ie in iedges])
        viewer.solid(verts)
        return iedges

def plot_bbedges(mesh, bb = 'all'):
    '''
    plot edges between boundary and boundary
       bb : 'all' : plot all edges
             35   : plot edges of battr = 35 splitted based on boundary connection
             (35 36): plot edges between  battr = 35 and 36

    '''
    from ifigure.interactive import figure
    
    edges, bb_edges = find_edges(mesh)
    bb_bdrs = bb_edges.keys()
    
    viewer = figure()
    viewer.threed('on')

    for bb_bdr in bb_bdrs:
        if bb != 'all':
            if isinstance(bb, tuple):
                if any([not x in bb_bdr for x in bb]): continue
            else:
               if not bb in  bb_bdr: continue
        print bb_bdr
        #if bb != 'all' and bb != bb_bdr: continue
        
        iedges = bb_edges[bb_bdr]
        verts = np.stack([np.vstack((mesh.GetVertexArray(mesh.GetEdgeVertices(ie)[0]),
                                     mesh.GetVertexArray(mesh.GetEdgeVertices(ie)[1])))
                                     for ie in iedges])
        viewer.solid(verts)
    
    
    
