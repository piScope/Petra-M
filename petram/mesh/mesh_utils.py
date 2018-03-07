import os
import numpy as np
from scipy.sparse import coo_matrix
from collections import OrderedDict, defaultdict

def distribute_shared_entity(pmesh):
    '''
    distribute entitiy numbering in master (owner) process
    '''
    from mpi4py import MPI
    myid = MPI.COMM_WORLD.rank
    nprc = MPI.COMM_WORLD.size   
    comm  = MPI.COMM_WORLD
        
    from mfem.common.mpi_debug import nicePrint, niceCall        
    
    from petram.helper.mpi_recipes import allgather, allgather_vector, gather_vector    
    master_entry = []
    local_data = {}
    master_data = {}
        
    offset_v = np.hstack([0, np.cumsum(allgather(pmesh.GetNV()))])    
    offset_e = np.hstack([0, np.cumsum(allgather(pmesh.GetNEdges()))])
    offset_f = np.hstack([0, np.cumsum(allgather(pmesh.GetNFaces()))])

    ng = pmesh.GetNGroups()

    from mfem.par import intp
    def GroupEdge(j, iv):
        edge = intp(); o = intp()
        pmesh.GroupEdge(j, iv, edge, o)
        return edge.value()
    def GroupFace(j, iv):
        face = intp(); o = intp()
        pmesh.GroupFace(j, iv, face, o)
        return face.value()
     
    for j in range(ng):
        if j == 0: continue
        nv = pmesh.GroupNVertices(j)
        sv = np.array([pmesh.GroupVertex(j, iv) for iv in range(nv)])
        ne = pmesh.GroupNEdges(j)
        se = np.array([GroupEdge(j, iv) for iv in range(ne)])
        nf = pmesh.GroupNFaces(j)
        sf = np.array([GroupFace(j, iv) for iv in range(nf)])

        data = (sv + offset_v[myid],
                se + offset_e[myid],
                sf + offset_f[myid])
        local_data[(pmesh.gtopo.GetGroupMasterRank(j),
                    pmesh.gtopo.GetGroupMasterGroup(j))] = data

        if pmesh.gtopo.IAmMaster(j):
            master_entry.append((myid, j,))
            mv = sv + offset_v[myid]
            me = se + offset_e[myid]
            mf = sf + offset_f[myid]
            data = (mv, me, mf)
        else:
            data = None
        master_data[(pmesh.gtopo.GetGroupMasterRank(j),
                     pmesh.gtopo.GetGroupMasterGroup(j))] = data
        
    master_entry = comm.gather(master_entry)
    if myid == 0:
        master_entry = sum(master_entry, [])
    master_entry = comm.bcast(master_entry)
    for entry in master_entry:
        master_id = entry[0]
        if master_id == myid:
            data = master_data[entry]
        else:
            data = None
        data = comm.bcast(data, root=master_id)
        if entry in master_data:
            master_data[entry] = data
            
    return local_data, master_data


def find_edge_corner(mesh):
    '''
    For 3D geometry
      find line (boundary between two bdr_attribute) and
      corner of lines
    '''
    use_parallel = hasattr(mesh, "GroupNVertices")
    
    if use_parallel:
        from mpi4py import MPI
        myid = MPI.COMM_WORLD.rank
        nprc = MPI.COMM_WORLD.size   
        comm  = MPI.COMM_WORLD
        
        from mfem.common.mpi_debug import nicePrint, niceCall        
        from petram.helper.mpi_recipes import allgather, allgather_vector, gather_vector
        from petram.mesh.mesh_utils import distribute_shared_entity        
        if not hasattr(mesh, "shared_info"):
            mesh.shared_info = distribute_shared_entity(mesh)
    else:
        myid = 0
    ndim =  mesh.Dimension()
    sdim =  mesh.SpaceDimension()    
    ne = mesh.GetNEdges()
    assert ndim == 3, "find_edge_corner is for 3D mesh"

    '''
    if ndim == 2:
        # 2D mesh
        get_edges = mesh.GetElementEdges
        get_attr  = mesh.GetAttribute
        iattr= mesh.GetAttributeArray()     # min of this array is 1
        nattr = np.max(iattr)
        nb = mesh.GetNE()        
    else:
    '''
    # 3D mesh
    get_edges = mesh.GetBdrElementEdges
    get_attr  = mesh.GetBdrAttribute
    iattr= mesh.GetBdrAttributeArray()  # min of this array is 1
    nattr = np.max(iattr)        
    nb = mesh.GetNBE()

    if use_parallel:
        offset = np.hstack([0, np.cumsum(allgather(mesh.GetNEdges()))])
        offsetf = np.hstack([0, np.cumsum(allgather(mesh.GetNFaces()))])
        offsetv = np.hstack([0, np.cumsum(allgather(mesh.GetNV()))])
        myoffset = offset[myid]
        myoffsetf = offsetf[myid]
        myoffsetv = offsetv[myid]                
        nattr = max(allgather(nattr))
        ne = sum(allgather(mesh.GetNEdges()))
    else:
        myoffset = 0
        myoffsetf = 0
        myoffsetv = 0        
        
    edges = defaultdict(list)
    iedges = np.arange(nb, dtype=int)
    
    if use_parallel:
        # eliminate slave faces from consideration
        iface = np.array([mesh.GetBdrElementEdgeIndex(i) for i in iedges],
                          dtype=int)+myoffsetf
        mask = np.array([True]*len(iface), dtype=bool)
        ld, md = mesh.shared_info
        for key in ld.keys():
            mid, g_in_master = key
            if mid == myid: continue
            iii = np.in1d(iedges, ld[key][2], invert = True)
            mask = np.logical_and(mask, iii)
        iedges = iedges[mask]
        
    # nicePrint(len(iedges)) np 1,2,4 gives 900... ok
    
    for i in iedges:
        ie, io = get_edges(i)
        ie += myoffset
        iattr = get_attr(i)
        edges[iattr].extend(list(ie))

    if use_parallel:
        # collect edges using master edge number
        # and gather it to a node.
        edgesc = {}
        ld, md = mesh.shared_info        
        for j in range(1, nattr+1):
            if j in edges:
               data = np.array(edges[j], dtype=int)
               for key in ld.keys():
                   mid, g_in_master = key
                   if mid == myid: continue
                   for le, me in zip(ld[key][1], md[key][1]):
                       iii =  np.where(data == le)[0]
                       data[iii] = me
            else:
                data = np.atleast_1d([]).astype(int)
            data = gather_vector(data, root = j % nprc)
            if data is not None: edgesc[j] = data
        edges = edgesc

    # for each iattr real edge appears only once
    for key in edges.keys():
        seen = defaultdict(int)
        for x in edges[key]: seen[x] +=1
        edges[key] = [k for k in seen if seen[k] == 1]
    
    #nicePrint('Num edges',
    nedge = sum([len(edges[k]) for k in edges])
    if nedge != 0:
        N = np.hstack([np.zeros(len(edges[k]), dtype=int)+k-1 for k in edges.keys()])
        M = np.hstack([np.array(edges[k]) for k in edges.keys()])
    else:
        N = np.atleast_1d([]).astype(int)
        M = np.atleast_1d([]).astype(int)        
    M = M.astype(int, copy = False)
    N = N.astype(int, copy = False)    

    if use_parallel:
        # send attribute to owner of edges
        for j in range(nprc):
            idx = np.logical_and(M >= offset[j], M < offset[j+1])
            Mpart = M[idx]
            Npart = N[idx]
            Mpart = gather_vector(Mpart, root = j)
            Npart = gather_vector(Npart, root = j)
            if j==myid: M2, N2 = Mpart, Npart
        M, N = M2, N2
        
    #nicePrint('unique edge', len(np.unique(M)))
    #nicePrint('N', len(N))    
    data = M*0+1
    table1 = coo_matrix((data, (M, N)), shape = (ne, nattr), dtype = int)

    csr = table1.tocsr()

    #embeded surface only touches to one iattr
    idx = np.where(np.diff(csr.indptr) >= 1)[0]
    csr = csr[idx, :]    

    # this is true bdr edges.
    bb_edges = defaultdict(list)
    indptr = csr.indptr; indices = csr.indices
    
    for i in range(csr.shape[0]):
        idxs = tuple(sorted(indices[indptr[i]:indptr[i+1]]+1))
        bb_edges[idxs].append(idx[i])
    bb_edges.default_factory = None

    # sort keys (= attribute set) 
    keys = bb_edges.keys()
    if use_parallel:  keys = comm.gather(keys)
    sorted_key = None
    if myid == 0:
        sorted_key = list(set(sum(keys, [])))
        sorted_key.sort(key = lambda x:(len(x), x))

    if use_parallel: sorted_key = comm.bcast(sorted_key, root=0)
    
    bb_edgess = OrderedDict()
    for k in sorted_key:
        if k in bb_edges:
            bb_edgess[k] = bb_edges[k]
        else:
            bb_edgess[k] = []  # in parallel, put empty so that key order is kept
    bb_edges = bb_edgess

    '''
    res = []
    for key in sorted_key:
        tmp = allgather(len(bb_edges[key]))
        if myid == 0:
            res.append((key, sum(tmp)))
    if myid == 0: print res
    '''
    # at this point each node has its own edges populated in bb_edges (no shadow)
    ivert = {}
    for k in sorted_key:
        if len(bb_edges[k])>0:
            ivert[k] = np.hstack([mesh.GetEdgeVertices(i-myoffset)+ myoffsetv
                              for i in np.unique(bb_edges[k])]).astype(int)
        else:
            ivert[k] = np.atleast_1d([]).astype(int)
            
    if use_parallel:
        # convert shadow vertex to real
        for k in sorted_key:
            data = ivert[k]
            for key in ld:
                if key[0] == myid: continue
                for le, me in zip(ld[key][0], md[key][0]):
                   iii =  np.where(data == le)[0]
                   data[iii] = me
            ivert[k] = data
        ivertc = {}
        for j, k in enumerate(sorted_key):
            data = gather_vector(ivert[k], root = j % nprc)
            if data is not None:
                ivertc[k] = data
        ivert = ivertc


    corners = {}
    for key in ivert:
        seen = defaultdict(int)
        for iiv in ivert[key]:
            seen[iiv] += 1
        corners[key] = [kk for kk in seen if seen[kk]==1]

    u = np.unique(np.hstack([corners[key]
                  for key in corners])).astype(int, copy=False)

    # collect vertex on each node and gather to node 0
    u_own = u
    if use_parallel:
         u = np.unique(allgather_vector(u))
         u_own = u.copy()
         for key in ld:
             if key[0] == myid: continue
             for lv, mv in zip(ld[key][0], md[key][0]):
                 iii =  np.where(u == mv)[0]
                 u[iii] = lv
         idx = np.logical_and(u >= offsetv[myid], u < offsetv[myid+1])         
         u= u[idx]  # u include shared vertex
         idx = np.logical_and(u_own >= offsetv[myid], u_own < offsetv[myid+1])
         u_own = u_own[idx] # u_own is only owned vertex
         
    #nicePrint('u_own',mesh.GetNV(),",",  u_own)
    if len(u_own) > 0:
        vtx = np.vstack([mesh.GetVertexArray(i - myoffsetv) for i in u_own])
    else:
        vtx = np.atleast_1d([]).astype(int).reshape(-1, sdim)
    if use_parallel:
        u_own = gather_vector(u_own)
        vtx   = gather_vector(vtx.flatten())

    # sort vertex  
    if myid == 0:
        vtx = vtx.reshape(-1, sdim)
        #print('vtx shape', vtx.shape)
        tmp = sorted([(k, tuple(x)) for k, x in enumerate(vtx)], key=lambda x:x[1])
        vtx = np.vstack([x[1] for x in tmp])
        u_own = np.hstack([[u_own[x[0]] for x in tmp]]).astype(int)
        ivert=np.arange(len(vtx), dtype=int)+1

    if use_parallel:
        #if myid != 0:
        #    u_own = None; vtx = None
        u_own = comm.bcast(u_own)
        ivert=np.arange(len(u_own), dtype=int)+1
        for key in ld:
            if key[0] == myid: continue
            for lv, mv in zip(ld[key][0], md[key][0]):
                iii =  np.where(u_own == mv)[0]
                u_own[iii] = lv
        idx = np.logical_and(u_own >= offsetv[myid], u_own < offsetv[myid+1])
        u_own = u_own[idx]
        vtx = comm.bcast(vtx)
        vtx = comm.bcast(vtx)[idx.flatten()]
        ivert = ivert[idx]                          

    vert2vert = {iv: iu-myoffsetv for iv, iu in zip(ivert, u_own)}
    #nicePrint('vert2vert', vert2vert)
    
    # mapping line index to vertex index (not MFFEM vertex id)
    line2vert = {}
    #nicePrint(corners)
    for j, key in enumerate(sorted_key):
        data = corners[key] if key in corners else None
        data = comm.bcast(data, root = j % nprc)
        data = np.array(data, dtype=int)
        if use_parallel:                     
            for key2 in ld:
                if key2[0] == myid: continue
                for lv, mv in zip(ld[key2][0], md[key2][0]):
                    iii =  np.where(data == mv)[0]
                    data[iii] = lv
            idx = np.logical_and(data >= offsetv[myid],
                                 data < offsetv[myid+1])
            data = data[idx]
        data = list(data - myoffsetv)
        line2vert[j+1] = [k for k in vert2vert
                          if vert2vert[k] in data]

    # finish-up edge data
    if use_parallel:
        # distribute edges, convert (add) from master to local
        # number
        for attr_set in bb_edges:
            data = sum(allgather(bb_edges[attr_set]), [])
            data = np.array(data, dtype=int)
            for key in ld:
                if key[0] == myid: continue
                for le, me in zip(ld[key][1], md[key][1]):
                   iii =  np.where(data == me)[0]
                   data[iii] = le
            
            idx = np.logical_and(data >= offset[myid], data < offset[myid+1])
            data = data[idx]
            bb_edges[attr_set] = list(data - myoffset)

        attrs = edges.keys()            
        attrsa = np.unique(sum(allgather(attrs), []))

        for a in attrsa:
            if a in attrs:
                data = np.array(edges[a], dtype=int)
            else:
                data = np.atleast_1d([]).astype(int)
            data = allgather_vector(data)
            
            for key in ld:
                if key[0] == myid: continue                
                for le, me in zip(ld[key][1], md[key][1]):
                   iii =  np.where(data == me)[0]
                   data[iii] = le
            idx = np.logical_and(data >= offset[myid], data < offset[myid+1])
            data = data[idx]
            edges[a] = list(data - myoffset)

    line2edge = {}
    for k, attr_set in enumerate(sorted_key):
        if attr_set in bb_edges:
            line2edge[k+1] = bb_edges[attr_set]
        else:
            line2edge[k+1] = []

    '''
    # debug find true (non-shadow) edges
    line2edge_true = {}
    for k, attr_set in enumerate(sorted_key):
        if attr_set in bb_edges:
            data = np.array(bb_edges[attr_set], dtype=int)
            for key in ld:
                if key[0] == myid: continue                                
                iii = np.in1d(data+myoffset, ld[key][1], invert = True)
                data = data[iii]
            line2edge_true[k+1] = data
        else:
            line2edge_true[k+1] = []
    nicePrint([sum(allgather(len(line2edge_true[key]))) for key in line2edge])
    '''
    surf2line = {k+1:[] for k in range(nattr)}
    for k, attr_set in enumerate(sorted_key):
        for a in attr_set: surf2line[a].append(k+1)


    nicePrint('line2vert', line2vert)
    #nicePrint('vert2vert', vert2vert)    
    '''
    comm.Barrier()
    for j in range(nprc):
        if j == myid:
            for k in line2vert:
#               print myid, k, [mesh.GetVertexArray(vert2vert[x]) for x in line2vert[k]]
            for x in vert2vert:
                 print myid, x, mesh.GetVertexArray(vert2vert[x])
        comm.Barrier()        
    '''
    '''
    edges : face index -> edge elements
    bb_edges : set of face index -> edge elements
    '''
    return surf2line, line2vert, line2edge, vert2vert
