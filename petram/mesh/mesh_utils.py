import os
import numpy as np
from scipy.sparse import coo_matrix
from collections import OrderedDict, defaultdict
from petram.helper.global_named_list import GlobalNamedList

debug = False

import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('mesh utils')

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
    MFEM3 = hasattr(pmesh, 'GroupFace')
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
    def GroupTriangle(j, iv):
        face = intp(); o = intp()
        pmesh.GroupTriangle(j, iv, face, o)
        return face.value()
    def GroupQuadrilateral(j, iv):
        face = intp(); o = intp()
        pmesh.GroupQuadrilateral(j, iv, face, o)
        return face.value()
     
    for j in range(ng):
        if j == 0: continue
        nv = pmesh.GroupNVertices(j)
        sv = np.array([pmesh.GroupVertex(j, iv) for iv in range(nv)])
        ne = pmesh.GroupNEdges(j)
        se = np.array([GroupEdge(j, iv) for iv in range(ne)])
        
        if MFEM3:        
            nf = pmesh.GroupNFaces(j)
            sf = np.array([GroupFace(j, iv) for iv in range(nf)])
        else:
            nt = pmesh.GroupNTriangles(j)
            nq = pmesh.GroupNQuadrilaterals(j)
            sf = np.array([GroupTriangle(j, iv) for iv in range(nt)] +
                          [GroupQuadrilateral(j, iv) for iv in range(nq)])

        data = (sv + offset_v[myid],
                se + offset_e[myid],
                sf + offset_f[myid])
            
            
        local_data[(pmesh.gtopo.GetGroupMasterRank(j),
                    pmesh.gtopo.GetGroupMasterGroup(j))] = data

        if pmesh.gtopo.IAmMaster(j):
            master_entry.append((myid, j,))
            '''
            mv = sv + offset_v[myid]
            me = se + offset_e[myid]
            mf = sf + offset_f[myid]
            data = (mv, me, mf)
            '''
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


def bdr_loop(mesh):
    use_parallel = hasattr(mesh, "GroupNVertices")
    
    ## fill line/surface loop info
    doms = np.unique(mesh.GetAttributeArray()).astype(int, copy=False)
    attrs =  mesh.GetBdrAttributeArray()
    
    if use_parallel:
        from petram.helper.mpi_recipes import allgather_vector
        
        doms = allgather_vector(doms).astype(int, copy=False)
        doms = np.unique(doms)
        
    loop= {k:[] for k in doms}

    for ibdr in range(mesh.GetNBE()):
        idx = mesh.GetBdrElementEdgeIndex(ibdr)
        elem1, elem2 = mesh.GetFaceElements(idx)
        if elem1 != -1:
            i = mesh.GetAttribute(elem1)
            loop[i].append(attrs[ibdr])
        if elem2 != -1:
            i = mesh.GetAttribute(elem2)
            loop[i].append(attrs[ibdr])
    for k in loop:
        loop[k] = list(set(loop[k]))

    return loop
    
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
        nprc = 1
    ndim =  mesh.Dimension()
    sdim =  mesh.SpaceDimension()    
    ne = mesh.GetNEdges()
    assert ndim == 3, "find_edge_corner is for 3D mesh"

    # 3D mesh
    get_edges = mesh.GetBdrElementEdges
    get_attr  = mesh.GetBdrAttribute
    iattr= mesh.GetBdrAttributeArray()  # min of this array is 1
    nattr = 0 if iattr.size == 0 else np.max(iattr)        
    nb = mesh.GetNBE()
    if mesh.GetNBE() == 0 and nprc == 1:    
        return {}, {}, {}, {}
    

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
        myoffset  = np.array(0, dtype=int)
        myoffsetf = np.array(0, dtype=int)
        myoffsetv = np.array(0, dtype=int)

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
    keys = list(bb_edges)
    if use_parallel:
        keys = comm.gather(keys)
        if myid == 0: keys = sum(keys, [])        
    sorted_key = None
    if myid == 0:
        sorted_key = list(set(keys))
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

    if len(corners) == 0:
       u = np.atleast_1d([]).astype(int)
    else:
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
        vtx = np.atleast_1d([]).reshape(-1, sdim)
    if use_parallel:
        u_own = gather_vector(u_own)
        vtx   = gather_vector(vtx.flatten())

    # sort vertex  
    if myid == 0:
        vtx = vtx.reshape(-1, sdim)
        #print('vtx shape', vtx.shape)
        tmp = sorted([(k, tuple(x)) for k, x in enumerate(vtx)], key=lambda x:x[1])
        if len(tmp) > 0:
            vtx = np.vstack([x[1] for x in tmp])
            u_own = np.hstack([[u_own[x[0]] for x in tmp]]).astype(int)
            ivert=np.arange(len(vtx), dtype=int)+1
        else:
            vtx = np.atleast_1d([]).astype(float)
            u_own = np.atleast_1d([]).astype(int)
            u_own = np.atleast_1d([]).astype(int)
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
        ivert = ivert[idx]
        #vtx = comm.bcast(vtx)
        #vtx = comm.bcast(vtx)[idx.flatten()]


    vert2vert = {iv: iu-myoffsetv for iv, iu in zip(ivert, u_own)}
    #nicePrint('vert2vert', vert2vert)

    vv_values = np.array([x[1] for x in vert2vert.items()])
    vv_keys = np.array([x[0] for x in vert2vert.items()])
    
    # mapping line index to vertex index (not MFFEM vertex id)
    line2vert = {}

    #nicePrint(corners)
    for j, key in enumerate(sorted_key):
        data = corners[key] if key in corners else None
        if use_parallel:
            data = comm.bcast(data, root = j % nprc)
            data = np.array(data, dtype=int)            
            for key2 in ld:
                if key2[0] == myid: continue
                for lv, mv in zip(ld[key2][0], md[key2][0]):
                    iii =  np.where(data == mv)[0]
                    data[iii] = lv
            idx = np.logical_and(data >= offsetv[myid],
                                 data < offsetv[myid+1])
            data = data[idx]
        else:
            data = np.array(data, dtype=int)                        
        data = list(data - myoffsetv)

        line2vert[j+1] = list(vv_keys[np.in1d(vv_values, data)])

        '''
        (this was origial very slow)
        line2vert[j+1] = [k for k in vert2vert
                          if vert2vert[k] in data]
        '''

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

        attrs = list(edges)
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

    if debug:
        g = GlobalNamedList(line2vert)
        g.sharekeys()
        gg = g.gather(nprc, overwrite = False).unique()
        if myid==0: print("debug (gathered line2vert)", gg)
    
    return surf2line, line2vert, line2edge, vert2vert

def find_corner(mesh):
    '''
    For 2D geometry
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
        nprc = 1
        
    ndim =  mesh.Dimension()
    sdim =  mesh.SpaceDimension()    
    ne = mesh.GetNEdges()
    assert ndim == 2, "find_edge_corner is for 3D mesh"

    get_edges = mesh.GetElementEdges
    get_attr  = mesh.GetAttribute
    iattr= mesh.GetAttributeArray()     # min of this array is 1
    nattr = 0 if iattr.size == 0 else np.max(iattr)            
    nb = mesh.GetNE()
    nbe = mesh.GetNBE()
    if use_parallel:
        nbe = sum(allgather(nbe))
    if nbe == 0:    
        return {}, {}, {}

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
        myoffset  = np.array(0, dtype=int)
        myoffsetf = np.array(0, dtype=int)
        myoffsetv = np.array(0, dtype=int)
        
    if mesh.GetNBE() == 0: # some parallel node may have zero boundary
        battrs =  []
        iedges = np.array([], dtype=int)
    else:
        battrs =  mesh.GetBdrAttributeArray()
        iedges = np.hstack([mesh.GetBdrElementEdgeIndex(ibdr) for ibdr in
                        range(mesh.GetNBE())]).astype(int, copy=False)
    
    line2edge = GlobalNamedList()
    line2edge.setlists(battrs, iedges)

    if use_parallel:
        ld, md = mesh.shared_info                
        iedges = iedges+myoffset

    if use_parallel:
        for key2 in ld:
            if key2[0] == myid: continue
            iii = np.in1d(iedges, ld[key2][1], invert = True)
            if len(iii) == 0: continue
            iedges = iedges[iii]
            battrs = battrs[iii]

    line2realedge = GlobalNamedList()            
    line2realedge.setlists(battrs, iedges)
    
    line2realvert = GlobalNamedList()
    for key in line2realedge:
        data = np.hstack([mesh.GetEdgeVertices(i-myoffset)+ myoffsetv
                           for i in line2realedge[key]])
        if use_parallel:        
            for key2 in ld:
                if key2[0] == myid: continue                
                for lv, mv in zip(ld[key2][0], md[key2][0]):
                   iii =  np.where(data == lv)[0]
                   data[iii] = mv
        line2realvert[key] = data

    line2realvert.sharekeys().gather(nprc, distribute=True)

    corners = GlobalNamedList()
    for key in line2realvert:
        seen = defaultdict(int)
        for iiv in line2realvert[key]:
            seen[iiv] += 1
        corners[key] = [kk for kk in seen if seen[kk]==1]

    sorted_key = corners.sharekeys().globalkeys
    
    corners.allgather()
    u_own = np.unique(np.hstack([corners[key] for key in corners]).astype(int, copy=False))

    if use_parallel:
        idx = np.logical_and(u_own >= offsetv[myid], u_own < offsetv[myid+1])
        u_own = u_own[idx]

    if len(u_own) > 0:
        vtx = np.hstack([mesh.GetVertexArray(i - myoffsetv) for i in u_own])
    else:
        vtx = np.atleast_1d([])
    if use_parallel:
        vtx = gather_vector(vtx)
        u_own = gather_vector(u_own)

    # sort vertex  
    if myid == 0:
        vtx = vtx.reshape(-1, sdim)
        tmp = sorted([(k, tuple(x)) for k, x in enumerate(vtx)], key=lambda x:x[1])
        if len(tmp) > 0:
            vtx = np.vstack([x[1] for x in tmp])
            u_own = np.hstack([[u_own[x[0]] for x in tmp]]).astype(int)
            ivert=np.arange(len(vtx), dtype=int)+1
        else:
            u_own = np.atleast_1d([]).astype(int)
            ivert = np.atleast_1d([]).astype(int)

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
    
    # mapping line index to vertex index (not MFFEM vertex id)
    line2vert = {}
    #nicePrint(corners)
    corners.bcast(nprc, distributed = True)
    for j, key in enumerate(sorted_key):
        data = corners[key]
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
   
    if debug:
        g = GlobalNamedList(line2vert)
        g.sharekeys()
        gg = g.gather(nprc, overwrite = False).unique()
        if myid==0: print(gg)
        for i in range(nprc):
            if use_parallel:comm.barrier()
            if myid == i:
                for k in vert2vert:
                    print(myid, k, mesh.GetVertexArray(vert2vert[k]))
            if use_parallel:comm.barrier()                  
    return line2vert, line2edge, vert2vert

def get_extended_connectivity(mesh):
    
    ndim = mesh.Dimension()
    if ndim == 3:
        v2s = bdr_loop(mesh)
        s2l, l2v, l2e, v2v = find_edge_corner(mesh)        

    elif ndim == 2:
        v2s = None
        s2l = bdr_loop(mesh)
        l2v, l2e, v2v = find_corner(mesh)
        
    else:
        lp = bdr_loop(mesh)        
        v2s = None; s2l = None
        attrs = mesh.GetAttributeArray()        
        l2e = {k:list(np.where(attrs==k)[0]) for k in lp.keys()}
        l2v = bdr_loop(mesh)
        v = np.unique(np.hstack([lp[k] for k in lp.keys()]))
        battrs = mesh.GetBdrAttributeArray()
        v2v = {battr:mesh.GetBdrElementVertices(k)[0]
               for k, battr in enumerate(battrs)}
        
    from mfem.common.mpi_debug import nicePrint, niceCall                
    #nicePrint('s2l', s2l)
    mesh.extended_connectivity = {}
    me = mesh.extended_connectivity
    me['vol2surf']  = v2s
    me['surf2line'] = s2l
    me['line2edge'] = l2e
    me['line2vert'] = l2v
    me['vert2vert'] = v2v

def get_reverse_connectivity(mesh):
    def reverse_dict(d):
        dd = defaultdict(list)        
        for k in d:
            for v in d[k]:
                dd[v].append(k)
        return dict(dd)
    
    me = mesh.extended_connectivity
    if me['vol2surf'] is not None:
        me['surf2vol'] = reverse_dict(me['vol2surf'])
    if me['surf2line'] is not None:
        me['line2surf'] = reverse_dict(me['surf2line'])
    if me['line2vert'] is not None:
        me['vert2line'] = reverse_dict(me['line2vert'])        

def vol2line(v2s, s2l):
    v2l =  {}
    for v in v2s:
        l = np.unique(np.hstack([s2l[s] for s in v2s[v]]))
        l = l.astype(int)
        v2l[v] = list(l)
    return v2l

def line2vol(v2l):
    l2v = defaultdict(list)
    for v in v2l:
        for l in v2l[v]: l2v[l].append(v)
    return dict(l2v)

def line2surf(s2l):
    l2s = defaultdict(list)
    for s in s2l:
        for l in s2l[s]: l2s[l].append(s)
    return dict(l2s)
    
def populate_plotdata(mesh, table, cells, cell_data):
    from petram.mesh.mesh_utils import  get_extended_connectivity
    if not hasattr(mesh, 'extended_connectivity'):
        get_extended_connectivity(mesh)

    ndim = mesh.Dimension()
    ec = mesh.extended_connectivity
   
    cell_data['line'] = {}
    cell_data['vertex'] = {}
    l2e = ec['line2edge']
    v2v = ec['vert2vert']

    kedge = []

    if mesh.Dimension() >= 2:
        method = mesh.GetEdgeVertices
    else:
        method = mesh.GetElementVertices
        
    if len(l2e) > 0:
        #kedge = np.array(sum([[key]*len(l2e[key]) for key in l2e], [])).astype(int)
        kedge = np.hstack([[key]*len(l2e[key]) for key in l2e]).astype(int, copy=False)
        iverts = np.vstack([method(ie)
                        for key in l2e for ie in l2e[key]])
    else:
        iverts = np.atleast_1d([]).astype(int)
    cells['line'] = table[iverts]
    cell_data['line']['physical'] = np.array(kedge)

    kvert = np.array([key for key in v2v]).astype(int, copy=False)
    iverts = np.array([v2v[key] for key in v2v]).astype(int, copy=False)

    cells['vertex'] = table[iverts]    
    cell_data['vertex']['physical'] = kvert
    if ndim == 3:
        l_s_loop = [ec['surf2line'], ec['vol2surf']]
    elif ndim == 2:
        l_s_loop = [ec['surf2line'], None]
    else:
        l_s_loop = [None, None]

    iedge2bb = None # is it used?
    return l_s_loop

#
#  FaceOf, EdgeOf, PointOf
#
#  Note: don't convert kwargs to mesh=None, since it provides are
#        safe-separation between args and kwargs.
#
def FaceOf(*args, **kwargs):
    mesh=kwargs.pop('mesh', None)
    vols = list(np.atleast_1d(args).flatten())
    ec = mesh.extended_connectivity['vol2surf']
    if ec is None: return []
    return list(set(sum([list(ec[k]) for k in vols],[])))

def EdgeOf(*args, **kwargs):
    mesh=kwargs.pop('mesh', None)    
    faces = list(np.atleast_1d(args).flatten())
    ec = mesh.extended_connectivity['surf2line']
    if ec is None: return []    
    return list(set(sum([list(ec[k]) for k in faces],[])))

def PointOf(*args, **kwargs):
    mesh=kwargs.pop('mesh', None)        
    edges = list(np.atleast_1d(args).flatten())
    ec = mesh.extended_connectivity['line2vert']
    if ec is None: return []    
    return list(set(sum([list(ec[k]) for k in edges],[])))

def NeighborFaceOf(*args, **kwargs):
    # return Neighboring Face
    mesh=kwargs.pop('mesh', None)
    faces = list(np.atleast_1d(args).flatten())
    s2l = mesh.extended_connectivity['surf2line']
    e1 = EdgeOf(faces, mesh = mesh)
    neighbor = [k for k in s2l if len(np.intersect1d(EdgeOf(k, mesh=mesh),e1))!=0]
    neighbor = [k for k in neighbor if not k in faces]
    return neighbor

def CommonEdgeOf(faces1, faces2, **kwargs):
    # common edge of face group 1 and 2
    mesh=kwargs.pop('mesh', None)
    f1 = EdgeOf(faces1, mesh=mesh)
    f2 = EdgeOf(faces2, mesh=mesh)
    return list(np.intersect1d(f1,f2))
