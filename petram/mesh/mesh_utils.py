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
    ne = mesh.GetNEdges()        

    if ndim == 2:
        # 2D mesh
        get_edges = mesh.GetElementEdges
        get_attr  = mesh.GetAttribute
        iattr= mesh.GetAttributeArray()     # min of this array is 1
        nattr = np.max(iattr)
        nb = mesh.GetNE()        
    else:
        # 3D mesh
        get_edges = mesh.GetBdrElementEdges
        get_attr  = mesh.GetBdrAttribute
        iattr= mesh.GetBdrAttributeArray()  # min of this array is 1
        nattr = np.max(iattr)        
        nb = mesh.GetNBE()

    if use_parallel:
        offset = np.hstack([0, np.cumsum(allgather(mesh.GetNEdges()))])
        offsetf = np.hstack([0, np.cumsum(allgather(mesh.GetNFaces()))])        
        myoffset = offset[myid]
        myoffsetf = offsetf[myid]        
        nattr = max(allgather(nattr))
        ne = sum(allgather(mesh.GetNEdges()))
    else:
        myoffset = 0
        
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
    
    #nicePrint('Num edges', sum([len(edges[k]) for k in edges]))
    N = np.hstack([np.zeros(len(edges[k]), dtype=int)+k-1 for k in edges.keys()])
    M = np.hstack([np.array(edges[k]) for k in edges.keys()])
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
    keys = bb_edges.keys()

    if use_parallel:  keys = comm.gather(keys)

    sorted_key = None
    if myid == 0:
        keys = sum(keys, [])
        d = {key: np.sum([k+i*nattr for i, k in enumerate(key)])
             for key in keys}
        sorted_key =  [item[0] for item in sorted(d.items(), key = lambda x:x[1])]

    if use_parallel: sorted_key = comm.bcast(sorted_key, root=0)
    
    bb_edgess = OrderedDict()
    for k in sorted_key:
        if k in bb_edges:
            bb_edgess[k] = bb_edges[k]
        else:
            bb_edgess[k] = []  # in parallel, put empty so that key order is kept
    bb_edges = bb_edgess

    #nicePrint([(key,len(bb_edges[key])) for key in sorted_key])

    '''
    res = []
    for key in sorted_key:
        tmp = allgather(len(bb_edges[key]))
        if myid == 0:
            res.append((key, sum(tmp)))
    if myid == 0: print res
    '''
    ## (to do) on the edge onwer node, need to collect vertex too
    ## then renumber it to global number and find corner...
    

    if use_parallel:
        # distribute edges, convert (add) from master to local
        # number
        for attr_set in bb_edges:
            data = sum(allgather(bb_edges[attr_set]), [])
            data = np.array(data, dtype=int)
            for key in ld:
                for le, me in zip(ld[key][1], md[key][1]):
                   iii =  np.where(data == me)[0]
                   data[iii] = le
            
            idx = np.logical_and(data >= offset[j], data < offset[j+1])
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
                for le, me in zip(ld[key][1], md[key][1]):
                   iii =  np.where(data == me)[0]
                   data[iii] = le
            idx = np.logical_and(data >= offset[j], data < offset[j+1])
            data = data[idx]
            bb_edges[a] = data - myoffset

    line2edge = {}
    for k, attr_set in enumerate(sorted_key):
        if attr_set in bb_edges:
            line2edge[k+1] = bb_edges[attr_set]
            
    surf2line = {k+1:[] for k in range(nattr)}
    for k, attr_set in enumerate(sorted_key):
        for a in attr_set: surf2line[a].append(k+1)
        
    '''
    edges : face index -> edge elements
    bb_edges : set of face index -> edge elements
    '''
    return surf2line, line2edge
