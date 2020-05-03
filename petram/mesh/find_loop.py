'''
   find_loop_par

   (parallel) find edge index surrouding a face

'''
import numpy as np
from collections import defaultdict

def find_loop_ser(mesh, *face):
    '''
    find_loop_ser(mesh, 1)                 # loop around boundary index = 1
    find_loop_ser(mesh, [1, 2, 3])           # loop around boundary made by union of index = 1,2,3
    find_loop_ser(mesh, [1,2,3], [4, 5]) # loop made by two set of boundaries 
                                              # (1,2,3) and (4, 5)

    '''
    if len(face) == 1:
        face = face[0]
        faces = np.atleast_1d(face)
        battrs = mesh.GetBdrAttributeArray()
        bidx = np.where(np.in1d(battrs, faces))[0]
        edges = [mesh.GetBdrElementEdges(i) for i in bidx]
        iedges = sum([e1[0] for e1 in edges], [])
        dirs =  sum([e1[1] for e1 in edges], [])
    else:
        battrs = mesh.GetBdrAttributeArray()
        
        bidx1 = np.where(np.in1d(battrs, face[0]))[0]
        edges1 = np.unique(np.hstack([mesh.GetBdrElementEdges(i)[0] for i in bidx1]))
        bidx2 = np.where(np.in1d(battrs, face[1]))[0]
        edges2 = np.unique(np.hstack([mesh.GetBdrElementEdges(i)[0] for i in bidx2]))

        iedges = np.intersect1d(edges1, edges2)

        #dirs = {}
        #for i in bidx2:
        #     print(mesh.GetBdrElementEdges(i))
        #     for ei, dir in zip(*mesh.GetBdrElementEdges(i)):
        #         if ei in dirs and  dirs[ei] != dir:
        #             print("orientation is not obtained by this way")
        #         dirs[ei] = dir
        #dirs = [dirs[i] for i in iedges]

    seens = defaultdict(int)
    seendirs  = defaultdict(int)
    
    for k, ie in enumerate(iedges):
        seens[ie] = seens[ie] + 1
        seendirs[ie] = dirs[k]
    seens.default_factory = None
     
    idx = []
    signs = []
    for k in seens.keys():
        if seens[k] == 1:
            idx.append(k)
            signs.append(seendirs[k])

    ll = {tuple(mesh.GetEdgeVertices(i)):i for i in idx}

    from petram.helper.geom import connect_pairs

    loop = connect_pairs(list(ll))

    dirs = {}
    for i in range(len(loop)-1):
        if (loop[i], loop[i+1]) in ll: dirs[(loop[i], loop[i+1])] = 1
        if (loop[i+1], loop[i]) in ll: dirs[(loop[i+1], loop[i])] = -1

    signs = [dirs[key] for key in ll.keys()]
    idx = [ll[key] for key in ll.keys()]    
        
    print("here",idx, signs)
    return idx, signs
    
def find_loop_par(pmesh, *face):
    '''
    find_loop_ser(mesh, 1)          # loop around boundary index = 1
    find_loop_ser(mesh, [1, 2, 3])    # loop around boundary made by union of index = 1,2,3
    '''
    
    import mfem.par as mfem
    from mpi4py import MPI

    myid = MPI.COMM_WORLD.rank
    nprc = MPI.COMM_WORLD.size   
    comm  = MPI.COMM_WORLD

    if nprc == 1:
        return find_loop_ser(pmesh, face)

    from mfem.common.mpi_debug import nicePrint
    from petram.helper.mpi_recipes import allgather, allgather_vector, gather_vector    

    battrs = pmesh.GetBdrAttributeArray()

    face = face[0]
    faces = np.atleast_1d(face)
    bidx = np.where(np.in1d(battrs,faces))[0]
    
    offset_e = np.hstack([0, np.cumsum(allgather(pmesh.GetNEdges()))])
    
    edges = [pmesh.GetBdrElementEdges(i) for i in bidx]
    iedges = np.array(sum([e1[0] for e1 in edges], []), dtype=int) + offset_e[myid]
    dirs =  np.array(sum([e1[1] for e1 in edges], []), dtype=int)

    from petram.mesh.mesh_utils import distribute_shared_entity

    shared_info = distribute_shared_entity(pmesh)

    keys = shared_info[0].keys()
    local_edges = np.hstack([shared_info[0][key][1] for key in keys])
    global_edges = np.hstack([shared_info[1][key][1] for key in keys])

    own_edges = []
    for k, ie in enumerate(iedges):
        iii = np.where(local_edges == ie)[0]
        if len(iii) != 0:
            if ie == global_edges[iii[0]]:
               own_edges.append(ie)          
            iedges[k] = global_edges[iii[0]]
        else:
            own_edges.append(ie)

    #nicePrint("iedges", iedges)
    iedges_all = allgather_vector(iedges)
    dirs = allgather_vector(dirs)

    seens = defaultdict(int)
    seendirs  = defaultdict(int)        
    for k, ie in enumerate(iedges_all):
        seens[ie] = seens[ie] + 1
        seendirs[ie] = dirs[k]
    seens.default_factory = None

    idx = []
    signs = []
    for k in seens.keys():
        if seens[k] == 1:
            idx.append(k)
            signs.append(seendirs[k])
    iedges_g = np.array(idx)
    # here idx is global numbering
    
    #nicePrint("global_index", idx, signs)
    #nicePrint("own_edges", own_edges)
    
    iedges_l = []
    signs_l = []
    for k, ie in enumerate(iedges_g):
        iii = np.where(own_edges == ie)[0]
        if len(iii) != 0:
            iedges_l.append(ie)
            signs_l.append(signs[k])
    iedges_l = np.array(iedges_l) - offset_e[myid]
    signs_l = np.array(signs_l)

    #nicePrint("local_index", iedges_l, signs_l)

    return iedges_l, signs_l
