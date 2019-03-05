'''
   find_loop_par

   (parallel) find edge index surrouding a face

'''
def find_loop_par(pmesh, face):
    import mfem.par as mfem
    from mpi4py import MPI

    myid = MPI.COMM_WORLD.rank
    nprc = MPI.COMM_WORLD.size   
    comm  = MPI.COMM_WORLD

    from mfem.common.mpi_debug import nicePrint
    from petram.helper.mpi_recipes import allgather, allgather_vector, gather_vector    
    import numpy as np

    battrs = pmesh.GetBdrAttributeArray()
    bidx = np.where(battrs == face)[0]
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

    nicePrint("iedges", iedges)
    iedges_all = allgather_vector(iedges)
    dirs = allgather_vector(dirs)

    from collections import defaultdict
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
    nicePrint("global_index", idx, signs)
    nicePrint("own_edges", own_edges)
    iedges_l = []
    signs_l = []
    for k, ie in enumerate(iedges_g):
        iii = np.where(own_edges == ie)[0]
        if len(iii) != 0:
            iedges_l.append(ie)
            signs_l.append(signs[k])
    iedges_l = np.array(iedges_l) - offset_e[myid]
    signs_l = np.array(signs_l)

    nicePrint("local_index", iedges_l, signs_l)

    return iedges_l, signs_l
