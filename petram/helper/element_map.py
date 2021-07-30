import numpy as np
import scipy

import petram.debug as debug
debug.debug_default_level = 1
dprint1, dprint2, dprint3 = debug.init_dprints('dof_map')

from petram.mfem_config import use_parallel
if use_parallel:
   from mpi4py import MPI
   comm = MPI.COMM_WORLD
   num_proc = MPI.COMM_WORLD.size
   myid     = MPI.COMM_WORLD.rank
   from petram.helper.mpi_recipes import *
   import mfem.par as mfem
   from mfem.common.mpi_debug import nicePrint
   
else:
   import mfem.ser as mfem
   num_proc = 1
   myid = 0
   def allgather(x): return [x]
   nicePrint = dprint1

'''
def map_dom2dom():


def map_dom2bdr():


def map_bdr2bdr():
'''

def find_element_ctr(mesh, idx, bdr):
    if len(idx) == 0:
        sdim = mesh.SpaceDimension()
        return np.empty(shape=(0,sdim))
    if bdr:
        m = mesh.GetBdrElementVertices
    else:
        m = mesh.GetElementVertices

    pts = np.mean(np.array([[mesh.GetVertexArray(i) for i in m(k)]
                             for k in idx]), 1)
    return pts
    

def find_element(mesh, use_bdr, attrs):
    if use_bdr:
        arr = mesh.GetBdrAttributeArray()
    else:
        arr = mesh.GetAttributeArray()
    idx = np.arange(len(arr))
        
    if attrs!='all':
        try:
            _void = attrs[0]
        except:
            attrs = [attrs]
        flag = np.in1d(arr, attrs)
        idx = idx[flag]
    return idx

def get_dombdr_flag(dim1, dim2, map_bdr):
    if dim1 == dim2:
        map_bdr1 = map_bdr
        map_bdr2 = map_bdr
    elif dim1 == dim2+1:
        map_bdr1 = map_bdr
        if not map_bdr:
            assert False, "can not map domain to a mesh with lower dim"
        else:
            map_bdr2 = False
    elif dim1 == dim2-1:
        map_bdr1 = map_bdr
        if not map_bdr:
            assert False, "can not map boundary to a mesh with higher dim"
        else:
            map_bdr2 = True
    else:
        assert False, "can not create element mapping between meshes"

    return map_bdr1, map_bdr2

def map_element(mesh1, mesh2, attr='all', map_bdr=False):
    '''
    create mapping between element
    '''

    dim1 = mesh1.Dimension()
    dim2 = mesh2.Dimension()

    map_bdr1, map_bdr2 = get_dombdr_flag(dim1, dim2, map_bdr)    
        
    el_arr1 = find_element(mesh1, map_bdr1, attr)
    el_arr2 = find_element(mesh2, map_bdr2, attr)

    ct1 = find_element_ctr(mesh1, el_arr1, map_bdr1)
    ct2 = find_element_ctr(mesh2, el_arr2, map_bdr2)    

                    
    if use_parallel:
        # share ibr2 (destination information among nodes...)
        ct1dim = ct1.shape[1] if ct1.size > 0 else 0
        ct1dim = comm.allgather(ct1dim)
        ct1 = np.atleast_2d(ct1).reshape(-1, max(ct1dim))
        ct2 = np.atleast_2d(ct2).reshape(-1, max(ct1dim))
        ct2_arr = [comm.bcast(ct2, root=i) for i in range(num_proc)]
    else:
        ct2_arr = [ct2]       
       
    from scipy.spatial import cKDTree

    dists = np.zeros(len(el_arr1))+np.inf

    nodes = np.zeros(len(el_arr1), dtype=int)
    el = np.zeros(len(el_arr1), dtype=int)-1
    num_target = 0
    for rank, ct2 in enumerate(ct2_arr):
        tree = cKDTree(ct2)
        ctr_dist, map_1_2 = tree.query(ct1)
        kk = (ctr_dist < dists)
        dists[kk] = ctr_dist[kk]
        nodes[kk] = rank
        el[kk] = map_1_2[kk]
        num_target += len(ct2)

    ## remove data whose distance is too big.
    ## need to find a threshold by gathering dist data.
    if use_parallel:
        dists_all = allgather_vector(dists)
    else:
        dists_all = dists

    sel = dists <= np.sort(dists_all)[num_target-1]
    el_arr1 = el_arr1[sel]
    el = el[sel]
    nodes = nodes[sel]
    
    el_map = {}
    for rank in range(num_proc):
        kk = (nodes == rank)
        el_map[rank] = dict(zip(el_arr1[kk], el[kk]))

    # reverse map
    rev_el_map = {}    
    if use_parallel:
        for i in range(num_proc):
            tmp = comm.gather(dict(zip(el_map[i].values(), list(el_map[i]))), root=i)
            if i == myid:
                data = tmp
                
        for rank in range(num_proc):
            rev_el_map[rank] = data[rank]
    else:
        rev_el_map[0] = dict(zip(el_map[0].values(), list(el_map[0])))
        
    '''
    format:
       el_map[mpi_rank] = {element_id in mesh1: element_id_mesh2}
         # elemnt_id_mesh1 is mapped in mpi_rank as elemet_id_mesh2
       el_mapr[mpi_rank] = {element_id_mesh2: element_id in mesh1}
         # elemnt_id_mesh2 is mapped in mpi_rank as elemet_id_mesh1
    '''
    return el_map, rev_el_map

    
