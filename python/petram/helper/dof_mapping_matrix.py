import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix

import petram
dprint1, dprint2, dprint3 = petram.debug.init_dprints('dof_mapping_matrix')

from petram.helper.find_dof_map_nd import find_dof_map_nd
from petram.helper.find_dof_map_h1 import find_dof_map_h1

# default mapper
def def_map_u(*args):
    return args[0]   # x -> u
def def_map_v(*args):
    return args[1]   # y -> v

from petram.mfem_config import use_parallel

if use_parallel:
   from petram.helper.mpi_recipes import allgather_vector
   import mfem.par as mfem
else:
   import mfem.ser as mfem
   num_proc = 1
   myid = 0

def dof_mapping_matrix(src,  dst,  fes, tdof, engine=None, dphase=0.0,
                       map_to_u = def_map_u, 
                       map_to_v = def_map_v,
                       smap_to_u = None,
                       smap_to_v = None,
                       tol = 1e-7):
    '''
     map: destinatiom mapping 
     smap: source mapping
    '''
    fec_name = fes.FEColl().Name()

    if fec_name.startswith('ND'):
        mapper = find_dof_map_nd
    elif fec_name.startswith('H1'):
        mapper = find_dof_map_h1
    else:
        raise NotImplementedError("mapping for " + fec_name)

    if smap_to_u is None:
        smap_to_u = map_to_u
    if smap_to_v is None:
        smap_to_v = map_to_v
 


    map = mapper(src,  dst,  map_to_u, map_to_v, smap_to_u, smap_to_v, 
                 fes, engine,  tdof, tol=tol)

    if (dphase == 0.):
        pass
    elif (dphase == 180.):
        map = -map
    else:   
        map = map.astype(complex)        
        map *= np.exp(-1j*np.pi/180*dphase)

    coo = coo_matrix(map)
    r = coo.row
    c = coo.col

    idx = range(map.shape[0])
    map.setdiag(1.0)
    for k in c :  map[k,k] = 0.0

    if use_parallel:
        start_row = fes.GetMyTDofOffset()
        end_row = fes.GetMyTDofOffset() + fes.GetTrueVSize()
        c =  allgather_vector(c)

        map = map.transpose()
        m1 = csr_matrix(map.real[start_row:end_row, :], dtype=float)
        m2 = csr_matrix(map.imag[start_row:end_row, :], dtype=float)
        m1.eliminate_zeros()
        m2.eliminate_zeros()
        from mfem.common.chypre import CHypreMat
        dprint1(m1.shape, m2.shape)
        M = CHypreMat(m1, m2).transpose()

    else:
        from petram.helper.block_matrix import convert_to_ScipyCoo        
        M = convert_to_ScipyCoo(coo_matrix(map, dtype=map.dtype))

    #idx = range(M.shape[0])
    #M.setDiag(idx, 1.0)
    #M.setDiag(c, 0.0)

    return M, r, c

