from petram.solver.mumps.hypre_to_mumps import PyZMatrix, PyDMatrix
import numpy as np
from mfem.common.chypre import *

import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('CHypre_2_PyMatrix')


'''
  [ A    B ][x]   [b]
  [        ][ ] = [ ]
  [ C    D ][l]   [c]

  B:  Vertical Vector/Matrix
  C:  Horizontal Vector/Matrix
'''

def Vec2MatH(v, complex=True): # horizontal vector
    vv =  CHypreVec2Array(v)
    if vv is None: return # other than root
    
    if complex:
       m1 = PyZMatrix(1, len(vv), len(vv))
       m1.set_rdata(vv.real)
       m1.set_idata(vv.imag)
    else:
       m1 = PyDMatrix(1, len(vv), len(vv))
       m1.set_data(vv)
    return m1

def Vec2MatV(v, complex=True):  # vertical vector
    vv =  CHypreVec2Array(v)
    if vv is None: return # other than root
    
    if complex:
       m1 = PyZMatrix(len(vv), 1, len(vv))
       m1.set_rdata(vv.real)
       m1.set_idata(vv.imag)
    else:
       m1 = PyDMatrix(len(vv), 1, len(vv))
       m1.set_data(vv)
    return m1

'''
following two are convertion to PyMatrix
'''
def Mat2CooH(m, complex = True, elimination = True): # horizontal matrix  (eliminate zero rows)
    from mpi4py import MPI
    comm     = MPI.COMM_WORLD
    myid     = MPI.COMM_WORLD.rank    

    N = m.shape[1]
    coo = CHypreMat2Coo(m)
    csr = coo.tocsr()
    num_nonzeros = np.diff(csr.indptr)
    k_nonzeros = np.where(num_nonzeros != 0)[0]
    
    if elimination : 
        rr = np.array(comm.allgather(len(k_nonzeros)))
    else:
        rr = np.array(comm.allgather(coo.shape[0]))
    rall = int(np.sum(rr))
    rstart = np.hstack((0, np.cumsum(rr)))

    if np.sum(num_nonzeros != 0) != 0:
        if elimination: csr = csr[np.where(num_nonzeros != 0)[0]]
        nnz = int(csr.getnnz())
        coo = csr.tocoo()
        if complex:
            m2 = PyZMatrix(rall, N, nnz, cmajor=2)
        else:
            m2 = PyDMatrix(rall, N, nnz, cmajor=2)            
        m2.set_irn(coo.row + rstart[myid])
        m2.set_jcn(coo.col)
        if complex:        
            m2.set_rdata(coo.data.real)        
            m2.set_idata(coo.data.imag)
        else:
            m2.set_data(coo.data.astype('float'))                
    else:
        if complex:
            m2 = PyZMatrix(rall, N,  0, cmajor=2)
        else:
            m2 = PyDMatrix(rall, N,  0, cmajor=2)
    return m2

def Mat2CooV(m, complex = True, elimination = True): # vertical offset  (eliminate zero cols)
    '''
    In order to eliminate col, it works on tranposed matrix,
    '''
    from mpi4py import MPI
    comm     = MPI.COMM_WORLD
    myid     = MPI.COMM_WORLD.rank
    
    M = m.shape[0]
    coo = CHypreMat2Coo(m.transpose())

    csr = coo.tocsr()
    
    num_nonzeros = np.diff(csr.indptr)
    k_nonzeros = np.where(num_nonzeros != 0)[0]

    if elimination : 
        rr = np.array(comm.allgather(len(k_nonzeros)))
    else:
        rr = np.array(comm.allgather(coo.shape[0]))
    rall = int(np.sum(rr))
    rstart = np.hstack((0, np.cumsum(rr)))

    if np.sum(num_nonzeros != 0) != 0:
        if elimination: csr = csr[np.where(num_nonzeros != 0)[0]]
        nnz = int(csr.getnnz())
        coo = csr.tocoo()
        if complex:
            m2 = PyZMatrix(M, rall,  nnz, cmajor=2)
        else:
            m2 = PyDMatrix(M, rall,  nnz, cmajor=2)            
        m2.set_jcn(coo.row + rstart[myid])
        m2.set_irn(coo.col)
        
        if complex:        
            m2.set_rdata(coo.data.real)        
            m2.set_idata(coo.data.imag)
        else:
            m2.set_data(coo.data.astype('float'))                
    else:
        if complex:
            m2 = PyZMatrix(M, rall,  0, cmajor=2)
        else:
            m2 = PyDMatrix(M, rall, 0, cmajor=2)
    return m2    


def NonZeroCol(m):
    from mpi4py import MPI
    comm     = MPI.COMM_WORLD
    myid     = MPI.COMM_WORLD.rank    

    N = m.shape[1]
    coo = CHypreMat2Coo(m.transpose())
    csr = coo.tocsr()
    num_zeros = np.diff(csr.indptr)
    k_zeros = np.where(num_zeros == 0)[0]
    dprint1("ZeroCol", coo.shape)
    rr = np.array(comm.allgather(len(k_zeros)))
    return rr
    
def allgather_vector(data, mpi_data_type = None, assemble=False):
    from mpi4py import MPI       
    from mfem.common.mpi_dtype import  get_mpi_datatype       
    if mpi_data_type is None:
         mpi_data_type = get_mpi_datatype(data)

    myid     = MPI.COMM_WORLD.rank
    rcounts = data.shape[0]
    rcounts = np.array(MPI.COMM_WORLD.allgather(rcounts))
    for x in data.shape[1:]: rcounts = rcounts * x        
    cm = np.hstack((0, np.cumsum(rcounts)))
    disps = list(cm[:-1])        
    length =  cm[-1]
    recvbuf = np.empty([length], dtype=data.dtype)
    recvdata = [recvbuf, rcounts, disps, mpi_data_type]
    senddata = [data.flatten(), data.flatten().shape[0]]        
    MPI.COMM_WORLD.Allgatherv(senddata, recvdata)
    return recvbuf.reshape(-1, *data.shape[1:])

def ZeroColIndex(m):
    from mpi4py import MPI
    comm     = MPI.COMM_WORLD
    myid     = MPI.COMM_WORLD.rank    

    N = m.shape[1]
    coo = CHypreMat2Coo(m.transpose())        
    rr = np.array(comm.allgather(coo.shape[0]))
    rall = int(np.sum(rr))
    rstart = np.hstack((0, np.cumsum(rr)))

    csr = coo.tocsr()
    num_zeros = np.diff(csr.indptr)
    k_zeros = np.where(num_zeros == 0)[0] + rstart[myid]
    #dprint1("ZeroCol", coo.shape, k_zeros)
    if MPI.INT.size == 4:
        dtype = np.int32
    else:
        dtype = np.int64
    k_zeros_all = allgather_vector(k_zeros.astype(dtype), MPI.INT, False)
    return k_zeros_all

