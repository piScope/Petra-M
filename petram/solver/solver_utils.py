from __future__ import print_function

import numpy as np
import scipy
import scipy.sparse.linalg

def null(a, rtol=1e-5):
    ''' 
    find null space of matrix a
    '''
    u, s, v = np.linalg.svd(a)
    rank = (s > rtol*s[0]).sum()
    return s, rank, v[rank:].T.copy()

# this one does not work...
def nulls(a, rtol=1e-5):
    d = min(a.shape)-1
    u, s, v = scipy.sparse.linalg.svds(a, k=d)
    print(v.shape)
    null_space = v.conj().ravel()
    rank = (s > rtol*s[0]).sum()    
    return s, rank, null_space

def nulls2(a, rtol=1e-12):
    from sparsesvd import sparsesvd
    smat = scipy.sparse.csc_matrix(a)
    ut, s, vt = sparsesvd(smat, np.min(a.shape))
    print(vt.shape)
    padding = max(0,max(np.shape(a))-np.shape(s)[0])
    null_mask = np.concatenate(((s <= rtol), np.ones((padding,),dtype=bool)),axis=0)
    print(null_mask.shape)
    null_space = scipy.compress(null_mask, vt, axis=0)
    rank = (s > rtol*s[0]).sum()
    return s, rank, scipy.transpose(null_space)

def add_constraints(A, b, mm, m=None):
    '''
     A   , [mm, 0]^t
 
    [mm 0]  [0]
    '''
    from scipy.sparse import csr_matrix, bmat, hstack
    asize = A.shape[1]
    mmsize = mm.shape[1]
    delta = asize - mmsize
    if delta > 0:
        mm = hstack((mm, csr_matrix((mm.shape[0], delta), dtype = A.dtype)), format = 'csr')
    d = csr_matrix((mm.shape[0], mm.shape[0]), dtype = A.dtype)
    if m is None:
        m = np.array([0]*mm.shape[0])

    return bmat([[A, mm.transpose()], [mm, d]], format = 'csr'), np.hstack((b, m))

def make_numpy_coo_matrix(A):
    '''
    solve A*x= b using mumps through PETSc
    '''
    print("!!!!! Deprecated use mfem.commons.sparse_utils.sparsemat_to_scipycsr")
    I = A.GetIArray()
    J = A.GetJArray()
    D = A.GetDataArray()
    return scipy.sparse.csr_matrix((D, J, I), (A.Size(),A.Size())).tocoo(False)

def gather_vector(fespace, A, data,  mpi_data_type, assemble=False ):
    '''
    gather vector to root node. 
    A: HypreParMatrix to know the total length of data
    B: Vector to be collected 
    '''
    from mpi4py import MPI
    myid     = MPI.COMM_WORLD.rank

    # Need to collect RHS to call mumps
    #disps = fespace.GetGlobalTDofNumber(0)
    #disps = MPI.COMM_WORLD.gather(disps, root = 0)
    #rcounts = fespace.GetTrueVSize()
    rcounts = data.shape[0]
    rcounts = MPI.COMM_WORLD.gather(rcounts, root = 0)
    cm = np.hstack((0, np.cumsum(rcounts)))
    disps = list(cm[:-1])        
    recvdata = None
    senddata = [data, data.shape[0]]
    
    if myid ==0:
       length =  cm[-1]
       recvbuf = np.empty([length], dtype=data.dtype)
       recvdata = [recvbuf, rcounts, disps, mpi_data_type]       
    else:
       recvdata = [None, rcounts, disps, mpi_data_type]
       recvbuf = None
    MPI.COMM_WORLD.Barrier()           
    MPI.COMM_WORLD.Gatherv(senddata, recvdata,  root = 0)
    if myid == 0:
        MPI.COMM_WORLD.Barrier()
        return np.array(recvbuf)
    MPI.COMM_WORLD.Barrier()
    return None
    

def scatter_vector(fespace, vector, mpi_data_type):
    from mpi4py import MPI    
    # Need to scatter solution
    senddata = None
    #disps = fespace.GetGlobalTDofNumber(0)
    #disps = MPI.COMM_WORLD.gather(disps, root = 0)
    rcounts = fespace.GetTrueVSize()
    rcounts = MPI.COMM_WORLD.gather(rcounts, root = 0)
    disps = list(np.hstack((0, np.cumsum(rcounts)))[:-1])
    recvdata = np.empty([fespace.GetTrueVSize()], dtype="float64")
    if vector is not None: 
        sol = np.array(vector, dtype="float64")
        senddata = [sol, rcounts, disps, mpi_data_type]
    MPI.COMM_WORLD.Scatterv(senddata, recvdata, root = 0)
    MPI.COMM_WORLD.Barrier()        
    return list(recvdata)    
