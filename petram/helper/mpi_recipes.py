'''

MPI utillities

'''
import numpy as np
from mpi4py import MPI
from  warnings import warn
from mfem.common.mpi_debug import nicePrint, niceCall

def allgather(data):
    comm     = MPI.COMM_WORLD     
    num_proc = MPI.COMM_WORLD.size
    myid     = MPI.COMM_WORLD.rank
    data = comm.allgather(data)
    MPI.COMM_WORLD.Barrier()           
    return data

def allgather_vector(data, mpi_data_type = None):
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

def gather_vector(data, mpi_data_type = None, root=0):
    '''
    gather vector to root node. 
    B: Vector to be collected 
    '''
    from mfem.common.mpi_dtype import  get_mpi_datatype
    if mpi_data_type is None:
       mpi_data_type = get_mpi_datatype(data)

    myid     = MPI.COMM_WORLD.rank
    rcounts = data.shape[0]
    rcounts = MPI.COMM_WORLD.gather(rcounts, root = root)
    cm = np.hstack((0, np.cumsum(rcounts)))
    disps = list(cm[:-1])        
    recvdata = None
    senddata = [data, data.shape[0]]

    if myid ==root:
        length =  cm[-1]
        recvbuf = np.empty([length], dtype=data.dtype)
        recvdata = [recvbuf, rcounts, disps, mpi_data_type]       
    else:
        recvdata = [None, rcounts, disps, mpi_data_type]
        recvbuf = None
    MPI.COMM_WORLD.Barrier()           
    MPI.COMM_WORLD.Gatherv(senddata, recvdata,  root = root)
    if myid == root:
        #print 'collected'
        MPI.COMM_WORLD.Barrier()
        return np.array(recvbuf)
    MPI.COMM_WORLD.Barrier()
    return None

def gather_vector(data, mpi_data_type = None, parent = False,
                  world = MPI.COMM_WORLD, root=0):
    '''
    gather vector to root
    B: Vector to be collected 

    for intra-communication, leave parent False. data is gatherd
    to root (myid = 0)

    for inter-communication:
       root group should call with parent = True 
       root group should call with data to tell the data type, like np.array(2)
       world should be specified

    '''
    from mfem.common.mpi_dtype import  get_mpi_datatype
    myid     = world.rank    

    if mpi_data_type is None:
       mpi_data_type = get_mpi_datatype(data)
    
    if world.Is_intra():
        if myid == root: parent = True
        rcounts = data.shape[0]
        senddata = [data, data.shape[0]]            
    elif parent:
        root = MPI.ROOT if myid == root else MPI.PROC_NULL
        rcounts = 0
        senddata = [np.array(()), 0]
    else:
        rcounts = data.shape[0]        
        senddata = [data, data.shape[0]]
        if myid == root: parent = True

    rcounts = world.allgather(rcounts)
    cm = np.hstack((0, np.cumsum(rcounts)))
    disps = list(cm[:-1])        

    if parent:
        length =  cm[-1]
        recvbuf = np.empty([length], dtype=data.dtype)
        recvdata = [recvbuf, rcounts, disps, mpi_data_type]       
    else:
        recvdata = [None, rcounts, disps, mpi_data_type]
        recvbuf = None
    world.Barrier()
    world.Gatherv(senddata, recvdata, root=root)
    if parent:
        #print 'collected'
        world.Barrier()
        return np.array(recvbuf)
    else:
        world.Barrier()
        return None

def scatter_vector(vector, mpi_data_type, rcounts):
    # scatter data
    #
    # rcounts indicats the amount of data which each process
    # receives
    #
    # for example:     rcounts = fespace.GetTrueVSize()
    senddata = None
    rcountss = MPI.COMM_WORLD.gather(rcounts, root = 0)
    #dprint1(rcountss)
    disps = list(np.hstack((0, np.cumsum(rcountss)))[:-1])
    recvdata = np.empty([rcounts], dtype="float64")
    if vector is not None: 
        sol = np.array(vector, dtype="float64")
        senddata = [sol, rcountss, disps, mpi_data_type]
    MPI.COMM_WORLD.Scatterv(senddata, recvdata, root = 0)
    MPI.COMM_WORLD.Barrier()        
    return recvdata

def scatter_vector2(vector, mpi_data_type, rcounts = None):
    ''' 
    scatter_vector2 hide difference between complex and real
    '''

    myid     = MPI.COMM_WORLD.rank        
    isComplex = check_complex(vector)
    if isComplex:
        if myid == 0:
           r = vector.real
           i = vector.imag
        else:
           r = None
           i = None
        return   (scatter_vector(r, mpi_data_type, rcounts = rcounts) + 
               1j*scatter_vector(i, mpi_data_type, rcounts = rcounts))
    else:
        if myid == 0:
           r = vector
        else:
           r = None
        return scatter_vector(r, mpi_data_type, rcounts = rcounts)

def check_complex(obj, root=0):
    return MPI.COMM_WORLD.bcast(np.iscomplexobj(obj), root=root)

def safe_flatstack(ll, dtype=int):
    if len(ll) > 0:
        return np.hstack(ll).astype(dtype, copy=False)
    else:
        return np.array([], dtype=dtype)
    
def get_row_partitioning(r_A):
    warn('get_row_partition is deplicated', DeprecationWarning,
                  stacklevel=2)
    comm     = MPI.COMM_WORLD     
    num_proc = MPI.COMM_WORLD.size
    myid     = MPI.COMM_WORLD.rank

    m = r_A.GetNumRows()
    m_array = comm.allgather(m)
    rows = [0] + list(np.cumsum(m_array))
    return rows

def get_col_partitioning(r_A):
    warn('get_col_partition is deplicated', DeprecationWarning,
                  stacklevel=2)    
    comm     = MPI.COMM_WORLD     
    num_proc = MPI.COMM_WORLD.size
    myid     = MPI.COMM_WORLD.rank

    n = r_A.GetNumCols()
    n_array = comm.allgather(n)
    cols = [0] + list(np.cumsum(n_array))
    return cols


def get_partition(A):
    comm     = MPI.COMM_WORLD     
    num_proc = MPI.COMM_WORLD.size
    myid     = MPI.COMM_WORLD.rank
    return np.linspace(0, A.shape[0], num_proc+1, dtype=int)
    
def distribute_vec_from_head(b):
    from mfem.common.mpi_dtype import  get_mpi_datatype
    
    comm     = MPI.COMM_WORLD     
    num_proc = MPI.COMM_WORLD.size
    myid     = MPI.COMM_WORLD.rank

    if myid == 0:
        partitioning = get_partition(b)        
        MPItype = get_mpi_datatype(b)                
        dtype = b.dtype        
        for i in range(num_proc-1):
            dest = i+1
            b0 = b[partitioning[dest]:partitioning[dest+1]]
            comm.send(b0.dtype, dest=dest, tag=dest)                                
            comm.send(b0.shape, dest=dest, tag=dest)                    
            comm.Send([b0, MPItype], dest=dest, tag=dest)
        b0 = b[partitioning[0]:partitioning[1]]
    else:
        dtype = comm.recv(source=0, tag=myid)
        shape = comm.recv( source=0, tag=myid)
        b0 = np.zeros(shape, dtype = dtype)
        MPItype = get_mpi_datatype(b0)        
        comm.Recv([b0, MPItype], source=0, tag=myid)
    return b0

def distribute_global_coo(A):
    from mfem.common.mpi_dtype import  get_mpi_datatype           

    comm     = MPI.COMM_WORLD     
    num_proc = MPI.COMM_WORLD.size
    myid     = MPI.COMM_WORLD.rank
    
    partitioning = get_partition(A)
    
    row = A.row
    col = A.col
    data = A.data

    ids = np.arange(num_proc)[::-1]

    dtype = data.dtype
    MPItype = get_mpi_datatype(data)

    row2 = []
    col2 = []
    data2 = []
    
    for i in range(num_proc):
        ids = np.roll(ids, 1)
        pair = ids[myid]
        
        idx = np.logical_and(row >= partitioning[pair], row < partitioning[pair+1])
        r0 = row[idx].astype(np.int32)
        c0 = col[idx].astype(np.int32)
        d0 = data[idx]

        if pair < myid: # send first
            comm.send(r0.shape, dest=pair, tag=i)
            comm.Send([r0, MPI.INT], dest=pair, tag=i)
            comm.Send([c0, MPI.INT], dest=pair, tag=i)        
            comm.Send([d0, MPItype], dest=pair, tag=i)

            shape = comm.recv(source=pair, tag=i)                
            r = np.zeros(shape, dtype = np.int32)
            c = np.zeros(shape, dtype = np.int32)
            d = np.zeros(shape, dtype = dtype)                
            comm.Recv([r, MPI.INT], source=pair, tag=i)
            comm.Recv([c, MPI.INT], source=pair, tag=i)
            comm.Recv([d, MPItype], source=pair, tag=i)        
        elif pair >  myid: # recv first
            shape = comm.recv(source=pair, tag=i)
            r = np.zeros(shape, dtype = np.int32)
            c = np.zeros(shape, dtype = np.int32)
            d = np.zeros(shape, dtype = dtype)               
            comm.Recv([r, MPI.INT], source=pair, tag=i)
            comm.Recv([c, MPI.INT], source=pair, tag=i)
            comm.Recv([d, MPItype], source=pair, tag=i)                
            
            comm.send(r0.shape, dest=pair, tag=i)        
            comm.Send([r0, MPI.INT], dest=pair, tag=i)
            comm.Send([c0, MPI.INT], dest=pair, tag=i)        
            comm.Send([d0, MPItype], dest=pair, tag=i)
        else:
            
            r = r0; c = c0; d = d0

        row2.append(r)
        col2.append(c)
        data2.append(d)

    from scipy.sparse import coo_matrix

    r = np.hstack(row2) - partitioning[myid]
    c = np.hstack(col2)
    d = np.hstack(data2)

    rsize = partitioning[myid+1] -  partitioning[myid]

    A = coo_matrix((d, (r, c)), shape=(rsize, A.shape[1]),
                   dtype = d.dtype)
    return A

        
    
        
