'''

MPI utillities

'''
import numpy as np
from mpi4py import MPI
from  warnings import warn

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

def gather_vector(data, mpi_data_type = None):
    '''
    gather vector to root node. 
    B: Vector to be collected 
    '''
    from mfem.common.mpi_dtype import  get_mpi_datatype
    if mpi_data_type is None:
       mpi_data_type = get_mpi_datatype(data)

    myid     = MPI.COMM_WORLD.rank
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
        #print 'collected'
        MPI.COMM_WORLD.Barrier()
        return np.array(recvbuf)
    MPI.COMM_WORLD.Barrier()
    return None

def gather_vector(data, mpi_data_type = None, parent = False,
                  world = MPI.COMM_WORLD):
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
    if mpi_data_type is None:
       mpi_data_type = get_mpi_datatype(data)
       
    myid     = world.rank
    root = 0
    
    if world.Is_intra():
        if myid == 0: parent = True
        rcounts = data.shape[0]
        senddata = [data, data.shape[0]]            
    elif parent:
        root = MPI.ROOT if myid == 0 else MPI.PROC_NULL
        rcounts = 0
        senddata = [np.array(()), 0]
    else:
        rcounts = data.shape[0]        
        senddata = [data, data.shape[0]]    

    rcounts = world.gather(rcounts, root = root)
    cm = np.hstack((0, np.cumsum(rcounts)))
    disps = list(cm[:-1])        
#    recvdata = None
    if parent:
        length =  cm[-1]
        recvbuf = np.empty([length], dtype=data.dtype)
        recvdata = [recvbuf, rcounts, disps, mpi_data_type]       
    else:
        recvdata = [None, rcounts, disps, mpi_data_type]
        recvbuf = None
    world.Barrier()           
    world.Gatherv(senddata, recvdata,  root = root)
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
    
