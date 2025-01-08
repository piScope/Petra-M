'''

MPI utillities

'''
import numpy as np
from mpi4py import MPI
from warnings import warn
from mfem.common.mpi_debug import nicePrint, niceCall

def mpi_type_flag(d):
    if d == MPI.LONG:
        return 1
    elif d == MPI.INT:
        return 2
    elif d == MPI.SHORT:
        return 3
    elif d == MPI.LONG_DOUBLE:
        return 4
    elif d == MPI.DOUBLE:
        return 5
    elif d == MPI.COMPLEX:
        return 6
    elif d == MPI.DOUBLE_COMPLEX:
        return 7
    elif d == MPI.BOOL:
        return 8
    elif d == MPI.FLOAT:
        return 9
    else:
        assert False, "unsupporte data type"


def mpi_type_from_flag(d):
    if d == 1:
        return MPI.LONG
    elif d == 2:
        return MPI.INT
    elif d == 3:
        return MPI.SHORT
    elif d == 4:
        return MPI.LONG_DOUBLE
    elif d == 5:
        return MPI.DOUBLE
    elif d == 6:
        return MPI.COMPLEX
    elif d == 7:
        return MPI.DOUBLE_COMPLEX
    elif d == 8:
        return MPI.BOOL
    elif d == 9:
        return MPI.FLOAT
    assert False, "unsupporte data type"


def mpi_type_to_dtype(mpi_dtype):
    if mpi_dtype == MPI.LONG:
        return np.int64
    if mpi_dtype == MPI.INT:
        return np.int32
    if mpi_dtype == MPI.SHORT:
        return np.int16
    if mpi_dtype == MPI.LONG_DOUBLE:
        return np.float128
    if mpi_dtype == MPI.DOUBLE:
        return np.float64
    if mpi_dtype == MPI.FLOAT:
        return np.float32
    if mpi_dtype == MPI.DOUBLE_COMPLEX:
        return np.complex128
    if mpi_dtype == MPI.COMPLEX:
        return np.complex64
    if mpi_dtype == MPI.BOOL:
        return np.bool
    assert False, "MPI data type is unknown for " + mpi_dtype


def allgather(data):
    comm = MPI.COMM_WORLD
    num_proc = MPI.COMM_WORLD.size
    myid = MPI.COMM_WORLD.rank
    data = comm.allgather(data)
    MPI.COMM_WORLD.Barrier()
    return data


def allgather_vector(data, mpi_data_type=None):
    from mfem.common.mpi_dtype import get_mpi_datatype
    if mpi_data_type is None:
        mpi_data_type = get_mpi_datatype(data)

    myid = MPI.COMM_WORLD.rank
    rcounts = data.shape[0]
    rcounts = np.array(MPI.COMM_WORLD.allgather(rcounts))

    for x in data.shape[1:]:
        rcounts = rcounts * x
    cm = np.hstack((0, np.cumsum(rcounts)))
    disps = list(cm[:-1])
    length = cm[-1]
    recvbuf = np.empty([length], dtype=data.dtype)
    recvdata = [recvbuf, rcounts, disps, mpi_data_type]
    senddata = [data.flatten(), data.flatten().shape[0]]
    MPI.COMM_WORLD.Allgatherv(senddata, recvdata)
    return recvbuf.reshape(-1, *data.shape[1:])


def gather_vector(data, mpi_data_type=None, root=0):
    '''
    gather vector to root node.
    B: Vector to be collected
    '''
    from mfem.common.mpi_dtype import get_mpi_datatype
    if mpi_data_type is None:
        mpi_data_type = get_mpi_datatype(data)

    myid = MPI.COMM_WORLD.rank
    rcounts = data.shape[0]
    rcounts = MPI.COMM_WORLD.gather(rcounts, root=root)
    cm = np.hstack((0, np.cumsum(rcounts)))
    disps = list(cm[:-1])
    recvdata = None
    senddata = [data, data.shape[0]]

    if myid == root:
        length = cm[-1]
        recvbuf = np.empty([length], dtype=data.dtype)
        recvdata = [recvbuf, rcounts, disps, mpi_data_type]
    else:
        recvdata = [None, rcounts, disps, mpi_data_type]
        recvbuf = None
    MPI.COMM_WORLD.Barrier()
    MPI.COMM_WORLD.Gatherv(senddata, recvdata, root=root)
    if myid == root:
        # print 'collected'
        MPI.COMM_WORLD.Barrier()
        return np.array(recvbuf)
    MPI.COMM_WORLD.Barrier()
    return None


def gather_vector(data, mpi_data_type=None, parent=False,
                  world=MPI.COMM_WORLD, root=0):
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
    from mfem.common.mpi_dtype import get_mpi_datatype
    myid = world.rank

    if mpi_data_type is None:
        mpi_data_type = get_mpi_datatype(data)

    if world.Is_intra():
        if myid == root:
            parent = True
        rcounts = data.shape[0]
        senddata = [data, data.shape[0]]
    elif parent:
        root = MPI.ROOT if myid == root else MPI.PROC_NULL
        rcounts = 0
        senddata = [np.array(()), 0]
    else:
        rcounts = data.shape[0]
        senddata = [data, data.shape[0]]
        if myid == root:
            parent = True

    rcounts = world.allgather(rcounts)
    cm = np.hstack((0, np.cumsum(rcounts)))
    disps = list(cm[:-1])

    if parent:
        length = cm[-1]
        recvbuf = np.empty([length], dtype=data.dtype)
        recvdata = [recvbuf, rcounts, disps, mpi_data_type]
    else:
        recvdata = [None, rcounts, disps, mpi_data_type]
        recvbuf = None
    world.Barrier()
    world.Gatherv(senddata, recvdata, root=root)
    if parent:
        # print 'collected'
        world.Barrier()
        return np.array(recvbuf)
    else:
        world.Barrier()
        return None


def scatter_vector(vector, mpi_data_type=None, rcounts=None, root=0):
    # scatter data
    #
    # rcounts indicats the amount of data which each process
    # receives
    #
    # for example:     rcounts = fespace.GetTrueVSize()
    if mpi_data_type is None:
        from mfem.common.mpi_dtype import get_mpi_datatype
        if MPI.COMM_WORLD.rank == root:
            mpi_data_type = get_mpi_datatype(vector)
            flag = mpi_type_flag(mpi_data_type)
        else:
            flag = None
        flag = MPI.COMM_WORLD.bcast(flag)
        mpi_data_type = mpi_type_from_flag(flag)

    rcountss = MPI.COMM_WORLD.gather(rcounts, root=root)
    # dprint1(rcountss)
    disps = list(np.hstack((0, np.cumsum(rcountss)))[:-1])
    recvdata = np.empty([rcounts], dtype=mpi_type_to_dtype(mpi_data_type))

    if vector is not None:
        senddata = [vector, rcountss, disps, mpi_data_type]
    else:
        senddata = None

    MPI.COMM_WORLD.Scatterv(senddata, recvdata, root=root)
    # MPI.COMM_WORLD.Barrier()
    return recvdata


'''
def scatter_vector2(vector, mpi_data_type=None, rcounts=None):
    from mfem.common.mpi_dtype import  get_mpi_datatype

    myid     = MPI.COMM_WORLD.rank
    isComplex = check_complex(vector)
    if mpi_data_type is None:
        if myid == 0:
            mpi_data_type = get_mpi_datatype(vector)
            flag = mpi_type_flag(mpi_data_type)
        else:
            flag = None
        flag = MPI.COMM_WORLD.bcast(flag)
        mpi_data_type = mpi_type_from_flag(flag)

    if isComplex:
        if myid == 0:
           r = vector.real +
           i = vector.imag
        else:
           r = None
           i = None
        return   (scatter_vector(r, mpi_data_type, rcounts=rcounts) +
               1j*scatter_vector(i, mpi_data_type, rcounts=rcounts))
    else:
        if myid == 0:
           r = vector
        else:
           r = None
        return scatter_vector(r, mpi_data_type, rcounts = rcounts)
'''


def alltoall_vector(data, datatype):
    '''
    data = list of np.array.
        each array goes to different MPI proc.
        length of data must be equal to MPI size
    '''
    from mfem.common.mpi_dtype import get_mpi_datatype

    num_proc = MPI.COMM_WORLD.size

    senddata = np.hstack(data).astype(datatype, copy=False)
    senddtype = get_mpi_datatype(senddata)
    senddtypes = MPI.COMM_WORLD.allgather(mpi_type_flag(senddtype))
    if len(np.unique(senddtypes)) != 1:
        assert False, "data type is not unique"

    sendsize = np.array([len(x) for x in data], dtype=int)
    senddisp = list(np.hstack((0, np.cumsum(sendsize)))[:-1])

    # (step 1) communicate the size of data
    if len(sendsize) != num_proc:
        assert False, "senddata size does not match with mpi size"
    recvsize = np.empty(num_proc, dtype=int)
    disp = list(range(num_proc))
    counts = [1] * num_proc
    dtype = get_mpi_datatype(recvsize)
    s1 = [sendsize, counts, disp, dtype]
    r1 = [recvsize, counts, disp, dtype]
    MPI.COMM_WORLD.Alltoallv(s1, r1)

    # (step 2) communicate the data
    recvsize = list(recvsize)
    recvdisp = list(np.hstack((0, np.cumsum(recvsize))))
    recvdata = np.empty(np.sum(recvsize), dtype=senddata.dtype)

    s1 = [senddata, sendsize, senddisp, senddtype]
    r1 = [recvdata, recvsize, recvdisp[:-1], senddtype]
    MPI.COMM_WORLD.Alltoallv(s1, r1)

    data = [recvdata[recvdisp[i]:recvdisp[i + 1]] for i in range(num_proc)]
    return data


def alltoall_vectorv(data, datatype):
    '''
    mulitidimension, arbitrary size version
    data = list of list of np.array.
        each list element goes to different MPI proc.
        length of data must be equal to MPI size
    '''
    from mfem.common.mpi_dtype import get_mpi_datatype

    num_proc = MPI.COMM_WORLD.size

    datashape = [np.array([y.shape for y in x]).flatten() for x in data]

    datadim = [np.hstack([len(y.shape) for y in x]) if len(x) > 0 else np.array([], dtype=int)
               for x in data]

    senddata = [np.hstack([y.flatten() for y in x]) if len(x) > 0
                else np.array([], dtype=datatype)
                for x in data]
    sendsize = np.array([len(x) for x in senddata], dtype=int)
    senddisp = list(np.hstack((0, np.cumsum(sendsize)))[:-1])
    senddata = np.hstack(senddata).astype(datatype, copy=False)

    senddtype = get_mpi_datatype(senddata)
    senddtypes = MPI.COMM_WORLD.allgather(mpi_type_flag(senddtype))

    if len(np.unique(senddtypes)) != 1:
        assert False, "data type is not unique"

    # (step 1) communicate the size of data

    datashape = alltoall_vector(datashape, int)
    datadim = alltoall_vector(datadim, int)

    # (step 2) communicate the data
    recvshapes = []
    for dims, shapes in zip(datadim, datashape):
        disps = np.hstack([0, np.cumsum(dims)])
        recvshapes.append([shapes[disps[i]:disps[i + 1]]
                           for i in range(len(disps) - 1)])

    recvsize = [np.sum([np.prod(s) for s in ss]).astype(int)
                for ss in recvshapes]

    recvdisp = list(np.hstack((0, np.cumsum(recvsize))))
    recvdata = np.empty(np.sum(recvsize), dtype=senddata.dtype)

    s1 = [senddata, sendsize, senddisp, senddtype]
    r1 = [recvdata, recvsize, recvdisp[:-1], senddtype]
    MPI.COMM_WORLD.Alltoallv(s1, r1)

    data0 = [recvdata[recvdisp[i]:recvdisp[i + 1]] for i in range(num_proc)]
    data = []
    for d, shapes in zip(data0, recvshapes):
        disps = np.hstack((0, np.cumsum([np.prod(s) for s in shapes])))
        data.append([d[disps[i]:disps[i + 1]].reshape(shapes[i])
                     for i in range(len(shapes))])

    return data


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
    comm = MPI.COMM_WORLD
    num_proc = MPI.COMM_WORLD.size
    myid = MPI.COMM_WORLD.rank

    m = r_A.GetNumRows()
    m_array = comm.allgather(m)
    rows = [0] + list(np.cumsum(m_array))
    return rows


def get_col_partitioning(r_A):
    warn('get_col_partition is deplicated', DeprecationWarning,
         stacklevel=2)
    comm = MPI.COMM_WORLD
    num_proc = MPI.COMM_WORLD.size
    myid = MPI.COMM_WORLD.rank

    n = r_A.GetNumCols()
    n_array = comm.allgather(n)
    cols = [0] + list(np.cumsum(n_array))
    return cols


def get_partition(A):
    comm = MPI.COMM_WORLD
    num_proc = MPI.COMM_WORLD.size
    myid = MPI.COMM_WORLD.rank
    return np.linspace(0, A.shape[0], num_proc + 1, dtype=int)


def distribute_vec_from_head(b):
    from mfem.common.mpi_dtype import get_mpi_datatype

    comm = MPI.COMM_WORLD
    num_proc = MPI.COMM_WORLD.size
    myid = MPI.COMM_WORLD.rank

    if myid == 0:
        partitioning = get_partition(b)
        MPItype = get_mpi_datatype(b)
        dtype = b.dtype
        for i in range(num_proc - 1):
            dest = i + 1
            b0 = b[partitioning[dest]:partitioning[dest + 1]]
            comm.send(b0.dtype, dest=dest, tag=dest)
            comm.send(b0.shape, dest=dest, tag=dest)
            comm.Send([b0, MPItype], dest=dest, tag=dest)
        b0 = b[partitioning[0]:partitioning[1]]
    else:
        dtype = comm.recv(source=0, tag=myid)
        shape = comm.recv(source=0, tag=myid)
        b0 = np.zeros(shape, dtype=dtype)
        MPItype = get_mpi_datatype(b0)
        comm.Recv([b0, MPItype], source=0, tag=myid)
    return b0


def distribute_global_coo(A):
    from mfem.common.mpi_dtype import get_mpi_datatype

    comm = MPI.COMM_WORLD
    num_proc = MPI.COMM_WORLD.size
    myid = MPI.COMM_WORLD.rank

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

        idx = np.logical_and(
            row >= partitioning[pair], row < partitioning[pair + 1])
        r0 = row[idx].astype(np.int32)
        c0 = col[idx].astype(np.int32)
        d0 = data[idx]

        if pair < myid:  # send first
            comm.send(r0.shape, dest=pair, tag=i)
            comm.Send([r0, MPI.INT], dest=pair, tag=i)
            comm.Send([c0, MPI.INT], dest=pair, tag=i)
            comm.Send([d0, MPItype], dest=pair, tag=i)

            shape = comm.recv(source=pair, tag=i)
            r = np.zeros(shape, dtype=np.int32)
            c = np.zeros(shape, dtype=np.int32)
            d = np.zeros(shape, dtype=dtype)
            comm.Recv([r, MPI.INT], source=pair, tag=i)
            comm.Recv([c, MPI.INT], source=pair, tag=i)
            comm.Recv([d, MPItype], source=pair, tag=i)
        elif pair > myid:  # recv first
            shape = comm.recv(source=pair, tag=i)
            r = np.zeros(shape, dtype=np.int32)
            c = np.zeros(shape, dtype=np.int32)
            d = np.zeros(shape, dtype=dtype)
            comm.Recv([r, MPI.INT], source=pair, tag=i)
            comm.Recv([c, MPI.INT], source=pair, tag=i)
            comm.Recv([d, MPItype], source=pair, tag=i)

            comm.send(r0.shape, dest=pair, tag=i)
            comm.Send([r0, MPI.INT], dest=pair, tag=i)
            comm.Send([c0, MPI.INT], dest=pair, tag=i)
            comm.Send([d0, MPItype], dest=pair, tag=i)
        else:

            r = r0
            c = c0
            d = d0

        row2.append(r)
        col2.append(c)
        data2.append(d)

    from scipy.sparse import coo_matrix

    r = np.hstack(row2) - partitioning[myid]
    c = np.hstack(col2)
    d = np.hstack(data2)

    rsize = partitioning[myid + 1] - partitioning[myid]

    A = coo_matrix((d, (r, c)), shape=(rsize, A.shape[1]),
                   dtype=d.dtype)
    return A
