
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
    padding = max(0, max(np.shape(a))-np.shape(s)[0])
    null_mask = np.concatenate(
        ((s <= rtol), np.ones((padding,), dtype=bool)), axis=0)
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
        mm = hstack(
            (mm, csr_matrix((mm.shape[0], delta), dtype=A.dtype)), format='csr')
    d = csr_matrix((mm.shape[0], mm.shape[0]), dtype=A.dtype)
    if m is None:
        m = np.array([0]*mm.shape[0])

    return bmat([[A, mm.transpose()], [mm, d]], format='csr'), np.hstack((b, m))


def make_numpy_coo_matrix(A):
    '''
    solve A*x= b using mumps through PETSc
    '''
    print("!!!!! Deprecated use mfem.commons.sparse_utils.sparsemat_to_scipycsr")
    I = A.GetIArray()
    J = A.GetJArray()
    D = A.GetDataArray()
    return scipy.sparse.csr_matrix((D, J, I), (A.Size(), A.Size())).tocoo(False)


def gather_vector(fespace, A, data,  mpi_data_type, assemble=False):
    '''
    gather vector to root node. 
    A: HypreParMatrix to know the total length of data
    B: Vector to be collected 
    '''
    from mpi4py import MPI
    myid = MPI.COMM_WORLD.rank

    # Need to collect RHS to call mumps
    #disps = fespace.GetGlobalTDofNumber(0)
    #disps = MPI.COMM_WORLD.gather(disps, root = 0)
    #rcounts = fespace.GetTrueVSize()
    rcounts = data.shape[0]
    rcounts = MPI.COMM_WORLD.gather(rcounts, root=0)
    cm = np.hstack((0, np.cumsum(rcounts)))
    disps = list(cm[:-1])
    recvdata = None
    senddata = [data, data.shape[0]]

    if myid == 0:
        length = cm[-1]
        recvbuf = np.empty([length], dtype=data.dtype)
        recvdata = [recvbuf, rcounts, disps, mpi_data_type]
    else:
        recvdata = [None, rcounts, disps, mpi_data_type]
        recvbuf = None
    MPI.COMM_WORLD.Barrier()
    MPI.COMM_WORLD.Gatherv(senddata, recvdata,  root=0)
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
    rcounts = MPI.COMM_WORLD.gather(rcounts, root=0)
    disps = list(np.hstack((0, np.cumsum(rcounts)))[:-1])
    recvdata = np.empty([fespace.GetTrueVSize()], dtype="float64")
    if vector is not None:
        sol = np.array(vector, dtype="float64")
        senddata = [sol, rcounts, disps, mpi_data_type]
    MPI.COMM_WORLD.Scatterv(senddata, recvdata, root=0)
    MPI.COMM_WORLD.Barrier()
    return list(recvdata)


def get_operator_block(Op, i, j):
    try:
        return Op._linked_op[(i, j)]
    except KeyError:
        return None


def write_blockoperator(A=None, b=None, x=None,
                        suffix="",
                        mat_base="matrix_",
                        b_base="rhs_",
                        x_base="x_"):
    '''

    '''
    from petram.mfem_config import use_parallel
    if use_parallel:
        import mfem.par as mfem
    else:
        import mfem.ser as mfem

    offset = A.RowOffsets()
    rows = A.NumRowBlocks()
    cols = A.NumColBlocks()
    if suffix != '':
        suffix = '.'+suffix

    if A is not None:
        for i in range(cols):
            for j in range(rows):
                m = get_operator_block(A, i, j)
                if m is None:
                    continue
                if isinstance(m, mfem.ComplexOperator):
                    m1 = m._real_operator
                    m2 = m._imag_operator
                    m1.Print(mat_base+"Re_" + str(i) + '_' + str(j))
                    m2.Print(mat_base+"Im_" + str(i) + '_' + str(j))
                else:
                    m.Print(mat_base + str(i) + '_' + str(j))
    if b is not None:
        for i, bb in enumerate(b):
            for j in range(rows):
                v = bb.GetBlock(j)
                v.Print(b_base + str(i) + '_' + str(j) + suffix)
    if x is not None:
        for j in range(rows):
            xx = x.GetBlock(j)
            xx.Print(x_base + str(i) + '_' + str(j) + suffix)


def check_block_operator(A):
    from petram.mfem_config import use_parallel
    if use_parallel:
        import mfem.par as mfem
    else:
        import mfem.ser as mfem

    rows = A.NumRowBlocks()
    cols = A.NumColBlocks()

    is_complex = False
    is_parallel = False
    for i in range(rows):
        for j in range(cols):
            m = get_operator_block(A, i, j)
            if m is None:
                continue
            if isinstance(m, mfem.ComplexOperator):
                is_complex = True
                check = m._real_operator
            else:
                check = m
            if not isinstance(check, mfem.SparseMatrix):
                is_parallel = True

    return is_complex, is_parallel


def check_operator_type(is_complex, A):
    """
    Identify real, complex, or complex-as-real block operators.
    """
    from petram.mfem_config import use_parallel
    if use_parallel:
        import mfem.par as mfem
    else:
        import mfem.ser as mfem

    rows = A.NumRowBlocks()
    cols = A.NumColBlocks()

    complexreal = False
    for i in range(rows):
        for j in range(cols):
            m = get_operator_block(A, i, j)
            if isinstance(m, mfem.ComplexOperator) and not is_complex:
                complexreal = True
                break

    if complexreal:
        return "complexreal"
    if is_complex:
        return "complex"
    return "real"


def build_csr_local(A, dtype, is_complex):
    """
    Build local CSR rows with rank-major global column numbering.

    Local rows retain their field-block order. Global numbering groups all
    fields on each rank, rather than grouping each field across all ranks.
    """
    from scipy.sparse import bmat, coo_matrix, csr_matrix
    from petram.mfem_config import use_parallel

    if use_parallel:
        from mpi4py import MPI
        from petram.helper.mpi_recipes import allgather_vector
        import mfem.par as mfem
        num_proc = MPI.COMM_WORLD.size
    else:
        import mfem.ser as mfem

    # print("build_csr_local", dtype, is_complex)
    offset = np.array(A.RowOffsets().ToList(), dtype=int)
    coffset = np.array(A.ColOffsets().ToList(), dtype=int)
    if is_complex:
        offset = offset // 2
    # nicePrint("offets", offset, coffset)
    rows = A.NumRowBlocks()
    cols = A.NumColBlocks()

    local_size = np.diff(offset)
    # nicePrint("local_size",local_size)

    if use_parallel:
        x = allgather_vector(local_size)
        global_size = np.sum(x.reshape(num_proc, -1), 0)
        global_offset = np.hstack(([0], np.cumsum(global_size)))
        # global_roffset = global_offset + offset
        # nicePrint("global offset/roffset", global_offset)
        new_offset = np.hstack(([0], np.cumsum(x)))[:-1]
        new_size = x.reshape(num_proc, -1)
        new_offset = new_offset.reshape(num_proc, -1)
        # nicePrint(new_size)
        # nicePrint(new_offset)
    else:
        global_size = local_size
        new_size = local_size.reshape(1, -1)
        new_offset = offset.reshape(1, -1)

    # index_mapping
    def blk_stm_idx_map(i):
        stm_idx = [new_offset[kk, i] +
                   np.arange(new_size[kk, i], dtype=int)
                   for kk in range(len(new_offset))]
        return np.hstack(stm_idx)

    def blk_stm_idx_map_complexreal(i):
        stm_idx = ([new_offset[kk, i] +
                   np.arange(new_size[kk, i]//2, dtype=int)
                   for kk in range(len(new_offset))] +
                   [new_offset[kk, i] + new_size[kk, i]//2 +
                   np.arange(new_size[kk, i]//2, dtype=int)
                   for kk in range(len(new_offset))])
        return np.hstack(stm_idx)

    def sparsemat2csr(m):
        w, h = m.Width(), m.Height()
        I = m.GetIArray()
        J = m.GetJArray()
        data = m.GetDataArray()
        m = csr_matrix((data, J, I), shape=(h, w),
                       dtype=data.dtype)
        return m

    def ToScipyCoo(mat):
        """
        Convert HypreParCSR to a local SciPy COO matrix.
        """
        num_rows, ilower, iupper, jlower, jupper, irn, jcn, data = mat.GetCooDataArray()
        m = iupper - ilower + 1
        n = mat.N()

        return coo_matrix((data, (irn - ilower, jcn)),
                          shape=(num_rows, n)), ilower

    op_type = check_operator_type(is_complex, A)
    if op_type == 'complexreal':
        map = [blk_stm_idx_map_complexreal(i) for i in range(rows)]
    else:
        map = [blk_stm_idx_map(i) for i in range(rows)]

    # nicePrint("map", map)
    newi = []
    newj = []
    newd = []
    nrows = np.sum(local_size)
    ncols = np.sum(global_size)

    elements = [[None] * cols for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            m = get_operator_block(A, i, j)
            if m is None:
                continue
            if use_parallel:
                if isinstance(m, mfem.ComplexOperator):
                    if not is_complex:
                        mr, ilower = ToScipyCoo(m._real_operator)
                        mi, ilower = ToScipyCoo(m._imag_operator)
                        m = bmat([[mr, -mi], [mi, mr]])
                    else:
                        mr, ilower = ToScipyCoo(m._real_operator)
                        mi, ilower = ToScipyCoo(m._imag_operator)
                        m = (mr + 1j * mi).tocoo()
                else:
                    m, ilower = ToScipyCoo(m)
            else:
                # this is not efficient but for now let's do this...
                if isinstance(m, mfem.ComplexOperator):
                    if not is_complex:
                        mr = m._real_operator
                        mi = m._imag_operator
                        mr = sparsemat2csr(mr)
                        mi = sparsemat2csr(mi)
                        m = bmat([[mr, -mi], [mi, mr]]).tocoo()
                    else:
                        mr = m._real_operator
                        mi = m._imag_operator
                        mr = sparsemat2csr(mr)
                        mi = sparsemat2csr(mi)
                        m = (mr + 1j * mi).tocoo()
                else:
                    m = sparsemat2csr(m).tocoo()

            elements[i][j] = m

    csr = bmat(elements, dtype=dtype).tocsr()

    # when block is squashed in vertical direction
    # we need to swap columns in order to match the order of x, rhs
    # elements

    csr2 = csr[:, np.argsort(np.hstack(map))]

    return csr2
