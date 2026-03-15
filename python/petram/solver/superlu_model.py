from petram.phys.vtable import VtableElement, Vtable
from petram.helper.matrix_file import write_matrix, write_vector, write_coo_matrix
from petram.mfem_config import use_parallel
from petram.solver.solver_model import LinearSolverModel, LinearSolver
from petram.solver.solver_model import Solver
from petram.namespace_mixin import NS_mixin


import sys
import numpy as np
import scipy
from scipy.sparse import coo_matrix, csr_matrix

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('SuperLU')


if use_parallel:
    from petram.helper.mpi_recipes import *
    from mfem.common.parcsr_extra import *
    import mfem.par as mfem
    default_kind = 'hypre'

    from mpi4py import MPI
    num_proc = MPI.COMM_WORLD.size
    myid = MPI.COMM_WORLD.rank
    smyid = '{:0>6d}'.format(myid)
    barrier = MPI.COMM_WORLD.Barrier
    from mfem.common.mpi_debug import nicePrint

else:
    import mfem.ser as mfem
    default_kind = 'scipy'
    num_proc = 1
    myid = 0

    def barrier():
        pass

    def nicePrint(*x):
        print(x)


class SuperLU(LinearSolverModel):
    hide_ns_menu = True
    has_2nd_panel = False
    accept_complex = True
    always_new_panel = False

    def __init__(self, *args, **kwargs):
        LinearSolverModel.__init__(self, *args, **kwargs)

    def init_solver(self):
        pass

    def attribute_set(self, v):
        v = super(SuperLU, self).attribute_set(v)
        v["col_permute_txt"] = "COLAMD"
        v["pivot_thr_txt"] = ""
        v["relax_txt"] = ""
        v["panel_size_txt"] = ""
        v["superlu_options_txt"] = ""
        v["use_single_precision"] = False
        v["write_mat"] = False
        return v

    def eval_options(self, input):
        if len(input.strip()) == 0:
            return None
        g = self._global_ns.copy()
        l = {}
        value = eval(input, g, l)
        return value

    def panel1_param(self):
        def validator(input, param, widget):
            try:
                self.eval_options(input)
            except:
                return False
            return True

        import wx
        ll = [["col. permute", "COLAMD", 4,
               {"choices": ["COLAMD", "NATURAL", "MMD_ATA", "MMD_AT_PLUS_A"],
                "style": wx.CB_READONLY}],
              ["pivot thr.", "None", 0, {}],
              ["relax.", "None", 0, {}],
              ["panel size", "None", 0, {}],
              ["options(=)", "None", 0, {"validator": validator,
                                         "validator_param": None}],
              ["use float32", False, 3, {"text": ""}],
              ["write matrix", False, 3, {"text": ""}],]
        return ll

    def get_panel1_value(self):
        val = [self.col_permute_txt,
               self.pivot_thr_txt,
               self.relax_txt,
               self.panel_size_txt,
               self.superlu_options_txt,
               self.use_single_precision,
               self.write_mat]
        return val

    def import_panel1_value(self, v):

        self.col_permute_txt = v[0]
        self.pivot_thr_txt = v[1]
        self.relax_txt = v[2]
        self.panel_size_txt = v[3]
        self.superlu_options_txt = v[4]
        self.use_single_precision = v[5]
        self.write_mat = v[6]

    def does_linearsolver_choose_linearsystem_type(self):
        return True

    def linear_system_type(self, assemble_real, phys_real):
        if phys_real:
            return 'blk_interleave'
        else:
            return 'blk_merged'

    def real_to_complex(self, solall, M):
        solver = self.get_solver()
        if solver.assemble_real:
            return self.real_to_complex_merged(solall, M)
        else:
            assert False, "should not come here"

    def real_to_complex_merged(self, solall, M):
        if use_parallel:
            of = M.RowOffsets().ToList()

            if not self.use_dist_sol:
                of = [np.sum(MPI.COMM_WORLD.allgather(np.int32(o)))
                      for o in of]
                if myid != 0:
                    return
        else:
            of = M.RowOffsets().ToList()

        # dprint1("merged block size", of)

        rows = M.NumRowBlocks()
        s = solall.shape
        i = 0
        pt = 0
        result = np.zeros((s[0] // 2, s[1]), dtype='complex')
        for i in range(rows):
            l = of[i + 1] - of[i]
            w = int(l // 2)

            result[pt:pt + w, :] = (solall[of[i]:of[i] + w, :]
                                    + 1j * solall[(of[i] + w):of[i + 1], :])
            pt = pt + w
        return result

    def allocate_solver(self, is_complex=False, engine=None):
        solver = SuperLUSolver(self, engine)

        pivot_thr = float(self.pivot_thr_txt) if len(
            self.pivot_thr_txt.strip()) > 0 else None
        relax = int(self.relax_txt) if len(
            self.relax_txt.strip()) > 0 else None
        panel_size = int(self.panel_size_txt) if len(
            self.panel_size_txt.strip()) > 0 else None
        options = self.eval_options(self.superlu_options_txt)

        solver.AllocSolver(is_complex,
                           self.use_single_precision,
                           self.col_permute_txt,
                           pivot_thr,
                           relax,
                           panel_size,
                           options)

        return solver


def check_operator_type(is_complex, A):
    from petram.solver.solver_utils import get_operator_block

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
    '''
    build CSR form of A as a single
    matrix
    '''
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
        '''
        convert HypreParCSR to Scipy Coo Matrix
        '''
        num_rows, ilower, iupper, jlower, jupper, irn, jcn, data = mat.GetCooDataArray()
        m = iupper - ilower + 1
        n = mat.N()

        return coo_matrix((data, (irn - ilower, jcn)),
                          shape=(num_rows, n)), ilower

    from scipy.sparse import bmat
    from petram.solver.solver_utils import get_operator_block

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

    # elements = [None] * rows
    # elements = [elements.copy() for x in range(cols)]
    # 2D None rows*cols
    elements = [[None for _ in range(rows)] for _ in range(cols)]

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


class SuperLUSolver(LinearSolver):
    def __init__(self, gui, engine):
        LinearSolver.__init__(self, gui, engine)

        self._superlu = None
        self.is_complex = False
        self.use_single_precision = False
        self.superlu_params = {}
        self.dtype = None
        self._superlu = None

    def AllocSolver(self, is_complex, use_single_precision, col_permute,
                    pivot_thr, relax, panel_size, options):

        # for SuperLU, allocation is done in SetOperator
        self.is_complex = is_complex
        self.use_single_precision = use_single_precision
        self.superlu_params = {"permc_spec": col_permute,
                               "diag_pivot_thresh": pivot_thr,
                               "relax": relax,
                               "panel_size": panel_size,
                               "options": options}
        if is_complex:
            if use_single_precision:
                dtype = np.complex64
            else:
                dtype = np.complex128
        else:
            if use_single_precision:
                dtype = np.float32
            else:
                dtype = np.float64
        self.dtype = dtype

    def SetOperator(self, A, dist, name=None):
        self.row_offsets = A.RowOffsets()
        # nicePrint("row offsets in SetOperator", self.row_offsets.ToList())

        self.op_type = check_operator_type(self.is_complex, A)
        AA = build_csr_local(A, self.dtype, self.is_complex)

        if self.gui.write_mat:
            write_coo_matrix('matrix', AA.tocoo())

        from scipy.sparse.linalg import splu

        if use_parallel:
            # in this case, we gather matrix to root node
            # and squash it.
            AAs = MPI.COMM_WORLD.gather(AA, root=0)
            if myid != 0:
                return
            AA = scipy.sparse.vstack(AAs)

        dprint1("creating SuperLU object. matrix shape=", AA.shape)
        self._superlu = splu(AA.tocsc(), **getattr(self, "superlu_params", {}))

    def Mult(self, b, x=None, case_base=0):

        if not self.gui.use_dist_sol:       
            assert False, "SuperLU model returns distributed solution vector. Other mode is not implemented"

        sol = []
        row_offsets = self.row_offsets.ToList()

        for kk, bb in enumerate(b):
            if x is None:
                xx = mfem.BlockVector(self.row_offsets)
                xx.Assign(0.0)
            else:
                xx = x

            if self.is_complex:
                tmp1 = []
                tmp2 = []
                for i in range(len(row_offsets) - 1):
                    bbv = bb.GetBlock(i).GetDataArray()
                    xxv = xx.GetBlock(i).GetDataArray()
                    ll = bbv.size
                    bbv = bbv[:ll // 2] + 1j * bbv[ll // 2:]
                    xxv = xxv[:ll // 2] + 1j * xxv[ll // 2:]
                    tmp1.append(bbv)
                    tmp2.append(xxv)
                bbv = np.hstack(tmp1)
                xxv = np.hstack(tmp2)
            else:
                bbv = bb.GetDataArray()
                xxv = xx.GetDataArray()

            if self.gui.write_mat:
                write_vector('rhs_' + str(kk), bbv)
                write_vector('x_' + str(kk), xxv)

            sys.stdout.flush()
            sys.stderr.flush()

            if use_parallel:
                nrow = len(bbv)
                if myid == 0:
                    bbv = gather_vector(bbv, parent=True)
                else:
                    gather_vector(bbv)

            if myid == 0:
                dprint1("calling solve", debug.format_memory_usage())
                xxv = self._superlu.solve(bbv)

            if use_parallel:
                xxv = scatter_vector(xxv, rcounts=nrow)

            barrier()
            sol.append(xxv)

        sol = np.transpose(np.vstack(sol))
        return sol


class SuperLUMFEMSolverModel(SuperLU):
    '''
    This one is to use STRUMPACK in iterative solver
    It creates MUMPSPreconditioner
    '''

    def prepare_solver(self, opr, engine):
        solver = SuperLUBlockPreconditioner(opr,
                                            gui=self,
                                            engine=engine,
                                            silent=True)
        solver.SetOperator(opr)
        return solver


class SuperLUBlockPreconditioner(mfem.Solver):
    def __init__(self, opr, gui=None, engine=None, silent=False, **kwargs):
        self.gui = gui
        self.engine = engine
        self.silent = silent

        self.is_complex_operator = False
        self.is_parallel = False

        self.solver = None
        super(StrumpackBlockPreconditioner, self).__init__()

    def Mult(self, x, y):
        s = self.solver.Mult([x])
        if self.is_complex_operator:
            assert False, "SuperLUMFEM for Complex is not yet implemented"
            # s = self.complex_to_real(s)

        y.Assign(s.flatten().astype(float, copy=False))

    def SetOperator(self, opr):
        from petram.solver.solver_utils import check_block_operator
        is_complex, is_parallel = check_block_operator(opr)

        solver = SuperLUSolver(self.gui, self.engine)
        solver.AllocSolver(is_complex, self.gui.use_single_precision)
        solver.SetOperator(opr, is_parallel)

        self.is_complex_operator = is_complex
        self.is_parallel = is_parallel
        self.solver = solver
