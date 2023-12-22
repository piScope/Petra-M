from __future__ import print_function
from petram.helper.matrix_file import write_matrix, write_vector, write_coo_matrix
from petram.mfem_config import use_parallel
from .solver_model import LinearSolverModel, LinearSolver
from .solver_model import Solver
from petram.namespace_mixin import NS_mixin


import sys
import numpy as np
import scipy
from scipy.sparse import coo_matrix, csr_matrix

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('StrumpackModel')


if use_parallel:
    from petram.helper.mpi_recipes import *
    from mfem.common.parcsr_extra import *
    import mfem.par as mfem
    default_kind = 'hypre'

    from mpi4py import MPI
    num_proc = MPI.COMM_WORLD.size
    myid = MPI.COMM_WORLD.rank
    smyid = '{:0>6d}'.format(myid)
    from mfem.common.mpi_debug import nicePrint

else:
    import mfem.ser as mfem
    default_kind = 'scipy'
    num_proc = 1
    myid = 0

    def nicePrint(*x):
        print(x)


attr_names = ['log_level',
              'ordering',
              'mc64job',
              'compression',
              'compression_rel_tol',
              'compression_abs_tol',
              'lossy_precision',
              'krylov',
              'rctol',
              'actol',
              'gmres_restart',
              'gstype',
              'maxiter',
              'use_gpu',
              'cuda_cutoff',
              'cuda_streams',
              'MUMPS_SYMQAMD',
              'agg_amalg',
              'indirect_sampling',
              'replace_tiny_pivots',
              'compression_min_sep_size',
              'compression_min_front_size',
              'compression_leaf_size',
              'separator_ordering_level',
              'hodlr_butterfly_levels',
              'use_single_precision',
              'use_64_int',
              'write_mat',
              'extra_options',
              'use_dist_sol', ]

ordering_modes = {"Natural": "STRUMPACK_NATURAL",
                  "Metis": "STRUMPACK_METIS",
                  "ParMetis": "STRUMPACK_PARMETIS",
                  "Scotch": "STRUMPACK_SCOTCH",
                  "PT-Scotch": "STRUMPACK_PTSCOTCH",
                  "RCM": "STRUMPACK_RCM"}

krylov_modes = {"Auto": "STRUMPACK_AUTO",
                "Direct": "STRUMPACK_DIRECT",
                "Refinement": "STRUMPACK_REFINE",
                "PGMRES": "STRUMPACK_PREC_GMRES",
                "GMRES": "STRUMPACK_GMRES",
                "PBICGSTAB": "STRUMPACK_PREC_BICGSTAB",
                "BICGSTAB": "BICGSTAB", }

compression_modes = {"None": "STRUMPACK_NONE",
                     "HSS": "STRUMPACK_HSS",
                     "BLR": "STRUMPACK_BLR",
                     "HOLDR": "STRUMPACK_HODLR",
                     "HOLBF": "STRUMPACK_HODLR",  # this is special
                     "LOESSLESS": "STRUMPACK_LOSSLESS",
                     "LOSSY": "STRUMPACK_LOSSY", }

GramSchmidt_types = {"Classical": "STRUMPACK_CLASSICAL",
                     "Modified": "STRUMPACK_MODIFIED"}


help_txt = ("0: no reordering for stability, this disables MC64/matching",
            "1: MC64(1): currently not supported",
            "2: MC64(2): maximize the smallest diagonal value",
            "3: MC64(3): maximize the smallest diagonal value, different strategy",
            "4: MC64(4): maximize sum of diagonal values",
            "5: MC64(5): maximize product of diagonal values and apply row and column scaling",
            "6: Combinatorial BLAS: approximate weight perfect matching",)


class Strumpack(LinearSolverModel):
    hide_ns_menu = True
    has_2nd_panel = False
    accept_complex = True
    always_new_panel = False

    def __init__(self, *args, **kwargs):
        LinearSolverModel.__init__(self, *args, **kwargs)

    def init_solver(self):
        pass

    def panel1_param(self):
        return [
            ["log_level", self.log_level, 400, {}],
            ["ordering", self.ordering, 4, {"readonly": True,
                                            "choices": list(ordering_modes), }],
            ["mc64 matching", self.mc64job, 4, {"readonly": True,
                                                "choices": ["0 (default)", "2", "3", "4", "5", "6"], }],
            ["compression", self.compression, 4, {"readonly": True,
                                                  "choices": list(compression_modes)}, ],
            ["compression rel. tol.", self.compression_rel_tol, 0, {}],
            ["compression abs. tol.", self.compression_abs_tol, 0, {}],
            ["lossy precision", self.lossy_precision, 0, {}],
            ["Krylov", self.krylov, 4, {"readonly": True,
                                        "choices": list(krylov_modes)}],
            ["rctol", self.rctol, 0, {}],
            ["actol", self.actol, 0, {}],
            ["gmres restrart", self.gmres_restart, 0, {}],
            ["GramSchmidt type", self.gstype, 4, {"readonly": True,
                                                  "choices": list(GramSchmidt_types), }],
            ["max iter.", self.maxiter, 0, {}],
            ["GPU", self.use_gpu, 3, {"text": ""}],
            ["CUDA cutoff", self.cuda_cutoff, 0, {}],
            ["CUDA streams", self.cuda_streams, 0, {}],
            ["MUMPS_SYMQAMD", self.MUMPS_SYMQAMD, 3, {"text": ""}],
            ["agg_amalg", self.agg_amalg, 3, {"text": ""}],
            ["indirect_sampling", self.indirect_sampling, 3, {"text": ""}],
            ["replace_tiny_pivots", self.replace_tiny_pivots, 3, {"text": ""}],
            ["compression min sep", self.compression_min_sep_size, 0, {}],
            ["compression min front", self.compression_min_front_size, 0, {}],
            ["compression leaf size", self.compression_leaf_size, 0, {}],
            ["separator ordering level", self.separator_ordering_level, 0, {}],
            ["hodlr butterfly levels", self.hodlr_butterfly_levels, 0, {}],
            ["single preceision", self.use_single_precision, 3, {"text": ""}],
            ["use 64bit integer", self.use_64_int, 3, {"text": ""}],
            ["write matrix", self.write_mat, 3, {"text": ""}],
            ["extra options", self.extra_options, 2235, {'nlines': 3}, ],
            ["use dist, SOL (dev.)", self.use_dist_sol, 3, {"text": ""}], ]

    def get_panel1_value(self):
        ans = []
        for n, p in zip(attr_names, self.panel1_param()):
            value = getattr(self, n)
            if p[3] == 3:
                value = bool(value)
            elif p[3] == 400:
                value = int(value)
            ans.append(value)
        return ans

    def import_panel1_value(self, v):
        for value, n, p in zip(v, attr_names, self.panel1_param()):
            if p[3] == 3:
                value = bool(value)
            if p[3] == 400:
                value = int(value)
            setattr(self, n, value)

    def attribute_set(self, v):
        v = super(Strumpack, self).attribute_set(v)
        v['log_level'] = 0
        v['write_mat'] = False
        v['gstype'] = 'Classical'
        v['rctol'] = "1e-6 (default)"
        v['actol'] = "1e-10 (default)"
        v['maxiter'] = "5000 (default)"
        v['gmres_restart'] = "30 (default)"
        v['mc64job'] = "0 (default)"
        v['krylov'] = 'Auto'
        v['use_gpu'] = False
        v['ordering'] = "Metis"
        v['compression'] = "None"
        v['use_single_precision'] = False
        v["cuda_cutoff"] = "500 (default)"
        v["cuda_streams"] = "10 (default)"
        v["MUMPS_SYMQAMD"] = False
        v["agg_amalg"] = False
        v["indirect_sampling"] = False
        v["replace_tiny_pivots"] = False
        v["compression_min_sep_size"] = "2147483647 (default)"
        v["compression_min_front_size"] = "2147483647 (default)"
        v["compression_leaf_size"] = "2147483647 (default)"
        v["separator_ordering_level"] = "1 (default)"
        v["hodlr_butterfly_levels"] = "100 (default)"
        v["compression_rel_tol"] = "1e-4 (default)"
        v["compression_abs_tol"] = "1e-8 (default)"
        v["lossy_precision"] = "16 (default)"
        v["extra_options"] = ""
        v["use_64_int"] = False
        v['use_dist_sol'] = True

        return v

    def does_linearsolver_choose_linearsystem_type(self):
        return True

    def linear_system_type(self, assemble_real, phys_real):
        if phys_real:
            return 'blk_interleave'
        else:
            return 'blk_merged'

    def real_to_complex(self, solall, M):
        if self.parent.assemble_real:
            return self.real_to_complex_merged(solall, M)
        else:
            assert False, "should not come here"

    def real_to_complex_merged(self, solall, M):
        if use_parallel:
            from mpi4py import MPI
            myid = MPI.COMM_WORLD.rank

            offset = M.RowOffsets().ToList()
            of = [np.sum(MPI.COMM_WORLD.allgather(np.int32(o)))
                  for o in offset]
            if myid != 0:
                return

        else:
            offset = M.RowOffsets()
            of = offset.ToList()
        dprint1(of)
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
        solver = StrumpackSolver(self, engine)
        solver.AllocSolver(is_complex, self.use_single_precision)
        return solver


def get_block(Op, i, j):
    try:
        return Op._linked_op[(i, j)]
    except KeyError:
        return None


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

    map = [blk_stm_idx_map(i) for i in range(rows)]
    # nicePrint("map", map)
    newi = []
    newj = []
    newd = []
    nrows = np.sum(local_size)
    ncols = np.sum(global_size)

    from scipy.sparse import bmat

    elements = [None] * rows
    elements = [elements.copy() for x in range(cols)]
    for i in range(rows):
        for j in range(cols):
            m = get_block(A, i, j)
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


class StrumpackSolver(LinearSolver):
    def __init__(self, gui, engine):
        LinearSolver.__init__(self, gui, engine)

    def spss_set_options(self):
        try:
            import STRUMPACK as ST
        except BaseException:
            assert False, "Can not load STRUMPACK"

        o = getattr(ST, ordering_modes[self.gui.ordering])
        self.spss.set_reordering_method(o)

        if self.gui.mc64job.find('default') == -1:
            job = int(self.gui.mc64job.split('(')[0])
            self.spss.set_matching(job)
        else:
            self.spss.set_matching(ST.STRUMPACK_MATCHING_NONE)

        # compression
        o = getattr(ST, compression_modes[self.gui.compression])
        self.spss.set_compression(o)
        if self.gui.compression_rel_tol.find('default') == -1:
            tol = float(self.gui.compression_rel_tol.split('(')[0])
            self.spss.set_compression_rel_tol(tol)
        if self.gui.compression_abs_tol.find('default') == -1:
            tol = float(self.gui.compression_abs_tol.split('(')[0])
            self.spss.set_compression_abs_tol(tol)
        if self.gui.compression == "HOLBF":
            l = int(self.gui.hodlr_butterfly_levels.split('(')[0])
            self.spss.set_compression_butterfly_levels(l)
        if self.gui.compression_min_sep_size.find('default') == -1:
            l = int(self.gui.compression_min_sep_size.split('(')[0])
            self.spss.set_compression_min_sep_size(l)
        if self.gui.compression_min_front_size.find('default') == -1:
            l = int(self.gui.compression_min_front_size.split('(')[0])
            self.spss.set_compression_min_front_size(l)
        if self.gui.compression_leaf_size.find('default') == -1:
            l = int(self.gui.compression_leaf_size.split('(')[0])
            self.spss.set_compression_leaf_size(l)

        # iterative
        o = getattr(ST, krylov_modes[self.gui.krylov])
        self.spss.set_Krylov_solver(o)
        if self.gui.rctol.find('default') == -1:
            tol = float(self.gui.rctol.split('(')[0])
            self.spss.set_rel_tol(tol)
        if self.gui.actol.find('default') == -1:
            tol = float(self.gui.actol.split('(')[0])
            self.spss.set_abs_tol(tol)
        if self.gui.gmres_restart.find('default') == -1:
            m = int(self.gui.gmres_restart.split('(')[0])
            self.spss.set_gmres_restart(m)
        gs = getattr(ST, GramSchmidt_types[self.gui.gstype])
        self.spss.set_GramSchmidt_type(gs)
        if self.gui.maxiter.find('default') == -1:
            it = int(self.gui.maxiter.split('(')[0])
            self.spss.set_maxit(it)

        # gpu
        if self.gui.use_gpu:
            self.spss.enable_gpu()
        else:
            self.spss.disable_gpu()

    def spss_options_args(self):
        opts = ["--sp_enable_METIS_NodeND", ]
#        opts = ["--sp_enable_METIS_NodeNDP", ]
#               "--sp_enable_METIS_NodeND"]
        if self.gui.lossy_precision.find('default') == -1:
            tol = int(self.gui.lossy_precision.split('(')[0])
            if tol != 16:
                opts.extend(["--sp_lossy_precision", str(tol)])

        if self.gui.use_gpu:
            if self.gui.cuda_cutoff.find('default') == -1:
                cutoff = int(self.gui.cuda_cutoff.split('(')[0])
                opts.extend(["--sp_cuda_cutoff", str(cutoff)])
            if self.gui.cuda_streams.find('default') == -1:
                streams = int(self.gui.cuda_streams.split('(')[0])
                opts.extend(["--sp_cuda_streams", str(streams)])

        if self.gui.MUMPS_SYMQAMD:
            opts.extend(["--sp_enable_MUMPS_SYMQAMD", ])
        else:
            opts.extend(["--sp_disable_MUMPS_SYMQAMD", ])

        if self.gui.agg_amalg:
            opts.extend(["--sp_enable_agg_amalg", ])
        else:
            opts.extend(["--sp_disable_agg_amalg", ])

        if self.gui.indirect_sampling:
            opts.extend(["--sp_enable_indirect_sampling", ])
        else:
            opts.extend(["--sp_disable_indirect_sampling", ])

        if self.gui.replace_tiny_pivots:
            opts.extend(["--sp_enable_replace_tiny_pivots", ])
        else:
            opts.extend(["--sp_disable_replace_tiny_pivots", ])

        if self.gui.separator_ordering_level.find('default') == -1:
            l = int(self.gui.separator_ordering_level.split('(')[0])
            opts.extend(["--sp_separator_ordering_level", str(l)])

        for x in self.gui.extra_options.split("\n"):
            opts.extend([x.strip()
                        for x in x.split(" ") if len(x.strip()) > 0])

        # opts.append("")
        return opts

    def AllocSolver(self, is_complex, use_single_precision):
        try:
            import STRUMPACK as ST
        except BaseException:
            assert False, "Can not load STRUMPACK"
        dprint1("AllocSolver", is_complex, use_single_precision)

        opts = self.spss_options_args()
        dprint1("options", opts, notrim=True)
        verbose = self.gui.log_level > 0
        if use_parallel:
            args = (MPI.COMM_WORLD, opts, verbose)
        else:
            args = (opts, verbose)

        if is_complex:
            if use_single_precision:
                dtype = np.complex64
                if self.gui.use_64_int:
                    spss = ST.C64StrumpackSolver(*args)
                else:
                    spss = ST.CStrumpackSolver(*args)
            else:
                dtype = np.complex128
                if self.gui.use_64_int:
                    spss = ST.Z64StrumpackSolver(*args)
                else:
                    spss = ST.ZStrumpackSolver(*args)
        else:
            if use_single_precision:
                dtype = np.float32
                if self.gui.use_64_int:
                    spss = ST.S64StrumpackSolver(*args)
                else:
                    spss = ST.SStrumpackSolver(*args)
            else:
                dtype = np.float64
                if self.gui.use_64_int:
                    spss = ST.D64StrumpackSolver(*args)
                else:
                    spss = ST.DStrumpackSolver(*args)

        assert spss.isValid(), "Failed to create STRUMPACK solver object"

        spss.set_from_options()

        self.dtype = dtype
        self.spss = spss
        self.is_complex = is_complex
        self.spss_set_options()

    def SetOperator(self, A, dist, name=None):
        try:
            from mpi4py import MPI
        except BaseException:
            from petram.helper.dummy_mpi import MPI
        myid = MPI.COMM_WORLD.rank
        nproc = MPI.COMM_WORLD.size

        self.row_offsets = A.RowOffsets()

        AA = build_csr_local(A, self.dtype, self.is_complex)

        if self.gui.write_mat:
            write_coo_matrix('matrix', AA.tocoo())

        if dist:
            self.spss.set_distributed_csr_matrix(AA)
        else:
            self.spss.set_csr_matrix(AA)
        self._matrix = AA

    def Mult(self, b, x=None, case_base=0):
        try:
            from mpi4py import MPI
        except BaseException:
            from petram.helper.dummy_mpi import MPI
        myid = MPI.COMM_WORLD.rank
        nproc = MPI.COMM_WORLD.size
        try:
            import STRUMPACK as ST
        except BaseException:
            assert False, "Can not load STRUMPACK"

        sol = []
        row_offsets = self.row_offsets.ToList()

        MPI.COMM_WORLD.Barrier()

        dprint1("calling reorder", debug.format_memory_usage())
        ret = self.spss.reorder()

        ret = np.sum(MPI.COMM_WORLD.allgather(int(ret != ST.STRUMPACK_SUCCESS)))
        if ret > 0:
            assert False, "error during recordering (Strumpack)"

        MPI.COMM_WORLD.Barrier()

        dprint1("calling factor", debug.format_memory_usage())
        ret = self.spss.factor()

        zero_pivot = np.sum(MPI.COMM_WORLD.allgather(
            int(ret == ST.STRUMPACK_ZERO_PIVOT)))
        if zero_pivot != 0:
            dprint1("!!!! Zero pivots are detected on " +
                    str(zero_pivot) + " processes")
            dprint1("!!!! continuing ....")

        ret = np.sum(MPI.COMM_WORLD.allgather(int(ret != ST.STRUMPACK_SUCCESS and
                                                  ret != ST.STRUMPACK_ZERO_PIVOT)))
        if ret > 0:
            assert False, "error during factor (Strumpack)"

        return_distributed = self.gui.use_dist_sol

        for kk, bb in enumerate(b):
            rows = MPI.COMM_WORLD.allgather(np.int32(bb.Size()))
            rowstarts = np.hstack((0, np.cumsum(rows)))
            # nicePrint("rowstarts/offser",rowstarts, row_offsets)
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

            MPI.COMM_WORLD.Barrier()
            dprint1("calling solve", debug.format_memory_usage())
            ret = self.spss.solve(bbv, xxv, False)

            ret = np.sum(MPI.COMM_WORLD.allgather(
                int(ret != ST.STRUMPACK_SUCCESS)))
            if ret > 0:
                assert False, "error during solve phase (Strumpack)"

            if return_distributed:
                sol.append(xxv)
            else:
                s = []
                for i in range(len(row_offsets) - 1):
                    r1 = row_offsets[i]
                    r2 = row_offsets[i + 1]

                    if self.is_complex:
                        r1 = r1 // 2
                        r2 = r2 // 2
                    xxvv = xxv[r1:r2]

                    if use_parallel:
                        vv = gather_vector(xxvv)
                    else:
                        vv = xxvv.copy()

                    if myid == 0:
                        s.append(vv)
                    else:
                        pass
                if myid == 0:
                    sol.append(np.hstack(s))

        if return_distributed:
            sol = np.transpose(np.vstack(sol))
            return sol
        else:
            if myid == 0:
                sol = np.transpose(np.vstack(sol))
                return sol
            else:
                return None
