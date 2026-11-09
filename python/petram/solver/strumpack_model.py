from petram.helper.check_module_installed import module_file_exists
from petram.helper.matrix_file import write_matrix, write_vector, write_coo_matrix
from petram.mfem_config import use_parallel
from .solver_model import LinearSolverModel, LinearSolver
from .solver_model import Solver
from petram.namespace_mixin import NS_mixin
from petram.solver.solver_utils import build_csr_local, check_operator_type


import os
import sys
from contextlib import contextmanager
import numpy as np
import scipy
from scipy.sparse import csr_matrix

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

has_strumpack = bool(module_file_exists(
    "petram.ext.strumpack.strumpack_solve"))


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
              'cuda_streams', ]

attr_names2 = ['MUMPS_SYMQAMD',
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
                "BICGSTAB": "STRUMPACK_BICGSTAB", }

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
            '3: MC64(3): maximize the smallest diagonal value, different\n'
            'strategy',
            "4: MC64(4): maximize sum of diagonal values",
            '5: MC64(5): maximize product of diagonal values and apply\n'
            'row and column scaling',
            "6: Combinatorial BLAS: approximate weight perfect matching",)


class Strumpack(LinearSolverModel):
    hide_ns_menu = True
    has_2nd_panel = True
    accept_complex = True
    always_new_panel = False
    supports_rank_reduction = True

    def __init__(self, *args, **kwargs):
        LinearSolverModel.__init__(self, *args, **kwargs)

    def init_solver(self):
        pass

    def _clean_properties_with_default(self):
        names = ("rctol", "actol", "maxiter", "gmres_restart", "mc64job", "cuda_streams",
                 "compression_min_sep_size", "compression_min_front_size", "compression_leaf_size",
                 "separator_ordering_level", "hodlr_butterfly_levels", "compression_rel_tol",
                 "compression_abs_tol", "lossy_precision")
        for name in names:
            if hasattr(self, name):
                value = getattr(self, name)
                if str(value).find("default") != -1:
                    setattr(self, name, "(default)")

    def panel1_param(self):
        self._clean_properties_with_default()
        panel = [
            ["log_level", self.log_level, 400, {}],
            ["ordering", self.ordering, 4, {"readonly": True,
                                            "choices": list(ordering_modes), }],
            ["mc64 matching", self.mc64job, 4, {"readonly": True,
                                                "choices": ["(default)", "0", "2", "3", "4", "5", "6"], }],
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
            ["GPU streams", self.cuda_streams, 0, {}], ]
        if self.supports_rank_reduction:
            panel.append(
                [None, (self.reduce_mpi_proc,
                        [int(self.solver_mpi_ranks), self.maximize_omp_threads]), 27,
                 ({"text": "Reduce MPI processes for solver"},
                  {"elp": [["Solver MPI size", int(self.solver_mpi_ranks), 400, {}],
                           ["Maximize OMP threads", self.maximize_omp_threads, 3,
                            {"text": ""}]],
                   "tip": ["-1 uses PETRAM_STRUMPACK_MPI_SIZE if set, otherwise all application ranks. A positive value overrides the environment.",
                           "Use the node-local application OpenMP thread budget on the ranks participating in STRUMPACK."]})])
        return panel

    def panel1_tip(self):
        tips = [
            'STRUMPACK output level: 0 suppresses solver output',
            'Ordering algorithm used to reduce fill-in during factorization.',
            ('Matching improves diagonal entries before factorization.\n'
             'Petra-M defaults to 0 (disabled)\n' + "\n".join(help_txt)),
            'Compression method for the factors. (default: none)',
            (
                'Relative tolerance for low-rank compression.\n'
                '(default[D/S]: HSS 0.01/0.1, BLR 1e-4/1e-2, HODLR 1e-4/1e-2)'
            ),
            (
                'Absolute tolerance for low-rank compression.\n'
                '(default[D/S]: HSS 1e-8/1e-5, BLR 1e-12/1e-6, HODLR 1e-10/1e-5)'
            ),
            'Bit precision for LOSSY compression.',
            'Linear solution method. (default: auto)',
            'Relative convergence tolerance for iterative solves.(default[D/S]: 1e-6/1e-4)',
            'Absolute convergence tolerance for iterative solves.(default[D/S]: 1e-10/1e-6)',
            'Number of GMRES iterations before restarting.(default: 30)',
            'Orthogonalization method used by GMRES. (default: Modified)',
            'Maximum number of iterative solver iterations. (default: 5000)\n',
            'Enable GPU acceleration.',
            'Number of GPU streams when GPU acceleration is enabled.(default: 4)',
        ]
        if self.supports_rank_reduction:
            tips.append((
                'Use fewer MPI ranks for this standalone solver. Set the\n'
                'solver MPI size and optionally maximize OpenMP threads on\n'
                'participating ranks.'
            ))
        return tips

    def get_panel1_value(self):
        ans = []
        for n, p in zip(attr_names, self.panel1_param()):
            value = getattr(self, n)
            if p[2] == 3:
                value = bool(value)
            elif p[2] == 400:
                value = int(value)
            ans.append(value)
        if self.supports_rank_reduction:
            ans.append((self.reduce_mpi_proc,
                        [int(self.solver_mpi_ranks), self.maximize_omp_threads]))
        return ans

    def import_panel1_value(self, v):
        for value, n, p in zip(v, attr_names, self.panel1_param()):
            if p[2] == 3:
                value = bool(value)
            if p[2] == 400:
                value = int(value)
            setattr(self, n, value)
        if self.supports_rank_reduction:
            self.reduce_mpi_proc, values = v[-1]
            self.solver_mpi_ranks = int(values[0])
            self.maximize_omp_threads = bool(values[1])

    def panel2_tabname(self):
        return "Advanced"

    def panel2_param(self):
        self._clean_properties_with_default()
        return [
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
            ["use dist. Sol", self.use_dist_sol, 3, {"text": ""}], ]

    def panel2_tip(self):
        return ['Use the MUMPS symmetric approximate minimum-degree ordering (MPI only)',
                'Enable aggressive amalgamation of supernodes.(default off)',
                'Enable indirect sampling for HSS compression.(default off)',
                'Replace tiny pivots during factorization to improve robustness.(default off)',
                ('Minimum separator size for compression.\n'
                 '(default: HSS 1000, BLR 512, HODLR 5000, Lossy/Lossless 8)'
                 ),
                'Minimum front size for compression. (default: 100000)',
                (
                    'Leaf size for hierarchical compression.\n'
                    '(default: HSS 512, BLR/HODLR 256, Lossy/Lossless 4)'
                ),
                'Level at which separator reordering starts for compression.(default: 1)'
                'Number of butterfly levels, applied only for HOLBF.(default: 0)',
                'Use single precision for factorization and solution.(default off)',
                'Use 64-bit matrix indices. (default off)',
                'Write the matrix and right-hand side to disk.',
                (
                    'Additional STRUMPACK command-line options.\n'
                    'Separate by spaces or newlines. Panel settings may override\n'
                    'these options; the GPU checkbox controls GPU enablement.'
                ),
                'Return the solution distributed across MPI ranks.',
                ]

    def get_panel2_value(self):
        ans = []
        for n, p in zip(attr_names2, self.panel2_param()):
            value = getattr(self, n)
            if p[2] == 3:
                value = bool(value)
            elif p[2] == 400:
                value = int(value)
            ans.append(value)
        return ans

    def import_panel2_value(self, v):
        for value, n, p in zip(v, attr_names2, self.panel2_param()):
            if p[2] == 3:
                value = bool(value)
            if p[2] == 400:
                value = int(value)
            setattr(self, n, value)

    def attribute_set(self, v):
        v = super(Strumpack, self).attribute_set(v)
        v['log_level'] = 0
        v['write_mat'] = False
        v['gstype'] = 'Modified'
        v['rctol'] = "(default)"
        v['actol'] = "(default)"
        v['maxiter'] = "(default)"
        v['gmres_restart'] = "(default)"
        v['mc64job'] = "0"
        v['krylov'] = 'Auto'
        v['use_gpu'] = False
        v['ordering'] = "Metis"
        v['compression'] = "None"
        v['use_single_precision'] = False
        v["cuda_streams"] = "(default)"
        v["MUMPS_SYMQAMD"] = False
        v["agg_amalg"] = False
        v["indirect_sampling"] = False
        v["replace_tiny_pivots"] = False
        v["compression_min_sep_size"] = "(default)"
        v["compression_min_front_size"] = "(default)"
        v["compression_leaf_size"] = "(default)"
        v["separator_ordering_level"] = "(default)"
        v["hodlr_butterfly_levels"] = "(default)"
        v["compression_rel_tol"] = "(default)"
        v["compression_abs_tol"] = "(default)"
        v["lossy_precision"] = "(default)"
        v["extra_options"] = ""
        v["use_64_int"] = False
        v['reduce_mpi_proc'] = False
        v['solver_mpi_ranks'] = -1
        v['maximize_omp_threads'] = False

        return v

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
            from mpi4py import MPI
            myid = MPI.COMM_WORLD.rank

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

    def resolve_solver_size(self, comm, environ=None):
        """
        Resolve the model/environment rank count without changing the model.
        """
        environ = os.environ if environ is None else environ
        size = int(str(self.solver_mpi_ranks).strip())
        source = 'model'
        if size == -1:
            value = environ.get('PETRAM_STRUMPACK_MPI_SIZE')
            source = 'default' if value is None else 'environment'
            size = comm.size if value is None else int(value.strip())
        if not 1 <= size <= comm.size:
            raise ValueError(
                f'Solver MPI ranks must be in 1..{comm.size}, got {size}')
        return size, source

    def allocate_solver(self, is_complex=False, engine=None):
        size = num_proc
        if use_parallel and self.supports_rank_reduction and self.reduce_mpi_proc:
            from petram.helper.mpi_recipes import collective_call
            size, source = collective_call(
                MPI.COMM_WORLD, lambda: self.resolve_solver_size(MPI.COMM_WORLD))
            sizes = MPI.COMM_WORLD.allgather(size)
            if len(set(sizes)) != 1:
                raise ValueError(
                    'STRUMPACK solver size differs between application ranks')
            dprint1(
                f'STRUMPACK MPI ranks: application={num_proc}, solver={size} ({source})')
        if size < num_proc:
            return ReducedStrumpackSolver(self, engine, MPI.COMM_WORLD, size, is_complex)

        solver = StrumpackSolver(self, engine)
        try:
            solver.AllocSolver(is_complex, self.use_single_precision)
        except Exception:
            solver.close()
            raise
        return solver


class StrumpackSolver(LinearSolver):
    def __init__(self, gui, engine, comm=None, rank_layout=None):
        LinearSolver.__init__(self, gui, engine)
        if comm is None:
            try:
                from mpi4py import MPI
            except BaseException:
                from petram.helper.dummy_mpi import MPI
            comm = MPI.COMM_WORLD
        self.comm = comm
        self.omp_threads = None
        if rank_layout is not None and self.gui.maximize_omp_threads:
            from petram.ext.strumpack.strumpack_solve import get_omp_max_threads
            self.omp_threads = (get_omp_max_threads() * rank_layout.node_app_size
                                // rank_layout.node_solver_size)
            dprint1(
                f'STRUMPACK OpenMP threads per solver rank: {self.omp_threads}')
        self.spss = None
        self.closed = False

    @contextmanager
    def _omp_scope(self):
        """
        Temporarily apply this solver's OpenMP limit and cached thread settings.
        """
        if self.omp_threads is None:
            yield
            return
        from petram.ext.strumpack.strumpack_solve import (get_omp_max_threads,
                                                          set_omp_num_threads)

        previous = get_omp_max_threads()
        try:
            set_omp_num_threads(self.omp_threads)
            yield
        finally:
            set_omp_num_threads(previous)

    def close(self):
        if not self.closed:
            if self.spss is not None:
                with self._omp_scope():
                    self.spss.close()
            self.spss = None
            self.closed = True

    def spss_set_options(self):
        try:
            from petram.ext.strumpack import strumpack_solve as ST
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
            if self.gui.hodlr_butterfly_levels.find('default') == -1:
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

    def spss_options_args(self):
        opts = ["--sp_enable_METIS_NodeND", ]
#        opts = ["--sp_enable_METIS_NodeNDP", ]
#               "--sp_enable_METIS_NodeND"]
        if self.gui.lossy_precision.find('default') == -1:
            tol = int(self.gui.lossy_precision.split('(')[0])
            opts.extend(["--sp_lossy_accuracy", "-1",
                         "--sp_lossy_precision", str(tol)])

        if self.gui.use_gpu:
            if self.gui.cuda_streams.find('default') == -1:
                streams = int(self.gui.cuda_streams.split('(')[0])
                opts.extend(["--sp_gpu_streams", str(streams)])

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

        # The checkbox controls GPU initialization, including extra options.
        opts.append(
            '--sp_enable_gpu' if self.gui.use_gpu else '--sp_disable_gpu')
        return opts

    def AllocSolver(self, is_complex, use_single_precision):
        self.is_complex = is_complex
        if is_complex:
            self.dtype = np.complex64 if use_single_precision else np.complex128
        else:
            self.dtype = np.float32 if use_single_precision else np.float64
        try:
            from petram.ext.strumpack import strumpack_solve as ST
        except BaseException:
            assert False, "Can not load STRUMPACK"
        dprint1("AllocSolver", is_complex, use_single_precision)

        opts = self.spss_options_args()
        dprint1("options", opts, notrim=True)
        verbose = self.gui.log_level > 0
        if use_parallel:
            args = (self.comm, opts, verbose)
        else:
            args = (opts, verbose)

        with self._omp_scope():
            if is_complex:
                if use_single_precision:
                    if self.gui.use_64_int:
                        spss = ST.C64StrumpackSolver(*args)
                    else:
                        spss = ST.CStrumpackSolver(*args)
                else:
                    if self.gui.use_64_int:
                        spss = ST.Z64StrumpackSolver(*args)
                    else:
                        spss = ST.ZStrumpackSolver(*args)
            else:
                if use_single_precision:
                    if self.gui.use_64_int:
                        spss = ST.S64StrumpackSolver(*args)
                    else:
                        spss = ST.SStrumpackSolver(*args)
                else:
                    if self.gui.use_64_int:
                        spss = ST.D64StrumpackSolver(*args)
                    else:
                        spss = ST.DStrumpackSolver(*args)

            dprint1("StrumpackSolvar created: " + str(spss.__class__))
            assert spss.isValid(), "Failed to create STRUMPACK solver object"

            self.spss = spss
            spss.set_from_options()
            self.spss_set_options()

    def SetOperator(self, A, dist, name=None):
        self.row_offsets = A.RowOffsets()
        self.op_type = check_operator_type(self.is_complex, A)
        AA = build_csr_local(A, self.dtype, self.is_complex)

        if self.gui.write_mat:
            write_coo_matrix('matrix', AA.tocoo())
        self.SetCSROperator(AA, dist)

    def SetCSROperator(self, A, dist):
        """
        Install CSR rows already numbered for this solver's communicator.
        """
        with self._omp_scope():
            if dist:
                self.spss.set_distributed_csr_matrix(A)
            else:
                self.spss.set_csr_matrix(A)
        self._matrix = A

    def Factor(self):
        from petram.ext.strumpack import strumpack_solve as ST

        self.comm.Barrier()

        dprint1("calling reorder", debug.format_memory_usage())
        with self._omp_scope():
            ret = self.spss.reorder()

        ret = np.sum(self.comm.allgather(
            int(ret != ST.STRUMPACK_SUCCESS)))
        if ret > 0:
            assert False, "error during recordering (Strumpack)"

        self.comm.Barrier()

        dprint1("calling factor", debug.format_memory_usage())
        with self._omp_scope():
            ret = self.spss.factor()

        zero_pivot = np.sum(self.comm.allgather(
            int(ret == ST.STRUMPACK_ZERO_PIVOT)))
        if zero_pivot != 0:
            dprint1("!!!! Zero pivots are detected on " +
                    str(zero_pivot) + " processes")
            dprint1("!!!! continuing ....")

        ret = np.sum(self.comm.allgather(
            int(ret != ST.STRUMPACK_SUCCESS and ret != ST.STRUMPACK_ZERO_PIVOT)))
        if ret > 0:
            assert False, "error during factor (Strumpack)"

    def Solve(self, b, x):
        from petram.ext.strumpack import strumpack_solve as ST

        self.comm.Barrier()
        dprint1("calling solve", debug.format_memory_usage())
        with self._omp_scope():
            ret = self.spss.solve(b, x, False)
        ret = np.sum(self.comm.allgather(int(ret != ST.STRUMPACK_SUCCESS)))
        if ret > 0:
            assert False, "error during solve phase (Strumpack)"

    def Mult(self, b, x=None, case_base=0):
        return _strumpack_mult(self, b, x)


class ReducedStrumpackSolver(LinearSolver):
    """
    Aggregate application CSR rows and return solutions to their owners.
    """

    def __init__(self, gui, engine, app_comm, size, is_complex):
        from petram.helper.solver_mpi import SolverRankLayout

        super().__init__(gui, engine)
        self.comm = app_comm
        self.ranks = SolverRankLayout(app_comm, size)
        self.is_complex = is_complex
        if is_complex:
            self.dtype = np.complex64 if gui.use_single_precision else np.complex128
        else:
            self.dtype = np.float32 if gui.use_single_precision else np.float64
        self.backend = None
        self.closed = False

        def allocate():
            self.backend = StrumpackSolver(gui, engine, comm=self.ranks.solver_comm,
                                           rank_layout=self.ranks)
            self.backend.AllocSolver(is_complex, gui.use_single_precision)

        try:
            self._call_backend(allocate)
        except Exception:
            self.close()
            raise

    def _call_backend(self, operation):
        if self.closed:
            raise RuntimeError('STRUMPACK solver is closed')
        return self.ranks.call_solver(operation)

    def SetOperator(self, A, dist, name=None):
        from petram.helper.mpi_recipes import gather_vector

        self.row_offsets = A.RowOffsets()
        self.op_type = check_operator_type(self.is_complex, A)
        AA = build_csr_local(A, self.dtype, self.is_complex)
        if self.gui.write_mat:
            write_coo_matrix('matrix', AA.tocoo())

        group = self.ranks.group_comm
        counts = np.asarray(self.comm.allgather(AA.shape[0]), dtype=np.int64)
        starts = np.hstack(([0], np.cumsum(counts)))
        leader = group.bcast(self.comm.rank if self.ranks.is_leader else None)
        members = self.comm.allgather((leader, group.rank))
        order = sorted(range(self.comm.size), key=lambda rank: members[rank])

        # Gathered rows follow leader/group order. Apply the same
        # permutation to columns, using rank offsets instead of a global map.
        new_starts = np.empty(self.comm.size, dtype=np.int64)
        new_starts[order] = np.hstack(([0], np.cumsum(counts[order])))[:-1]
        owners = np.searchsorted(starts[1:], AA.indices, side='right')
        columns = new_starts[owners] + AA.indices - starts[owners]

        lengths = gather_vector(np.asarray(
            np.diff(AA.indptr), dtype=np.int64), world=group)
        columns = gather_vector(np.ascontiguousarray(columns), world=group)
        values = gather_vector(AA.data, world=group)

        def set_matrix():
            indptr = np.hstack(([0], np.cumsum(lengths)))
            matrix = csr_matrix((values, columns, indptr),
                                shape=(len(lengths), int(starts[-1])))
            self.backend.SetCSROperator(matrix, True)

        self._call_backend(set_matrix)

    def Factor(self):
        self._call_backend(lambda: self.backend.Factor())

    def Solve(self, b, x):
        from petram.helper.mpi_recipes import gather_vector, scatter_vector

        group = self.ranks.group_comm
        rhs = gather_vector(b, world=group)
        sol = gather_vector(x, world=group)
        self._call_backend(lambda: self.backend.Solve(rhs, sol))
        x[:] = scatter_vector(sol, rcounts=x.size, comm=group)

    def Mult(self, b, x=None, case_base=0):
        # _strumpack_mult calls Solve, where theRHS is gathered
        return _strumpack_mult(self, b, x)

    def close(self):
        if not self.closed:
            self._call_backend(lambda: self.backend.close()
                               if self.backend is not None else None)
            self.backend = None
            self.ranks.close()
            self.closed = True


def _strumpack_mult(solver, b, x):
    """
    Pack MFEM block vectors and format full-rank or reduced-rank solutions.
    """
    myid = solver.comm.rank
    sol = []
    row_offsets = solver.row_offsets.ToList()
    solver.Factor()

    return_distributed = solver.gui.use_dist_sol

    for kk, bb in enumerate(b):
        if x is None:
            xx = mfem.BlockVector(solver.row_offsets)
            xx.Assign(0.0)
        else:
            xx = x

        if solver.is_complex:
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

        bbv = np.ascontiguousarray(bbv, dtype=solver.dtype)
        xxv = np.ascontiguousarray(xxv, dtype=solver.dtype)

        if solver.gui.write_mat:
            write_vector('rhs_' + str(kk), bbv)
            write_vector('x_' + str(kk), xxv)

        sys.stdout.flush()
        sys.stderr.flush()

        solver.Solve(bbv, xxv)

        # MFEM stores solution vectors in double precision.
        dtype = np.complex128 if solver.is_complex else np.float64
        xxv = xxv.astype(dtype, copy=False)
        if not solver.is_complex:
            xx.GetDataArray()[:] = xxv

        if return_distributed:
            # A supplied x is reused for the next RHS; retain this result.
            sol.append(xxv.copy())
        else:
            s = []
            for i in range(len(row_offsets) - 1):
                r1 = row_offsets[i]
                r2 = row_offsets[i + 1]

                if solver.op_type == "real":
                    xxvv = xxv[r1:r2]
                    if use_parallel:
                        vv = gather_vector(xxvv, world=solver.comm)
                    else:
                        vv = xxvv.copy()

                elif solver.op_type == "complex":
                    r1 = r1 // 2
                    r2 = r2 // 2
                    xxvv = xxv[r1:r2]
                    if use_parallel:
                        vv = gather_vector(xxvv, world=solver.comm)
                    else:
                        vv = xxvv.copy()

                elif solver.op_type == "complexreal":
                    r1 = r1
                    width = (r2-r1)//2
                    xxvv = xxv[r1:r1+width]
                    if use_parallel:
                        vv1 = gather_vector(xxvv, world=solver.comm)
                    else:
                        vv1 = xxvv.copy()
                    xxvv2 = xxv[r1+width:r1+width+width]
                    if use_parallel:
                        vv2 = gather_vector(xxvv2, world=solver.comm)
                    else:
                        vv2 = xxvv2.copy()
                    vv = np.hstack((vv1, vv2))
                else:
                    assert False, "unknonw operator type"

                # nicePrint("xxvv", xxvv.shape)

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


class StrumpackMFEMSolverModel(Strumpack):
    '''
    This one is to use STRUMPACK in iterative solver
    It creates MUMPSPreconditioner
    '''
    supports_rank_reduction = False

    def prepare_solver(self, opr, engine):
        solver = StrumpackBlockPreconditioner(opr,
                                              gui=self,
                                              engine=engine,
                                              silent=True)
        solver.SetOperator(opr)
        return solver


class StrumpackBlockPreconditioner(mfem.Solver):
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
            assert False, "StrumpackMFEM for Complex is not yet implemented"
            # s = self.complex_to_real(s)

        y.Assign(s.flatten().astype(float, copy=False))

    def SetOperator(self, opr):
        print('opr', opr)
        from petram.solver.solver_utils import check_block_operator
        is_complex, is_parallel = check_block_operator(opr)

        solver = StrumpackSolver(self.gui, self.engine)
        solver.AllocSolver(is_complex, self.gui.use_single_precision)
        solver.SetOperator(opr, is_parallel)

        self.is_complex_operator = is_complex
        self.is_parallel = is_parallel
        self.solver = solver
