from petram.phys.vtable import VtableElement, Vtable
from petram.helper.matrix_file import write_matrix, write_vector, write_coo_matrix
from petram.mfem_config import use_parallel
from petram.solver.solver_model import LinearSolverModel, LinearSolver
from petram.solver.solver_model import Solver
from petram.namespace_mixin import NS_mixin
from petram.solver.solver_utils import build_csr_local, check_operator_type


import sys
import numpy as np
import scipy

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('SuperLU')

DIST_ORDERINGS = {"METIS": "metis", "NATURAL": "natural",
                  "MMD_ATA": "mmd_ata", "MMD_AT_PLUS_A": "mmd_at_plus_a"}
DIST_OPTIONS = ('dist_equil',
                'dist_replace_tiny_pivot', 'dist_refinement', 'dist_log_statistics')
SCIPY_OPTIONS = ('pivot_thr_txt', 'relax_txt', 'panel_size_txt',
                 'superlu_options_txt', 'use_single_precision')


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
        v["superlu_backend"] = "SciPy"
        v["col_permute_txt"] = "COLAMD"
        v["pivot_thr_txt"] = ""
        v["relax_txt"] = ""
        v["panel_size_txt"] = ""
        v["superlu_options_txt"] = ""
        v["use_single_precision"] = False
        v["write_mat"] = False
        v["dist_equil"] = True
        v["dist_replace_tiny_pivot"] = False
        v["dist_refinement"] = False
        v["dist_log_statistics"] = False
        return v

    def verify_setting(self):
        result = super().verify_setting()
        if not result[0]:
            return result

        backend = self.superlu_backend
        if backend not in ('SciPy', 'SuperLU_DIST'):
            return False, 'Unknown SuperLU backend', str(backend)

        orderings = (('COLAMD', 'NATURAL', 'MMD_ATA', 'MMD_AT_PLUS_A')
                     if backend == 'SciPy' else tuple(DIST_ORDERINGS))
        if self.col_permute_txt not in orderings:
            return (False, 'Unsupported SuperLU ordering',
                    f'{backend} supports: {", ".join(orderings)}. '
                    'METIS is available only with SuperLU_DIST; '
                    'COLAMD is available only with SciPy.')

        return result

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
        scipy_panel = [
            ["pivot thr.", "None", 0, {}],
            ["relax.", "None", 0, {}],
            ["panel size", "None", 0, {}],
            ["options(=)", "None", 0, {"validator": validator,
                                      "validator_param": None}],
            ["use float32", False, 3, {"text": ""}]]
        dist_panel = [
            ["equilibration", True, 3, {"text": ""}],
            ["replace tiny pivots", False, 3, {"text": ""}],
            ["iterative refinement", False, 3, {"text": ""}],
            ["log statistics", False, 3, {"text": ""}]]
        scipy_tips = [
            'Threshold for accepting a diagonal entry as a pivot. (Leave blank for default)',
            'Relaxation parameter for grouping columns into supernodes. (Leave blank for default)',
            'Number of columns in a factorization panel. (Leave blank for default)',
            'Python dictionary of additional options passed to scipy.sparse.linalg.splu.',
            'Use single precision for factorization and solves.']
        dist_tips = [
            'Scale matrix rows and columns to improve numerical conditioning. (default: on)',
            'Replace tiny pivots with a small nonzero value. This modifies the factorization. (default: off)',
            'Apply double-precision iterative refinement after solving. (default: off)',
            'Print SuperLU_DIST memory usage, timing, and flop statistics to the run output.']

        return [
            [None, None, 27, ({"text": "Use MPI"},
                             {"elp": dist_panel, "tip": dist_tips},
                             {"elp": scipy_panel, "tip": scipy_tips})],
            ["col. permute", "COLAMD", 4,
             {"choices": ["COLAMD", "NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "METIS"],
              "style": wx.CB_READONLY}],
            ["write matrix", False, 3, {"text": ""}]]

    def panel1_tip(self):
        return [
            'Use MPI calls SuperLU_DIST. Otherwise the sequential SuperLU in SciPy is used.',
            'Column ordering used to reduce fill-in during factorization.\nMETIS requires MPI; COLAMD requires Sequential version.',
            'Write the assembled matrix and right-hand side to files for inspection.']

    def get_panel1_value(self):
        switch = (self.superlu_backend == 'SuperLU_DIST',
                  [getattr(self, name) for name in DIST_OPTIONS],
                  [getattr(self, name) for name in SCIPY_OPTIONS])
        return [switch, self.col_permute_txt, self.write_mat]

    def import_panel1_value(self, v):
        use_mpi, dist_values, scipy_values = v[0]
        self.superlu_backend = 'SuperLU_DIST' if use_mpi else 'SciPy'
        for name, value in zip(DIST_OPTIONS, dist_values):
            setattr(self, name, value)
        for name, value in zip(SCIPY_OPTIONS, scipy_values):
            setattr(self, name, value)
        self.col_permute_txt = v[1]
        self.write_mat = v[2]

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
        success, text, detail = self.verify_setting()
        if not success:
            raise ValueError(text + ': ' + detail)

        if self.superlu_backend == 'SuperLU_DIST':
            solver = SuperLUDistSolver(self, engine)
        else:
            solver = SuperLUSolver(self, engine)

        solver.AllocSolver(is_complex)
        return solver


class SuperLUSolverBase(LinearSolver):
    """
    Shared MFEM conversion and local vector handling for SuperLU backends.
    """

    def __init__(self, gui, engine):
        super().__init__(gui, engine)
        self._superlu = None
        self.is_complex = False
        self.dtype = None

    def SetOperator(self, A, dist, name=None):
        self.close()
        self.row_offsets = A.RowOffsets()
        # nicePrint("row offsets in SetOperator", self.row_offsets.ToList())

        self.op_type = check_operator_type(self.is_complex, A)
        AA = build_csr_local(A, self.dtype, self.is_complex)

        if self.gui.write_mat:
            write_coo_matrix('matrix', AA.tocoo())

        self._factor_matrix(AA)

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

            xxv = self._solve_rhs(bbv)

            barrier()
            sol.append(xxv)

        sol = np.transpose(np.vstack(sol))
        return sol


class SuperLUSolver(SuperLUSolverBase):
    def __init__(self, gui, engine):
        super().__init__(gui, engine)
        self.use_single_precision = False
        self.superlu_params = {}

    def AllocSolver(self, is_complex):
        gui = self.gui
        pivot_thr = float(gui.pivot_thr_txt) if len(
            gui.pivot_thr_txt.strip()) > 0 else None
        relax = int(gui.relax_txt) if len(
            gui.relax_txt.strip()) > 0 else None
        panel_size = int(gui.panel_size_txt) if len(
            gui.panel_size_txt.strip()) > 0 else None
        options = gui.eval_options(gui.superlu_options_txt)

        # for SuperLU, allocation is done in SetOperator
        self.is_complex = is_complex
        self.use_single_precision = gui.use_single_precision
        self.superlu_params = {"permc_spec": gui.col_permute_txt,
                               "diag_pivot_thresh": pivot_thr,
                               "relax": relax,
                               "panel_size": panel_size,
                               "options": options}
        if is_complex:
            if self.use_single_precision:
                dtype = np.complex64
            else:
                dtype = np.complex128
        else:
            if self.use_single_precision:
                dtype = np.float32
            else:
                dtype = np.float64
        self.dtype = dtype

    def _factor_matrix(self, matrix):
        from scipy.sparse.linalg import splu

        if use_parallel:
            parts = MPI.COMM_WORLD.gather(matrix, root=0)
            if myid == 0:
                matrix = scipy.sparse.vstack(parts, format='csr')

        if myid == 0:
            dprint1("creating SciPy SuperLU object. matrix shape=", matrix.shape)
            self._superlu = splu(matrix.tocsc(), **self.superlu_params)

    def _solve_rhs(self, rhs):
        if use_parallel:
            nrow = len(rhs)
            rhs = gather_vector(rhs, parent=(myid == 0))

        solution = None
        if myid == 0:
            dprint1("calling SciPy solve", debug.format_memory_usage())
            solution = self._superlu.solve(rhs)

        if use_parallel:
            solution = scatter_vector(solution, rcounts=nrow)
        return solution

    def close(self):
        self._superlu = None


class SuperLUDistSolver(SuperLUSolverBase):
    """
    Pass Petra-M's local CSR rows and vectors directly to SuperLU_DIST.
    """

    def __init__(self, gui, engine):
        from mpi4py import MPI

        super().__init__(gui, engine)
        self.comm = MPI.COMM_WORLD if use_parallel else MPI.COMM_SELF
        self.backward_error = None

    def AllocSolver(self, is_complex):
        self.is_complex = is_complex
        self.dtype = np.complex128 if is_complex else np.float64

    def _factor_matrix(self, matrix):
        import time
        from petram.ext.superlu.factor import factor
        from petram.helper.mpi_recipes import collective_call

        def canonicalize():
            matrix.sum_duplicates()
            matrix.sort_indices()

        collective_call(self.comm, canonicalize)
        counts = self.comm.allgather(matrix.shape[0])
        first = sum(counts[:self.comm.rank])

        start = time.perf_counter()
        self._superlu = factor(
            matrix, comm=self.comm, first_row=first,
            ordering=DIST_ORDERINGS[self.gui.col_permute_txt],
            equil=self.gui.dist_equil,
            replace_tiny_pivot=self.gui.dist_replace_tiny_pivot,
            refinement=self.gui.dist_refinement,
            log_statistics=self.gui.dist_log_statistics)
        dprint1('SuperLU_DIST factorization:', time.perf_counter()-start,
                's; MPI ranks:', self.comm.size, 'grid:', self._superlu.process_grid,
                debug.format_memory_usage())

    def _solve_rhs(self, rhs):
        local_solution = self._superlu.solve(rhs)
        self.backward_error = self._superlu.backward_error.copy()
        return local_solution

    def close(self):
        if self._superlu is not None:
            self._superlu.close()
            self._superlu = None


class SuperLUMFEMSolverModel(SuperLU):
    '''
    This model creates SuperLU solver as a preconditioner in iterative solver
    '''

    def verify_setting(self):
        if self.superlu_backend != 'SciPy':
            return (False, 'SuperLU preconditioner requires SciPy',
                    'SuperLU_DIST is supported only as a standalone solver.')
        return super().verify_setting()

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
        super(SuperLUBlockPreconditioner, self).__init__()

    def Mult(self, x, y):
        s = self.solver.Mult([x])
        if self.is_complex_operator:
            assert False, "SuperLUMFEM for Complex is not yet implemented"
            # s = self.complex_to_real(s)

        y.Assign(s.flatten().astype(float, copy=False))

    def SetOperator(self, opr):
        from petram.solver.solver_utils import check_block_operator
        is_complex, is_parallel = check_block_operator(opr)

        solver = self.gui.allocate_solver(is_complex=is_comples, engine=self.engine)
        solver.SetOperator(opr, is_parallel)

        self.is_complex_operator = is_complex
        self.is_parallel = is_parallel
        self.solver = solver
