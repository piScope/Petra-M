from petram.mfem_config import use_parallel
from .solver_model import LinearSolverModel, LinearSolver
from petram.helper.matrix_file import write_matrix, write_vector, write_coo_matrix
from petram.helper.solver_mpi import SolverLayout, SolverRankLayout
import os
import numpy as np
from scipy.sparse import coo_matrix
import weakref

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('MUMPSModel')

from petram.helper.check_module_installed import module_file_exists
has_mumps = bool(module_file_exists("petram.ext.mumps.mumps_solve"))


if use_parallel:
    import mfem.par as mfem
    from mfem.common.mpi_debug import nicePrint

else:
    import mfem.ser as mfem
    nicePrint = dprint1

def convert2float(txt):
    try:
        return float(txt)
    except BaseException:
        assert False, "can not convert to float. Input text is " + txt


def convert2int(txt):
    try:
        return int(txt)
    except BaseException:
        assert False, "can not convert to float. Input text is " + txt


class MUMPSBase(LinearSolverModel):
    has_2nd_panel = False
    accept_complex = True
    is_iterative = False

    def __init__(self):
        self.s = None
        LinearSolverModel.__init__(self)

    def init_solver(self):
        pass

    def panel1_param(self):
        return [["log_level(0-2)", self.log_level, 400, {}],
                ["ordering", self.ordering, 4, {"readonly": True,
                                                "choices": ["auto", "AMF", "PORD", "QAMD",
                                                            "Metis", "Scotch",
                                                            "ParMetis", "PT-Scotch"]}],
                ["out-of-core", self.out_of_core, 3, {"text": ""}],
                ["error analysis", self.error_ana, 4, {"readonly": True,
                                                       "choices": ["none", "full stat.", "main stat."]}],
                ["write matrix", self.write_mat, 3, {"text": ""}],
                ["write factor", self.write_fac, 3, {"text": ""}],
                ["restore factor", self.restore_fac, 3, {"text": ""}],
                ["factor path/prefix", self.factor_path, 0, {}],
                ["use BLR", self.use_blr, 3, {"text": ""}],
                ["BLR drop parameter", self.blr_drop, 300, {}],
                ["WS Inc. (ICNTL14)", self.icntl14, 0, {}],
                ["WS Size (ICNTL23)", self.icntl23, 0, {}],
                ["numerical pivot thr. (CNTL1)", self.cntl1, 0, {}],
                ["static pivot thr. (CNTL4)", self.cntl4, 0, {}],
                ["Itr. refinement (ICNTL10)", self.icntl10, 0, {}],
                ["refinement stop Cond. (CNTL2)", self.cntl2, 0, {}],
                ["permutation/scaling Opt.(ICNTL6)", self.icntl6, 0, {}],
                ["scaling strategy (ICNTL8)", self.icntl8, 0, {}],
                ["write inverse", self.write_inv, 3, {"text": ""}],
                ["use float32", self.use_single_precision, 3, {"text": ""}],
                ["use dist, RHS (dev.)",
                 self.use_dist_rhs, 3, {"text": ""}],
                ["use dist, SOL (dev.)", self.use_dist_sol, 3, {"text": ""}], ]

    def get_panel1_value(self):
        return (int(self.log_level), self.ordering, self.out_of_core,
                self.error_ana, self.write_mat, self.write_fac,
                self.restore_fac, self.factor_path,
                self.use_blr, self.blr_drop, str(self.icntl14),
                str(self.icntl23),
                self.cntl1, self.cntl4, self.icntl10, self.cntl2,
                self.icntl6, self.icntl8, self.write_inv,
                self.use_single_precision, self.use_dist_rhs, self.use_dist_sol)

    def import_panel1_value(self, v):
        self.log_level = int(v[0])
        self.ordering = str(v[1])
        self.out_of_core = v[2]
        self.error_ana = v[3]
        self.write_mat = v[4]
        self.write_fac = v[5]
        self.restore_fac = v[6]
        self.factor_path = v[7]
        self.use_blr = v[8]
        self.blr_drop = v[9]
        self.icntl14 = v[10]
        self.icntl23 = v[11]
        self.cntl1 = v[12]
        self.cntl4 = v[13]
        self.icntl10 = v[14]
        self.cntl2 = v[15]
        self.icntl6 = v[16]
        self.icntl8 = v[17]
        self.write_inv = v[18]
        self.use_single_precision = v[19]
        self.use_dist_rhs = v[20]
        self.use_dist_sol = v[21]

    def attribute_set(self, v):
        v = super(MUMPSBase, self).attribute_set(v)

        v['log_level'] = 0
        '''
        1 : Only error messages printed.
        2 : Errors, warnings, and main statistics printed.
        3 : Errors and warnings and terse diagnostics
            (only first ten entries of arrays) printed.
        4 : Errors, warnings and information on input, output parameters printed
        '''
        v['out_of_core'] = False
        v['write_mat'] = False
        v['write_fac'] = False
        v['restore_fac'] = False
        v['factor_path'] = "~/mumps_data/factored_matrix*"
        v['central_mat'] = False
        v['ordering'] = 'auto'
        v['error_ana'] = 'none'
        v['use_blr'] = False
        v['blr_drop'] = 0.0
        v['icntl14'] = '20'
        v['icntl23'] = '0'
        v['cntl1'] = 'default'
        v['cntl4'] = 'default'
        v['icntl10'] = 'default'
        v['cntl2'] = 'default'
        v['icntl6'] = 'default'
        v['icntl8'] = 'default'
        v['use_single_precision'] = False
        v['write_inv'] = False
        v['use_dist_rhs'] = False

        # make sure that old data type (data was stored as int) is converted to
        # string
        if hasattr(self, "icntl14"):
            self.icntl14 = str(self.icntl14)
        if hasattr(self, "icntl23"):
            self.icntl23 = str(self.icntl23)

        # this flag needs to be set, so that destcuctor works when
        # model tree is loaded from pickled file
        v['s'] = None
        return v

    def allocate_solver(self, is_complex=False, engine=None):
        # engine not used
        ls_type = self.get_solve_root().get_linearsystem_type_from_modeltree()

        if ls_type.startswith('blk'):
            solver = MUMPSBlockSolver(self, engine)
        else:
            solver = MUMPSSolver(self, engine)
            solver.AllocSolver(is_complex, self.use_single_precision)
        return solver

    def solve(self, engine, A, b):
        solver = self.allocate_solver((A.dtype == 'complex'), engine)
        try:
            solver.SetOperator(A, dist=engine.is_matrix_distributed)
            return solver.Mult(b, case_base=engine.case_base)
        finally:
            solver.close()

    def real_to_complex(self, solall, M=None):
        try:
            from mpi4py import MPI
        except BaseException:
            from petram.helper.dummy_mpi import MPI
        myid = MPI.COMM_WORLD.rank
        nproc = MPI.COMM_WORLD.size

        if solall is not None:
            s = solall.shape[0]
            solall = solall[:s // 2, :] + 1j * solall[s // 2:, :]
            return solall

    def __del__(self):
        if self.s is not None:
            self.s.finish()
        self.s = None


class MUMPS(MUMPSBase):
    def attribute_set(self, v):
        v = super().attribute_set(v)
        v['solver_mpi_ranks'] = -1
        v['use_dist_rhs'] = True
        return v

    def panel1_param(self):
        return super().panel1_param() + [["Solver MPI ranks", self.solver_mpi_ranks, 400, {}]]

    def get_panel1_value(self):
        return tuple(super().get_panel1_value()) + (self.solver_mpi_ranks,)

    def import_panel1_value(self, values):
        super().import_panel1_value(values[:-1])
        self.solver_mpi_ranks = int(values[-1])

    def resolve_solver_size(self, comm, environ=None):
        """
        Resolve the shared model/environment setting without changing the model.
        """
        environ = os.environ if environ is None else environ

        n = int(str(self.solver_mpi_ranks).strip())
        source = 'model'
        if n == -1:
            env = environ.get('PETRAM_MUMPS_MPI_SIZE')
            source = 'default' if env is None else 'environment'
            n = comm.size if env is None else int(env.strip())
        if not 1 <= n <= comm.size:
            raise ValueError(f'Solver MPI ranks must be in 1..{comm.size}, got {n}')
        return n, source

    def allocate_solver(self, is_complex=False, engine=None):
        ls_type = self.get_solve_root().get_linearsystem_type_from_modeltree()
        if not use_parallel or ls_type.startswith('blk'):
            return super().allocate_solver(is_complex, engine)

        from mpi4py import MPI

        size, source = self.resolve_solver_size(MPI.COMM_WORLD)
        dprint1(f'MUMPS MPI ranks: application={MPI.COMM_WORLD.size}, solver={size} ({source})')
        

        if size == MPI.COMM_WORLD.size:
            solver = MUMPSSolver(self, engine, comm=MPI.COMM_WORLD, 
                                 rhs_is_local=True)
            solver.AllocSolver(is_complex, self.use_single_precision)
            return solver

        return ReducedMUMPSSolver(self, engine, MPI.COMM_WORLD, size, is_complex)

    def does_linearsolver_choose_linearsystem_type(self):
        from petram.solver.solver_model import Solver

        # for the sake of backword compabitibiliy. it is true only when
        # it is the top level linearsolver

#        if isinstance(self.parent, Solver):
        from petram.solver.solver_controls import SolveControl

        if self.parent.parent is self.get_solve_root():
            return True
        elif isinstance(self.parent.parent, SolveControl):
            if self.parent.parent.parent is self.get_solve_root():
                return True
            else:
                return False
        else:
            return False

    def supported_linear_system_type(self):
        return 'ANY'

    def linear_system_type(self, assemble_real, phys_real):
        if phys_real:
            return 'coo'
        if assemble_real:
            return 'coo_real'
        return 'coo'


class MUMPSMFEMSolverModel(MUMPSBase):
    '''
    This one is to use MUMPS in iterative solver
    It creates MUMPSBlockPreconditioner
    '''
    @classmethod
    def fancy_menu_name(self):
        return 'MUMPS'

    @classmethod
    def fancy_tree_name(self):
        return 'MUMPS'

    def does_linearsolver_choose_linearsystem_type(self):
        return False

    def supported_linear_system_type(self):
        return ["blk_interleave",
                "blk_merged_s",
                "blk_merged", ]

    def prepare_solver(self, opr, engine, name=None):
        solver = MUMPSBlockPreconditioner(opr,
                                          gui=self,
                                          engine=engine,
                                          silent=True)
        solver.SetOperator(opr)
        return solver


class MUMPSPreconditionerModel(MUMPSBase):
    '''
    This one is to use MUMPS in iterative solver
    It creates MUMPSPreconditioner
    '''
    @classmethod
    def fancy_menu_name(self):
        return 'MUMPS'

    @classmethod
    def fancy_tree_name(self):
        return 'MUMPS'

    def does_linearsolver_choose_linearsystem_type(self):
        return False

    def supported_linear_system_type(self):
        return ["blk_interleave",
                "blk_merged_s",
                "blk_merged", ]

    def prepare_solver(self, opr, engine, name=None):
        prc = MUMPSPreconditioner(opr,
                                  gui=self,
                                  engine=engine,
                                  silent=True)
        return prc


class MUMPSSolver(LinearSolver):
    is_iterative = False

    def __init__(self, *args, comm=None, rhs_is_local=False, **kwargs):
        super(MUMPSSolver, self).__init__(*args, **kwargs)
        if comm is None:
            if use_parallel:
                from mpi4py import MPI
            else:
                from petram.helper.dummy_mpi import MPI
            comm = MPI.COMM_WORLD
        self.comm = comm
        self.s = None
        self.silent = False
        self.keep_sol_distributed = False
        # Standalone COO finalization supplies local values in equation-ID order.
        # Preconditioners retain their existing centralized/distributed contract.
        self.rhs_is_local = rhs_is_local
        self._rhs_transfer = None
        self._solution_transfer = None

    def close(self):
        """
        Finish the native solver collectively on its own communicator.
        """
        if self.s is not None:
            self.s.finish()
            self.s = None

    def _run_job(self, job):
        """
        Run a native collective and check the global MUMPS status.
        """
        self.s.set_job(job)
        self.s.run()
        status = self.s.get_infog(1)
        if status < 0:
            raise RuntimeError(f'MUMPS job {job}: INFOG(1)={status}, INFOG(2)={self.s.get_infog(2)}')
        if status > 0:
            dprint1(f'MUMPS job {job} warning: INFOG(1)={status}, INFOG(2)={self.s.get_infog(2)}')
        return status

    def split_dir_prefix(self, txt):
        from petram.helper.mpi_recipes import collective_call
        txt = os.path.abspath(os.path.expanduser(txt))
        txt = txt.strip().split('*')[0]
        path = os.path.dirname(txt)
        prefix = os.path.basename(txt)
        try:
            from mpi4py import MPI
            myid = self.comm.rank
        except BaseException:
            myid = 0
            from petram.helper.dummy_mpi import MPI

        def create_directory():
            if myid == 0 and not os.path.exists(path):
                dprint1("Creating path for MUMPS data", path)
                os.makedirs(path)

        collective_call(self.comm, create_directory)
        return prefix, path

    def set_silent(self, silent):
        self.silent = silent

    def set_ordering_flag(self, s):
        from petram.mfem_config import use_parallel
        gui = self.gui
        if gui.ordering == 'auto':
            pass
        elif gui.ordering == 'QAMD':
            # if use_parallel:
            #    dprint1(
            #        "!!! QAMD ordering is selected. But solver is not in serial mode. Ignored")
            # else:
            s.set_icntl(28, 1)
            s.set_icntl(7, 6)
        elif gui.ordering == 'AMF':
            # if use_parallel:
            #    dprint1(
            #        "!!! AMF ordering is selected. But solver is not in serial mode. Ignored")
            # else:
            s.set_icntl(28, 1)
            s.set_icntl(7, 2)
        elif gui.ordering == 'PORD':
            # if use_parallel:
            #    dprint1(
            #        "!!! PORD ordering is selected. But solver is not in serial mode. Ignored")
            # else:
            s.set_icntl(28, 1)
            s.set_icntl(7, 4)
        elif gui.ordering == 'Metis':
            # if use_parallel:
            #    dprint1(
            #        "!!! Metis ordering is selected. But solver is not in serial mode. Ignored")
            # else:
            s.set_icntl(28, 1)
            s.set_icntl(7, 5)
        elif gui.ordering == 'Scotch':
            # if use_parallel:
            #    dprint1(
            #        "!!! Scotch ordering is selected. But solver is not in serial mode. Ignored")
            # else:
            s.set_icntl(28, 1)
            s.set_icntl(7, 3)
        elif gui.ordering == 'ParMetis':
            if use_parallel:
                s.set_icntl(28, 2)
                s.set_icntl(29, 2)
            else:
                dprint1(
                    "!!! ParMetis ordering is selected. But solver is not in parallel mode. Ignored")
        elif gui.ordering == 'PT-Scotch':
            if use_parallel:
                s.set_icntl(28, 2)
                s.set_icntl(29, 1)
            else:
                dprint1(
                    "!!! PT-Scotch ordering is selected. But solver is not in parallel mode. Ignored")
        else:
            pass
        # s.set_icntl(28,  2)

    def set_error_analysis(self, s):
        from petram.mfem_config import use_parallel
        gui = self.gui
        if gui.error_ana == 'none':
            s.set_icntl(11, 0)
        elif gui.error_ana == 'main stat.':
            s.set_icntl(11, 2)
        elif gui.error_ana == 'full stat.':
            s.set_icntl(11, 1)
        else:
            pass

    def AllocSolver(self, is_complex, use_single_precision):
        if use_parallel:
            from mpi4py import MPI
            myid = self.comm.rank
            args = (1, 0, self.comm)
        else:
            myid = 0
            args = (1, 0)

        if is_complex:
            if use_single_precision:
                from petram.ext.mumps.mumps_solve import c_array as data_array
                from petram.ext.mumps.mumps_solve import CMUMPS
                s = CMUMPS(*args)
            else:
                from petram.ext.mumps.mumps_solve import z_array as data_array
                from petram.ext.mumps.mumps_solve import ZMUMPS
                s = ZMUMPS(*args)
        else:
            if use_single_precision:
                from petram.ext.mumps.mumps_solve import s_array as data_array
                from petram.ext.mumps.mumps_solve import SMUMPS
                s = SMUMPS(*args)
            else:
                from petram.ext.mumps.mumps_solve import d_array as data_array
                from petram.ext.mumps.mumps_solve import DMUMPS
                s = DMUMPS(*args)

        self.s = s
        self.is_complex = is_complex
        self.data_array = data_array

        gui = self.gui
        # No outputs
        if gui.log_level <= 0:
            s.set_icntl(1, -1)
            s.set_icntl(2, -1)
            s.set_icntl(3, -1)
            s.set_icntl(4, 0)
        elif gui.log_level == 1:
            pass
        else:
            if myid == 0:
                s.set_icntl(1, 6)
                s.set_icntl(2, 6)
                s.set_icntl(3, 6)
                s.set_icntl(4, 6)

    def _data_type(self):
        if self.gui.use_single_precision:
            if self.is_complex:
                return np.complex64
            else:
                return np.float32
        else:
            if self.is_complex:
                return np.complex128
            else:
                return np.float64

    def _int_type(self):
        import petram.ext.mumps.mumps_solve as mumps_solve
        dtype_int = 'int' + str(mumps_solve.SIZEOF_MUMPS_INT() * 8)
        return dtype_int

    def make_matrix_entries(self, A):
        datatype = self._data_type()
        AA = A.data.astype(datatype, copy=False)
        return AA
        '''
        if self.gui.use_single_precision:
            if self.is_complex:
                AA = A.data.astype(np.complex64, copy=False)
            else:
                AA = A.data.astype(np.float32, copy=False)
        else:
            if self.is_complex:
                AA = A.data.astype(np.complex128, copy=False)
            else:
                AA = A.data.astype(np.float64, copy=False)
        return AA
        '''

    def make_vector_entries(self, B):
        datatype = self._data_type()
        return B.astype(datatype, copy=False)
        '''
        if self.gui.use_single_precision:
            if self.is_complex:
                return B.astype(np.complex64, copy=False)
            else:
                return B.astype(np.float32, copy=False)
        else:
            if self.is_complex:
                return B.astype(np.complex128, copy=False)
            else:
                return B.astype(np.float64, copy=False)
        '''

    def _merge_coo_matrix(self, myid, rank):
        def filename(myid):
            smyid = '{:0>6d}'.format(myid)
            return "matrix." + smyid + ".npz"

        if use_parallel:
            from mpi4py import MPI
            self.comm.Barrier()

            myid = self.comm.rank
        else:
            myid = 0

        if myid != 0:
            return
        f = filename(myid)
        mat = np.load(f, allow_pickle=True)
        os.remove(f)
        mat = mat['A'][()]

        for i in range(1, rank):
            f = filename(i)
            mat2 = np.load(f, allow_pickle=True)
            os.remove(f)
            mat = mat + mat2['A'][()]

        np.savez("matrix", A=mat)

    def set_distributed_sol(self, s, nrhs):
        '''
        setup parameters to receive a distributed solution

        must be called after Factorization
        '''
        from petram.ext.mumps.mumps_solve import i_array

        n_pivots = s.get_info(23)

        sol_loc = np.zeros(n_pivots * nrhs, dtype=self._data_type())
        isol_loc = np.zeros(n_pivots, dtype=self._int_type())

        s.set_sol_loc(self.data_array(sol_loc),
                      n_pivots,
                      i_array(isol_loc))

        s.set_icntl(21, 1)
        return n_pivots, isol_loc, sol_loc

    def read_rhs_distribution(self, s):
        from petram.ext.mumps.mumps_solve import i_array

        lrhs_loc = s.get_info(23)

        irhs_loc = np.empty(lrhs_loc, dtype=self._int_type())

        s.set_irhs_loc(lrhs_loc, i_array(irhs_loc))

        if not self.silent:
            dprint1("job9")
        s.set_job(9)
        s.run()
        return irhs_loc

    def redistributed_array(self, sol, isol_loc, requested_ids, nrhs):
        """
        Return solution values in the requested global-index order.
        """
        from petram.helper.mpi_recipes import IndexedTransfer
        if self._solution_transfer is None:
            self._solution_transfer = IndexedTransfer(self.comm, isol_loc, requested_ids)
        values = sol.reshape(nrhs, -1).T
        return self._solution_transfer.apply(values).T

    def SetOperator(self, A, dist, name=None, ifactor=0):
        from petram.helper.mpi_recipes import collective_call
        try:
            from mpi4py import MPI
        except BaseException:
            from petram.helper.dummy_mpi import MPI

        myid = self.comm.rank
        nproc = self.comm.size
        layout = getattr(A, 'solver_layout', None)

        from petram.ext.mumps.mumps_solve import i_array
        gui = self.gui
        s = self.s

        if dist:
            dprint1("SetOperator distributed matrix")
            A.eliminate_zeros()
            if gui.write_mat:
                write_coo_matrix('matrix', A, comm=self.comm)
            if gui.write_inv:
                smyid = '{:0>6d}'.format(myid)
                np.savez("matrix." + smyid, A=A)
                self._merge_coo_matrix(myid, nproc)

            import petram.ext.mumps.mumps_solve as mumps_solve
            dprint1('!!!these two must be consistent')
            dprint1('sizeof(MUMPS_INT) ', mumps_solve.SIZEOF_MUMPS_INT())

            # set matrix format
            s.set_icntl(5, 0)
            s.set_icntl(18, 3)

            dprint1("NNZ local: ", A.nnz)
            nnz_array = np.array(self.comm.allgather(A.nnz))
            if myid == 0:
                s.set_n(A.shape[1])
            dprint1("NNZ total: ", np.sum(nnz_array))
            dprint1("NNZ all: ", nnz_array)

            dtype_int = 'int' + str(mumps_solve.SIZEOF_MUMPS_INT() * 8)
            row = A.row
            col = A.col
            row = row.astype(dtype_int) + 1
            col = col.astype(dtype_int) + 1
            AA = self.make_matrix_entries(A)

            if len(col) > 0:
                dprint1('index data size ', type(col[0]))
                dprint1('matrix data type ', type(AA[0]))

            s.set_nz_loc(len(A.data))
            s.set_irn_loc(i_array(row))
            s.set_jcn_loc(i_array(col))
            s.set_a_loc(self.data_array(AA))

            self.dataset = (AA, row, col)

            self.irhs_loc = np.unique(row)

        else:
            A = A.tocoo(False)  # .astype('complex')
            import petram.ext.mumps.mumps_solve as mumps_solve
            dprint1('!!!these two must be consistent')
            dprint1('sizeof(MUMPS_INT) ', mumps_solve.SIZEOF_MUMPS_INT())

            dtype_int = 'int' + str(mumps_solve.SIZEOF_MUMPS_INT() * 8)

            if gui.write_mat:
                # tocsr().tocoo() forces the output is row sorted.
                collective_call(self.comm, lambda: write_coo_matrix('matrix', A.tocsr().tocoo(), comm=self.comm))
            # No outputs
            if myid == 0:
                row = A.row
                col = A.col
                row = row.astype(dtype_int) + 1
                col = col.astype(dtype_int) + 1
                AA = self.make_matrix_entries(A)

                if len(col) > 0:
                    dprint1('index data size ', type(col[0]))
                    dprint1('matrix data type ', type(AA[0]))

                s.set_n(A.shape[0])
                s.set_nz(len(A.data))
                s.set_irn(i_array(row))
                s.set_jcn(i_array(col))
                s.set_a(self.data_array(AA))
                self.dataset = (AA, row, col)
                self.irhs_loc = np.unique(row)
            else:
                self.irhs_loc = None

        self.N_global = A.shape[0]
        self.solution_ids = self.irhs_loc
        if layout is not None:
            self.irhs_loc = np.asarray(layout.equation_ids, dtype=self._int_type()) + 1
            self.solution_ids = np.asarray(layout.solution_ids, dtype=self._int_type()) + 1
        self._rhs_transfer = None
        self._solution_transfer = None
        if self.skip_solve:
            return

        # blr
        if gui.use_blr:
            s.set_icntl(35, 1)
            s.set_cntl(7, float(gui.blr_drop))

        # out-of-core
        if gui.out_of_core:
            s.set_icntl(22, 1)

        if gui.icntl14.lower() != 'default':
            # percentage increase in the estimated workingspace
            s.set_icntl(14, convert2int(gui.icntl14))

        if gui.icntl23.lower() != 'default':
            # maximum size of the working memory
            s.set_icntl(23, convert2int(gui.icntl23))

        if gui.icntl8.lower() != 'default':
            # the scaling strategy
            s.set_icntl(8, convert2int(gui.icntl8))

        if gui.icntl6.lower() != 'default':
            # permutes the matrix to  azero-freediagonal and/or
            # scale the matrix
            s.set_icntl(6, convert2int(gui.icntl6))

        if gui.cntl1.lower() != 'default':
            # relative threshold for numerical pivoting
            s.set_cntl(1, convert2float(gui.cntl1))

        if gui.cntl4.lower() != 'default':
            # threshold for static pivoting
            s.set_cntl(4, convert2float(gui.cntl4))

        s.set_icntl(24, 0)  # No Null detection
        self.set_ordering_flag(s)

        self.comm.Barrier()
        if not gui.restore_fac:
            if not self.silent:
                dprint1("job1", debug.format_memory_usage())
            self._run_job(1)
            if not self.silent:
                dprint1("job2", debug.format_memory_usage())
            s.set_icntl(13, 0)
            s.set_icntl(5, 0)

            self._run_job(2)
            if gui.write_fac:
                if not self.silent:
                    dprint1("job7 (save)")
                prefix, path = self.split_dir_prefix(gui.factor_path)
                s.set_saveparam(prefix, path)
                s.set_oocparam("ooc_", path)

                # wait here to make sure path is created.
                self.comm.Barrier()

                self._run_job(7)

        else:
            if not self.silent:
                dprint1("job8 (restore)")
            pathes = gui.factor_path.split(',')
            prefix, path = self.split_dir_prefix(pathes[ifactor].strip())

            s.set_saveparam(prefix, path)
            s.set_oocparam("ooc_", path)

            self.comm.Barrier()  # wait here to make sure path is created.
            # s.set_job(-1)
            # s.run()
            self._run_job(8)

    def write_inverse(self, s):
        from petram.helper.mpi_recipes import collective_call
        if use_parallel:
            from mpi4py import MPI
        else:
            from petram.helper.dummy_mpi import MPI

        myid = self.comm.rank
        smyid = '{:0>6d}'.format(myid)
        mpi_size = self.comm.size

        datatype = self._data_type()

        nrhs = self.N_global
        controls = {i: s.get_icntl(i) for i in (11, 20, 21)}
        s.set_icntl(20, 0)
        s.set_lrhs_nrhs(nrhs, nrhs)

        if myid == 0:
            dprint1("nrhs", nrhs)
            bb = np.zeros((nrhs, nrhs), dtype=datatype)
            for i in range(nrhs):
                bb[i, i] = 1.0
            bstack = np.hstack(np.transpose(bb))
            bstack = self.make_vector_entries(bstack)
            s.set_rhs(self.data_array(bstack))

        lsol_loc, isol_loc, sol_loc = self.set_distributed_sol(s, nrhs)
        s.set_icntl(11, 0)

        # set to distributed sol mode
        use_distributed_save = True

        self._run_job(3)

        # sol = s.get_sol_loc()
        # if use_distributed_save:
        sol = np.transpose(sol_loc.reshape(nrhs, lsol_loc))
        def save():
            np.savez("matrix_inv_idx." + smyid, A_inv_idx=isol_loc)
            np.savez("matrix_inv." + smyid, A_inv=sol)

        collective_call(self.comm, save)

        for i, value in controls.items():
            s.set_icntl(i, value)

    def Mult(self, b, x=None, case_base=0):
        from petram.helper.mpi_recipes import IndexedTransfer, collective_call
        if use_parallel:
            from mpi4py import MPI
        else:
            from petram.helper.dummy_mpi import MPI

        from petram.ext.mumps.mumps_solve import i_array

        myid = self.comm.rank
        nproc = self.comm.size

        gui = self.gui
        s = self.s
        nrhs = self.comm.bcast(b.shape[1] if myid == 0 else None, root=0)
        distributed_rhs = bool(gui.use_dist_rhs and use_parallel)
        distributed_sol = bool(use_parallel and
                               (self.keep_sol_distributed or gui.use_dist_sol))

        if self.rhs_is_local and not distributed_rhs:
            if self._rhs_transfer is None:
                requested = (np.arange(1, self.N_global + 1, dtype=np.int64)
                             if myid == 0 else np.empty(0, dtype=np.int64))
                self._rhs_transfer = IndexedTransfer(self.comm, self.irhs_loc, requested)
            b = self._rhs_transfer.apply(b)

        if gui.write_mat:
            def write_rhs():
                for ib in range(nrhs):
                    filename = 'rhs_' + str(ib + case_base)
                    if distributed_rhs:
                        rhs_coo = coo_matrix(
                            (b[:, ib], (self.irhs_loc - 1, np.zeros(len(b), dtype=int))),
                            shape=(self.N_global, 1))
                        write_coo_matrix(filename, rhs_coo, comm=self.comm)
                    elif myid == 0:
                        write_vector(filename, b[:, ib], comm=self.comm)

            collective_call(self.comm, write_rhs)

        if self.gui.write_inv and not self.skip_solve:
            self.write_inverse(s)

        if self.skip_solve:
            if distributed_sol:
                result = np.zeros((len(self.solution_ids), nrhs), dtype=self._data_type())
                return result if gui.use_dist_sol else result.T
            return np.zeros((self.N_global, nrhs), dtype=self._data_type()) if myid == 0 else None

        s.set_lrhs_nrhs(self.N_global, nrhs)
        s.set_icntl(21, int(distributed_sol))
        if distributed_rhs:
            s.set_icntl(20, 10)

            # Native MUMPS stores each RHS column consecutively. Empty ranks
            # still need a positive leading dimension.
            nloc_rhs = len(self.irhs_loc)
            lrhs_loc = max(1, nloc_rhs)
            rhs_loc = np.zeros((nrhs, lrhs_loc), dtype=self._data_type())
            rhs_loc[:, :nloc_rhs] = np.asarray(b).T
            irhs_loc_read = self.irhs_loc.astype(dtype=self._int_type())

            rhs_locc = self.data_array(rhs_loc.reshape(-1))
            irhs_locc = i_array(irhs_loc_read)

            s.set_nrhs_lrhs_irhs_rhs_loc(
                nloc_rhs, lrhs_loc, irhs_locc, rhs_locc)

        else:
            s.set_icntl(20, 0)
            if myid == 0:
                s.set_lrhs_nrhs(b.shape[0], b.shape[1])
                bstack = np.hstack(np.transpose(b))
                bstack = self.make_vector_entries(bstack)
                s.set_rhs(self.data_array(bstack))

        if distributed_sol:
            lsol_loc, isol_loc, sol_loc = self.set_distributed_sol(s, nrhs)

        else:
            if distributed_rhs and myid == 0:
                rhs = np.empty(self.N_global * nrhs, dtype=self._data_type())
                s.set_rhs(self.data_array(rhs))

        if self.gui.icntl10.lower() != 'default':       # iterative refinement
            s.set_icntl(10, convert2int(self.gui.icntl10))

        if self.gui.cntl2.lower() != 'default':
            # stopping criterion for iterative refinement
            s.set_cntl(2, convert2float(self.gui.cntl2))

        if self.gui.log_level >= 10 or self.gui.log_level <= 0:
            s.set_icntl(1, -1)
            s.set_icntl(2, -1)
            s.set_icntl(3, -1)
            s.set_icntl(4, 0)
        else:
            if not self.silent:
                dprint1("job3", debug.format_memory_usage())

        if not distributed_rhs and not distributed_sol:
            self.set_error_analysis(s)
        else:
            s.set_icntl(11, 0)

        self._run_job(3)

        rsol = None
        isol = None
        sol_extra = None

        if distributed_sol:
            sol = self.redistributed_array(
                sol_loc.flatten(), isol_loc, self.solution_ids, nrhs)

            if self.gui.use_dist_sol:
                sol = np.transpose(sol.reshape(nrhs, -1))
            else:
                # preconditioner mode. keep sol as it is
                pass
        else:
            if myid == 0:
                if self.is_complex:
                    sol = s.get_real_rhs() + 1j * s.get_imag_rhs()
                else:
                    sol = s.get_real_rhs()

                sol = np.transpose(sol.reshape(nrhs, self.N_global))
            else:
                sol = None

        # nicePrint('doen with dist_RHS, dist_sol', distributed_rhs, distributed_sol)

        return sol


class ReducedMUMPSSolver(LinearSolver):
    """
    Aggregate application data for MUMPS and return the solution to its owners.
    """
    is_iterative = False

    def __init__(self, gui, engine, app_comm, size, is_complex):
        super().__init__(gui, engine)
        self.comm = app_comm
        self.ranks = SolverRankLayout(app_comm, size)
        self.is_complex = is_complex
        self.backend = None
        self.closed = False

    def _call_backend(self, operation):
        from petram.helper.mpi_recipes import collective_call
        return collective_call(self.comm, lambda: operation() if self.ranks.is_leader else None)

    def SetOperator(self, A, dist=False, name=None, ifactor=0):
        from petram.helper.mpi_recipes import collective_call, gather_vector

        def prepare():
            coo = A.tocoo()
            if coo.shape[0] != coo.shape[1]:
                raise ValueError('MUMPS requires a square matrix')
            layout = getattr(A, 'solver_layout', None)
            if layout is None:
                if dist:
                    raise ValueError('Distributed COO input requires SolverLayout metadata')
                ids = np.arange(coo.shape[0], dtype=np.int64) if self.comm.rank == 0 else np.empty(0, dtype=np.int64)
                layout = SolverLayout(ids, ids)
            for ids in (layout.equation_ids, layout.solution_ids):
                if np.any(np.asarray(ids) < 0) or np.any(np.asarray(ids) >= coo.shape[0]):
                    raise ValueError('Solver layout index out of range')
            return coo, layout

        coo, self.layout = collective_call(self.comm, prepare)
        meta = self.comm.allgather((coo.shape, coo.dtype.str))
        if any(item != meta[0] for item in meta):
            raise ValueError('Inconsistent matrix dimensions or dtype')
        self.shape = coo.shape
        group = self.ranks.group_comm
        # Centralized input is supplied once, even if callers replicate it.
        take = dist or self.comm.rank == 0
        row = gather_vector(np.asarray(coo.row if take else [], dtype=np.int64), world=group)
        col = gather_vector(np.asarray(coo.col if take else [], dtype=np.int64), world=group)
        data = gather_vector(coo.data if take else np.empty(0, dtype=coo.dtype), world=group)
        equations = gather_vector(np.ascontiguousarray(self.layout.equation_ids, dtype=np.int64),
                                  world=group)
        solutions = gather_vector(np.ascontiguousarray(self.layout.solution_ids, dtype=np.int64),
                                  world=group)

        def setup():
            matrix = coo_matrix((data, (row, col)), shape=self.shape)
            matrix.solver_layout = SolverLayout(equations, solutions)

            if self.backend is None:
                solver = MUMPSSolver(self.gui, self.engine, 
                                     comm=self.ranks.solver_comm,
                                     rhs_is_local=True)
                solver.AllocSolver(self.is_complex, self.gui.use_single_precision)
                self.backend = solver

            self.backend.skip_solve = self.skip_solve
            self.backend.SetOperator(matrix, dist=True, name=name, ifactor=ifactor)

        self._call_backend(setup)

    def Mult(self, b, x=None, case_base=0):
        from petram.helper.mpi_recipes import collective_call, gather_vector, scatter_vector
        group = self.ranks.group_comm

        def prepare():
            values = np.asarray(b)
            rows = len(self.layout.equation_ids)
            if values.ndim != 2 or values.shape[0] != rows or values.shape[1] < 1:
                raise ValueError('RHS dimensions do not match the solver layout')
            return np.ascontiguousarray(values)

        values = collective_call(self.comm, prepare)
        meta = self.comm.allgather((values.shape[1], values.dtype.str))
        if len(set(meta)) != 1:
            raise ValueError('Inconsistent RHS count or dtype')
        nrhs = values.shape[1]
        packed = gather_vector(values.reshape(-1), world=group)
        bb = packed.reshape(-1, nrhs) if self.ranks.is_leader else None

        def solve():
            self.backend.skip_solve = self.skip_solve
            return self.backend.Mult(bb, x=x, case_base=case_base)

        sol = self._call_backend(solve)
        if not self.gui.use_dist_sol:
            return sol if self.comm.rank == 0 else None
        send = np.ascontiguousarray(sol).reshape(-1) if self.ranks.is_leader else None
        rows = len(self.layout.solution_ids)
        result = scatter_vector(send, rcounts=rows * nrhs, comm=group)
        return result.reshape(rows, nrhs)

    def close(self):
        if not self.closed:
            self._call_backend(lambda: self.backend.close() if self.backend is not None else None)
            self.ranks.close()
            self.closed = True



class MUMPSPreconditioner(mfem.PyOperator):
    def __init__(self, A0, gui=None, engine=None, silent=True, **kwargs):
        mfem.PyOperator.__init__(self, A0.Height())
        self.gui = gui
        self.engine = engine
        self.silent = silent
        self.irhs_loc = None
        self.is_complex_operator = False

        if 'single' in kwargs and not 'double' in kwargs:
            self.single = kwargs.pop('single')
            self.gui.use_single_precision = self.single
        elif not 'single' in kwargs and 'double' in kwargs:
            self.single = not kwargs.pop('double')
            self.gui.use_single_precision = self.single
        elif 'single' in kwargs and 'double' in kwargs:
            assert False, "singel and double can not be uset together"
        else:
            pass

        self.SetOperator(A0)

    def SetOperator(self, opr):
        def isSparseMatrix(opr):
            return isinstance(opr, mfem.SparseMatrix)

        check = opr._real_operator if isinstance(
            opr, mfem.ComplexOperator) else opr

        if isSparseMatrix(check):
            from mfem.common.sparse_utils import sparsemat_to_scipycsr

            if isinstance(opr, mfem.ComplexOperator):
                mat = (sparsemat_to_scipycsr(opr._real_operator, float) +
                       sparsemat_to_scipycsr(opr._imag_operator, float) * 1j)
                coo_opr = mat.tocoo()
                self.is_complex_operator = True
            else:
                coo_opr = sparsemat_to_scipycsr(opr, float).tocoo()

            self.solver = MUMPSSolver(self.gui, self.engine)
            self.solver.AllocSolver(self.is_complex_operator,
                                    self.gui.use_single_precision)
            self.solver.SetOperator(coo_opr, False)
            self.solver.keep_sol_distributed = True
            self.row_part = [-1, -1]

        else:
            from mfem.common.parcsr_extra import ToScipyCoo

            if isinstance(opr, mfem.ComplexOperator):
                lcsr = (ToScipyCoo(opr._real_operator).tocsr() +
                        ToScipyCoo(opr._imag_operator).tocsr() * 1j)
                lcoo = lcsr.tocoo()
                shape = (opr._real_operator.GetGlobalNumRows(),
                         opr._real_operator.GetGlobalNumCols())

                rpart = opr._real_operator.GetRowPartArray()
                self.is_complex_operator = True
                self.row_part = rpart
            else:
                lcoo = ToScipyCoo(opr)
                shape = (opr.GetGlobalNumRows(), opr.GetGlobalNumCols())
                rpart = opr.GetRowPartArray()
                self.row_part = rpart

            gcoo = coo_matrix(shape)
            gcoo.data = lcoo.data
            gcoo.row = lcoo.row + rpart[0]
            gcoo.col = lcoo.col
            self.solver = MUMPSSolver(self.gui, self.engine)
            self.solver.AllocSolver(self.is_complex_operator,
                                    self.gui.use_single_precision)
            self.solver.SetOperator(gcoo, True)
            self.solver.keep_sol_distributed = True
            self.is_parallel = True

        self.solver.set_silent(self.silent)

    def Mult(self, x, y):
        # in the parallel enviroment, we need to collect x and
        # redistribute y
        # we keep RowPart array from opr since here y is
        # vector not ParVector even in the parallel env.
        if use_parallel:
            from mpi4py import MPI
        else:
            from petram.helper.dummy_mpi import MPI

        myid = MPI.COMM_WORLD.rank
        nproc = MPI.COMM_WORLD.size

        if self.is_complex_operator:
            vec = x.GetDataArray()
            ll = vec.size
            vec = vec[:ll // 2] + 1j * vec[ll // 2:]
        else:
            vec = x.GetDataArray()

        if self.row_part[0] == -1:
            xx = np.atleast_2d(vec).transpose()
        else:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            from petram.helper.mpi_recipes import gather_vector
            if self.gui.use_dist_rhs:
                xx = np.atleast_2d(vec).transpose()
            else:
                xx = gather_vector(vec)
                if myid == 0:
                    xx = np.atleast_2d(xx).transpose()

        # if myid == 0:
        #    print("xx shape (at node-0)", xx.shape)

        s = self.solver.Mult(xx)

        if self.row_part[0] != -1:
            distributed_sol = use_parallel and nproc > 1
            if not distributed_sol:
                comm = MPI.COMM_WORLD
                s = comm.bcast(s)
                s = s[self.row_part[0]:self.row_part[1]]

        if self.is_complex_operator:
            s = np.hstack((s.real.flatten(), s.imag.flatten()))

        # nicePrint(s.shape, y.Size())
        y.Assign(s.flatten().astype(float, copy=False))


class MUMPSBlockPreconditioner(mfem.Solver):
    def __init__(self, opr, gui=None, engine=None, silent=False, **kwargs):
        self.gui = gui
        self.engine = engine
        self.silent = silent
        self.is_complex_operator = False
        if opr is not None:
            self._opr = weakref.ref(opr)
        else:
            self._opr = None
        self.irhs_loc = None
        super(MUMPSBlockPreconditioner, self).__init__()

    def real_to_complex(self, x):
        rows = len(self.block_size)
        of = self.block_offset
        pt = 0
        x2 = np.zeros(of[-1] // 2, dtype=self.dtype)

        for i in range(rows):
            l = of[i + 1] - of[i]
            w = int(l // 2)
            x2[pt:pt + w] = x[of[i]:of[i] + w] + 1j * x[(of[i] + w):of[i + 1]]
            pt = pt + w

        return x2

    def complex_to_real(self, y):
        rows = len(self.block_size)
        of = self.block_offset
        pt = 0

        if self.gui.use_single_precision:
            dtype = np.float32
        else:
            dtype = np.float64

        y2 = np.zeros(of[-1], dtype=dtype)
        for i in range(rows):
            l = of[i + 1] - of[i]
            w = int(l // 2)
            y2[of[i]:(of[i] + w)] = y[pt:pt + w].real
            y2[(of[i] + w):(of[i] + w + w)] = y[pt:pt + w].imag
            pt = pt + w

        return y2

    def Mult(self, x, y):
        if use_parallel:
            from petram.helper.mpi_recipes import (gather_vector,
                                                   scatter_vector,
                                                   allgather_vector)
            from mpi4py import MPI
        else:
            from petram.helper.dummy_mpi import MPI
        myid = MPI.COMM_WORLD.rank
        nproc = MPI.COMM_WORLD.size

        if self.is_complex_operator:
            vec = x.GetDataArray()
            vec = self.real_to_complex(vec)
        else:
            vec = x.GetDataArray()

        if self.row_offset == -1:
            xx = np.atleast_2d(vec).transpose()
        else:
            if self.gui.use_dist_rhs:
                xx = np.atleast_2d(vec).transpose()
            else:
                xx = gather_vector(vec)
                if myid == 0:
                    xx = np.atleast_2d(xx).transpose()

        s = [solver.Mult(xx) for solver in self.solver]

        if self.row_offset != -1:
            # if myid == 0:
            # w = [0.8*np.exp(1j*77/180*np.pi), np.exp(-1j*60/180*np.pi)*1.2]
            w = [1]*len(s)
            s = [xx.flatten()*w[i] for i, xx in enumerate(s)]
            s = np.mean(s, 0)
            # else:
            #    s = None
            distributed_sol = use_parallel and nproc > 1
            if not distributed_sol:
                size = np.sum(self.all_block_size, 0)[myid]
                s = scatter_vector(s, rcounts=size)

            if self.is_complex_operator:
                s = self.complex_to_real(s)

        else:
            s = [xx.flatten() for xx in s]
            s = np.mean(s, 0)
            if self.is_complex_operator:
                s = self.complex_to_real(s)

        y.Assign(s.flatten().astype(float, copy=False))

    def SetOperator(self, opr):
        if use_parallel:
            from mpi4py import MPI
            from petram.helper.mpi_recipes import (gather_vector,
                                                   allgather_vector)
            nprc = MPI.COMM_WORLD.size
            myid = MPI.COMM_WORLD.rank
            smyid = '{:0>6d}'.format(myid)

        # opr = mfem.Opr2BlockOpr(opr)
        # self._opr = weakref.ref(opr)

        opr = self._opr()
        from petram.solver.strumpack_model import build_csr_local
        from petram.solver.solver_utils import check_block_operator
        is_complex, is_parallel = check_block_operator(opr)

        if is_complex:
            if self.gui.use_single_precision:
                dtype = np.complex64
            else:
                dtype = np.complex128
        else:
            if self.gui.use_single_precision:
                dtype = np.float32
            else:
                dtype = np.float64
        self.dtype = dtype
        self.is_complex_operator = is_complex

        lcsr = build_csr_local(opr, dtype, is_complex)

        if use_parallel:

            lcoo = lcsr.tocoo()
            rows = MPI.COMM_WORLD.allgather(lcoo.shape[0])
            global_offsets = np.hstack((0, np.cumsum(rows)))
            self.row_offset = global_offsets[myid]

            shape = (np.sum(rows), lcoo.shape[1])
            gcoo = coo_matrix(shape)
            gcoo.data = lcoo.data
            gcoo.row = lcoo.row + self.row_offset
            gcoo.col = lcoo.col
        else:
            gcoo = lcsr.tocoo()
            self.row_offset = -1

        self.block_offset = np.array(opr.RowOffsets().ToList())
        self.block_size = self.block_offset[1:] - self.block_offset[:-1]

        if use_parallel:
            # create this to distibute solution vector
            self.all_block_size = allgather_vector(self.block_size)
            self.all_block_size = self.all_block_size.reshape(
                nprc, -1).transpose()
            if is_complex:
                self.all_block_size //= 2
            self.mpi_block_size = np.sum(
                self.all_block_size, 0)   # size for each mpi

        else:
            self.irhs_loc = None

        self.solver = []

        if not self.gui.restore_fac:
            solver = MUMPSSolver(self.gui, self.engine)
            solver.set_silent(self.silent)
            solver.AllocSolver(self.is_complex_operator,
                               self.gui.use_single_precision)
            solver.SetOperator(gcoo, is_parallel)
            solver.keep_sol_distributed = True
            self.solver.append(solver)
        else:
            pathes = self.gui.factor_path.split(',')
            for i in range(len(pathes)):
                solver = MUMPSSolver(self.gui, self.engine)
                solver.set_silent(self.silent)
                solver.AllocSolver(self.is_complex_operator,
                                   self.gui.use_single_precision)
                solver.SetOperator(gcoo, is_parallel, ifactor=i)
                solver.keep_sol_distributed = True
                self.solver.append(solver)

        self.is_parallel = is_parallel

        if is_parallel:
            pass
        else:
            self.row_part = [-1, -1]

        for x in self.solver:
            x.set_silent(self.silent)


class MUMPSBlockSolver(LinearSolver):
    '''
    MUMPS LinearSolverInstance which internally uses BlockSolver

    This is used from DerivedValue solver in TimeDependent solver.

    Not recommended to use in future, since it collect solution in
    root node.
    '''
    is_iterative = True

    def __init__(self, *args, **kwargs):
        super(MUMPSBlockSolver, self).__init__(*args, **kwargs)
        self.silent = False

    def SetOperator(self, A, dist, name=None, ifactor=0):
        solver = MUMPSBlockPreconditioner(A,
                                          gui=self.gui,
                                          engine=self.engine,)
        solver.SetOperator(A)
        self._solver = solver

    def Mult(self, b, x=None, case_base=0):
        self._solver.Mult(b[0], x)

        if use_parallel:
            from mpi4py import MPI
            from petram.helper.mpi_recipes import (gather_vector,
                                                   allgather_vector)
            myid = MPI.COMM_WORLD.rank
            if self.gui.use_dist_sol:
                xx = x.GetDataArray()
            else:
                xx = gather_vector(x.GetDataArray())
            if xx is not None:
                xx = np.atleast_2d(xx).transpose()
        else:
            xx = x.GetDataArray().copy().reshape(-1, 1)

        return xx
