from petram.mfem_config import use_parallel
from .solver_model import LinearSolverModel, LinearSolver
from petram.helper.matrix_file import write_matrix, write_vector, write_coo_matrix
import os
import numpy as np
import scipy
import weakref

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('MUMPSModel')


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
                ["use float32", self.use_single_precision, 3, {"text": ""}], ]

    def get_panel1_value(self):
        return (int(self.log_level), self.ordering, self.out_of_core,
                self.error_ana, self.write_mat, self.write_fac,
                self.restore_fac, self.factor_path,
                self.use_blr, self.blr_drop, str(self.icntl14),
                str(self.icntl23),
                self.cntl1, self.cntl4, self.icntl10, self.cntl2,
                self.icntl6, self.icntl8, self.write_inv,
                self.use_single_precision)

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
        solver = MUMPSSolver(self, engine)
        solver.AllocSolver(is_complex, self.use_single_precision)
        return solver

    def solve(self, engine, A, b):
        solver = self.allocate_solver((A.dtype == 'complex'), engine)
        solver.SetOperator(A, dist=engine.is_matrix_distributed)
        solall = solver.Mult(b, case_base=engine.case_base)
        return solall

    def real_to_complex(self, solall, M=None):
        try:
            from mpi4py import MPI
        except BaseException:
            from petram.helper.dummy_mpi import MPI
        myid = MPI.COMM_WORLD.rank
        nproc = MPI.COMM_WORLD.size

        if myid == 0:
            s = solall.shape[0]
            solall = solall[:s // 2, :] + 1j * solall[s // 2:, :]
            return solall

    def __del__(self):
        if self.s is not None:
            self.s.finish()
        self.s = None


class MUMPS(MUMPSBase):
    def does_linearsolver_choose_linearsystem_type(self):
        from petram.solver.solver_model import Solver

        # for the sake of backword compabitibiliy. it is true only when
        # it is the top level linearsolver

        if isinstance(self.parent, Solver):
            return True
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

    def prepare_solver(self, opr, engine):
        solver = MUMPSBlockPreconditioner(opr,
                                          gui=self,
                                          engine=engine,)
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

    def prepare_solver(self, opr, engine):
        prc = MUMPSPreconditioner(opr,
                                  gui=self,
                                  engine=engine)
        return prc


class MUMPSSolver(LinearSolver):
    is_iterative = False

    def __init__(self, *args, **kwargs):
        super(MUMPSSolver, self).__init__(*args, **kwargs)
        self.silent = False

    @staticmethod
    def split_dir_prefix(txt):
        txt = os.path.abspath(os.path.expanduser(txt))
        txt = txt.strip().split('*')[0]
        path = os.path.dirname(txt)
        prefix = os.path.basename(txt)
        try:
            from mpi4py import MPI
            myid = MPI.COMM_WORLD.rank
        except BaseException:
            myid = 0
            from petram.helper.dummy_mpi import MPI

        if myid == 0:
            if not os.path.exists(path):
                dprint1("Creating path for MUMPS data", path)
                os.makedirs(path)
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
        #s.set_icntl(28,  2)

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
        if is_complex:
            if use_single_precision:
                from petram.ext.mumps.mumps_solve import c_array as data_array
                from petram.ext.mumps.mumps_solve import CMUMPS
                s = CMUMPS()
            else:
                from petram.ext.mumps.mumps_solve import z_array as data_array
                from petram.ext.mumps.mumps_solve import ZMUMPS
                s = ZMUMPS()
        else:
            if use_single_precision:
                from petram.ext.mumps.mumps_solve import s_array as data_array
                from petram.ext.mumps.mumps_solve import SMUMPS
                s = SMUMPS()
            else:
                from petram.ext.mumps.mumps_solve import d_array as data_array
                from petram.ext.mumps.mumps_solve import DMUMPS
                s = DMUMPS()

        self.s = s
        self.is_complex = is_complex
        self.data_array = data_array

        gui = self.gui
        # No outputs
        if gui.log_level == 0:
            s.set_icntl(1, -1)
            s.set_icntl(2, -1)
            s.set_icntl(3, -1)
            s.set_icntl(4, 0)
        elif gui.log_level == 1:
            pass
        else:
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
        MPI.COMM_WORLD.Barrier()

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

    def SetOperator(self, A, dist, name=None, ifactor=0):
        try:
            from mpi4py import MPI
        except BaseException:
            from petram.helper.dummy_mpi import MPI
        myid = MPI.COMM_WORLD.rank
        nproc = MPI.COMM_WORLD.size

        from petram.ext.mumps.mumps_solve import i_array
        gui = self.gui
        s = self.s
        if dist:
            dprint1("SetOperator distributed matrix")
            A.eliminate_zeros()
            if gui.write_mat:
                write_coo_matrix('matrix', A)
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
            nnz_array = np.array(MPI.COMM_WORLD.allgather(A.nnz))
            if myid == 0:
                dprint1("NNZ all: ", nnz_array, np.sum(nnz_array))
                s.set_n(A.shape[1])
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

            s.set_icntl(2, 1)

            self.dataset = (A.data, row, col)
        else:
            A = A.tocoo(False)  # .astype('complex')
            import petram.ext.mumps.mumps_solve as mumps_solve
            dprint1('!!!these two must be consistent')
            dprint1('sizeof(MUMPS_INT) ', mumps_solve.SIZEOF_MUMPS_INT())

            dtype_int = 'int' + str(mumps_solve.SIZEOF_MUMPS_INT() * 8)

            if gui.write_mat:
                # tocsr().tocoo() forces the output is row sorted.
                write_coo_matrix('matrix', A.tocsr().tocoo())
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
                self.dataset = (A.data, row, col)

        # blr
        if gui.use_blr:
            s.set_icntl(35, 1)
            s.set_cntl(7, float(gui.blr_drop))

        # out-of-core
        if gui.out_of_core:
            s.set_icntl(22, 1)

        if gui.icntl14.lower() != 'default':       # percentage increase in the estimated workingspace
            s.set_icntl(14, convert2int(gui.icntl14))

        if gui.icntl23.lower() != 'default':       # maximum size of the working memory
            s.set_icntl(23, convert2int(gui.icntl23))

        if gui.icntl8.lower() != 'default':        # the scaling strategy
            s.set_icntl(8, convert2int(gui.icntl8))

        if gui.icntl6.lower() != 'default':        # permutes the matrix to  azero-freediagonal and/or
            s.set_icntl(6, convert2int(gui.icntl6))  # scale the matrix

        if gui.cntl1.lower() != 'default':         # relative threshold for numerical pivoting
            s.set_cntl(1, convert2float(gui.cntl1))

        if gui.cntl4.lower() != 'default':         # threshold for static pivoting
            s.set_cntl(4, convert2float(gui.cntl4))

        self.set_ordering_flag(s)

        MPI.COMM_WORLD.Barrier()
        if not gui.restore_fac:
            if not self.silent:
                dprint1("job1")
            s.set_job(1)
            s.run()
            info1 = s.get_info(1)

            if info1 != 0:
                assert False, "MUMPS call (job1) failed. Check error log"

            if not self.silent:
                dprint1("job2")
            s.set_icntl(24, 1)

            s.set_job(2)
            s.run()
            info1 = s.get_info(1)
            if info1 != 0:
                assert False, "MUMPS call (job2) failed. Check error log"
            if gui.write_fac:
                if not self.silent:
                    dprint1("job7 (save)")
                prefix, path = self.split_dir_prefix(gui.factor_path)
                s.set_saveparam(prefix, path)
                s.set_oocparam("ooc_", path)

                # wait here to make sure path is created.
                MPI.COMM_WORLD.Barrier()

                s.set_job(7)
                s.run()
                info1 = s.get_info(1)
                if info1 != 0:
                    assert False, "MUMPS call (job7) failed. Check error log"

        else:
            if not self.silent:
                dprint1("job8 (restore)")
            pathes = gui.factor_path.split(',')
            prefix, path = self.split_dir_prefix(pathes[ifactor].strip())

            s.set_saveparam(prefix, path)
            s.set_oocparam("ooc_", path)

            MPI.COMM_WORLD.Barrier()  # wait here to make sure path is created.
            # s.set_job(-1)
            # s.run()
            s.set_job(8)
            s.run()
            info1 = s.get_info(1)
            if info1 != 0:
                assert False, "MUMPS call (job8) failed. Check error log"

    def write_inverse(self, s, b):
        try:
            from mpi4py import MPI
        except BaseException:
            from petram.helper.dummy_mpi import MPI
        myid = MPI.COMM_WORLD.rank
        smyid = '{:0>6d}'.format(myid)
        mpi_size = MPI.COMM_WORLD.size

        datatype = self._data_type()

        nrhs = b.shape[0] if myid == 0 else 0
        #nrhs = 30

        if mpi_size > 1:
            nrhs = MPI.COMM_WORLD.bcast(nrhs)

        if myid == 0:
            dprint1("nrhs", nrhs)
            s.set_lrhs_nrhs(b.shape[0], nrhs)
            bb = np.zeros((b.shape[0], nrhs), dtype=datatype)
            for i in range(nrhs):
                bb[i, i] = 1.0
            bstack = np.hstack(np.transpose(bb))
            bstack = self.make_vector_entries(bstack)
            s.set_rhs(self.data_array(bstack))

        n_pivots = s.get_info(23)
        #print("n_pivots", n_pivots)

        sol_loc = np.zeros(n_pivots * nrhs, dtype=self._data_type())
        isol_loc = np.zeros(n_pivots, dtype=self._int_type())

        from petram.ext.mumps.mumps_solve import i_array
        s.set_sol_loc(self.data_array(sol_loc),
                      n_pivots,
                      i_array(isol_loc))

        self.set_error_analysis(s)

        # set to distributed sol mode
        use_distributed_save = True
        if use_distributed_save:
            s.set_icntl(21, 1)
        s.set_job(3)
        s.run()
        info1 = s.get_info(1)
        if info1 != 0:
            assert False, "MUMPS call (job3) failed. Check error log"

        #sol = s.get_sol_loc()
        if use_distributed_save:
            sol = np.transpose(sol_loc.reshape(-1, n_pivots))
            np.savez("matrix_inv_idx." + smyid, A_inv_idx=isol_loc)
            np.savez("matrix_inv." + smyid, A_inv=sol)
        else:
            if myid == 0:
                if self.is_complex:
                    sol = s.get_real_rhs() + 1j * s.get_imag_rhs()
                else:
                    sol = s.get_real_rhs()
                sol = np.transpose(sol.reshape(-1, len(bb)))
                np.savez("matrix_inv", A_inv=sol)

        # set to gathered sol mode
        s.set_icntl(21, 0)

    def Mult(self, b, x=None, case_base=0):
        try:
            from mpi4py import MPI
        except BaseException:
            from petram.helper.dummy_mpi import MPI
        myid = MPI.COMM_WORLD.rank
        nproc = MPI.COMM_WORLD.size

        #self.SetOperator(A, b, True, engine)
        gui = self.gui
        s = self.s
        if gui.write_mat:
            if myid == 0:
                for ib in range(b.shape[1]):
                    write_vector('rhs_' + str(ib + case_base), b[:, ib])
                case_base = case_base + b.shape[1]
            else:
                case_base = None
            case_base = MPI.COMM_WORLD.bcast(case_base, root=0)

        if self.gui.write_inv:
            self.write_inverse(s, b)
        if myid == 0:
            s.set_lrhs_nrhs(b.shape[0], b.shape[1])
            bstack = np.hstack(np.transpose(b))
            bstack = self.make_vector_entries(bstack)
            s.set_rhs(self.data_array(bstack))

        if self.silent:
            s.set_icntl(1, -1)
            s.set_icntl(2, -1)
            s.set_icntl(3, -1)
            s.set_icntl(4, 0)

        if self.gui.icntl10.lower() != 'default':       # iterative refinement
            s.set_icntl(10, convert2int(self.gui.icntl10))

        if self.gui.cntl2.lower() != 'default':
            # stopping criterion for iterative refinement
            s.set_cntl(2, convert2float(self.gui.cntl2))

        if not self.silent:
            dprint1("job3")
        self.set_error_analysis(s)
        s.set_job(3)
        s.run()

        info1 = s.get_info(1)
        if info1 != 0:
            assert False, "MUMPS call (job3) failed. Check error log"

        # s.finish()

        rsol = None
        isol = None
        sol_extra = None
        if (myid == 0):
            if self.is_complex:
                sol = s.get_real_rhs() + 1j * s.get_imag_rhs()
            else:
                sol = s.get_real_rhs()

            sol = np.transpose(sol.reshape(-1, len(b)))
            return sol


if use_parallel:
    from petram.helper.mpi_recipes import *
    from mfem.common.parcsr_extra import *
    import mfem.par as mfem
else:
    import mfem.ser as mfem


class MUMPSPreconditioner(mfem.PyOperator):
    def __init__(self, A0, gui=None, engine=None, silent=True, **kwargs):
        mfem.PyOperator.__init__(self, A0.Height())
        self.gui = gui
        self.engine = engine
        self.silent = silent
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
            self.row_part = [-1, -1]

        else:
            from mfem.common.parcsr_extra import ToScipyCoo
            from scipy.sparse import coo_matrix

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
            self.is_parallel = True

        self.solver.set_silent(self.silent)

    def Mult(self, x, y):
        # in the parallel enviroment, we need to collect x and
        # redistribute y
        # we keep RowPart array from opr since here y is
        # vector not ParVector even in the parallel env.
        try:
            from mpi4py import MPI
        except BaseException:
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
            xx = gather_vector(vec)
            if myid == 0:
                xx = np.atleast_2d(xx).transpose()

        print("xx shape", xx.shape)
        s = self.solver.Mult(xx)

        if self.row_part[0] != -1:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            s = comm.bcast(s)
            s = s[self.row_part[0]:self.row_part[1]]

        if self.is_complex_operator:
            s = np.hstack((s.real.flatten(), s.imag.flatten()))

        y.Assign(s.flatten())


def check_block_operator(A):
    from petram.solver.strumpack_model import get_block

    rows = A.NumRowBlocks()
    cols = A.NumColBlocks()

    is_complex = False
    is_parallel = False
    for i in range(rows):
        for j in range(cols):
            m = get_block(A, i, j)
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


class MUMPSBlockPreconditioner(mfem.Solver):
    def __init__(self, opr, gui=None, engine=None, silent=True, **kwargs):
        self.gui = gui
        self.engine = engine
        self.silent = silent
        self.is_complex_operator = False
        self._opr = weakref.ref(opr)
        super(MUMPSBlockPreconditioner, self).__init__()

    def real_to_complex(self, x):
        '''
        if use_parallel:
           from mpi4py import MPI
           myid     = MPI.COMM_WORLD.rank


           offset = M.RowOffsets().ToList()
           of = [np.sum(MPI.COMM_WORLD.allgather(np.int32(o))) for o in offset]
           if myid != 0: return

        else:
            pass
        '''

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
        '''
        if use_parallel:
           from mpi4py import MPI
           myid     = MPI.COMM_WORLD.rank


           offset = M.RowOffsets().ToList()
           of = [np.sum(MPI.COMM_WORLD.allgather(np.int32(o))) for o in offset]
           if myid != 0: return

        else:
            pass
        '''
        rows = len(self.block_size)
        of = self.block_offset
        pt = 0
        y2 = np.zeros(of[-1], dtype=np.float)
        for i in range(rows):
            l = of[i + 1] - of[i]
            w = int(l // 2)
            y2[of[i]:(of[i] + w)] = y[pt:pt + w].real
            y2[(of[i] + w):(of[i] + w + w)] = y[pt:pt + w].imag
            pt = pt + w

        #assert False, "test"
        return y2

    def Mult(self, x, y):
        try:
            from mpi4py import MPI
        except BaseException:
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
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            from petram.helper.mpi_recipes import gather_vector
            xx = gather_vector(vec)
            if myid == 0:
                xx = np.atleast_2d(xx).transpose()

        s = [solver.Mult(xx) for solver in self.solver]

        if self.row_offset != -1:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            nprc = MPI.COMM_WORLD.size
            myid = MPI.COMM_WORLD.rank

            if myid == 0:
                #w = [0.8*np.exp(1j*77/180*np.pi), np.exp(-1j*60/180*np.pi)*1.2]
                w = [1]*len(s)
                s = [xx.flatten()*w[i] for i, xx in enumerate(s)]
                s = np.mean(s, 0)
            else:
                s = None

            size = np.sum(self.all_block_size, 0)[myid]
            s = scatter_vector(s, rcounts=size)
            if self.is_complex_operator:
                s = self.complex_to_real(s)

        else:
            s = [xx.flatten() for xx in s]
            s = np.mean(s, 0)
            if self.is_complex_operator:
                s = self.complex_to_real(s)

        y.Assign(s.flatten())

    def SetOperator(self, opr):
        if use_parallel:
            from mpi4py import MPI
            from petram.helper.mpi_recipes import gather_vector
            nprc = MPI.COMM_WORLD.size
            myid = MPI.COMM_WORLD.rank
            smyid = '{:0>6d}'.format(myid)
            from mfem.common.mpi_debug import nicePrint
        else:
            def nicePrint(*x):
                print(x)

        #opr = mfem.Opr2BlockOpr(opr)
        #self._opr = weakref.ref(opr)

        opr = self._opr()
        from petram.solver.strumpack_model import build_csr_local

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
            from scipy.sparse import coo_matrix

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
            self.mpi_block_size = np.sum(
                self.all_block_size, 0)   # size for each mpi
            if is_complex:
                self.all_block_size //= 2
            # nicePrint(self.all_block_size)

        self.solver = []

        if not self.gui.restore_fac:
            solver = MUMPSSolver(self.gui, self.engine)
            solver.AllocSolver(self.is_complex_operator,
                               self.gui.use_single_precision)
            solver.SetOperator(gcoo, is_parallel)
            self.solver.append(solver)
        else:
            pathes = self.gui.factor_path.split(',')
            for i in range(len(pathes)):
                solver = MUMPSSolver(self.gui, self.engine)
                solver.AllocSolver(self.is_complex_operator,
                                   self.gui.use_single_precision)
                solver.SetOperator(gcoo, is_parallel, ifactor=i)
                self.solver.append(solver)

        self.is_parallel = is_parallel

        if is_parallel:
            pass
        else:
            self.row_part = [-1, -1]

        for x in self.solver:
            x.set_silent(self.silent)
