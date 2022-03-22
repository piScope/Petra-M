'''

Krylov Model. 
  Krylov solver
  This model is supposeed to work inside Iteratife solver chain.
  This model itsel does not define an iterative solver itself.

'''
from petram.solver.mumps_model import MUMPSPreconditioner
from petram.mfem_config import use_parallel
import numpy as np

from petram.debug import flush_stdout
from petram.namespace_mixin import NS_mixin
from .solver_model import LinearSolverModel, LinearSolver

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('KrylovModel')

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
    smyid = ''

choices_a = ['CG', 'GMRES', 'FGMRES', 'BiCGSTAB', 'MINRES']

single1_elp = [["log_level", -1, 400, {}],
               ["max  iter.", 200, 400, {}],
               ["rel. tol", 1e-7, 300, {}],
               ["abs. tol.", 1e-7, 300, {}], ]
single2_elp = [["log_level", -1, 400, {}],
               ["max  iter.", 200, 400, {}],
               ["rel. tol", 1e-7, 300, {}],
               ["abs. tol.", 1e-7, 300, {}],
               ["restart(kdim)", 50, 400, {}]]


class KrylovModel(LinearSolverModel, NS_mixin):
    hide_ns_menu = True
    has_2nd_panel = False
    accept_complex = False
    always_new_panel = False

    def __init__(self, *args, **kwargs):
        LinearSolverModel.__init__(self, *args, **kwargs)
        NS_mixin.__init__(self, *args, **kwargs)

    def init_solver(self):
        pass

    def panel1_param(self):
        return [[None, None, 34, [{'text': "Solver", 'choices': choices_a},
                                  {'elp': single1_elp},  # CG
                                  {'elp': single2_elp},  # GMRES
                                  {'elp': single2_elp},  # FGMRES
                                  {'elp': single1_elp},  # BiCGSTAB
                                  {'elp': single1_elp},  # MINRES
                                  ], ],
                [None, self.write_mat, 3, {"text": "write matrix"}],
                [None, self.assert_no_convergence, 3,
                    {"text": "check converegence"}], ]

    def get_panel1_value(self):
        # this will set _mat_weight
        from petram.solver.solver_model import SolveStep
        p = self.parent
        while not isinstance(p, SolveStep):
            p = p.parent
            if p is None:
                assert False, "Solver is not under SolveStep"

        single1 = [int(self.log_level), int(self.maxiter),
                   self.reltol, self.abstol]
        single2 = [int(self.log_level), int(self.maxiter),
                   self.reltol, self.abstol, int(self.kdim)]
        value = ([self.solver_type, single1, single2, single2,
                  single1, single1],
                 self.write_mat, self.assert_no_convergence,)

        return value

    def import_panel1_value(self, v):
        self.solver_type = str(v[0][0])
        idx = choices_a.index(self.solver_type)
        vv = v[0][idx + 1]
        self.log_level = int(vv[0])
        self.maxiter = int(vv[1])
        self.reltol = vv[2]
        self.abstol = vv[3]

        if len(vv) > 4:
            self.kdim = int(vv[4])

        self.write_mat = bool(v[1])
        self.assert_no_convergence = bool(v[2])

    def attribute_set(self, v):
        v = super(KrylovModel, self).attribute_set(v)
        v['solver_type'] = 'GMRES'
        v['log_level'] = 0
        v['maxiter'] = 200
        v['reltol'] = 1e-7
        v['abstol'] = 1e-7
        v['kdim'] = 50

        v['printit'] = 1
        v['write_mat'] = False
        v['assert_no_convergence'] = True
        return v

    def get_possible_child(self):
        from petram.solver.mumps_model import MUMPSMFEMSolverModel as MUMPS
        from petram.solver.block_smoother import DiagonalPreconditioner

        return KrylovModel, MUMPS, DiagonalPreconditioner

    def get_possible_child_menu(self):
        from petram.solver.mumps_model import MUMPSMFEMSolverModel as MUMPS
        from petram.solver.block_smoother import DiagonalPreconditioner

        choice = [("Preconditioner", KrylovSmoother),
                  ("", MUMPS),
                  ("!", DiagonalPreconditioner), ]
        return choice

    def verify_setting(self):
        return True, "", ""

    def real_to_complex(self, solall, M):
        if self.merge_real_imag:
            return self.real_to_complex_merged(solall, M)
        else:
            return self.real_to_complex_interleaved(solall, M)

    def real_to_complex_interleaved(self, solall, M):
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

        rows = M.NumRowBlocks()
        s = solall.shape
        nb = rows // 2
        i = 0
        pt = 0
        result = np.zeros((s[0] // 2, s[1]), dtype='complex')
        for j in range(nb):
            l = of[i + 1] - of[i]
            result[pt:pt + l, :] = (solall[of[i]:of[i + 1], :]
                                    + 1j * solall[of[i + 1]:of[i + 2], :])
            i = i + 2
            pt = pt + l

        return result

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

    def prepare_preconditioner(self, opr, engine):
        for x in self.iter_enabled():
            if isinstance(x, LinearSolverModel):
                return x.prepare_solver(opr, engine)


    def do_prepare_solver(self, opr, engine):
        cls = getattr(mfem, self.solver_type + 'Solver')
        args = (MPI.COMM_WORLD,) if use_parallel else ()

        solver = cls(*args)
        solver.SetAbsTol(self.abstol)
        solver.SetRelTol(self.reltol)
        solver.SetMaxIter(self.maxiter)
        solver.SetPrintLevel(self.log_level)
        if self.solver_type in ['GMRES', 'FGMRES']:
            solver.SetKDim(int(self.kdim))

        solver.iterative_mode = True
        solver.SetOperator(opr)
        if self.write_mat:
            self.write_matrix(opr)

        M = self.prepare_preconditioner(opr, engine)

        if M is not None:
            solver.SetPreconditioner(M)
            solver._prc = M
        return solver

    def write_matrix(self, A=None, b=None, x=None, suffix=smyid):
        def get_block(Op, i, j):
            try:
                return Op._linked_op[(i, j)]
            except KeyError:
                return None

        offset = A.RowOffsets()
        rows = A.NumRowBlocks()
        cols = A.NumColBlocks()
        if suffix != '':
            suffix = '.'+suffix

        if A is not None:
            for i in range(cols):
                for j in range(rows):
                    m = get_block(A, i, j)
                    if m is None:
                        continue
                    if isinstance(m, mfem.ComplexOperator):
                        m1 = m._real_operator
                        m2 = m._imag_operator
                        m1.Print('matrix_Re_' + str(i) + '_' + str(j))
                        m2.Print('matrix_Im_' + str(i) + '_' + str(j))
                    else:
                        m.Print('matrix_' + str(i) + '_' + str(j))
        if b is not None:
            for i, bb in enumerate(b):
                for j in range(rows):
                    v = bb.GetBlock(j)
                    v.Print('rhs_' + str(i) + '_' + str(j) + suffix)
        if x is not None:
            for j in range(rows):
                xx = x.GetBlock(j)
                xx.Print('x_' + str(i) + '_' + str(j) + suffix)

    def prepare_solver(self, opr, engine):
        solver = self.do_prepare_solver(opr, engine)
        solver.iterative_mode = True

        return solver

    @classmethod
    def fancy_menu_name(self):
        return 'KrylovSolver'

    @classmethod
    def fancy_tree_name(self):
        return 'Krylov'

    def get_info_str(self):
        return 'Solver'

    def does_linearsolver_choose_linearsystem_type(self):
        return False

    def supported_linear_system_type(self):
        return ["blk_interleave",
                "blk_merged_s",
                "blk_merged", ]


class KrylovSmoother(KrylovModel):
    @classmethod
    def fancy_menu_name(self):
        return 'KrylovSmoother'

    @classmethod
    def fancy_tree_name(self):
        return 'Krylov'

    def get_info_str(self):
        return 'Smoother'

    def attribute_set(self, v):
        v = KrylovModel.attribute_set(self, v)
        v['log_level'] = -1
        v['abstol'] = 0.0
        v['assert_no_convergence'] = False
        return v

    def prepare_solver(self, opr, engine):
        solver = self.do_prepare_solver(opr, engine)
        solver.iterative_mode = False
        return solver

    def prepare_solver_with_multtranspose(self, opr, engine):
        '''
        This is called from multi-level refinement
        '''
        class MyPySolver(mfem.PyIterativeSolver):
            def __init__(self, solver_type):
                args = (MPI.COMM_WORLD,) if use_parallel else ()
                cls = getattr(mfem, solver_type + 'Solver')
                self.s1 = cls(*args)
                self.s2 = cls(*args)
                mfem.PyIterativeSolver.__init__(self, *args)

            def SetAbsTol(self, tol):
                self.s1.SetAbsTol(tol)
                self.s2.SetAbsTol(tol)

            def SetRelTol(self, tol):
                self.s1.SetRelTol(tol)
                self.s2.SetRelTol(tol)

            def SetMaxIter(self, iter):
                self.s1.SetMaxIter(iter)
                self.s2.SetMaxIter(iter)

            def SetPrintLevel(self, lvl):
                self.s1.SetPrintLevel(lvl)
                self.s2.SetPrintLevel(lvl)

            def SetKDim(self, d):
                self.s1.SetKDim(d)
                self.s2.SetKDim(d)

            def SetOperator(self, opr):
                self.s1.SetOperator(opr)
                opr2 = mfem.TransposeOperator(opr)
                self.s2.SetOperator(opr2)
                self._oprs = (opr, opr2)

            def SetPreconditioner(self, opr):
                self.s1.SetPreconditioner(opr)
                #opr2 = mfem.TransposeOperator(opr)
                # self.s2.SetPreconditioner(opr2)
                self._prcs = (opr, )

            def Mult(self, x, y):
                self.s1.Mult(x, y)

            def MultTranspose(self, x, y):
                self.s1.Mult(x, y)

            @property
            def iterative_mode(self):
                return self.s1.iterative_mode

            @iterative_mode.setter
            def iterative_mode(self, v):
                self.s1.iterative_mode = v
                self.s2.iterative_mode = v

        args = (MPI.COMM_WORLD,) if use_parallel else ()
        solver = MyPySolver(self.solver_type)
        solver.SetAbsTol(self.abstol)
        solver.SetRelTol(self.reltol)
        solver.SetMaxIter(self.maxiter)
        solver.SetPrintLevel(self.log_level)
        if self.solver_type in ['GMRES', 'FGMRES']:
            solver.SetKDim(int(self.kdim))

        M = self.prepare_preconditioner(opr, engine)
        if M is not None:
            solver.SetPreconditioner(M)

        solver.iterative_mode = False
        solver.SetOperator(opr)
        return solver


class StationaryRefinementModel(KrylovModel):
    hide_ns_menu = True
    has_2nd_panel = False
    accept_complex = False
    always_new_panel = False

    def panel1_param(self):
        p1 = [["log_level", -1, 400, {}],
              ["max  iter.", 200, 400, {}],
              ["rel. tol", 1e-7, 300, {}],
              ["abs. tol.", 1e-7, 300, {}], ]
        return p1

    def get_panel1_value(self):
        # this will set _mat_weight
        from petram.solver.solver_model import SolveStep
        p = self.parent
        while not isinstance(p, SolveStep):
            p = p.parent
            if p is None:
                assert False, "Solver is not under SolveStep"

        single1 = [int(self.log_level), int(self.maxiter),
                   self.reltol, self.abstol]
        return single1

    def import_panel1_value(self, v):
        self.log_level = int(v[0])
        self.maxiter = int(v[1])
        self.reltol = v[2]
        self.abstol = v[3]

    def attribute_set(self, v):
        v = super(StationaryRefinementModel, self).attribute_set(v)
        v['log_level'] = 0
        v['maxiter'] = 200
        v['reltol'] = 1e-7
        v['abstol'] = 1e-7
        v['printit'] = 1
        v['write_mat'] = False
        v['assert_no_convergence'] = False
        return v

    def do_prepare_solver(self, opr, engine):
        cls = mfem.SLISolver
        args = (MPI.COMM_WORLD,) if use_parallel else ()

        solver = cls(*args)
        solver.SetAbsTol(self.abstol)
        solver.SetRelTol(self.reltol)
        solver.SetMaxIter(self.maxiter)
        solver.SetPrintLevel(self.log_level)

        solver.iterative_mode = True
        solver.SetOperator(opr)

        M = self.prepare_preconditioner(opr, engine)

        if M is not None:
            solver.SetPreconditioner(M)
            solver._prc = M
        return solver

    def prepare_solver(self, opr, engine):
        solver = self.do_prepare_solver(opr, engine)
        solver.iterative_mode = True

        return solver

    @classmethod
    def fancy_menu_name(self):
        return 'StationaryRefinement'

    @classmethod
    def fancy_tree_name(self):
        return 'Refinement'

    def get_info_str(self):
        return 'Refinement'

    def does_linearsolver_choose_linearsystem_type(self):
        return False

    def supported_linear_system_type(self):
        return ["blk_interleave",
                "blk_merged_s",
                "blk_merged", ]


class StationaryRefinementSmoother(StationaryRefinementModel):
    def get_info_str(self):
        return 'Smoother'

    def attribute_set(self, v):
        v = KrylovModel.attribute_set(self, v)
        v['log_level'] = -1
        v['abstol'] = 0.0
        v['assert_no_convergence'] = False
        return v

    def prepare_solver(self, opr, engine):
        solver = self.do_prepare_solver(opr, engine)
        solver.iterative_mode = False
        return solver
