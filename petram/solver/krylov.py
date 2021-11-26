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
        class MyGMRESSolver(mfem.PyGMRESSolver):
            def __init__(self, *args, **kwags):
                mfem.PyGMRESSolver.__init__(self, *args, **kwags)

            def Mult(self, x, y):
                mfem.PyGMRESSolver.Mult(self, x, y)

            def MultTranspose(self, x, y):
                mfem.PyGMRESSolver.Mult(self, x, y)
            # def SetOperator(self, x):
            #     mfem.GMRESSolver.SetOperator(self, x)

        args = (MPI.COMM_WORLD,) if use_parallel else ()
        solver = MyGMRESSolver(*args)
        solver.SetAbsTol(self.abstol)
        solver.SetRelTol(self.reltol)
        solver.SetMaxIter(self.maxiter)
        solver.SetPrintLevel(self.log_level)
        if self.solver_type in ['GMRES', 'FGMRES']:
            solver.SetKDim(int(self.kdim))

        solver.iterative_mode = False
        solver.SetOperator(opr)
        return solver
