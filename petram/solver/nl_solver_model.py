'''
 
  non-linear stationary solver

'''
from petram.solver.solver_model import SolverInstance
import os
import numpy as np

from petram.model import Model
from petram.solver.solver_model import Solver
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('NLSolver')
rprint = debug.regular_print('StdSolver')


class NLSolver(Solver):
    can_delete = True
    has_2nd_panel = False

    @classmethod
    def fancy_menu_name(self):
        return 'Stationary(NL)'

    @classmethod
    def fancy_tree_name(self):
        return 'NLStationary'

    def attribute_set(self, v):
        super(NLSolver, self).attribute_set(v)
        v["nl_scheme"] = "Newton"
        v["nl_maxiter"] = 10
        v["nl_reltol"] = 0.001
        v["nl_abstol"] = 0.0
        v["nl_damping"] = 0.2
        v["nl_verbose"] = True
        return v

    def panel1_param(self):
        return [  # ["Initial value setting",   self.init_setting,  0, {},],
            ["physics model",   self.phys_model,  0, {}, ],
            ["Non-linear solver", None, 1, {
                "values": ["FixedPoint", "Newton"]}],
            ["Max iteration", self.nl_maxiter, 400, {}],
            ["NL rel. tol.", self.nl_reltol, 300, {}],
            ["NL abs. tol.", self.nl_abstol, 300, {}],
            ["NL damping", self.nl_damping, 300, {}],
            [None, self.nl_verbose, 3, {
                "text": "verbose output for non-linear iteration"}],
            [None, self.init_only,  3, {"text": "initialize solution only"}],
            [None,
             self.clear_wdir,  3, {"text": "clear working directory"}],
            [None,
             self.assemble_real,  3, {"text": "convert to real matrix (complex prob.)"}],
            [None,
             self.save_parmesh,  3, {"text": "save parallel mesh"}],
            [None,
             self.use_profiler,  3, {"text": "use profiler"}],
            [None, self.skip_solve,  3, {"text": "skip linear solve"}],
            [None, self.load_sol,  3, {
                "text": "load sol file (linear solver is not called)"}],
            [None, self.sol_file,  0, None], ]

    def get_panel1_value(self):
        return (  # self.init_setting,
            self.phys_model,
            self.nl_scheme,
            self.nl_maxiter,
            self.nl_reltol,
            self.nl_abstol,
            self.nl_damping,
            self.nl_verbose,
            self.init_only,
            self.clear_wdir,
            self.assemble_real,
            self.save_parmesh,
            self.use_profiler,
            self.skip_solve,
            self.load_sol,
            self.sol_file)

    def import_panel1_value(self, v):
        #self.init_setting = str(v[0])
        self.phys_model = str(v[0])
        self.nl_scheme = v[1]
        self.nl_maxiter = v[2]
        self.nl_reltol = v[3]
        self.nl_abstol = v[4]
        self.nl_damping = v[5]
        self.nl_verbose = v[6]

        self.init_only = v[7]
        self.clear_wdir = v[8]
        self.assemble_real = v[9]
        self.save_parmesh = v[10]
        self.use_profiler = v[11]
        self.skip_solve = v[12]
        self.load_sol = v[13]
        self.sol_file = v[14]

    def get_editor_menus(self):
        return []

    def get_possible_child(self):
        choice = []
        try:
            from petram.solver.mumps_model import MUMPS
            choice.append(MUMPS)
        except ImportError:
            pass

        # try:
        #    from petram.solver.gmres_model import GMRES
        #    choice.append(GMRES)
        # except ImportError:
        #    pass

        try:
            from petram.solver.iterative_model import Iterative
            choice.append(Iterative)
        except ImportError:
            pass

        try:
            from petram.solver.strumpack_model import Strumpack
            choice.append(Strumpack)
        except ImportError:
            pass
        return choice

    def allocate_solver_instance(self, engine):
        if self.clear_wdir:
            engine.remove_solfiles()

        if self.nl_scheme == 'Newton':
            instance = NewtonSolver(
                self, engine) if self.instance is None else self.instance
        elif self.nl_scheme == 'FixedPoint':
            instance = FixedPointSolver(
                self, engine) if self.instance is None else self.instance
        else:
            assert False, "Unknown Nonlinear solver:" + self.nl_scheme

        return instance

    def get_matrix_weight(self, timestep_config):  # , timestep_weight):
        # this solver uses y, and grad(y)
        if timestep_config[0]:
            if self.nl_scheme == 'Newton':
                return [1, 0, 0, 1]
            else:
                return [1, 0, 0]
        else:
            return [0, 0, 0]

    @debug.use_profiler
    def run(self, engine, is_first=True, return_instance=False):
        dprint1("Entering run (is_first= ", is_first, ") ", self.fullpath())

        instance = self.allocate_solver_instance(engine)

        instance.set_blk_mask()
        if return_instance:
            return instance

        instance.configure_probes(self.probe)
        engine.sol = engine.assembled_blocks[1][0]
        instance.sol = engine.sol

        if self.init_only:
            pass
        elif self.load_sol:
            if is_first:
                instance.assemble()
                is_first = False
            instance.load_sol(self.sol_file)
        else:
            kiter = 0
            instance.reset_count(self.nl_maxiter,
                                 self.nl_abstol,
                                 self.nl_reltol)
            instance.set_damping(self.nl_damping)

            while not instance.done():
                if is_first:
                    instance.assemble()
                    is_first = False
                else:
                    instance.assemble(update=True)

                update_operator = engine.check_block_matrix_changed(
                    instance.blk_mask)
                instance.solve(update_operator=update_operator)

        instance.save_solution(ksol=0,
                               skip_mesh=False,
                               mesh_only=False,
                               save_parmesh=self.save_parmesh)
        engine.sol = instance.sol

        instance.save_probe()

        self.instance = instance

        dprint1(debug.format_memory_usage())
        return is_first


class NonlinearBaseSolver(SolverInstance):
    def __init__(self, gui, engine):
        SolverInstance.__init__(self, gui, engine)
        self.assembled = False
        self.linearsolver = None
        self._operator_set = False
        self._kiter = 0
        self._alpha = 1.0
        self._beta = 0.0
        self._done = False
        self._converged = False

    @property
    def blocks(self):
        return self.engine.assembled_blocks

    @property
    def kiter(self):
        return self._kiter
    
    @property
    def done(self):
        return self._done

    def set_damping(self, damping):
        assert False, "Must be implemented in child"

    def compute_A(self, M, B, X, mask_M, mask_B):
        assert False, "Must be implemented in subclass"

    def compute_rhs(self, M, B, X):
        assert False, "Must be implemented in subclass"
        
    def assemble_rhs(self):
        assert False, "assemble_rhs should not be called"

    def reset_count(self, maxiter, abstol, reltol):
        self._kiter = 0
        self._maxiter = maxiter
        self._current_error = (np.infty, np.infty)
        self._reltol = reltol
        self._abstol = abstol
        self._done = False
        self._converged = False

    def assemble(self, inplace=True, update=False):
        engine = self.engine
        phys_target = self.get_phys()
        phys_range = self.get_phys_range()

        # use get_phys to apply essential to all phys in solvestep
        dprint1("Asembling system matrix",
                [x.name() for x in phys_target],
                [x.name() for x in phys_range])

        if not update:
            engine.run_verify_setting(phys_target, self.gui)
        else:
            engine.set_update_flag('TimeDependent')

        M_updated = engine.run_assemble_mat(
            phys_target, phys_range, update=update)
        B_updated = engine.run_assemble_b(phys_target, update=update)

        engine.run_apply_essential(phys_target, phys_range, update=update)
        engine.run_fill_X_block(update=update)

        _blocks, M_changed = self.engine.run_assemble_blocks(self.compute_A,
                                                             self.compute_rhs,
                                                             inplace=inplace,
                                                             update=update,)
        #A, X, RHS, Ae, B, M, names = blocks
        self.assembled = True
        return M_changed

    def do_solve(self, update_operator=True):
        update_operator = update_operator or not self._operator_set
        engine = self.engine

        # if not self.assembled:
        #    assert False, "assmeble must have been called"

        A, X, RHS, Ae, B, M, depvars = self.blocks
        mask = self.blk_mask
        engine.copy_block_mask(mask)

        depvars = [x for i, x in enumerate(depvars) if mask[0][i]]

        if update_operator:
            AA = engine.finalize_matrix(A, mask, not self.phys_real,
                                        format=self.ls_type)

        BB = engine.finalize_rhs([RHS], A, X[0], mask, not self.phys_real,
                                 format=self.ls_type)

        if self.linearsolver is None:
            linearsolver = self.allocate_linearsolver(
                self.gui.is_complex(), self. engine)
            self.linearsolver = linearsolver
        else:
            linearsolver = self.linearsolver

        linearsolver.skip_solve = self.gui.skip_solve

        if update_operator:
            linearsolver.SetOperator(AA,
                                     dist=engine.is_matrix_distributed,
                                     name=depvars)
            self._operator_set = True

        if linearsolver.is_iterative:
            XX = engine.finalize_x(X[0], RHS, mask, not self.phys_real,
                                   format=self.ls_type)
        else:
            XX = None

        solall = linearsolver.Mult(BB, x=XX, case_base=0)
        if solall is not None:
            dprint1("solall.shape", solall.shape)

        #linearsolver.SetOperator(AA, dist = engine.is_matrix_distributed)
        #solall = linearsolver.Mult(BB, case_base=0)

        if not self.phys_real and self.gui.assemble_real:
            solall = self.linearsolver_model.real_to_complex(solall, AA)

        if solall is not None:
            sol_norm = np.sqrt(solall*np.conj(solall))
        from petram.mfem_config import use_parallel
        if use_parallel:
            from mpi4py import MPI
            if myid == 0:
                sol_norm = MPI.COMM_WORLD.bcast(sol_norm, root=0)
            else:
                sol_norm = MPI.COMM_WORLD.bcast(None, root=0)
        self.sol_norm = sol_norm

        A.reformat_central_mat(
            solall, 0, X[0], mask, alpha=self._alpha, beta=self._beta)
        self.sol = X[0]

        # store probe signal (use t=0.0 in std_solver)
        for p in self.probe:
            p.append_sol(X[0])

        self._kiter = self._kiter + 1

    def load_sol(self, solfile):
        from petram.mfem_config import use_parallel
        if use_parallel:
            from mpi4py import MPI
        else:
            from petram.helper.dummy_mpi import MPI
        myid = MPI.COMM_WORLD.rank

        if myid == 0:
            solall = np.load(solfile)
        else:
            solall = None

        A, X, RHS, Ae, B, M, depvars = self.blocks
        mask = self.blk_mask
        A.reformat_central_mat(solall, 0, X[0], mask)
        self.sol = X[0]

        # store probe signal (use t=0.0 in std_solver)
        for p in self.probe:
            p.append_sol(X[0])

        return True

class NewtonSolver(NonlinearBaseSolver):
    def __init__(self, gui, engine):
        NonlinearBaseSolver.__init__(self, gui, engine)

    def set_damping(self, damping):
        self._alpha = (1.0 - damping)
        self._beta = 1.0

    def compute_A(self, M, B, X, mask_M, mask_B):
        '''
        return A and isAnew
        '''
        if self.kiter == 0:
            A = M[0]
        else:
            A = M[0] + M[3]
        return A, np.any(mask_M[0]) or np.any(mask_M[3])

    def compute_rhs(self, M, B, X):
        '''
        RHS = Ax - b
        '''
        RHS = M[0].dot(self.engine.sol) - B
        return RHS

    def assemble(self, inplace=True, update=False):
        if self.kiter == 0:
            self.engine.set_enabled_matrix([True, False, False, False])
        else:
            self.engine.set_enabled_matrix([True, False, False, True])
        NonlinearBaseSolver.assemble(self, inplace=inplace, update=update)

    def solve(self, update_operator=True):
        A, X, RHS, Ae, B, M, depvars = self.blocks

        if self.kiter == 0:
            self.norm0 = X[0].norm()
        elif self.kiter == 1:
            self.norm1 = X[1].norm()

        self.do_solve(update_operator=update_operator)

        if self.kiter == 1:
            self.correction0 = sol_norm * self._alpha
        else:
            correction = sol_norm * self._alpha
            if correction < self.correction0*self._reltol:
                self._converged = True
                self._done = True

        if self._kiter >= self._maxiter:
            self._done = True

        self.engine.add_FESvariable_to_NS(self.get_phys())


class FixedPointSolver(NonlinearBaseSolver):
    def __init__(self, gui, engine):
        NonlinearBaseSolver.__init__(self, gui, engine)

    def set_damping(self, damping):
        self._alpha = (1.0 - damping)
        self._beta = damping

    def compute_A(self, M, B, X, mask_M, mask_B):
        return M[0], np.any(mask_M[0])

    def compute_rhs(self, M, B, X):
        return B

    def solve(self, update_operator=True):
        A, X, RHS, Ae, B, M, depvars = self.blocks

        if self.kiter == 0:
            self.norm0 = X[0].norm()

        xdata = self.copy_x(X[0])
        self.do_solve(update_operator=update_operator)

        diffnorm = self.diff_norm(X[0], xdata)
        norm = X[0].norm

        if self.kiter == 1:
            self.correction0 = norm
        else:
            if diffnorm < self.correction0*self._reltol:
                self._converged = True
                self._done = True
            if abs(norm - self.norm0)/abs(norm + self.norm0) < self._reltol:
                self._converged = True
                self._done = True

        if self._kiter >= self._maxiter:
            self._done = True

        self.engine.add_FESvariable_to_NS(self.get_phys())

    def copy_x(self, X):
        shape = X.shape
        xdata = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                v = X[i, j]
                if isinstance(v, chypre.CHypreVec):
                    vec = v.toarray()
                elif isinstance(v, ScipyCoo):
                    vec = v
                else:
                    assert False, "not supported"
                xdata.append(vec.copy())
        return xdata

    def diff_norm(self, X, xdata):
        shape = X.shape
        xdata = []
        idx = 0
        norm = 0
        for i in range(shape[0]):
            for j in range(shape[1]):
                v = X[i, j]
                if isinstance(v, chypre.CHypreVec):
                    vec = v.toarray()
                elif isinstance(v, ScipyCoo):
                    vec = v
                else:
                    assert False, "not supported"

                delta = xdata[idx] - vec
                norm += delta * np.conj(delta)
                idx = idx+1

        from petram.mfem_config import use_parallel
        if use_parallel:
            norm = np.sum(np.array(MPI.COMM_WORLD.allgather(norm)))
        return np.sqrt(norm)
