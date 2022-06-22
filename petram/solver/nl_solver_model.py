'''
 
  non-linear stationary solver

'''
import mfem.common.chypre as chypre
import petram.helper.block_matrix as bm
from petram.solver.solver_model import SolverInstance
import os
import numpy as np

from petram.model import Model
from petram.solver.solver_model import Solver
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('NLSolver')


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
        v["nl_damping"] = 1.0
        v["nl_verbose"] = True
        v['dwc_name'] = ''
        v['use_dwc_nl'] = False
        v['dwc_nl_arg'] = ''

        return v

    def panel1_param(self):
        ret = [["dwc",   self.dwc_name,   0, {}],
               ["args.",   self.dwc_nl_arg,   0, {}]]
        value = [self.dwc_name, self.dwc_nl_arg]
        return [  # ["Initial value setting",   self.init_setting,  0, {},],
            ["physics model",   self.phys_model,  0, {}, ],
            ["Non-linear solver", None, 1, {
                "values": ["FixedPoint", "Newton"]}],
            ["Max iteration", self.nl_maxiter, 400, {}],
            ["NL rel. tol.", self.nl_reltol, 300, {}],
            ["NL abs. tol.", self.nl_abstol, 300, {}],
            ["NL damping", self.nl_damping, 300, {}],
            [None, [False, value], 27, [{'text': 'Use DWC (nlcheckpoint)'},
                                        {'elp': ret}]],
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
            [self.use_dwc_nl, [self.dwc_name, self.dwc_nl_arg, ]],
            self.nl_verbose,
            self.init_only,
            self.clear_wdir,
            self.assemble_real,
            self.save_parmesh,
            self.use_profiler,
            self.skip_solve,
            self.load_sol,
            self.sol_file,)

    def import_panel1_value(self, v):
        #self.init_setting = str(v[0])
        self.phys_model = str(v[0])
        self.nl_scheme = v[1]
        self.nl_maxiter = v[2]
        self.nl_reltol = v[3]
        self.nl_abstol = v[4]
        self.nl_damping = v[5]
        self.use_dwc_nl = v[6][0]
        self.dwc_name = v[6][1][0]
        self.dwc_nl_arg = v[6][1][1]

        self.nl_verbose = v[7]

        self.init_only = v[8]
        self.clear_wdir = v[9]
        self.assemble_real = v[10]
        self.save_parmesh = v[11]
        self.use_profiler = v[12]
        self.skip_solve = v[13]
        self.load_sol = v[14]
        self.sol_file = v[15]

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
        from petram.engine import max_matrix_num
        weight = [0]*max_matrix_num

        if timestep_config[0]:
            weight[0] = 1
            if self.nl_scheme == 'Newton':
                weight[max_matrix_num//2] = 1
        return weight

    @debug.use_profiler
    def run(self, engine, is_first=True, return_instance=False):
        dprint1("Entering run (is_first= ", is_first, ") ", self.fullpath())

        instance = self.allocate_solver_instance(engine)
        instance.set_verbose(self.nl_verbose)

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
            instance.reset_count(self.nl_maxiter,
                                 self.nl_abstol,
                                 self.nl_reltol)
            instance.set_damping(self.nl_damping)
            dprint1("Starting non-linear iteration")
            while not instance.done:
                dprint1("="*72)
                dprint1("NL iteration step=", instance.kiter)
                if is_first:
                    instance.assemble()
                    is_first = False
                else:
                    instance.assemble(update=True)

                update_operator = engine.check_block_matrix_changed(
                    instance.blk_mask)
                instance.solve(update_operator=update_operator)

                if not instance.done:
                    # we do this only if we are going into the next loop
                    # since the same is done in save_solution
                    instance.recover_solution(ksol=0)

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
        self._verbose = False
        self.debug_data = []

    @property
    def blocks(self):
        return self.engine.assembled_blocks

    @property
    def kiter(self):
        return self._kiter

    @property
    def done(self):
        return self._done

    @property
    def verbose(self):
        return self._verbose

    def set_verbose(self, verbose):
        self._verbose = verbose

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
        self.debug_data = []

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
        #linearsolver.SetOperator(AA, dist = engine.is_matrix_distributed)
        #solall = linearsolver.Mult(BB, case_base=0)

        if not self.phys_real and self.gui.assemble_real:
            solall = self.linearsolver_model.real_to_complex(solall, AA)

        if solall is not None:
            sol_norm = np.sum(solall*np.conj(solall))
        from petram.mfem_config import use_parallel
        if use_parallel:
            from mpi4py import MPI
            if MPI.COMM_WORLD.rank == 0:
                sol_norm = MPI.COMM_WORLD.bcast(sol_norm, root=0)
            else:
                sol_norm = MPI.COMM_WORLD.bcast(None, root=0)
            sol_norm = np.sum(sol_norm)
        self.sol_norm = np.sqrt(sol_norm)

        A.reformat_central_mat(
            solall, 0, X[0], mask, alpha=self._alpha, beta=self._beta)
        self.sol = X[0]

        # store probe signal (use t=0.0 in std_solver)
        for p in self.probe:
            p.append_sol(X[0])

        self._kiter = self._kiter + 1

    def call_dwc_nliteration(self):
        if self.gui.use_dwc_nl:
            converged = self.engine.call_dwc(self.gui.get_phys_range(),
                                             method="nliteration",
                                             callername=self.gui.name(),
                                             dwcname=self.gui.dwc_name,
                                             args=self.gui.dwc_nl_arg,
                                             count=self.kiter-1,)
            if converged:
                self._done = True
                self._conveged = True

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
        self.norm0 = 0.0
        self.norm1 = 0.0

    def set_damping(self, damping):
        self._alpha = damping
        self._beta = 1.0

    def compute_A(self, M, B, X, mask_M, mask_B):
        '''
        return A and isAnew
        '''
        from petram.engine import max_matrix_num

        if self.kiter == 0:
            A = M[0]
        else:
            A = M[0] + M[max_matrix_num//2]
        return A, np.any(mask_M[0]) or np.any(mask_M[max_matrix_num//2])

    def compute_rhs(self, M, B, X):
        '''
        RHS = Ax - b
        '''
        #RHS = M[0].dot(self.engine.sol) - B
        RHS = B - M[0].dot(X[0])
        return RHS

    def assemble(self, inplace=True, update=False):
        from petram.engine import max_matrix_num

        if self.kiter == 0:
            self.engine.deactivate_matrix(max_matrix_num//2)
        else:
            self.engine.activate_matrix(max_matrix_num//2)

        NonlinearBaseSolver.assemble(self, inplace=inplace, update=update)

    def solve(self, update_operator=True):
        A, X, RHS, Ae, B, M, depvars = self.blocks

        if self.kiter == 0:
            self.norm0 = X[0].norm()

        elif self.kiter == 1:
            self.norm1 = X[0].norm()

        if self.verbose:
            dprint1("Linear solve...step=", self.kiter)

        self.do_solve(update_operator=update_operator)
        self.engine.add_FESvariable_to_NS(self.get_phys())

        if self.verbose:
            dprint1("|X0|, |X1| and [dX| = ",
                    self.norm0, self.norm1, self.sol_norm)

        if self.kiter == 1:
            self.correction0 = self.sol_norm * abs(self._alpha)
            self.debug_data.append(self.correction0)
        else:
            correction = self.sol_norm * abs(self._alpha)
            self.debug_data.append(correction)
            if correction < self.correction0*self._reltol:
                self._converged = True
                self._done = True

        if self._kiter >= self._maxiter:
            self._done = True

        self.call_dwc_nliteration()

        if self._done:
            if self._converged:
                dprint1("converged (newton) #iter=", self.kiter)
            else:
                dprint1("no convergence (newton interation)")

            if self.verbose:
                dprint1("reference norms |X0|, |X1|=", self.norm0, self.norm1)
                dprint1("correction alpha*|dX| = ", self.debug_data)


class FixedPointSolver(NonlinearBaseSolver):
    def __init__(self, gui, engine):
        NonlinearBaseSolver.__init__(self, gui, engine)

    def set_damping(self, damping):
        self._alpha = damping
        self._beta = (1.0 - damping)

    def compute_A(self, M, B, X, mask_M, mask_B):
        return M[0], np.any(mask_M[0])

    def compute_rhs(self, M, B, X):
        return B

    def solve(self, update_operator=True):
        A, X, RHS, Ae, B, M, depvars = self.blocks

        if self.kiter == 0:
            self.norm0 = np.abs(X[0].norm())

        xdata = self.copy_x(X[0])

        if self.verbose:
            dprint1("Linear solve...step=", self.kiter)

        self.do_solve(update_operator=update_operator)
        self.engine.add_FESvariable_to_NS(self.get_phys())

        diffnorm = self.diff_norm(X[0], xdata)
        norm = np.abs(X[0].norm())
        self.debug_data.append((norm, diffnorm))

        if self.kiter == 1:
            self.correction0 = diffnorm
            if self.verbose:
                dprint1("reference correction", self.correction0)

        else:
            if diffnorm < norm*self._reltol:
                self._converged = True
                self._done = True

        if self._kiter >= self._maxiter:
            self._done = True

        self.call_dwc_nliteration()

        if self._done:
            if self._converged:
                dprint1("converged (fixed-point) #iter=", self.kiter)
            else:
                dprint1("no convergence (fixed-point interation)")

            if self.verbose:
                dprint1("reference correction", self.correction0)
                dprint1("norms", [x[0] for x in self.debug_data])
                dprint1("dnorms", [x[1] for x in self.debug_data])

    def copy_x(self, X):
        shape = X.shape

        xdata = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                v = X[i, j]
                if isinstance(v, chypre.CHypreVec):
                    vec = v.toarray()
                elif isinstance(v, bm.ScipyCoo):
                    vec = v.toarray().flatten()
                else:
                    assert False, "not supported"
                xdata.append(vec.copy())
        from mfem.common.mpi_debug import nicePrint

        return xdata

    def diff_norm(self, X, xdata):
        shape = X.shape
        idx = 0
        norm = 0

        for i in range(shape[0]):
            for j in range(shape[1]):
                v = X[i, j]
                if isinstance(v, chypre.CHypreVec):
                    vec = v.toarray()
                elif isinstance(v, bm.ScipyCoo):
                    vec = v.toarray().flatten()
                else:
                    assert False, "not supported"

                delta = xdata[idx] - vec
                norm += np.sum(delta * np.conj(delta))
                idx = idx+1

        from petram.mfem_config import use_parallel
        if use_parallel:
            from mpi4py import MPI
            norm = np.sum(np.array(MPI.COMM_WORLD.allgather(norm)))
        return np.abs(np.sqrt(norm))
