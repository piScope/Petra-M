'''
 
  non-linear stationary solver

'''
import mfem.common.chypre as chypre
import petram.helper.block_matrix as bm
from petram.solver.solver_model import SolverInstance
import os
import numpy as np
from scipy.sparse import coo_matrix

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
        v["nl_damping_min"] = 0.01
        v["nl_damping_fixed"] = False
        v["nl_tolthr"] = 0.01
        v["nl_stopcond"] = "error"
        v["nl_stopresidual"] = "all"
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
            ["NL inital damping", self.nl_damping, 300, {}],
            ["NL min damping", self.nl_damping_min, 300, {}],
            [None, not self.nl_damping_fixed,  3,
                {"text": "auto adjust damping"}],
            [None, [False, value], 27, [{'text': 'Use DWC (nl_start/nl_checkpoint/nl_end)'},
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
            self.nl_damping,
            self.nl_damping_min,
            not self.nl_damping_fixed,
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
        self.nl_damping = v[4]
        self.nl_damping_min = v[5]
        self.nl_damping_fixed = not v[6]
        self.use_dwc_nl = v[7][0]
        self.dwc_name = v[7][1][0]
        self.dwc_nl_arg = v[7][1][1]

        self.nl_verbose = v[8]

        self.init_only = v[9]
        self.clear_wdir = v[10]
        self.assemble_real = v[11]
        self.save_parmesh = v[12]
        self.use_profiler = v[13]
        self.skip_solve = v[14]
        self.load_sol = v[15]
        self.sol_file = v[16]

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
        instance.set_tol(self.nl_reltol, self.nl_tolthr)
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
            instance.reset_count(self.nl_maxiter)
            instance.set_damping(self.nl_damping,
                                 minimum=self.nl_damping_min,
                                 fixed=self.nl_damping_fixed)
            dprint1("Starting non-linear iteration")

            if self.use_dwc_nl:
                engine.call_dwc(self.get_phys_range(),
                                method="nl_start",
                                callername=self.name(),
                                dwcname=self.dwc_name,
                                args=self.dwc_nl_arg)

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

            if self.use_dwc_nl:
                engine.call_dwc(self.get_phys_range(),
                                method="nl_end",
                                callername=self.name(),
                                dwcname=self.dwc_name,
                                args=self.dwc_nl_arg)

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
        self.debug_data2 = []
        self.damping_record = []
        self.residual_record = []

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

    def set_tol(self, reltol, tolthr):
        self._reltol = reltol
        self._tolthr = tolthr
        dprint1("NL iteration tolelance/threshold: ",
                self._reltol, self._tolthr)

    def set_damping(self, damping, minimum=None, fixed=None):
        assert False, "Must be implemented in child"

    def compute_A(self, M, B, X, mask_M, mask_B):
        assert False, "Must be implemented in subclass"

    def compute_rhs(self, M, B, X):
        assert False, "Must be implemented in subclass"

    def assemble_rhs(self):
        assert False, "assemble_rhs should not be called"

    def reset_count(self, maxiter):
        self._kiter = 0
        self._maxiter = maxiter
        self._current_error = (np.infty, np.infty)
        self._done = False
        self._converged = False
        self.debug_data = []
        self.error_record = []
        self.damping_record = []

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

        #A, X, RHS, Ae, B, M, names = _blocks

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
            opr = self.adjust_operator(A, M, mask)   # add Jacobian for Newtown
            AA = engine.finalize_matrix(opr, mask, not self.phys_real,
                                        format=self.ls_type)

        BB = engine.finalize_rhs([RHS], A, X[0], mask, not self.phys_real,
                                 format=self.ls_type,
                                 use_residual=True)

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

        return solall

    def update_x(self, solall):
        A, X, RHS, Ae, B, M, depvars = self.blocks
        mask = self.blk_mask

        A.reformat_central_mat(
            solall, 0, X[0], mask, alpha=self._alpha, beta=self._beta)
        self.sol = X[0]

        # store probe signal (use t=0.0 in std_solver)
        for p in self.probe:
            p.append_sol(X[0])

        self._kiter = self._kiter + 1

    def do_error_estimate(self):
        '''
        call do_solve without updating operator
        '''
        damping = self.damping
        self._alpha = 1.0
        self._beta = 0.0

        tmp = self.do_solve(update_operator=False)
        self.update_x(tmp)

        self.set_damping(damping)
        self._kiter = self._kiter - 1

    def call_dwc_nliteration(self):
        if self.gui.use_dwc_nl:
            stopit = self.engine.call_dwc(self.gui.get_phys_range(),
                                          method="nl_checkpoint",
                                          callername=self.gui.name(),
                                          dwcname=self.gui.dwc_name,
                                          args=self.gui.dwc_nl_arg,
                                          count=self.kiter-1,)
            return stopit
        else:
            return False

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

    def copy_x(self, X):
        '''
        extract data from X (BlockMatrix)
        '''
        shape = X.shape

        xdata = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                v = X[i, j]
                if isinstance(v, chypre.CHypreVec):
                    vec = v.toarray()
                elif isinstance(v, bm.ScipyCoo):
                    vec = v.toarray()
                else:
                    assert False, "not supported"
                xdata.append(vec.copy())
        return xdata

    def copyback_x(self, X, xdata):
        '''
        put data back to X (BlockMatrix)
        '''
        shape = X.shape

        for i in range(shape[0]):
            for j in range(shape[1]):
                v = X[i, j]
                if isinstance(v, chypre.CHypreVec):
                    if v.isComplex():
                        v.real.Assign(xdata[i].real)
                        v.imag.Assign(xdata[i].imag)
                    else:
                        v.real.Assign(xdata[i])
                elif isinstance(v, bm.ScipyCoo):
                    coo = coo_matrix(xdata[i])
                    v.data = coo.data
                    v.row = coo.row
                    v.col = coo.col
                else:
                    assert False, "not supported"

        return xdata


class NewtonSolver(NonlinearBaseSolver):
    def __init__(self, gui, engine):
        NonlinearBaseSolver.__init__(self, gui, engine)
        self.scheme_name = "newton"
        self.minimum_damping = 0.05
        self._err_before = 100.
        self._err_guidance = 100.
        self._err_inc_count = 0
        self._fixed_damping = False
        self._stall_counter = 0
        self.max_stall = 10

        self._dwidth = 0.15
        self.dwidth1 = 1 + self._dwidth
        self.dwidth2 = 1 - self._dwidth

    def reset_count(self, maxiter):
        NonlinearBaseSolver.reset_count(self, maxiter)
        self._err_before = 100.

    @property
    def damping(self):
        return self._alpha

    def set_damping(self, damping, minimum=None, fixed=None):
        self._alpha = min(damping, 1.0)
        self._beta = 1.0
        if minimum is not None:
            self.minimum_damping = minimum
        if fixed is not None:
            self._fixed_damping = fixed

    def reset_count(self, maxiter):
        NonlinearBaseSolver.reset_count(self, maxiter)

    def compute_A(self, M, B, X, mask_M, mask_B):
        '''
        return A and isAnew
        '''
        A = M[0]
        return A, np.any(mask_M[0])

    def compute_rhs(self, M, B, X):
        '''
        M[0] x = B
        '''
        return B

    def compute_err(self, sol, sol_ave_norm, Res):
        '''
        sol is list of dense array
        Res is BlockMatrix
        '''
        shape = Res.shape
        assert shape[1] == 1, "multilpe vectors are not supported"

        err = np.zeros(shape[0])
        length = np.zeros(shape[0])

        for i in range(shape[0]):
            for j in [0]:
                res = Res[i]
                x_vec = sol[i]
                if isinstance(res, chypre.CHypreVec):
                    res_vec = res.toarray()
                elif isinstance(res, bm.ScipyCoo):
                    res_vec = res.toarray()
                else:
                    assert False, "not supported"
                w = np.abs(x_vec)
                if sol_ave_norm[i] != 0:
                    thr = sol_ave_norm[i]*self._tolthr
                else:
                    thr = np.mean(sol_ave_norm)*self._tolthr
                w[w < thr] = thr
                err_est = np.abs(res_vec)

                err[i] = np.sum((err_est/w)**2)
                length[i] = np.prod(x_vec.shape)

        from petram.mfem_config import use_parallel
        if use_parallel:
            from mpi4py import MPI
            allgather = MPI.COMM_WORLD.allgather
            length = np.sum(allgather(length), 0)
            err = np.sum(allgather(err), 0)

        err = err/length
        self.debug_data.append(err)

        err_total = np.sqrt(np.sum(err))/np.sqrt(shape[0])

        return err_total

    def compute_residual(self, RHS):
        shape = RHS.shape
        w = np.array([0]*shape[0])
        w[0] = 1

        if self.kiter == 0:
            self._res_sq0 = RHS.average_norm(sq=True)
            return self._res_sq0
        if self.kiter == 1:
            self._res_sq1 = RHS.average_norm(sq=True)
            self._res_w = (self._res_sq0 + self._res_sq1)/2.0
            self._res_w = [w if w != 0 else np.mean(self._res_w)
                           for w in self._res_w]
            return self._res_sq1
        if self.kiter > 1:
            nsq = RHS.normsq()
            dprint1(np.sqrt(np.sum(w*nsq/self._res_w))/np.sqrt(shape[0]),
                    nsq)
            return np.sqrt(np.sum(w*nsq/self._res_w))/np.sqrt(shape[0])

    def assemble(self, inplace=True, update=False):
        from petram.engine import max_matrix_num

        if self.kiter == 0:
            self.engine.deactivate_matrix(max_matrix_num//2)
        else:
            self.engine.activate_matrix(max_matrix_num//2)

        M_changed = NonlinearBaseSolver.assemble(
            self, inplace=inplace, update=update)

        if self.kiter > 0:
            A, X, RHS, Ae, B, M, depvars = self.blocks
            self.engine.eliminateJac(M[max_matrix_num//2])

    def solve(self, update_operator=True):
        A, X, RHS, Ae, B, M, depvars = self.blocks

        if self.verbose:
            dprint1("Linear solve...step=", self.kiter)

        residual = self.compute_residual(B - M[0].dot(X[0]))
        if self.kiter == 2:
            self._resitual0 = residual
        if self.verbose:
            dprint1("Current residual..", residual)

        if self.kiter > 0:

            sol_ave_norm = X[0].average_norm()
            soldata = self.copy_x(X[0])

            self.do_error_estimate()
            err = self.compute_err(soldata, sol_ave_norm, X[0])
            self.copyback_x(X[0], soldata)

            if self.verbose:
                dprint1("estimated error, err_before, err_guidance, err_inc_count,  damping)",
                        err, self._err_before, self._err_guidance, self._err_inc_count, self.damping)
            if err < self._reltol:
                self._converged = True
                self._done = True
            if abs((err - self._err_before)/err) < 1e-3:
                self._stall_counter = self._stall_counter + 1
            else:
                self._stall_counter = 0
            if self._stall_counter > self.max_stall:
                dprint1("no convergence (stall)")
                self._done = True

            if self.kiter == 1:
                self._err_before = err
                self._err_guidance = err
            elif self._fixed_damping:
                pass

            elif (err > self._err_guidance*1.05 or
                  # err > self._err_before*self.dwidth1 or
                  err > self._err_before*1.05 or
                  self._err_inc_count > 2):

                self.set_damping(self.damping*self.dwidth2)
                self._err_guidance = self._err_guidance*self.dwidth1
                # this makes reduction of damping milder...not sure if this is better....
                # self._err_guidance = self._err_before

                self.copyback_x(X[0], self._solbackup)
                self.update_x(self._delta)
                self._kiter = self._kiter - 1
                self._err_inc_count = 0

                if self.damping < self.minimum_damping:
                    # Let's give up ...(sad face)
                    self._done = True
                else:
                    dprint1("new damping (reduced), ref_error, current_error",
                            self.damping, self._err_before, err)

                    self._err_before = err
                    if self.scheme_name != "fixed-point":
                        return

            elif err < self._err_guidance*self.dwidth2 and self.damping < 1.0:
                self._err_guidance = err
                self._err_before = err
                self.set_damping(self.damping*self.dwidth1)
#                self.set_damping(self.damping*1.2)
                dprint1("new damping (increased)", self.damping)

            else:
                if err > self._err_before:
                    self._err_inc_count = self._err_inc_count + 1
                else:
                    self._err_inc_count = 0
                self._err_before = err
                self._err_guidance = self._err_guidance*1.02

            self.error_record.append(err)

            if self._kiter >= self._maxiter:
                self._done = True

        stopit = self.call_dwc_nliteration()
        if stopit:
            self._done = True

        self.residual_record.append(residual)

        if self.kiter > 5 and residual < 3e-6:
            self._done = True

        if not self._converged and not self._done:
            self.damping_record.append(self.damping)

            self._solbackup = self.copy_x(X[0])
            self._delta = self.do_solve(update_operator=update_operator)
            self.update_x(self._delta)
            self.engine.add_FESvariable_to_NS(self.get_phys())

            if self._kiter >= self._maxiter:
                self._done = True

        if self._done and not stopit:
            if self.damping != 1.0 and self.damping > self.minimum_damping:
                self.set_damping(1.0)
                self._done = False
                self._converged = False
                self._fixed_damping = True

        if self._done:
            if self._converged:
                dprint1("converged ("+self.scheme_name + ") #iter=", self.kiter)
                dprint1("final error |err| = ", err)
                dprint1("damping parameters", self.damping_record)

            else:
                dprint1("no convergence ("+self.scheme_name+" interation)")
                dprint1("damping parameters", self.damping_record)

            if self.verbose:
                dprint1("err history = ", self.error_record)
                dprint1("err^2 history (decomposition) = ", self.debug_data)
                dprint1("residuals", self.residual_record)

    def save_probe(self):
        from petram.mfem_config import use_parallel
        if use_parallel:
            from mpi4py import MPI
        else:
            from petram.helper.dummy_mpi import MPI
        myid = MPI.COMM_WORLD.rank

        if myid != 0:
            return

        from petram.sol.probe import Probe

        p1 = Probe(self.gui.name()+'_error')
        p2 = Probe(self.gui.name()+'_damping')

        for i, v in enumerate(self.error_record):
            p1.append_value(v, t=i)
        for i, v in enumerate(self.damping_record):
            p2.append_value(v, t=i)

        p1.write_file()
        p2.write_file()

        NonlinearBaseSolver.save_probe(self)

    def adjust_operator(self, A, M, mask):
        '''
        Add Jacobian term to operator
        '''
        from petram.engine import max_matrix_num

        M = (M[max_matrix_num//2]).get_subblock(mask[0], mask[1])

        if self.kiter == 0:
            pass
        else:
            A = A + M
        return A


class FixedPointSolver(NewtonSolver):
    def __init__(self, *args, **kwargs):
        NewtonSolver.__init__(self, *args, **kwargs)
        self.scheme_name = "fixed-point"

    def assemble(self, inplace=True, update=False):
        NonlinearBaseSolver.assemble(self, inplace=inplace, update=update)

    def adjust_operator(self, A, M, mask):
        return A
