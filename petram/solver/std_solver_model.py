from petram.solver.solver_model import SolverInstance
import os
import numpy as np

from petram.model import Model
from petram.solver.solver_model import Solver
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('StdSolver')
rprint = debug.regular_print('StdSolver')


class StdSolver(Solver):
    can_delete = True
    has_2nd_panel = False

    @classmethod
    def fancy_menu_name(self):
        return 'Stationary'

    @classmethod
    def fancy_tree_name(self):
        return 'Stationary'

    def verify_setting(self):
        try:
            phys = self.get_phys()
        except KeyError:
            return False, "", "Physics does not exist (KeyError)"
        if len(phys) == 0:
            return False, "", "No Physics is chosen to solve"

        return True, "", ""

    def attribute_set(self, v):
        super(StdSolver, self).attribute_set(v)
        return v

    def panel1_param(self):
        return [  # ["Initial value setting",   self.init_setting,  0, {},],
            ["physics model",   self.phys_model,  0, {}, ],
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
            self.init_only,
            self.clear_wdir,
            self.assemble_real,
            self.save_parmesh,
            self.use_profiler,
            self.skip_solve,
            self.load_sol,
            self.sol_file)

    def import_panel1_value(self, v):
        # self.init_setting = str(v[0])
        self.phys_model = str(v[0])
        self.init_only = v[1]
        self.clear_wdir = v[2]
        self.assemble_real = v[3]
        self.save_parmesh = v[4]
        self.use_profiler = v[5]
        self.skip_solve = v[6]
        self.load_sol = v[7]
        self.sol_file = v[8]

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

        instance = StandardSolver(self, engine)
        return instance

    def get_matrix_weight(self, timestep_config):  # , timestep_weight):
        if timestep_config[0]:
            return [1, 0, 0]
        else:
            return [0, 0, 0]

    @debug.use_profiler
    def run(self, engine, is_first=True, return_instance=False):
        dprint1("Entering run (is_first= ", is_first, ") ", self.fullpath())
        if self.clear_wdir:
            engine.remove_solfiles()

        instance = StandardSolver(
            self, engine) if self.instance is None else self.instance

        instance.set_blk_mask()
        if return_instance:
            return instance

        instance.configure_probes(self.probe)

        if self.init_only:
            engine.sol = engine.assembled_blocks[1][0]
            instance.sol = engine.sol

        elif self.load_sol:
            if is_first:
                instance.assemble()
                is_first = False
            instance.load_sol(self.sol_file)
        else:
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


class StandardSolver(SolverInstance):
    def __init__(self, gui, engine):
        SolverInstance.__init__(self, gui, engine)
        self.assembled = False
        self.linearsolver = None
        self._operator_set = False

    @property
    def blocks(self):
        return self.engine.assembled_blocks

    def compute_A(self, M, B, X, mask_M, mask_B):
        '''
        M[0] x = B

        return A and isAnew
        '''
        return M[0], np.any(mask_M[0])

    def compute_rhs(self, M, B, X):
        '''
        M[0] x = B
        '''
        return B

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
        # A, X, RHS, Ae, B, M, names = blocks
        self.assembled = True
        return M_changed

    def assemble_rhs(self):
        engine = self.engine
        phys_target = self.get_phys()
        engine.run_assemble_b(phys_target)
        B = self.engine.run_update_B_blocks()
        self.blocks[4] = B
        self.assembled = True

    def solve(self, update_operator=True):
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

        from petram.mfem_config import use_parallel
        if use_parallel:
            from mpi4py import MPI
        else:
            from petram.helper.dummy_mpi import MPI
        myid = MPI.COMM_WORLD.rank
        size = MPI.COMM_WORLD.size

        ss = str(solall.shape) if solall is not None else "x"
        ss = MPI.COMM_WORLD.gather(ss)
        if ss is not None:
            dprint1("solshape", ', '.join(ss))

        # linearsolver.SetOperator(AA, dist = engine.is_matrix_distributed)
        # solall = linearsolver.Mult(BB, case_base=0)

        is_sol_central = any(MPI.COMM_WORLD.allgather(solall is None))
        if is_sol_central:
            if not self.phys_real and self.gui.assemble_real:
                solall = self.linearsolver_model.real_to_complex(solall, AA)
            A.reformat_central_mat(solall, 0, X[0], mask)
        else:
            if not self.phys_real and self.gui.assemble_real:
                assert False, "this operation is not permitted"
            A.reformat_distributed_mat(solall, 0, X[0], mask)

        self.sol = X[0]

        # store probe signal (use t=0.0 in std_solver)
        for p in self.probe:
            p.append_sol(X[0])

        return True

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
