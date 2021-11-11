from petram.solver.iterative_model import Iterative
from petram.solver.solver_model import SolverInstance
import os
import numpy as np

from petram.model import Model
from petram.solver.solver_model import Solver, SolverInstance
from petram.solver.std_solver_model import StdSolver
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('MGSolver')
rprint = debug.regular_print('MGSolver')


class CoarseSolver:
    pass


class FineSolver:
    pass


class CoarseIterative(Iterative, CoarseSolver):
    @classmethod
    def fancy_menu_name(self):
        return 'Iterative'

    @classmethod
    def fancy_tree_name(self):
        return 'Iterative'

    def get_info_str(self):
        return 'Coarse'


class FineIterative(Iterative, FineSolver):
    @classmethod
    def fancy_menu_name(self):
        return 'Iterative'

    @classmethod
    def fancy_tree_name(self):
        return 'Iterative'

    def get_info_str(self):
        return 'Fine'


class MGSolver(StdSolver):
    def attribute_set(self, v):
        super(MGSolver, self).attribute_set(v)
        v["refinement_levels"] = "2"
        v["refinement_type"] = "P(order)"
        return v

    def panel1_param(self):
        panels = super(MGSolver, self).panel1_param()
        panels.extend([["refinement type", self.refinement_type, 1,
                        {"values": ["P(order)", "H(mesh)"]}],
                       ["number of levels", "", 0, {}], ])
        return panels

    def get_panel1_value(self):
        value = list(super(MGSolver, self).get_panel1_value())
        value.append(self.refinement_type)
        value.append(self.refinement_levels)
        return value

    def import_panel1_value(self, v):
        super(MGSolver, self).import_panel1_value(v[:-2])
        self.refinement_type = v[-2]
        self.refinement_levels = v[-1]

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

    def get_num_levels(self):
        return int(self.refinement_levels)

    def set_model_level(self, klevel):
        '''
        change physcis model setting to assemble operator at
        differnet level
        '''
        return None

    def reset_model_level(self):
        '''
        revert physcis model setting to oritinal
        '''
        return None

    def get_possible_child(self):
        return FineIterative, CoarseIterative

    def get_possible_child_menu(self):
        choice = [("Coarse Lv. Solver", CoarseIterative),
                  ("!", None),
                  ("Fine Lv. Solver", FineIterative),
                  ("!", None)]
        return choice

    def create_refined_levels(self, engine, lvl):
        '''
        lvl : refined level number (1, 2, 3, ....)
              1 means "AFTER" 1 refinement
        '''
        if lvl >= int(self.refinement_levels):
            return False

        target_phys = self.get_target_phys()
        for phys in target_phys:
            dprint1("Adding refined level for " + phys.name())
            engine.prepare_refined_level(phys, 'P', inc=1)

        engine.level_idx = lvl
        for phys in target_phys:
            engine.get_true_v_sizes(phys)

        return True

    @debug.use_profiler
    def run(self, engine, is_first=True, return_instance=False):
        dprint1("Entering run (is_first=", is_first, ")", self.fullpath())
        if self.clear_wdir:
            engine.remove_solfiles()

        instance = MGInstance(self, engine)
        instance.set_blk_mask()
        if return_instance:
            return instance

        instance.configure_probes(self.probe)

        if self.init_only:
            engine.sol = engine.assembled_blocks[1][0]
            instance.sol = engine.sol
        else:
            if is_first:
                instance.assemble()
                is_first = False
            instance.solve()

        instance.save_solution(ksol=0,
                               skip_mesh=False,
                               mesh_only=False,
                               save_parmesh=self.save_parmesh)
        engine.sol = instance.sol

        instance.save_probe()

        dprint1(debug.format_memory_usage())
        return is_first


class MGInstance(SolverInstance):
    def __init__(self, gui, engine):
        SolverInstance.__init__(self, gui, engine)
        self.assembled = False
        self.linearsolver = None

    @property
    def blocks(self):
        return self.engine.assembled_blocks

    def compute_A(self, M, B, X, mask_M, mask_B):
        '''
        M[0] x = B

        return A and isAnew
        '''
        return M[0], True

    def compute_rhs(self, M, B, X):
        '''
        M[0] x = B
        '''
        return B

    def do_assemble(self, inplace=True):
        engine = self.engine
        phys_target = self.get_phys()
        phys_range = self.get_phys_range()

        # use get_phys to apply essential to all phys in solvestep
        dprint1("Asembling system matrix",
                [x.name() for x in phys_target],
                [x.name() for x in phys_range])
        engine.run_verify_setting(phys_target, self.gui)
        engine.run_assemble_mat(phys_target, phys_range)
        engine.run_assemble_b(phys_target)
        engine.run_fill_X_block()

        self.engine.run_assemble_blocks(self.compute_A,
                                        self.compute_rhs,
                                        inplace=inplace)
        #A, X, RHS, Ae, B, M, names = blocks

    def assemble(self, inplace=True):
        engine = self.engine

        engine.level_idx = 0
        self.do_assemble(inplace)

        engine.level_idx = 1
        self.do_assemble(inplace)

        self.assembled = True

    def set_linearsolver_model(self):
        fine_solver = self.gui.get_active_solver(cls=FineSolver)
        if fine_solver is None:
            assert False, "Fine level solver is not chosen"
        coarse_solver = self.gui.get_active_solver(cls=CoarseSolver)
        if coarse_solver is None:
            assert False, "Coarse level solver is not chosen"

        phys_target = self.get_phys()

        self.linearsolver_models = [coarse_solver, fine_solver]
        self.phys_real = all([not p.is_complex() for p in phys_target])
        ls_type1 = coarse_solver.linear_system_type(self.gui.assemble_real,
                                                    self.phys_real)
        ls_type2 = fine_solver.linear_system_type(self.gui.assemble_real,
                                                  self.phys_real)
        if ls_type1 != ls_type2:
            assert False, "Fine/Coarse solvers must assmelbe the same linear system type"
        self.ls_type = ls_type1

    def finalize_linear_system(self, level):
        engine = self.engine

        engine.level_idx = level
        solver_model = self.linearsolver_models[level]
        # if not self.assembled:
        #    assert False, "assmeble must have been called"

        A, X, RHS, Ae, B, M, depvars = self.blocks
        mask = self.blk_mask
        engine.copy_block_mask(mask)

        depvars = [x for i, x in enumerate(depvars) if mask[0][i]]

        AA = engine.finalize_matrix(A, mask, not self.phys_real,
                                    format=self.ls_type)
        BB = engine.finalize_rhs([RHS], A, X[0], mask, not self.phys_real,
                                 format=self.ls_type)

        linearsolver = self.allocate_linearsolver(self.gui.is_complex(),
                                                  self.engine,
                                                  solver_model=solver_model)

        linearsolver.SetOperator(AA,
                                 dist=engine.is_matrix_distributed,
                                 name=depvars)

        if linearsolver.is_iterative:
            XX = engine.finalize_x(X[0], RHS, mask, not self.phys_real,
                                   format=self.ls_type)
        else:
            XX = None
        return linearsolver

    def assemble_rhs(self):
        assert False, "assemble_rhs is not implemented"
        '''
        engine = self.engine
        phys_target = self.get_phys()
        engine.run_assemble_b(phys_target)
        B = self.engine.run_update_B_blocks()
        self.blocks[4] = B
        self.assembled = True
        '''

    def solve(self, update_operator=True):
        engine = self.engine

        solver0 = self.finalize_linear_system(0)
        solver1 = self.finalize_linear_system(1)
        '''
        solall = linearsolver.Mult(BB, x=XX, case_base=0)

        #linearsolver.SetOperator(AA, dist = engine.is_matrix_distributed)
        #solall = linearsolver.Mult(BB, case_base=0)

        if not self.phys_real and self.gui.assemble_real:
            solall = self.linearsolver_model.real_to_complex(solall, AA)

        A.reformat_central_mat(solall, 0, X[0], mask)
        self.sol = X[0]

        # store probe signal (use t=0.0 in std_solver)
        for p in self.probe:
            p.append_sol(X[0])
        '''
        return True
