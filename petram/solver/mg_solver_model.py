import os
import numpy as np

from petram.solver.iterative_model import (Iterative,
                                           IterativeSolver)
from petram.solver.solver_model import SolverInstance

from petram.model import Model
from petram.solver.solver_model import Solver, SolverInstance
from petram.solver.std_solver_model import StdSolver

from petram.mfem_config import use_parallel
if use_parallel:
    from petram.helper.mpi_recipes import *
    import mfem.par as mfem
else:
    import mfem.ser as mfem

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
        return linearsolver, BB, XX, AA

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

        ls0, _BB, _XX, _AA = self.finalize_linear_system(0)     # coarse
        ls1, BB, XX, AA = self.finalize_linear_system(1)     # fine

        print(ls0.solver)
        print(ls1.solver)

        P_matrix = engine.fill_prolongation_operator(0, ls1.A)
        prolongations = [P_matrix]
        smoothers = [ls0.solver, engine.genearate_smoother(0, ls1.A)]
        operators = [ls0.A, ls1.A]

        print(ls1.A)

        #solall = linearsolver.Mult(BB, case_base=0)
        mg = MG(operators, smoothers, prolongations)

        # very small value
        # ls1.solver.SetPreconditioner(mg.solver)
        #solall = ls1.Mult(BB, XX)

        # transfer looks okay
        #solall0 = ls0.Mult(_BB, _XX)
        #P_matrix.Mult(_XX, XX)
        #print("here", type(_XX), _XX.GetDataArray().shape, XX.GetDataArray().shape)
        solall = np.transpose(np.vstack([XX.GetDataArray()]))
        #

        # mg alone seems okay. but smoother destor
        #mg.solver.Mult(BB[0], XX)
        #solall = np.transpose(np.vstack([XX.GetDataArray()]))

        class MyPreconditioner(mfem.Solver):
            def __init__(self):
                mfem.Solver.__init__(self)

            def Mult(self, x, y):
                np.save('original_b', _BB[0].GetDataArray())
                P_matrix.MultTranspose(x, _BB[0])
                np.save('restricted_b', _BB[0].GetDataArray())
                ls0.Mult(_BB, _XX)
                P_matrix.Mult(_XX, y)
                #print(x, y)
                #assert False, "faile for now"

            def SetOperator(self, opr):
                pass

        prc = MyPreconditioner()
        # write solver here...
        solver = mfem.FGMRESSolver()
        solver.SetRelTol(1e-12)
        solver.SetMaxIter(1)
        solver.SetPrintLevel(1)
        solver.SetOperator(ls1.A)
        solver.SetPreconditioner(prc)
        solver.Mult(BB[0], XX)
        solall = np.transpose(np.vstack([XX.GetDataArray()]))

        '''
        solall = linearsolver.Mult(BB, x=XX, case_base=0)

        #linearsolver.SetOperator(AA, dist = engine.is_matrix_distributed)
        #solall = linearsolver.Mult(BB, case_base=0)
        '''
        if not self.phys_real and self.gui.assemble_real:
            solall = self.linearsolver_models[-1].real_to_complex(solall, AA)

        engine.level_idx = 1
        A = engine.assembled_blocks[0]
        X = engine.assembled_blocks[1]
        A.reformat_central_mat(solall, 0, X[0], self.blk_mask)
        print(X[0])
        self.sol = X[0]

        # store probe signal (use t=0.0 in std_solver)
        for p in self.probe:
            p.append_sol(X[0])

        return True


class MG(IterativeSolver):   # LinearSolver
    def __init__(self, operators, smoothers, prolongations):

        own_operators = [False]*len(operators)
        own_smoothers = [False]*len(smoothers)
        own_prolongations = [False]*len(prolongations)

        mg = mfem.Multigrid(operators, smoothers, prolongations,
                            own_operators, own_smoothers, own_prolongations)
        mg.SetCycleType(mfem.Multigrid.CycleType_VCYCLE, 0, 0)
        self.solver = mg
        #self.A = operators[-1]
