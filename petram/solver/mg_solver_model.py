from petram.solver.solver_model import SolverInstance
import os
import numpy as np

from petram.model import Model
from petram.solver.solver_model import Solver
from petram.solver.std_solver_model import StdSolver
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('StdSolver')
rprint = debug.regular_print('StdSolver')


class MGSolver(StdSolver):
    def attribute_set(self, v):
        super(MGSolver, self).attribute_set(v)
        v["refinement_levels"] = "2"
        v["refinement_type"] = "P(order)"
        v["fine_solver"] = ""
        v["coarse_solver"] = ""
        return v

    def panel1_param(self):
        panels = super(MGSolver, self).panel1_param()
        panels.extend([["refinement type", self.refinement_type, 1,
                        {"values": ["P(order)", "H(mesh)"]}],
                       ["number of levels", "", 0, {}],
                       ["fine solver", "", 0, {}],
                       ["coasest solver", "", 0, {}]])
        return panels

    def get_panel1_value(self):
        value = list(super(MGSolver, self).get_panel1_value())
        value.append(self.refinement_type)
        value.append(self.refinement_levels)
        value.append(self.fine_solver)
        value.append(self.coarse_solver)
        return value

    def import_panel1_value(self, v):
        super(MGSolver, self).import_panel1_value(v[:-4])
        self.refinement_type = v[-4]
        self.refinement_levels = v[-3]
        self.fine_solver = v[-2]
        self.coarse_solver = v[-1]

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

    def create_multilevel_setting(self, engine):
        target_phys = self.get_target_phys()
        for phys in target_phys:
            dprint1("Adding refined level for " + phys.name())        
            engine.add_refined_fespace(phys, 'P', inc=1)


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
        choice = []
        try:
            from petram.solver.iterative_model import Iterative
            choice.append(Iterative)
        except ImportError:
            pass
        return choice

    @debug.use_profiler
    def run(self, engine, is_first=True, return_instance=False):
        dprint1("Entering run (is_first=", is_first, ")", self.fullpath())
        if self.clear_wdir:
            engine.remove_solfiles()

        instance = StandardSolver(self, engine)
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
            self.create_multilevel_setting(engine)
            instance.solve()

        instance.save_solution(ksol=0,
                               skip_mesh=False,
                               mesh_only=False,
                               save_parmesh=self.save_parmesh)
        engine.sol = instance.sol

        instance.save_probe()

        dprint1(debug.format_memory_usage())
        return is_first


class StandardSolver(SolverInstance):
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

    def assemble(self, inplace=True):
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
        self.assembled = True

    def assemble_rhs(self):
        engine = self.engine
        phys_target = self.get_phys()
        engine.run_assemble_b(phys_target)
        B = self.engine.run_update_B_blocks()
        self.blocks[4] = B
        self.assembled = True

    def solve(self, update_operator=True):
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

        if update_operator:
            linearsolver.SetOperator(AA,
                                     dist=engine.is_matrix_distributed,
                                     name=depvars)

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

        A.reformat_central_mat(solall, 0, X[0], mask)
        self.sol = X[0]

        # store probe signal (use t=0.0 in std_solver)
        for p in self.probe:
            p.append_sol(X[0])

        return True
