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


class Smoother(Iterative, FineSolver):
    @classmethod
    def fancy_menu_name(self):
        return 'Smoother'

    @classmethod
    def fancy_tree_name(self):
        return 'Smoother'

    def get_info_str(self):
        return 'Fine'


class MGSolver(StdSolver):
    def attribute_set(self, v):
        super(MGSolver, self).attribute_set(v)
        v["refinement_levels"] = "2"
        v["refinement_type"] = "P(order)"
        v["presmoother_count"] = "1"
        v["postsmoother_count"] = "1"
        return v

    def panel1_param(self):
        panels = super(MGSolver, self).panel1_param()
        panels.extend([["refinement type", self.refinement_type, 1,
                        {"values": ["P(order)", "H(mesh)"]}],
                       ["number of levels", "", 0, {}],
                       ["pre-smoother #count", "", 0, {}],
                       ["post-smoother #count", "", 0, {}], ])
        return panels

    def get_panel1_value(self):
        value = list(super(MGSolver, self).get_panel1_value())
        value.append(self.refinement_type)
        value.append(self.refinement_levels)
        value.append(self.presmoother_count)
        value.append(self.postsmoother_count)
        return value

    def import_panel1_value(self, v):
        super(MGSolver, self).import_panel1_value(v[:-4])
        self.refinement_type = v[-4]
        self.refinement_levels = v[-3]
        self.presmoother_count = v[-2]
        self.postsmoother_count = v[-1]

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
        refine_type = self.refinement_type[0]  # P or H
        for phys in target_phys:
            dprint1("Adding refined level for " + phys.name())
            engine.prepare_refined_level(phys, refine_type, inc=1)

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

        class MyGMRESSolver(mfem.PyGMRESSolver):
            def __init__(self, *args, **kwags):
                mfem.PyGMRESSolver.__init__(self, *args, **kwags)

            def Mult(self, x, y):
                print('----- MyGMRESSolver.Mult')
                mfem.PyGMRESSolver.Mult(self, x, y)

            def MultTranspose(self, x, y):
                print('MyGMRESSolver.MultTranspose')
                mfem.PyGMRESSolver.Mult(self, x, y)
            # def SetOperator(self, x):
            #     mfem.GMRESSolver.SetOperator(self, x)

        solver1 = MyGMRESSolver()
        solver1.SetRelTol(ls0.gui.reltol)
        solver1.SetAbsTol(0.0)
        solver1.SetMaxIter(50)
        solver1.SetPrintLevel(ls0.gui.log_level)
        solver1.SetOperator(ls1.A)

        P_matrix = fill_prolongation_operator(engine, 0, ls1.A)
        prolongations = [P_matrix]
        #smoothers = [ls0.solver, genearate_smoother(engine, 0, ls1.A)]
        smoothers = [ls0.solver, solver1]
        operators = [ls0.A, ls1.A]

        #solall = linearsolver.Mult(BB, case_base=0)
        mg = MG(operators, smoothers, prolongations,
                presmoother_count=int(self.gui.presmoother_count),
                postsmoother_count=int(self.gui.postsmoother_count))

        # very small value
        # ls1.solver.SetPreconditioner(mg.solver)
        #solall = ls1.Mult(BB, XX)

        # transfer looks okay
        #solall0 = ls0.Mult(_BB, _XX)
        #np.save('saved_block_vector', _XX.GetDataArray())

        '''
        solall0 = ls0.Mult(_BB, _XX)
        engine.level_idx = 0
        A = engine.assembled_blocks[0]
        X = engine.assembled_blocks[1]
        solall = np.transpose(np.vstack([_XX.GetDataArray()]))        
        if not self.phys_real and self.gui.assemble_real:
            solall = self.linearsolver_models[-1].real_to_complex(solall, _AA)

        A.reformat_central_mat(solall, 0, X[0], self.blk_mask)
        self.sol = X[0]
        return True
        '''
        #

        # this seems good after removing ReorientTetMesh and
        # changed to use H refinement
        '''
        solall0 = ls0.Mult(_BB, _XX)
        P_matrix.Mult(_XX, XX)
        np.save('saved_block_vector', XX.GetDataArray())        
        engine.level_idx = 1
        A = engine.assembled_blocks[0]
        X = engine.assembled_blocks[1]
        solall = np.transpose(np.vstack([XX.GetDataArray()]))        
        if not self.phys_real and self.gui.assemble_real:
            solall = self.linearsolver_models[-1].real_to_complex(solall, AA)

        A.reformat_central_mat(solall, 0, X[0], self.blk_mask)
        self.sol = X[0]
        return True
        '''
        #print("here", type(_XX), _XX.GetDataArray().shape, XX.GetDataArray().shape)
        #solall = np.transpose(np.vstack([XX.GetDataArray()]))
        #

        # mg alone seems okay. but smoother destor
        #mg.solver.Mult(BB[0], XX)
        #solall = np.transpose(np.vstack([XX.GetDataArray()]))
        '''
        class MyPreconditioner(mfem.Solver):
            def __init__(self):
                mfem.Solver.__init__(self)

            def Mult(self, x, y):
                print('MyPreconditioner.Mult')                
                np.save('original_b', _BB[0].GetDataArray())
                np.save('original_x', x.GetDataArray())                
                P_matrix.MultTranspose(x, _BB[0])
                np.save('restricted_b', _BB[0].GetDataArray())
                ls0.Mult(_BB, _XX)
                P_matrix.Mult(_XX, y)
                #print(x, y)
                #assert False, "faile for now"

            def SetOperator(self, opr):
                pass

        class MyFGMRESSolver(mfem.FGMRESSolver):
            def Mult(self, x, y):
                print('MyFGMRESSolver.Mult')
                mfem.FGMRESSolver.Mult(self, x, y)

        prc = MyPreconditioner()
        # write solver here...
        solver = MyFGMRESSolver()
        solver.SetRelTol(1e-18)
        solver.SetAbsTol(0)
        solver.SetMaxIter(1)
        solver.SetPrintLevel(1)
        solver.SetOperator(ls1.A)
        solver.SetPreconditioner(mg.solver)
        #solver.SetPreconditioner(prc)
        np.save('initial_x', XX.GetDataArray())        
        solver.Mult(BB[0], XX)
        solall = np.transpose(np.vstack([XX.GetDataArray()]))
        '''
        #solall = ls1.Mult(BB, XX)
        '''
        solall = ls1.Mult(BB, XX)        
        mg.solver.Mult(BB[0], XX)

        '''
        np.save('saved_block_vector_input', BB[0].GetDataArray())
        if ls1.gui.maxiter > 0:
            ls1.solver.SetPreconditioner(mg.solver)
            solall = ls1.Mult(BB, XX)
        else:
            mg.solver.Mult(BB[0], XX)
            solall = np.transpose(np.vstack([XX.GetDataArray()]))

        np.save('saved_block_vector_mg', XX.GetDataArray())

        # check fine level linear system...
        #solall = ls1.Mult(BB, XX)

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

        self.sol = X[0]

        # store probe signal (use t=0.0 in std_solver)
        for p in self.probe:
            p.append_sol(X[0])

        return True


class MG(IterativeSolver):   # LinearSolver
    def __init__(self, operators, smoothers, prolongations,
                 presmoother_count=1,
                 postsmoother_count=1):

        own_operators = [False]*len(operators)
        own_smoothers = [False]*len(smoothers)
        own_prolongations = [False]*len(prolongations)

        mg = mfem.Multigrid(operators, smoothers, prolongations,
                            own_operators, own_smoothers, own_prolongations)
        mg.SetCycleType(mfem.Multigrid.CycleType_VCYCLE,
                        presmoother_count, postsmoother_count)

        self.solver = mg
        #self.A = operators[-1]


def fill_prolongation_operator(engine, level, blk_opr):
    engine.access_idx = 0
    P = None
    diags = []

    A, _X, _RHS, _Ae,  _B,  _M, _dep_vars = engine.assembled_blocks
    use_complex_opr = A.complex and (A.shape[0] == blk_opr.NumRowBlocks())

    hights = A.get_local_row_heights()
    widths = A.get_local_col_widths()

    print(hights, widths)
    cols = [0]
    rows = [0]

    for dep_var in engine.r_dep_vars:
        offset = engine.r_dep_var_offset(dep_var)

        tmp_cols = []
        tmp_rows = []
        tmp_diags = []

        if use_complex_opr:
            mat = blk_opr._linked_op[(offset, offset)]
            conv = mat.GetConvention()
            conv == (1 if mfem.ComplexOperator.HERMITIAN else -1)
        else:
            conv = 1

        if engine.r_isFESvar(dep_var):
            h = engine.fespaces.get_hierarchy(dep_var)
            P = h.GetProlongationAtLevel(level)
            tmp_cols.append(P.Width())
            tmp_rows.append(P.Height())
            tmp_diags.append(P)
            if A.complex:
                tmp_cols.append(P.Width())
                tmp_rows.append(P.Height())
                if conv == -1:
                    oo2 = mfem.ScaleOperator(P, -1)
                    oo2._opr = P
                    tmp_diags.append(oo2)
                else:
                    tmp_diags.append(P)
        else:
            tmp_cols.append(widths[offset])
            tmp_rows.append(widths[offset])
            tmp_diags.append(mfem.IdentityOperator(widths[offset]))

            if A.complex:
                tmp_cols.append(widths[offset])
                tmp_rows.append(widths[offset])
                oo = mfem.IdentityOperator(widths[offset])
                if conv == -1:
                    oo2 = mfem.ScaleOperator(oo, -1)
                    oo2._opr = oo
                    tmp_diags.append(oo2)
                else:
                    tmp_diags.append(oo)
        if use_complex_opr:
            tmp_cols = [0] + tmp_cols
            tmp_rows = [0] + tmp_rows
            offset_c = mfem.intArray(tmp_cols)
            offset_r = mfem.intArray(tmp_rows)
            offset_c.PartialSum()
            offset_r.PartialSum()
            smoother = mfem.BlockOperator(offset_r, offset_c)
            smoother.SetDiagonalBlock(0, tmp_diags[0])
            smoother.SetDiagonalBlock(1, tmp_diags[1])
            smoother._smoother = tmp_diags
            cols.append(tmp_cols[1]*2)
            rows.append(tmp_rows[1]*2)
            diags.append(smoother)
        else:
            cols.extend(tmp_cols)
            rows.extend(tmp_rows)
            diags.extend(tmp_diags)

    ro = mfem.intArray(rows)
    co = mfem.intArray(cols)
    ro.PartialSum()
    co.PartialSum()

    P = mfem.BlockOperator(ro, co)
    for i, d in enumerate(diags):
        P.SetBlock(i, i, d)
    P._diags = diags
    return P


def genearate_smoother(engine, level, blk_opr):
    from petram.engine import ParallelEngine
    assert not isinstance(
        engine, ParallelEngine), "Parallel is not supported"

    engine.access_idx = 0
    P = None
    diags = []
    cols = [0]
    ess_tdofs = []
    A, _X, _RHS, _Ae,  _B,  _M, _dep_vars = engine.assembled_blocks
    widths = A.get_local_col_widths()

    use_complex_opr = A.complex and (A.shape[0] == blk_opr.NumRowBlocks())

    for dep_var in engine.r_dep_vars:
        offset = engine.r_dep_var_offset(dep_var)

        tmp_cols = []
        tmp_diags = []
        if engine.r_isFESvar(dep_var):
            ess_tdof = mfem.intArray(engine.ess_tdofs[dep_var])
            ess_tdofs.append(ess_tdofs)
            opr = A[offset, offset]

            if use_complex_opr:
                mat = blk_opr._linked_op[(offset, offset)]
                conv = mat.GetConvention()
                conv == 1 if mfem.ComplexOperator.HERMITIAN else -1
                mat1 = mat._real_operator
                mat2 = mat._imag_operator
            else:
                conv = 1
                if A.complex:
                    mat1 = blk_opr.GetBlock(offset*2, offset*2)
                    mat2 = blk_opr.GetBlock(offset*2 + 1, offset*2 + 1)
                else:
                    mat1 = blk_opr.GetBlock(offset, offset)

            if A.complex:
                dd = opr.diagonal()
                diag = mfem.Vector(list(dd.real))
                print("real", dd.real)
                rsmoother = mfem.OperatorChebyshevSmoother(mat1,
                                                           diag,
                                                           ess_tdof,
                                                           2)
                dd = opr.diagonal()*conv
                diag = mfem.Vector(list(dd.real))
                print("imag", dd.imag)
                ismoother = mfem.OperatorChebyshevSmoother(mat1,
                                                           diag,
                                                           ess_tdof,
                                                           2)
                tmp_cols.append(opr.shape[0])
                tmp_cols.append(opr.shape[0])
                tmp_diags.append(rsmoother)
                tmp_diags.append(ismoother)

            else:
                dd = opr.diagonal()
                diag = mfem.Vector(list(dd))
                smoother = mfem.OperatorChebyshevSmoother(mat1,
                                                          diag,
                                                          ess_tdof,
                                                          2)
                tmp_diags.append(smoother)
                tmp_cols.append(opr.shape[0])

        else:
            print("Non FESvar", dep_var, offset)
            tmp_cols.append(widths[offset])
            tmp_diags.append(mfem.IdentityOperator(widths[offset]))
            if A.complex:
                tmp_cols.append(widths[offset])
                oo = mfem.IdentityOperator(widths[offset])
                if conv == -1:
                    oo2 = mfem.ScaleOperator(oo, -1)
                    oo2._opr = oo
                    tmp_diags.append(oo2)
                else:
                    tmp_diags.append(oo)

        if use_complex_opr:
            tmp_cols = [0] + tmp_cols
            blockOffsets = mfem.intArray(tmp_cols)
            blockOffsets.PartialSum()
            smoother = mfem.BlockDiagonalPreconditioner(blockOffsets)
            smoother.SetDiagonalBlock(0, tmp_diags[0])
            smoother.SetDiagonalBlock(1, tmp_diags[1])
            smoother._smoother = tmp_diags
            cols.append(tmp_cols[1]*2)
            diags.append(smoother)
        else:
            cols.extend(tmp_cols)
            diags.extend(tmp_diags)

    co = mfem.intArray(cols)
    co.PartialSum()

    P = mfem.BlockDiagonalPreconditioner(co)
    for i, d in enumerate(diags):
        P.SetDiagonalBlock(i,  d)
    P._diags = diags
    P._ess_tdofs = ess_tdofs
    return P
