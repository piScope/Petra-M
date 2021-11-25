from petram.solver.strumpack_model import Strumpack
from petram.solver.mumps_model import MUMPSMFEMSolverModel
from petram.solver.krylov import KrylovModel
import os
import warnings

import numpy as np

from petram.namespace_mixin import NS_mixin
from petram.solver.iterative_model import (Iterative,
                                           IterativeSolver)
from petram.solver.solver_model import (SolverBase,
                                        SolverInstance)

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


class FineLevel(NS_mixin):
    def attribute_set(self, v):
        super(FineLevel, self).attribute_set(v)
        v["grid_level"] = 1
        return v

    def get_info_str(self):
        return 'Lv:' + str(self.grid_level)

    def get_special_menu(self, evt):
        return [["Set level", self.onSetLevel, None, ], ]

    def onSetLevel(self, evt):
        import wx
        from ifigure.utils.edit_list import DialogEditList

        diag = evt.GetEventObject().GetTopLevelParent()

        list6 = [["New level", self.grid_level, 0], ]
        value = DialogEditList(list6,
                               modal=True,
                               style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
                               tip=None,
                               parent=diag,
                               title='Set level...')
        if not value[0]:
            return

        self.grid_level = int(value[1][0])


class CoarsestLvlSolver:
    grid_level = 0


class FinestLvlSolver:
    ...


class RefinedLevel(FineLevel, SolverBase):
    hide_ns_menu = True
    has_2nd_panel = False

    def __init__(self, *args, **kwags):
        SolverBase.__init__(self, *args, **kwags)
        FineLevel.__init__(self, *args, **kwags)

    def attribute_set(self, v):
        v = FineLevel.attribute_set(self, v)
        v = SolverBase.attribute_set(self, v)
        v["level_inc"] = "1"
        v["refinement_type"] = "P(order)"
        return v

    def panel1_param(self):
        panels = super(RefinedLevel, self).panel1_param()
        panels.extend([["refinement type", self.refinement_type, 1,
                        {"values": ["P(order)", "H(mesh)"]}],
                       ["level increment", "", 0, {}], ])
        return panels

    def get_panel1_value(self):
        value = list(super(RefinedLevel, self).get_panel1_value())
        value.append(self.refinement_type)
        value.append(self.level_inc)
        return value

    def import_panel1_value(self, v):
        super(RefinedLevel, self).import_panel1_value(v[:-2])
        self.refinement_type = v[-2]
        self.level_inc = v[-1]

    def get_possible_child(self):
        from petram.solver.krylov import KrylovSmoother
        from petram.solver.block_smoother import DiagonalSmoother
        return [KrylovSmoother, DiagonalSmoother]

    def get_phys(self):
        my_solve_step = self.get_solve_root()
        return my_solve_step.get_phys()

    def get_phys_range(self):
        my_solve_step = self.get_solve_root()
        return my_solve_step.get_phys_range()

    def prepare_solver(self, opr, engine):
        for x in self.iter_enabled():
            return x.prepare_solver(opr, engine)

    @classmethod
    def fancy_tree_name(self):
        return 'Refined'

    @property
    def is_iterative(self):
        return True


class CoarseIterative(KrylovModel, CoarsestLvlSolver):
    @classmethod
    def fancy_menu_name(self):
        return 'Kryrov'

    @classmethod
    def fancy_tree_name(self):
        return 'Kryrov'

    def get_info_str(self):
        return 'Coarsest:Lv0'


class CoarseMUMPS(MUMPSMFEMSolverModel, CoarsestLvlSolver):
    @classmethod
    def fancy_menu_name(self):
        return 'MUMPS'

    @classmethod
    def fancy_tree_name(self):
        return 'Direct'

    def get_info_str(self):
        return 'MUMPS:Lv0'


class CoarseStrumpack(Strumpack, CoarsestLvlSolver):
    @classmethod
    def fancy_menu_name(self):
        return 'STRUMPACK'

    @classmethod
    def fancy_tree_name(self):
        return 'Direct'

    def get_info_str(self):
        return 'STRUMPACK:Lv0'


class FineIterative(KrylovModel, FinestLvlSolver):
    @classmethod
    def fancy_menu_name(self):
        return 'Finest level solver'

    @classmethod
    def fancy_tree_name(self):
        return 'Iterative'

    def get_info_str(self):
        return 'Finest'

    def get_possible_child(self):
        return []

    def get_possible_child_menu(self):
        return []


class MultiLvlStationarySolver(StdSolver):
    @classmethod
    def fancy_menu_name(self):
        return 'Stationary(MultiLevel)'

    @classmethod
    def fancy_tree_name(self):
        return 'Stationary'

    def attribute_set(self, v):
        super(MultiLvlSolver, self).attribute_set(v)
        v['merge_real_imag'] = True
        v['use_block_symmetric'] = False
        v["presmoother_count"] = "1"
        v["postsmoother_count"] = "1"
        v['assemble_real'] = True
        return v

    def panel1_param(self):
        panels = super(MultiLvlStationarySolver, self).panel1_param()

        mm = [[None, self.use_block_symmetric, 3,
               {"text": "block symmetric format"}], ]

        p2 = [[None, (self.merge_real_imag, (self.use_block_symmetric,)),
               27, ({"text": "Use ComplexOperator"}, {"elp": mm},)], ]
        panels.extend(p2)

        p3 = [["pre-smoother #count", "", 0, {}],
              ["post-smoother #count", "", 0, {}], ]
        panels.extend(p3)

        return panels

    def get_panel1_value(self):
        value = list(super(MultiLvlStationarySolver, self).get_panel1_value())
        value.append((self.merge_real_imag, [self.use_block_symmetric, ]))
        value.append(self.presmoother_count)
        value.append(self.postsmoother_count)
        return value

    def import_panel1_value(self, v):
        super(MultiLvlStationarySolver, self).import_panel1_value(v[:-3])
        self.merge_real_imag = bool(v[-3][0])
        self.use_block_symmetric = bool(v[-3][1][0])
        self.presmoother_count = v[-2]
        self.postsmoother_count = v[-1]

    def allocate_solver_instance(self, engine):
        if self.clear_wdir:
            engine.remove_solfiles()

        instance = MLInstance(self, engine)
        return instance

    def get_matrix_weight(self, timestep_config):  # , timestep_weight):
        if timestep_config[0]:
            return [1, 0, 0]
        else:
            return [0, 0, 0]

    def does_solver_choose_linearsystem_type(self):
        return True

    def get_linearsystem_type_from_solvermodel(self):
        assemble_real = self.assemble_real
        phys_real = self.get_solve_root().is_allphys_real()

        if phys_real:
            if assemble_real:
                dprint1("Use assemble-real is only for complex value problem !!!!")
                return 'blk_interleave'
            else:
                return 'blk_interleave'

        # below phys is complex

        # merge_real_imag -> complex operator
        if self.merge_real_imag and self.use_block_symmetric:
            return 'blk_merged_s'
        elif self.merge_real_imag and not self.use_block_symmetric:
            return 'blk_merged'
        elif assemble_real:
            return 'blk_interleave'
        else:
            assert False, "complex problem must assembled using complex operator or expand as real value problem"
        # return None

    def verify_setting(self):
        '''
        has to have one coarse solver
        '''
        isvalid = True
        txt = ''
        txt_long = ''
        cs = self.get_coarsest_solvers()
        if len(cs) != 1:
            isvalid = False
            txt = 'Invlid MultiLvlSolver configuration'
            txt_long = 'Number of active coarse level solver must be one.'

        levels, klevels = self._get_level_solvers()

        if list(range(len(levels))) != klevels:
            isvalid = False
            txt = 'Invlid MultiLvlSolver configuration'
            txt_long = 'Grid levels are not set properly'

        finest = self.get_active_solver(cls=FinestLvlSolver)
        if finest is not None and len(levels) == 1:
            isvalid = False
            txt = 'Invlid MultiLvlSolver configuration'
            txt_long = 'There is no multi-level setting, while finest solver is set'

        return isvalid, txt, txt_long

    def get_possible_child(self):
        return CoarseMUMPS, CoarseStrumpack, CoarseIterative, RefinedLevel, FineIterative

    def get_possible_child_menu(self):
        choice = [("CoarseSolver", CoarseMUMPS),
                  ("", CoarseStrumpack),
                  ("!", CoarseIterative),
                  ("", RefinedLevel),
                  ("", FineIterative), ]
        return choice

    def get_coarsest_solvers(self):
        s = []
        for x in self:
            child = self[x]
            if not child.is_enabled():
                continue
            if isinstance(child, CoarsestLvlSolver):
                s.append(child)
        return s

    def _get_level_solvers(self):
        levels = [self.get_coarsest_solvers()[0]]
        klevels = [0]
        for x in self:
            child = self[x]
            if not child.is_enabled():
                continue
            if isinstance(child, CoarsestLvlSolver):
                continue
            if isinstance(child, FineLevel):
                levels.append(child)
                klevels.append(child.grid_level)

        idx = np.argsort(klevels)
        levels = [levels[i] for i in idx]
        klevels = [int(klevels[i]) for i in idx]
        return levels, klevels

    def get_level_solvers(self):
        return self._get_level_solvers()[0]

    def create_refined_levels(self, engine, lvl):
        '''
        lvl : refined level number (1, 2, 3, ....)
              1 means "AFTER" 1 refinement
        '''
        levels = self.get_level_solvers()

        if lvl >= len(levels):
            return False

        level = levels[lvl]

        target_phys = self.get_target_phys()
        refine_type = level.refinement_type[0]  # P or H
        refine_inc = int(level.level_inc)

        for phys in target_phys:
            dprint1("Adding refined level for " + phys.name())
            engine.prepare_refined_level(phys, refine_type, inc=refine_inc)

        engine.level_idx = lvl
        for phys in target_phys:
            engine.get_true_v_sizes(phys)

        return True

    @debug.use_profiler
    def run(self, engine, is_first=True, return_instance=False):
        dprint1("Entering run (is_first=", is_first, ")", self.fullpath())
        if self.clear_wdir:
            engine.remove_solfiles()

        instance = self.allocate_solver_instance(engine)
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


MultiLvlSolver = MultiLvlStationarySolver


class MLInstance(SolverInstance):
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

        levels = self.gui.get_level_solvers()

        for l, lvl in enumerate(levels):
            engine.level_idx = l
            self.do_assemble(inplace)

        self.assembled = True

    @property
    def is_iterative(self):
        levels = self.gui.get_level_solvers()
        check = [x.is_iterative for x in levels]
        return np.any(check)

    def finalize_linear_system(self, level):
        engine = self.engine
        engine.level_idx = level
        ls_type = self.ls_type

        A, X, RHS, Ae, B, M, depvars = self.blocks

        mask = self.blk_mask
        engine.copy_block_mask(mask)

        depvars = [x for i, x in enumerate(depvars) if mask[0][i]]

        AA = engine.finalize_matrix(A, mask, not self.phys_real,
                                    format=self.ls_type)
        BB = engine.finalize_rhs([RHS], A, X[0], mask, not self.phys_real,
                                 format=ls_type)

        XX = engine.finalize_x(X[0], RHS, mask, not self.phys_real,
                               format=ls_type)
        return BB, XX, AA

    def finalize_linear_systems(self):
        '''
        call finalize_linear_system for all levels
        '''
        levels = self.gui.get_level_solvers()

        finalized_ls = []
        for k, _lvl in enumerate(levels):
            BB, XX, AA = self.finalize_linear_system(k)
            finalized_ls.append((BB, XX, AA))

        self.finalized_ls = finalized_ls

    def create_solvers(self):
        engine = self.engine
        levels = self.gui.get_level_solvers()

        solvers = []
        for lvl, solver_model in enumerate(levels):
            opr = self.finalized_ls[lvl][-1]
            s = solver_model.prepare_solver(opr, engine)
            solvers.append(s)

        return solvers

    def create_prolongations(self):
        engine = self.engine
        levels = self.gui.get_level_solvers()

        prolongations = []
        for k, _lvl in enumerate(levels):
            if k == len(levels)-1:
                break
            _BB, XX, AA = self.finalized_ls[k]
            P = fill_prolongation_operator(
                engine, k, XX, AA, self.ls_type, self.phys_real)
            prolongations.append(P)

        return prolongations

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

        self.finalize_linear_systems()

        smoothers = self.create_solvers()
        prolongations = self.create_prolongations()

        operators = [x[-1] for x in self.finalized_ls]

        '''
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
        '''
        #solall = linearsolver.Mult(BB, case_base=0)
        if len(smoothers) > 1:
            coarsest = None
            mg = MG(operators, smoothers, prolongations,
                    presmoother_count=int(self.gui.presmoother_count),
                    postsmoother_count=int(self.gui.postsmoother_count))

        else:
            coarsest = smoothers[0]

        finest = self.gui.get_active_solver(cls=FinestLvlSolver)
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
        BB, XX, AA = self.finalized_ls[-1]
        np.save('saved_block_vector_input', BB[0].GetDataArray())
        if finest is not None:
            finest.create_solver()
            finest.solver.SetOperator(self.finalized_ls[-1][-1])
            finest.solver.SetPreconditioner(mg.solver)
            solall = ls1.Mult(BB, XX)
        elif coarsest is not None:
            coarsest.Mult(BB[0], XX)
            solall = np.transpose(np.vstack([XX.GetDataArray()]))
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
        if not self.phys_real:
            from petram.solver.solver_model import convert_realblocks_to_complex
            merge_real_imag = self.ls_type in ["blk_merged", "blk_merged_s"]
            solall = convert_realblocks_to_complex(solall, AA, merge_real_imag)

        engine.level_idx = len(self.finalized_ls)-1
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


def fill_prolongation_operator(engine, level, XX, AA, ls_type, phys_real):
    engine.access_idx = 0
    P = None
    diags = []

    print("level", level)
    print(XX, AA)
    print(type(XX))
    use_complex_opr = ls_type in ['blk_merged', 'blk_merged_s']

    widths = [XX.BlockSize(i) for i in range(XX.NumBlocks())]

    print("widths", widths)
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
            if not phys_real:
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

            if not phys_real:
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
