from petram.solver.strumpack_model import Strumpack
from petram.solver.mumps_model import MUMPSMFEMSolverModel
from petram.solver.krylov import KrylovModel, StationaryRefinementModel
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

    def __init__(self):
        self.is_preconditioner = True


class FinestLvlSolver:
    ...


class RefinedLevel(FineLevel, SolverBase):
    hide_ns_menu = True
    has_2nd_panel = True

    def __init__(self, *args, **kwags):
        SolverBase.__init__(self, *args, **kwags)
        FineLevel.__init__(self, *args, **kwags)
        self.smoother_count = None

    def attribute_set(self, v):
        v = FineLevel.attribute_set(self, v)
        v = SolverBase.attribute_set(self, v)
        v["level_inc"] = "1"
        v["refinement_type"] = "P(order)"
        v["extra_constraints1"] = ""
        v["extra_constraints2"] = ""
        v["select_extra_constreint"] = False
        v["unselect_extra_constreint"] = False
        v["use_domain_refinement"] = False
        v["presmoother_count"] = "default"
        v["postsmoother_count"] = "default"
        v["cycle_max"] = "default"

        return v

    def panel1_param(self):
        panels = super(RefinedLevel, self).panel1_param()

        mm = [["constraints", "", 0, {}], ]

        p2 = [None, (self.select_extra_constreint, (self.extra_constraints1,)),
              27, ({"text": "Enable contributions"}, {"elp": mm},)]
        p3 = [None, (self.unselect_extra_constreint, (self.extra_constraints2,)),
              27, ({"text": "Disable contributions"}, {"elp": mm},)]
        p4 = [["pre-smoother #count", "", 0, {}],
              ["post-smoother #count", "", 0, {}],
              ["cycle max (V:1, W;2)", "", 0, {}], ]

        panels.extend([["refinement type", self.refinement_type, 1,
                        {"values": ["P(order)", "H(mesh)"]}],
                       ["level increment", "", 0, {}],
                       p2, p3])
        panels.extend(p4)
        return panels

    def get_panel1_value(self):
        value = list(super(RefinedLevel, self).get_panel1_value())
        value.append(self.refinement_type)
        value.append(self.level_inc)
        v2 = (self.select_extra_constreint, (self.extra_constraints1,))
        v3 = (self.unselect_extra_constreint, (self.extra_constraints2,))
        v4 = [self.presmoother_count,
              self.postsmoother_count,
              self.cycle_max]

        value.append(v2)
        value.append(v3)
        value.extend(v4)

        return value

    def import_panel1_value(self, v):
        super(RefinedLevel, self).import_panel1_value(v[:-7])
        self.refinement_type = v[-7]
        self.level_inc = v[-6]
        self.select_extra_constreint = v[-5][0]
        self.extra_constraints1 = v[-5][1][0]
        self.unselect_extra_constreint = v[-4][0]
        self.extra_constraints2 = v[-4][1][0]
        self.presmoother_count = v[-3]
        self.postsmoother_count = v[-2]
        self.cycle_max = v[-1]

    def panel2_param(self):
        mm = [["constraints", "", 0, {}], ]
        p2 = [None, (self.use_domain_refinement, (self._sel_index,)),
              27, ({"text": "Resrict domain (H-only)"}, {"elp": mm},)]

        return [p2, ]

    def get_panel2_value(self):
        txt = ",".join([str(x) for x in self._sel_index])
        v1 = (self.use_domain_refinement, (txt,))
        return (v1,)

    def import_panel2_value(self, v):
        self.use_domain_refinement = v[0][0]
        self._sel_index = [x for x in v[0][1][0].split(',')]

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
        if self.smoother_count[1] == 0:
            for x in self.iter_enabled():
                return x.prepare_solver(opr, engine)
        else:
            for x in self.iter_enabled():
                return x.prepare_solver_with_multtranspose(opr, engine)
            # return x.prepare_solver(opr, engine)

    def adjust_physics(self, phys):
        names = [x.strip() for x in self.extra_constraints1.split(',')]
        if self.select_extra_constreint:
            for o in phys.walk():
                fname = o.fullname()
                for n in names:
                    if fname.find(n) != -1:
                        o.enabled = True
                        dprint1("!!!!! enabling", fname)

        names = [x.strip() for x in self.extra_constraints2.split(',')]
        if self.unselect_extra_constreint:
            for o in phys.walk():
                fname = o.fullname()
                for n in names:
                    if fname.find(n) != -1:
                        o.enabled = False
                        dprint1("!!!!!disabling", fname)

    @classmethod
    def fancy_tree_name(self):
        return 'Refined'

    @property
    def is_iterative(self):
        return True


class CoarseIterative(KrylovModel, CoarsestLvlSolver):

    def __init__(self, *args, **kwargs):
        KrylovModel.__init__(self, *args, **kwargs)
        CoarsestLvlSolver.__init__(self)

    @classmethod
    def fancy_menu_name(self):
        return 'Kryrov'

    @classmethod
    def fancy_tree_name(self):
        return 'Kryrov'

    def get_info_str(self):
        return 'Coarsest:Lv0'

    def prepare_solver(self, opr, engine):
        solver = self.do_prepare_solver(opr, engine)

        if self.is_preconditioner:
            solver.iterative_mode = False
        else:
            solver.iterative_mode = True

        return solver


class CoarseRefinement(StationaryRefinementModel, CoarsestLvlSolver):
    def __init__(self, *args, **kwargs):
        StationaryRefinementModel.__init__(self, *args, **kwargs)
        CoarsestLvlSolver.__init__(self)

    @classmethod
    def fancy_menu_name(self):
        return 'Refinement'

    @classmethod
    def fancy_tree_name(self):
        return 'Refinement'

    def get_info_str(self):
        return 'Coarsest:Lv0'


class CoarseMUMPS(MUMPSMFEMSolverModel, CoarsestLvlSolver):
    def __init__(self, *args, **kwargs):
        MUMPSMFEMSolverModel.__init__(self, *args, **kwargs)
        CoarsestLvlSolver.__init__(self)

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
        return 'Kryrov'

    @classmethod
    def fancy_tree_name(self):
        return 'Iterative'

    def get_info_str(self):
        return 'Finest'

    def get_possible_child(self):
        return []

    def get_possible_child_menu(self):
        return []

    def prepare_solver(self, opr, engine):
        solver = self.do_prepare_solver(opr, engine)
        solver.iterative_mode = True

        return solver


class FineRefinement(StationaryRefinementModel, FinestLvlSolver):
    @classmethod
    def fancy_menu_name(self):
        return 'StationaryRefinement'

    @classmethod
    def fancy_tree_name(self):
        return 'Refinement'

    def get_info_str(self):
        return 'Finest'

    def get_possible_child(self):
        return []

    def get_possible_child_menu(self):
        return []

    def prepare_solver(self, opr, engine):
        solver = self.do_prepare_solver(opr, engine)
        solver.iterative_mode = True

        return solver


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
        v["cycle_max"] = "2"
        v['assemble_real'] = True
        v["debug_cycle_level"] = False
        v["debug_residuals"] = False

        return v

    def panel1_param(self):
        panels = super(MultiLvlStationarySolver, self).panel1_param()

        mm = [[None, self.use_block_symmetric, 3,
               {"text": "block symmetric format"}], ]

        p2 = [[None, (self.merge_real_imag, (self.use_block_symmetric,)),
               27, ({"text": "Use ComplexOperator"}, {"elp": mm},)], ]
        panels.extend(p2)

        p3 = [["pre-smoother #count", "", 0, {}],
              ["post-smoother #count", "", 0, {}],
              ["cycle max (V:1, W;2)", "", 0, {}], ]
        panels.extend(p3)

        return panels

    def get_panel1_value(self):
        value = list(super(MultiLvlStationarySolver, self).get_panel1_value())
        value.append((self.merge_real_imag, [self.use_block_symmetric, ]))
        value.append(self.presmoother_count)
        value.append(self.postsmoother_count)
        value.append(self.cycle_max)
        return value

    def import_panel1_value(self, v):
        super(MultiLvlStationarySolver, self).import_panel1_value(v[:-4])
        self.merge_real_imag = bool(v[-4][0])
        self.use_block_symmetric = bool(v[-4][1][0])
        self.presmoother_count = v[-3]
        self.postsmoother_count = v[-2]
        self.cycle_max = v[-1]

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

        levels, klevels, setting = self._get_level_solvers()

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
        return (CoarseMUMPS,
                CoarseStrumpack,
                CoarseRefinement,
                CoarseIterative,
                RefinedLevel,
                FineIterative,
                FineRefinement)

    def get_possible_child_menu(self):
        choice = [("CoarseSolver", CoarseMUMPS),
                  ("", CoarseStrumpack),
                  ("", CoarseIterative),
                  ("!", CoarseRefinement),
                  ("", RefinedLevel),
                  ("FineSolver", FineIterative),
                  ("!", FineRefinement), ]
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

        level_setting = {}

        for x in self:
            child = self[x]
            if not child.is_enabled():
                continue
            if isinstance(child, CoarsestLvlSolver):
                continue
            if isinstance(child, RefinedLevel):
                levels.append(child)
                klevels.append(child.grid_level)

                presmoother_count = int(self.presmoother_count
                                        if child.presmoother_count.lower() == 'default'
                                        else child.presmooth_count)
                postsmoother_count = int(self.postsmoother_count
                                         if child.postsmoother_count.lower() == 'default'
                                         else child.presmooth_count)
                cycle_max = int(self.cycle_max
                                if child.cycle_max.lower() == 'default'
                                else child.cycle_max)

                level_setting[child.grid_level] = (
                    presmoother_count, postsmoother_count, cycle_max)

        idx = np.argsort(klevels)
        levels = [levels[i] for i in idx]
        klevels = [int(klevels[i]) for i in idx]
        return levels, klevels, level_setting

    def get_level_solvers(self):
        return self._get_level_solvers()[0]

    def get_level_setting(self):
        return self._get_level_solvers()[-1]

    def create_refined_levels(self, engine, lvl):
        '''
        lvl : refined level number (1, 2, 3, ....)
              1 means "AFTER" 1 refinement
        '''
        levels = self.get_level_solvers()
        for l in levels:
            l.smoother_count = (int(self.presmoother_count),
                                int(self.postsmoother_count))

        if lvl >= len(levels):
            return False

        level = levels[lvl]

        target_phys = self.get_target_phys()
        refine_type = level.refinement_type[0]  # P or H
        refine_inc = int(level.level_inc)
        if level.use_domain_refinement:
            refine_dom = [int(x) for x in level._sel_index]
        else:
            refine_dom = None

        for phys in target_phys:
            dprint1("Adding refined level for " + phys.name())
            level.adjust_physics(phys)
            engine.prepare_refined_level(phys, refine_type,
                                         inc=refine_inc,
                                         refine_dom=refine_dom)

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
        return M[0], np.any(mask_M[0])

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

        enabled_flag = engine.model.gather_enebled_flags(engine.model['Phys'])

        for l, lvl in enumerate(levels):
            engine.level_idx = l

            if isinstance(lvl, RefinedLevel):
                for phys in self.get_phys():
                    lvl.adjust_physics(phys)
                engine.assign_sel_index()

            self.do_assemble(inplace)

        self.assembled = True

        engine.model.apply_enebled_flags(engine.model['Phys'], enabled_flag)

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
        finest = self.gui.get_active_solver(cls=FinestLvlSolver)

        if finest is None and len(levels) == 1:
            levels[0].is_preconditioner = False

        solvers = []
        for lvl, solver_model in enumerate(levels):
            engine.level_idx = lvl
            opr = self.finalized_ls[lvl][-1]
            s = solver_model.prepare_solver(opr, engine)
            solvers.append(s)

        if finest is not None:
            opr = self.finalized_ls[lvl][-1]
            finestsolver = finest.prepare_solver(opr, engine)
        else:
            finestsolver = None
        return solvers, finestsolver

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

    def _solve(self, update_operator=True):
        engine = self.engine

        self.finalize_linear_systems()

        smoothers, finestsolver = self.create_solvers()
        prolongations = self.create_prolongations()

        operators = [x[-1] for x in self.finalized_ls]

        #solall = linearsolver.Mult(BB, case_base=0)
        if len(smoothers) > 1:
            mg = generate_MG(operators, smoothers, prolongations,
                             presmoother_count=int(self.gui.presmoother_count),
                             postsmoother_count=int(self.gui.postsmoother_count))

        else:
            mg = None

        BB, XX, AA = self.finalized_ls[-1]

        if finestsolver is not None:
            finestsolver.SetPreconditioner(mg)
            finestsolver.Mult(BB[0], XX)
            solall = np.transpose(np.vstack([XX.GetDataArray()]))
        elif mg is not None:
            mg.Mult(BB[0], XX)
            solall = np.transpose(np.vstack([XX.GetDataArray()]))
        else:
            # this makes sense if coarsest smoother is direct solver
            smoothers[0].Mult(BB[0], XX)
            solall = np.transpose(np.vstack([XX.GetDataArray()]))

        if not self.phys_real:
            from petram.solver.solver_model import convert_realblocks_to_complex
            merge_real_imag = self.ls_type in ["blk_merged", "blk_merged_s"]
            solall = convert_realblocks_to_complex(solall, AA, merge_real_imag)

        engine.level_idx = len(self.finalized_ls)-1
        A = engine.assembled_blocks[0]
        X = engine.assembled_blocks[1]
        A.reformat_distributed_mat(solall, 0, X[0], self.blk_mask)

        self.sol = X[0]

        # store probe signal (use t=0.0 in std_solver)
        for p in self.probe:
            p.append_sol(X[0])

        return True

    def solve(self, update_operator=True):
        '''
        test version calls one V cycle written in Python
        '''
        engine = self.engine

        self.finalize_linear_systems()

        smoothers, finestsolver = self.create_solvers()
        prolongations = self.create_prolongations()

        operators = [x[-1] for x in self.finalized_ls]

        #offsets0 = operators[0].RowOffsets().ToList()
        #offsets1 = operators[1].RowOffsets().ToList()

        if len(smoothers) > 1:
            lvl = engine.level_idx
            is_complex = not self.phys_real

            ess_tdofs_arr = []
            for lvl in range(len(operators)):
                engine.level_idx = lvl
                esstdofs0 = engine.collect_local_ess_TDofs(
                    operators[lvl], self.ls_type, is_complex)
                ess_tdofs_arr.append(esstdofs0)

            #solall = linearsolver.Mult(BB, case_base=0)
            level_setting = self.gui.get_level_setting()

            presmoother_count = {
                key: level_setting[key][0] for key in level_setting}
            postsmoother_count = {
                key: level_setting[key][1] for key in level_setting}
            cycle_max = {key: level_setting[key][2] for key in level_setting}

            mg = PyMG(operators,
                      smoothers,
                      prolongations,
                      ess_tdofs=ess_tdofs_arr,
                      presmoother_count=presmoother_count,
                      postsmoother_count=postsmoother_count,
                      cycle_max=cycle_max,
                      debug1=self.gui.debug_cycle_level,
                      debug2=self.gui.debug_residuals)

        else:
            mg = None

        BB, XX, AA = self.finalized_ls[-1]

        if finestsolver is not None:
            finestsolver.SetPreconditioner(mg)
            finestsolver.Mult(BB[0], XX)
            solall = np.transpose(np.vstack([XX.GetDataArray()]))
        elif mg is not None:
            mg.Mult(BB[0], XX)
            solall = np.transpose(np.vstack([XX.GetDataArray()]))
        else:
            # this makes sense if coarsest smoother is direct solver
            smoothers[0].Mult(BB[0], XX)
            solall = np.transpose(np.vstack([XX.GetDataArray()]))

        solall = np.transpose(np.vstack([XX.GetDataArray()]))

        if not self.phys_real:
            from petram.solver.solver_model import convert_realblocks_to_complex
            merge_real_imag = self.ls_type in ["blk_merged", "blk_merged_s"]
            solall = convert_realblocks_to_complex(solall, AA, merge_real_imag)

        engine.level_idx = len(self.finalized_ls)-1
        A = engine.assembled_blocks[0]
        X = engine.assembled_blocks[1]
        A.reformat_distributed_mat(solall, 0, X[0], self.blk_mask)

        self.sol = X[0]

        # store probe signal (use t=0.0 in std_solver)
        for p in self.probe:
            p.append_sol(X[0])

        return True


def generate_MG(operators, smoothers, prolongations,
                presmoother_count=1,
                postsmoother_count=1):

    own_operators = [False]*len(operators)
    own_smoothers = [False]*len(smoothers)
    own_prolongations = mfem.boolArray([False]*len(prolongations))

    ops = mfem.OperatorPtrArray(operators)
    # mg = mfem.Multigrid(operators, smoothers, prolongations,
    mg = mfem.Multigrid(ops, smoothers, prolongations,
                        own_operators, own_smoothers, own_prolongations)

    mg.SetCycleType(mfem.Multigrid.CycleType_VCYCLE,
                    presmoother_count, postsmoother_count)

    return mg


class PyMG(mfem.PyIterativeSolver):
    def __init__(self, operators, smoothers, prolongations,
                 ess_tdofs=None, presmoother_count=1, postsmoother_count=1,
                 debug1=True, debug2=True, cycle_max=1):
        self.operators = operators
        self.smoothers = smoothers
        self.prolongations = prolongations
        self.presmoother_count = presmoother_count
        self.postsmoother_count = postsmoother_count
        self.ess_tdofs = ess_tdofs

        self.debug1 = debug1
        self.debug2 = debug2

        self.cycle_rel_tol = 0.01
        self.cycle_max = cycle_max

        if use_parallel:
            from mpi4py import MPI
            args = (MPI.COMM_WORLD,)
        else:
            args = tuple()
        mfem.PyIterativeSolver.__init__(self, *args)

    def Mult(self, x, y):
        '''
        call cycle_max times of perform_one_cycle
        '''
        y.Assign(0.0)

        lvl = len(self.ess_tdofs)-1

        correction = mfem.Vector(y.Size())
        correction.Assign(0.0)
        err = mfem.Vector(x.Size())
        err.Assign(x)
        tmp = mfem.Vector(x.Size())

        for i in range(self.cycle_max[lvl]):
            self.perform_one_cycle(err, correction)

            self.operators[lvl].Mult(correction, tmp)
            tmp *= -1
            tmp += err
            err.Assign(tmp)

            y += correction

    def perform_one_cycle(self, x, y, lvl=None):

        if lvl is None:
            # start from finest level
            lvl = len(self.ess_tdofs)-1

        if self.debug1:
            dprint1("========\n")
            dprint1("Entering Cycle lvl =  ", lvl)
        if lvl == 0:
            if self.debug2:
                dprint1("    - error on essential at level = 0",
                        np.sum(np.abs(x.GetDataArray()[self.ess_tdofs[0]])))
                dprint1("    - NormInf before level0 solve", x.Normlinf())

            y.Assign(0.0)
            self.smoothers[0].Mult(x, y)

            if self.debug2:
                dprint1("    - NormInf after level0 solve", y.Normlinf())
                dprint1("    - correction on essential at coarse level",
                        np.sum(np.abs(y.GetDataArray()[self.ess_tdofs[0]])))

                tmp = mfem.Vector(y.Size())
                self.operators[0].Mult(y, tmp)
                tmp -= x
                dprint1("    - level0 linear inverse error (L2): ", tmp.Norml2())

            if self.debug1:
                dprint1("Exiting Cycle lvl =  ", lvl)
                dprint1("========\n")
            return

        if self.debug2:
            dprint1("")
            dprint1("  - residual on essential at the start of level",
                    np.sum(np.abs(x.GetDataArray()[self.ess_tdofs[1]])))
            dprint1("  - initial residual L2", x.Norml2())

        err = mfem.Vector(x.Size())
        err.Assign(x)

        y0 = mfem.Vector(x.Size())
        y.Assign(0.0)

        for jjj in range(self.presmoother_count[lvl]):
            if self.debug1:
                txt = str(jjj) + '/' + str(self.presmoother_count[lvl])
                dprint1("   -  Performing pre-smooth " +
                        txt + " : level = " + str(lvl))

            self.smoothers[lvl].iterative_mode = False
            y0.Assign(0.0)

            if self.debug2:
                dprint1("    - resdidual on essential before presmooth",
                        np.sum(np.abs(err.GetDataArray()[self.ess_tdofs[lvl]])))

            self.smoothers[lvl].Mult(err, y0)

            y += y0
            self.operators[lvl].Mult(y, err)
            err *= -1
            err += x

            if self.debug2:
                dprint1("    - residual on essential after pre-smooth",
                        np.sum(np.abs(err.GetDataArray()[self.ess_tdofs[lvl]])))

        if self.debug2:
            dprint1(
                "  correction (L2) before adding prolonged correction: ", y.Norml2())

        # prepare err passed to lower level
        lvl2 = lvl - 1
        lvl2_width = self.operators[lvl2].Width()
        err2 = mfem.Vector(lvl2_width)
        self.prolongations[lvl2].MultTranspose(err, err2)
        y2 = mfem.Vector(lvl2_width)

        # (zeroing the error sent to the lower level)   <--- this works
        err2.GetDataArray()[self.ess_tdofs[lvl2]] = 0.0

        if self.debug2:
            dprint1("    - error on essential given to a coarse level",
                    np.sum(np.abs(err2.GetDataArray()[self.ess_tdofs[lvl2]])))

        # calling lower levels
        #   cycle max = 1 (V-cycle)
        #   cycle max = 2 (W-cycle)

        x2 = mfem.Vector(lvl2_width)
        x2.Assign(0.0)
        err2_L2 = err2.Norml2()
        if use_parallel:
            err2_L2 = np.sqrt(
                np.sum([x**2 for x in MPI.COMM_WORLD.allgather(err2_L2)]))
        rel_improve0 = 1.0
        for i in range(self.cycle_max[lvl]):
            self.perform_one_cycle(err2, y2, lvl=lvl2)
            x2 += y2
            tmp = mfem.Vector(x2.Size())
            self.operators[lvl2].Mult(y2, tmp)
            err2 -= tmp
            err2norm = err2.Norml2()
            if use_parallel:
                err2norm = np.sqrt(
                    np.sum([x**2 for x in MPI.COMM_WORLD.allgather(err2norm)]))

            rel_improve = err2norm/err2_L2

            if self.debug2:
                dprint1(str(i)+" th cycle checking cycle error lvl = :" + str(lvl))
                dprint1("correction L2/ rel_improve", y2.Norml2(), rel_improve)
                dprint1("change of improvement", np.abs(
                    np.abs(rel_improve0/rel_improve)-1))

            if rel_improve < self.cycle_rel_tol:
                break
            if np.abs(np.abs(rel_improve0/rel_improve)-1) < 0.1:
                break
            rel_improve0 = rel_improve

        y2 = x2

        # (zeroing the correction given from the lower level)   <--- this works
        y2.GetDataArray()[self.ess_tdofs[lvl2]] = 0.0

        self.prolongations[lvl2].Mult(y2, y0)
        # (zeroing after MUMPS after prolongation)
        # tmp.GetDataArray()[self.ess_tdofs[1]] = 0.0  # <--- does not works
        y += y0

        if self.debug2:
            dprint1("    - correction on essential prolonged from lower level",
                    np.sum(np.abs(y0.GetDataArray()[self.ess_tdofs[lvl]])))
            dprint1(
                "    - correction (L2) after adding prolonged correction: ", y.Norml2())

        for jjj in range(self.postsmoother_count[lvl]):
            if self.debug1:
                txt = str(jjj) + '/' + str(self.postsmoother_count[lvl])
                dprint1("   -  Performing post-smooth " +
                        txt + " : level = " + str(lvl))

            # compute error
            self.operators[lvl].Mult(y, err)
            err *= -1
            err += x

            if self.debug2:
                dprint1("    - residual on essential before postsmooth",
                        np.sum(np.abs(err.GetDataArray()[self.ess_tdofs[lvl]])))
                dprint1("    - residual L2 before before postsmooth", err.Norml2())

            y0.Assign(0.0)
            self.smoothers[lvl].Mult(err, y0)
            # (zeroing the essentials of final correction)
            # y0.GetDataArray()[self.ess_tdofs[1]] = 0.0   <--- does not works
            y += y0

            if self.debug2:
                dprint1("    - correction on essential",
                        np.sum(np.abs(y0.GetDataArray()[self.ess_tdofs[lvl]])))
                dprint1("    - correction norm (L2) after postsmooth", y0.Norml2())

        if self.debug2:
            self.operators[lvl].Mult(y, err)
            err *= -1
            err += x

            dprint1("  - final correction on essential",
                    np.sum(np.abs(y.GetDataArray()[self.ess_tdofs[lvl]])))
            dprint1("  - finial norm of correction (L2): ", y.Norml2())
            dprint1("  - final residual L2", err.Norml2())
        if self.debug1:
            dprint1("Exiting Cycle lvl =  ", lvl)
            dprint1("========\n")


def fill_prolongation_operator(engine, level, XX, AA, ls_type, phys_real):
    engine.access_idx = 0
    P = None
    diags = []

    use_complex_opr = ls_type in ['blk_merged', 'blk_merged_s']

    widths = [XX.BlockSize(i) for i in range(XX.NumBlocks())]

    if not phys_real:
        if use_complex_opr:
            widths = [x//2 for x in widths]
        else:
            widths = [widths[i*2] for i in range(len(widths)//2)]

    cols = [0]
    rows = [0]

    for dep_var in engine.r_dep_vars:
        offset = engine.r_dep_var_offset(dep_var)

        tmp_cols = []
        tmp_rows = []
        tmp_diags = []

        # if use_complex_opr:
        #    mat = AA._linked_op[(offset, offset)]
        #    conv = mat.GetConvention()
        #    conv = (1 if mfem.ComplexOperator.HERMITIAN else -1)
        # else:
        #    conv = 1

        if engine.r_isFESvar(dep_var):
            h = engine.fespaces.get_hierarchy(dep_var)
            P = h.GetProlongationAtLevel(level)
            tmp_cols.append(P.Width())
            tmp_rows.append(P.Height())
            tmp_diags.append(P)
            if not phys_real:
                tmp_cols.append(P.Width())
                tmp_rows.append(P.Height())
                # if conv == -1:
                #    oo2 = mfem.ScaledOperator(P, -1)
                #    oo2._opr = P
                #    tmp_diags.append(oo2)
                # else:
                tmp_diags.append(P)
        else:
            tmp_cols.append(widths[offset])
            tmp_rows.append(widths[offset])
            tmp_diags.append(mfem.IdentityOperator(widths[offset]))

            if not phys_real:
                tmp_cols.append(widths[offset])
                tmp_rows.append(widths[offset])
                oo = mfem.IdentityOperator(widths[offset])
                # if conv == -1:
                #    oo2 = mfem.ScaledOperator(oo, -1)
                #    oo2._opr = oo
                #    tmp_diags.append(oo2)
                # else:
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


def generate_smoother(engine, level, blk_opr):
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
                rsmoother = mfem.OperatorChebyshevSmoother(mat1,
                                                           diag,
                                                           ess_tdof,
                                                           2)
                dd = opr.diagonal()*conv
                diag = mfem.Vector(list(dd.real))
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
