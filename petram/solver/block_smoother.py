'''
BlcokPreconditioner Model. 
'''
from petram.solver.mumps_model import MUMPSPreconditioner
from petram.mfem_config import use_parallel
import numpy as np

from petram.debug import flush_stdout
from petram.namespace_mixin import NS_mixin
from .solver_model import LinearSolverModel, LinearSolver

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('BlockSmoother')

if use_parallel:
    from petram.helper.mpi_recipes import *
    from mfem.common.parcsr_extra import *
    import mfem.par as mfem
    default_kind = 'hypre'

    from mpi4py import MPI
    num_proc = MPI.COMM_WORLD.size
    myid = MPI.COMM_WORLD.rank
    smyid = '{:0>6d}'.format(myid)
    from mfem.common.mpi_debug import nicePrint

else:
    import mfem.ser as mfem
    default_kind = 'scipy'

class BlockSmoother(LinearSolverModel, NS_mixin):
    hide_ns_menu = True
    has_2nd_panel = False
    accept_complex = False
    always_new_panel = False
    
    @classmethod
    def fancy_menu_name(self):
        return 'BlockPreconditioner'

    @classmethod
    def fancy_tree_name(self):
        return 'BlockPreconditioner'        

    def does_linearsolver_choose_linearsystem_type(self):
        return False

    def supported_linear_system_type(self):
        return ["blk_interleave",
                "blk_merged_s",
                "blk_merged",]

class DiagonalPreconditioner(BlockSmoother):
    @classmethod
    def fancy_menu_name(self):
        return 'DiagonalPreconditioner'

    @classmethod
    def fancy_tree_name(self):
        return 'DiagonalPreconditioner'

    def panel1_param(self):
        import wx
        from petram.pi.widget_smoother import WidgetSmoother

        smp1 = [None, None, 99, {"UI": WidgetSmoother, "span": (1, 2)}]

        return [[None, [False, [''], [[], ]], 27, [{'text': 'advanced mode'},
                                                   {'elp': [
                                                       ['preconditioner', '', 0, None], ]},
                                                   {'elp': [smp1, ]}], ],]

    def get_panel1_value(self):
        # this will set _mat_weight
        from petram.solver.solver_model import SolveStep
        p = self.parent
        while not isinstance(p, SolveStep):
            p = p.parent
            if p is None:
                assert False, "Solver is not under SolveStep"
        num_matrix = p.get_num_matrix(self.get_phys())

        all_dep_vars = self.root()['Phys'].all_dependent_vars(num_matrix,
                                                              self.get_phys(),
                                                              self.get_phys_range())

        prec = [x for x in self.preconditioners if x[0] in all_dep_vars]
        names = [x[0] for x in prec]
        for n in all_dep_vars:
            if not n in names:
                prec.append((n, ['None', 'None']))
        self.preconditioners = prec

        value = ((self.adv_mode, [self.adv_prc, ], [self.preconditioners, ]),)

        return value

    def import_panel1_value(self, v):
        self.preconditioners = v[0][2][0]
        self.adv_mode = v[0][0]
        self.adv_prc = v[0][1][0]

    def attribute_set(self, v):
        v = super(DiagonalPreconditioner, self).attribute_set(v)
        v['preconditioner'] = ''
        v['preconditioners'] = []
        v['adv_mode'] = False
        v['adv_prc'] = ''
        return v

    def get_possible_child(self):
        from petram.solver.mumps_model import MUMPS
        from petram.solver.krylov import KrylovModel, KrylovSmoother
        return KrylovSmoother, MUMPS


    def get_possible_child_menu(self):
        from petram.solver.mumps_model import MUMPS
        from petram.solver.krylov import KrylovModel, KrylovSmoother
        choice = [("Blocks", KrylovSmoother),
                  ("!", MUMPS),]
        return choice
    

class DiagonalSmoother(BlockSmoother):
    @classmethod
    def fancy_menu_name(self):
        return 'DiagonalSmoother'

    @classmethod
    def fancy_tree_name(self):
        return 'DiagonalSmoother'

class DiagonalLinearSolver(LinearSolver):
    is_iterative = True
    

    
