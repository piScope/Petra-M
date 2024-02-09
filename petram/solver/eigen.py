import os
import numpy as np

from petram.namespace_mixin import NS_mixin
from petram.solver_model import (LinearSolverModel,
                                 LinearSolver)
from petram.solver.std_solver_model import (StdSolver,
                                            StandardSolver)

from petram.model import Model
from petram.solver.solver_model import Solver
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('StdSolver')
rprint = debug.regular_print('StdSolver')


class EgnSolver(StdSolver):
    @classmethod
    def fancy_menu_name(self):
        return 'Eigenmode'

    @classmethod
    def fancy_tree_name(self):
        return 'Eigenmode'
    
    def get_possible_child(self):
        choice = [HypreLOBPCG, HypreAME]
        return choice

class EigenValueSolver(StandardSolver):
    pass


class HypreAME(LinearSolverModel, NS_mixin):
    pass
class HypreLOBPCG(LinearSolverModel, NS_mixin):
    pass
