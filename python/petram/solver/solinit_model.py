import numpy as np

from petram.model import Model
from .solver_model import Solver
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('StdSolver')
rprint = debug.regular_print('StdSolver')

class SolInit(Solver):
    can_delete = True
    has_2nd_panel = False
    mustbe_firstchild =True        

    def attribute_set(self, v):
        v['phys_model']   = ''
        super(SolInit, self).attribute_set(v)
        return v
    
    def panel1_param(self):
        return [["physics model",   self.phys_model,  0, {},],]

    def get_panel1_value(self):
        return (self.phys_model,)
    
    def import_panel1_value(self, v):
        self.phys_model = str(v[0])

    def get_editor_menus(self):
        return []

    def get_phys(self):
        names = self.phys_model.split(',')
        names = [n.strip() for n in names]
        return [self.root()['Phys'][n] for n in names]

    def init_sol(self, engine):
        phys_targets = self.get_phys()
        engine.run_init_sol(phys_targets)
        return

    def run(self, engine):
        phys_target = self.get_phys()
        self.init_sol(engine)

