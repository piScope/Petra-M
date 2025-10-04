import os
import numpy as np

from petram.model import Model
from petram.solver.solver_model import Solver
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('SetVar')
rprint = debug.regular_print('StdVar')

class SetVar(Solver):
    has_2nd_panel = False        
    def panel1_param(self):
        return [["physics model",   self.phys_model,  0, {},],
                ["Initial value setting",   self.init_setting,   0, {},],]

    def import_panel1_value(self, v):
        self.phys_model = str(v[0])        
        self.init_setting = str(v[1])
        
    def attribute_set(self, v):
        v['init_setting']   = ''
        super(SetVar, self).attribute_set(v)
        return v

    def get_panel1_value(self):
        return (self.phys_model, self.init_setting, )
    
    def get_init_setting(self):
        names = self.init_setting.split(',')
        names = [n.strip() for n in names if n.strip() != '']        
        return [self.root()['InitialValue'][n] for n in names]

    def get_matrix_weight(self, timestep_config):#, timestep_weight):
        return [0, 0, 0]

    def run(self, engine, is_first = True):

        self.init_only = True
        
        from petram.solver.solver_model import SolverInstance
        instance = SolverInstance(self, engine)
        
        phys_range = self.get_phys_range()                
        inits = self.get_init_setting()
        dprint1("Setting variable :" + self.name(), inits)        
        engine.run_apply_init(phys_range, inits=inits)

        phys_target = self.get_phys()        
        engine.save_sol_to_file(phys_target, 
                                skip_mesh = False,
                                mesh_only = False,
                                save_parmesh = self.save_parmesh)
        

