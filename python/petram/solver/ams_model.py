from .solver_model import Solver
import numpy as np

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('MUMPSModel')

class AMS(Solver):
    has_2nd_panel = False
    accept_complex = False
    def init_solver(self):
        pass
    
    def panel1_param(self):
        return [["log_level",   self.log_level,  400, {}],
                ["max  iter.",  self.maxiter,  300, {}],
                ["rel. tol",    self.reltol,  300,  {}],
                ["abs. tol.",   self.abstol,  300, {}],
                ["restart(kdim)", self.kdim,     400, {}],
                ["preconditioner", self.preconditioner,     0, {}],]    
    
    def get_panel1_value(self):
        return (long(self.log_level), self.maxiter,
                self.reltol, self.abstol, self.kdim,
                self.preconditioner)
    
    def import_panel1_value(self, v):
        self.log_level = long(v[0])
        self.maxiter = v[1]
        self.reltol = v[2]
        self.abstol = v[3]
        self.kdim = v[4]        
        self.preconditioner = v[5]
        
    def attribute_set(self, v):
        v = super(AMS, self).attribute_set(v)
        v['log_level'] = 0
        v['maxiter'] = 200
        v['reltol']  = 1e-7
        v['abstol'] = 1e-7
        v['kdim'] =   50
        v['preconditioner'] = 'AMS'
        return v
    
    def verify_setting(self):
        if not self.parent.assemble_real:
            root = self.root
            phys = root['Phys'][self.parent.phys_model]
            if phys.is_complex: return False, "Complex Problem not supported.", "AMS does not support complex problem"
        return True, "", ""
