import os
import traceback
import gc

from petram.model import Model
from petram.solver.solver_model import Solver, SolveStep
from petram.namespace_mixin import NS_mixin
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('Parametric')
format_memory_usage = debug.format_memory_usage

assembly_methods = {'Full assemble': 0,
                    'Reuse matrix' : 1}

def def_start_cond(count, max_count,  *args, **kwargs):
    return False

def def_stop_cond(count, max_count, *args, **kwargs):
    return count >= max_count
    
class Loop(SolveStep, NS_mixin):
    has_2nd_panel = False
    
    def __init__(self, *args, **kwargs):
        SolveStep.__init__(self, *args, **kwargs)
        NS_mixin.__init__(self, *args, **kwargs)
    
    def attribute_set(self, v):
        v['phys_model']   = ''
        v['init_setting']   = ''
        v['postprocess_sol']   = ''        
        v['use_dwc_pp']   = False
        v['dwc_pp_arg']   = ''

        v['max_count'] = 1
        v['start_cond_func'] = ''        
        v['stop_cond_func'] = ''
        super(Loop, self).attribute_set(v)
        return v
    
    def panel1_param(self):

        ret = super(Loop, self).panel1_param()
        ret = ret +[["Max loop count'",    self.max_count,   400, {},],
                    ["Start cond.",    self.start_cond_func,  0, {},],        
                    ["Stop cond.",     self.stop_cond_func,   0, {},],]
                   
        return ret

    def get_panel1_value(self):
        ret = list(super(Loop, self).get_panel1_value())

        ret = ret + [self.max_count,
                     self.start_cond_func,
                     self.stop_cond_func]
        
        return ret

    def import_panel1_value(self, v):
        super(Loop, self).import_panel1_value(v[:-3])
        self.max_count = int(v[-3])
        self.start_cond_func = v[-2]
        self.stop_cond_func = v[-1]

    def run(self, engine, is_first = True):
        dprint1("Entering SolveStep :" + self.name())
        solvers = self.get_active_solvers()

        # initialize and assemble here

        # in run method..
        #   std solver : make sub block matrix and solve
        #   time-domain solver : do step

        start_func = def_start_cond
        stop_func = def_stop_cond

        if self.start_cond_func in self._global_ns:
             start_func = self._global_ns[self.start_cond_func]
        if self.stop_cond_func in self._global_ns:
             stop_func = self._global_ns[self.stop_cond_func]
                                         
        for i in range(self.max_count):
            self.prepare_form_sol_variables(engine)
            self.init(engine)

            is_first = True            
            if start_func(i, self.max_count): break
            for solver in solvers:
                 is_first = solver.run(engine, is_first=is_first)
                 engine.add_FESvariable_to_NS(self.get_phys()) 
                 engine.store_x()
                 if self.solve_error[0]:
                     dprint1("Loop failed " + self.name() + ":"  + self.solve_error[1])
                     break
            if stop_func(i, self.max_count): break
             
        postprocess = self.get_pp_setting()
        engine.run_postprocess(postprocess, name = self.name())
        
        if self.use_dwc_pp:
            engine.call_dwc(self.get_phys_range(),
                            method="postprocess",
                            callername = self.name(),
                            args = self.dwc_pp_arg)
    
