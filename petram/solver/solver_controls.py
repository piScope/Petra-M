from petram.namespace_mixin import NS_mixin
from .solver_model import SolverBase

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('SolveControl')
format_memory_usage = debug.format_memory_usage

class SolveControl(SolverBase):
    has_2nd_panel = False
    
def def_start_cond(count, max_count,  *args, **kwargs):
    return False

def def_stop_cond(count, max_count, *args, **kwargs):
    return count >= max_count

def def_continue_stop():
    return False

def def_break_stop():
    return False

from petram.namespace_mixin import NS_mixin
class ForLoop(SolveControl, NS_mixin):
    def __init__(self, *args, **kwargs):
        SolveControl.__init__(self, *args, **kwargs)
        NS_mixin.__init__(self, *args, **kwargs)
    
    def attribute_set(self, v):
        v['phys_model'] = ''
        v['init_setting'] = ''
        v['postprocess_sol'] = ''
        v['use_dwc_pp'] = False
        v['dwc_pp_arg'] = ''

        v['max_count'] = 1
        v['start_cond_func'] = ''        
        v['stop_cond_func'] = ''
        super(ForLoop, self).attribute_set(v)
        return v

    def get_possible_child(self):
        from petram.solver.solver_model import SolveStep
        from petram.solver.parametric import Parametric        
        return [SolveStep, Parametric, Break, Continue]

    def panel1_param(self):
        return [#["Initial value setting", self.init_setting, 0, {},],
                ["Postporcess solution", self.postprocess_sol, 0, {},],
                ["Max loop count'", self.max_count, 400, {},],
                ["Start cond.", self.start_cond_func, 0, {},],
                ["Stop cond.", self.stop_cond_func, 0, {},],]

    def get_panel1_value(self):
        return (#self.init_setting,
                self.postprocess_sol,
                self.max_count,
                self.start_cond_func,
                self.stop_cond_func)

    def import_panel1_value(self, v):
        #self.init_setting = v[0]
        self.postprocess_sol = v[0]
        self.max_count = int(v[1])
        self.start_cond_func = v[2]
        self.stop_cond_func = v[3]

    def get_all_phys_range(self):
        steps = self.get_active_steps()
        ret = sum([s.get_phys_range() for s in steps], [])
        return list(set(ret))

    def get_active_steps(self):
        steps = []
        for x in self.iter_enabled():
            if len(list(x.iter_enabled())) > 0:
                steps.append(x)
        
        return steps

    def get_pp_setting(self):
        names = self.postprocess_sol.split(',')
        names = [n.strip() for n in names if n.strip() != '']        
        return [self.root()['PostProcess'][n] for n in names]
    
    def run(self, engine, is_first = True):
        dprint1("!!!!! Entering SolveLoop :" + self.name() + " !!!!!")

        steps = self.get_active_steps()

        start_func = def_start_cond
        stop_func = def_stop_cond

        if self.start_cond_func in self._global_ns:
             start_func = self._global_ns[self.start_cond_func]
        if self.stop_cond_func in self._global_ns:
             stop_func = self._global_ns[self.stop_cond_func]

        for i in range(self.max_count):
            dprint1("!!!!! SolveLoop : Count = " + str(i))
            if start_func(i, self.max_count): break
            for s in steps:
                do_break = False
                do_continue = False
                if isinstance(s, Break):
                    do_break = s.run(engine, i)
                elif isinstance(s, Continue):
                    do_continue = s.run(engine, i)
                else:
                    s.run(engine, is_first=is_first)
                if s.solve_error[0]:
                    dprint1("Loop failed " + s.name() + ":"  + s.solve_error[1])
                    break
                is_first = False
                if do_break or do_continue:
                    break
            if stop_func(i, self.max_count):
                break
            if do_break:
                break

        postprocess = self.get_pp_setting()
        engine.run_postprocess(postprocess, name=self.name())

        if self.use_dwc_pp:
            engine.call_dwc(self.get_all_phys_range(),
                            method="postprocess",
                            callername=self.name(),
                            args=self.dwc_pp_arg)

class Break(SolveControl, NS_mixin):
    def __init__(self, *args, **kwargs):
        SolveControl.__init__(self, *args, **kwargs)
        NS_mixin.__init__(self, *args, **kwargs)
    
    def attribute_set(self, v):
        v['break_cond_func'] = ''
        v['use_dwc'] = False
        v['dwc_name'] = ''
        v['dwc_args'] = ''                
        return super(Break, self).attribute_set(v)
    
    def panel1_param(self):
        ret = [["dwc",   self.dwc_name,   0, {},],
               "args.",   self.dwc_arg,   0, {},]
        value = [self.dwc_name, self.dwc_arg]
        return [["Break cond.",    self.break_cond_func,  0, {},],
                [None, [False, value], 27, [{'text':'Use DWC (loopcontrol)'},
                                              {'elp': ret}]],]
    
    def import_panel1_value(self, v):
        self.break_cond_func = v[0]
        self.use_dwc = v[1][0]
        self.dwc_name = v[1][1][0]
        self.dwc_arg = v[1][1][1]

    def get_panel1_value(self):
        return (self.break_cond_func,
                [self.use_dwc, [self.dwc_name, self.dwc_arg,]])

    def get_all_phys_range(self):        
        return self.parent().get_all_phys_range()

    def run(self, engine, count):
        break_func = def_break_cond

        if self.use_dwc:
            return engine.call_dwc(self.get_all_phys_range(),
                                   method="loopcontrol",
                                   callername=self.name(),
                                   dwcname=self.dwc_name,
                                   args=self.dwc_arg,)
        else:
            if self.break_cond_func in self._global_ns:
                 break_func = self._global_ns[self.break_cond_func]
            g = self._global_ns
            code = 'break_func(count)'
            return exec(code, g, {})
  
class Continue(SolveControl, NS_mixin):
    def __init__(self, *args, **kwargs):
        SolveControl.__init__(self, *args, **kwargs)
        NS_mixin.__init__(self, *args, **kwargs)
    
    def attribute_set(self, v):
        v['continue_cond'] = ''
        v['use_dwc'] = False
        v['dwc_name'] = ''
        v['dwc_args'] = ''                
        return super(Continue, self).attribute_set(v)

    def panel1_param(self):
        ret = [["dwc",   self.dwc_name,   0, {},],
               "args.",   self.dwc_arg,   0, {},]
        value = [self.dwc_name, self.dwc_arg]
        return [["Continue cond.",    self.continue_cond_func,  0, {},],
                [None, [False, value], 27, [{'text':'Use DWC (loopcontrol)'},
                                              {'elp': ret}]],]

    def import_panel1_value(self, v):
        self.continue_cond_func = v[0]
        self.use_dwc = v[1][0]
        self.dwc_name = v[1][1][0]
        self.dwc_arg = v[1][1][1]

    def get_panel1_value(self):
        return (self.continue_cond_func,
                [self.use_dwc, [self.dwc_name, self.dwc_arg,]])

    def get_all_phys_range(self):
        return self.parent().get_all_phys_range()

    def run(self, engine, count):
        c_func = def_continue_cond

        if self.use_dwc:
            return engine.call_dwc(self.get_all_phys_range(),
                                   method="loopcontrol",
                                   callername=self.name(),
                                   dwcname=self.dwc_name,
                                   args=self.dwc_arg,)
        else:
            if self.break_cond_func in self._global_ns:
                 c_func = self._global_ns[self.continue_cond_func]
            g = self._global_ns
            code = 'c_func(count)'
            return exec(code, g, {})


