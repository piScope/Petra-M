from petram.phys.vtable import VtableElement, Vtable, Vtable_mixin
from petram.namespace_mixin import NS_mixin
from .solver_model import SolverBase

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('SolveControl')
format_memory_usage = debug.format_memory_usage


class SolveControl(SolverBase):
    has_2nd_panel = False


data = [("max_count", VtableElement("max_count",
                                    type='int',
                                    guilabel="Max count",
                                    default=3,
                                    tip="parameter range",))]


class ForLoop(SolveControl, NS_mixin, Vtable_mixin):
    vt_loop = Vtable(data)

    def __init__(self, *args, **kwargs):
        SolveControl.__init__(self, *args, **kwargs)
        NS_mixin.__init__(self, *args, **kwargs)

    def attribute_set(self, v):
        v['phys_model'] = ''
        v['init_setting'] = ''
        v['postprocess_sol'] = ''
        v['use_dwc_pp'] = False
        v['dwc_pp_arg'] = ''
        v['counter_name'] = 'loop_counter'
        self.vt_loop.attribute_set(v)

        super(ForLoop, self).attribute_set(v)
        return v

    def get_possible_child(self):
        from petram.solver.solver_model import SolveStep
        from petram.solver.parametric import Parametric
        return [SolveStep, Parametric, Break, Continue]

    def panel1_param(self):
        panels = self.vt_loop.panel_param(self)
        return [  # ["Initial value setting", self.init_setting, 0, {},],
            ["Postporcess solution", self.postprocess_sol, 0, {}, ],
            ["Counter name", self.counter_name, 0, {}, ]] + panels

    def get_panel1_value(self):
        val = self.vt_loop.get_panel_value(self)

        return (  # self.init_setting,
            self.postprocess_sol,
            self.counter_name,
            val[0])

    def import_panel1_value(self, v):
        #self.init_setting = v[0]
        self.postprocess_sol = v[0]
        self.counter_name = v[1]
        self.vt_loop.import_panel_value(self, (v[2],))

    def get_all_phys_range(self):
        steps = self.get_active_steps()
        ret = sum([s.get_phys_range() for s in steps], [])
        return list(set(ret))

    def get_active_steps(self):
        steps = []
        for x in self.iter_enabled():
            if not x.enabled:
                continue

            if isinstance(x, Break):
                steps.append(x)
            elif isinstance(x, Continue):
                steps.append(x)
            elif len(list(x.iter_enabled())) > 0:
                steps.append(x)

        return steps

    def get_pp_setting(self):
        names = self.postprocess_sol.split(',')
        names = [n.strip() for n in names if n.strip() != '']
        return [self.root()['PostProcess'][n] for n in names]

    def run(self, engine, is_first=True):
        dprint1("!!!!! Entering SolveLoop :" + self.name() + " !!!!!")

        steps = self.get_active_steps()
        self.vt_loop.preprocess_params(self)
        max_count = self.vt_loop.make_value_or_expression(self)[0]

        for i in range(max_count):
            dprint1("!!!!! SolveLoop : Count = " + str(i))
            g = self._global_ns[self.counter_name] = i
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
                        dprint1(
                            "Loop failed " +
                            s.name() +
                            ":" +
                            s.solve_error[1])
                        break
                    is_first = False
                print("do_break", do_break)
                if do_break or do_continue:
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
        v['break_cond'] = ''
        v['use_dwc'] = False
        v['dwc_name'] = ''
        v['dwc_args'] = ''
        return super(Break, self).attribute_set(v)

    def panel1_param(self):
        ret = [["dwc", self.dwc_name, 0, {}, ],
               ["args.", self.dwc_args, 0, {}, ], ]
        value = [self.dwc_name, self.dwc_args]
        return [["Break cond.", self.break_cond, 0, {}, ],
                [None, [False, value], 27, [{'text': 'Use DWC (loopcontrol)'},
                                            {'elp': ret}]], ]

    def import_panel1_value(self, v):
        self.break_cond = v[0]
        self.use_dwc = v[1][0]
        self.dwc_name = v[1][1][0]
        self.dwc_args = v[1][1][1]

    def get_panel1_value(self):
        return (self.break_cond,
                [self.use_dwc, [self.dwc_name, self.dwc_args, ]])

    def get_all_phys_range(self):
        return self.parent().get_all_phys_range()

    def run(self, engine, count):
        print("Checking break condition")
        print(self._global_ns.keys())

        if self.use_dwc:
            return engine.call_dwc(self.get_all_phys_range(),
                                   method="loopcontrol",
                                   callername=self.name(),
                                   dwcname=self.dwc_name,
                                   args=self.dwc_args,)
        else:
            if self.break_cond in self._global_ns:
                break_func = self._global_ns[self.break_cond]
            else:
                assert False, self.break_cond + " is not defined"
            g = self._global_ns
            code = "check =" + self.break_cond + '(count)'
            ll = {'count': count}
            exec(code, g, ll)
            print(ll)
            return ll['check']


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
        ret = [["dwc", self.dwc_name, 0, {}, ],
               ["args.", self.dwc_args, 0, {}, ], ]
        value = [self.dwc_name, self.dwc_arg]
        return [["Continue cond.", self.continue_cond, 0, {}, ],
                [None, [False, value], 27, [{'text': 'Use DWC (loopcontrol)'},
                                            {'elp': ret}]], ]

    def import_panel1_value(self, v):
        self.continue_cond = v[0]
        self.use_dwc = v[1][0]
        self.dwc_name = v[1][1][0]
        self.dwc_arg = v[1][1][1]

    def get_panel1_value(self):
        return (self.continue_cond,
                [self.use_dwc, [self.dwc_name, self.dwc_args, ]])

    def get_all_phys_range(self):
        return self.parent().get_all_phys_range()

    def run(self, engine, count):
        if self.use_dwc:
            return engine.call_dwc(self.get_all_phys_range(),
                                   method="loopcontrol",
                                   callername=self.name(),
                                   dwcname=self.dwc_name,
                                   args=self.dwc_args,)
        else:
            if self.continue_cond in self._global_ns:
                c_func = self._global_ns[self.continue_cond]
            else:
                assert False, self.continue_cond + " is not defined"

            g = self._global_ns
            code = "check=" + self.continue_cond + '(count)'
            ll = {'count': count}
            exec(code, g, ll)
            print(ll)
            return ll['check']
