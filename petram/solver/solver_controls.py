from petram.phys.vtable import VtableElement, Vtable, Vtable_mixin
from petram.namespace_mixin import NS_mixin
from .solver_model import SolverBase

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('SolveControl')
format_memory_usage = debug.format_memory_usage


class SolveControl(SolverBase):
    has_2nd_panel = False

    def get_phys(self):
        my_solve_step = self.get_solve_root()
        return my_solve_step.get_phys()

    def get_phys_range(self):
        my_solve_step = self.get_solve_root()
        return my_solve_step.get_phys_range()


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
        v['dwc_name'] = ''
        v['use_dwc'] = False
        v['dwc_args'] = ''
        v['counter_name'] = 'loop_counter'
        self.vt_loop.attribute_set(v)

        super(ForLoop, self).attribute_set(v)
        return v

    def get_possible_child(self):
        from petram.solver.solver_model import SolveStep
        from petram.solver.parametric import Parametric
        return [SolveStep, Parametric, Break, Continue, DWCCall]

    def panel1_param(self):
        panels = self.vt_loop.panel_param(self)

        ret1 = [["dwc", self.dwc_name, 0, {}, ],
                ["args.", self.dwc_args, 0, {}, ], ]
        value1 = [self.dwc_name, self.dwc_args]
        panel2 = [[None, [False, value1, ], 27, [{'text': 'use DWC (postprocess)'},
                                                 {'elp': ret1}, ]], ]

        return ([["Postprocess solution", self.postprocess_sol, 0, {}, ],
                 ["Counter name", self.counter_name, 0, {}, ]] + panels + panel2)

    def get_panel1_value(self):
        val = self.vt_loop.get_panel_value(self)

        return (  # self.init_setting,
            self.postprocess_sol,
            self.counter_name,
            val[0],
            [self.use_dwc, [self.dwc_name, self.dwc_args]])

    def import_panel1_value(self, v):
        #self.init_setting = v[0]
        self.postprocess_sol = v[0]
        self.counter_name = v[1]
        self.vt_loop.import_panel_value(self, (v[2],))
        self.use_dwc = v[3][0]
        self.dwc_name = v[3][1][0]
        self.dwc_args = v[3][1][1]

    def get_all_phys_range(self):
        steps = self.get_active_steps()
        ret0 = sum([s.get_phys_range() for s in steps], [])

        ret = []
        for x in ret0:
            if not x in ret:
                ret.append(x)

        return ret

    def get_active_steps(self, with_control=False):
        steps = []
        for x in self.iter_enabled():
            if not x.enabled:
                continue

            if isinstance(x, Break) and with_control:
                steps.append(x)
            elif isinstance(x, Continue) and with_control:
                steps.append(x)
            elif isinstance(x, DWCCall) and with_control:
                steps.append(x)
            elif len(list(x.iter_enabled())) > 0:
                steps.append(x)

        return steps

    def get_pp_setting(self):
        names = self.postprocess_sol.split(',')
        names = [n.strip() for n in names if n.strip() != '']
        return [self.root()['PostProcess'][n] for n in names]

    def iter_active_solvers(self, with_control=False):
        steps = self.get_active_steps(with_control=False)
        for s in steps:
            for s2 in s.get_active_solvers():
                yield s2

    def get_target_phys(self):
        ret = []
        for s in self.iter_active_solvers():
            for item in s.get_target_phys():
                if item not in ret:
                    ret.append(item)
        return ret

    def get_child_solver(self):
        ret = []
        for s in self.iter_active_solvers():
            ret.extend(s.get_child_solver())
        return ret

    def get_custom_init(self):
        ret = []
        for s in self.iter_active_solvers():
            ret.extend(s.get_custom_init())
        return ret

    def get_matrix_weight(self, timestep_config):  # , timestep_weight):
        ret = []
        s1, s2, s3 = 0, 0, 0
        for s in self.iter_active_solvers():
            n1, n2, n3 = s.get_matrix_weight(timestep_config)
            s1 = max(s1, n1)
            s2 = max(s2, n2)
            s3 = max(s3, n3)
        return s1, s2, s3

    def run(self, engine, is_first=True):
        dprint1("!!!!! Entering SolveLoop : (is_first =",
                is_first, ") " + self.name() + " !!!!!")

        steps = self.get_active_steps(with_control=True)
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
                elif isinstance(s, DWCCall):
                    do_continue = s.run(engine, i)
                else:
                    is_first = s.run(engine, is_first=is_first)
                    if s.solve_error[0]:
                        dprint1(
                            "Loop failed " +
                            s.name() +
                            ":" +
                            s.solve_error[1])
                        break
                    is_first = False

                if do_break or do_continue:
                    break
            if do_break:
                break

        postprocess = self.get_pp_setting()
        engine.run_postprocess(postprocess, name=self.name())

        if self.use_dwc:
            engine.call_dwc(self.get_all_phys_range(),
                            method="postprocess",
                            callername=self.name(),
                            dwcname=self.dwc_name,
                            args=self.dwc_args)


class InnerForLoop(ForLoop):
    '''
    InnerForLoop is placed insde SolveStep
    Loop supports only standard stationary solver
    '''

    def get_active_solvers(self, with_control=False):
        steps = []
        for x in self.iter_enabled():
            if not x.enabled:
                continue

            if isinstance(x, Break) and with_control:
                steps.append(x)
            elif isinstance(x, Continue) and with_control:
                steps.append(x)
            elif isinstance(x, DWCCall) and with_control:
                steps.append(x)
            elif len(list(x.iter_enabled())) > 0:
                steps.append(x)

        return steps

    def iter_active_solvers(self):
        solvers = self.get_active_solvers(with_control=False)
        for s in solvers:
            yield s

    def free_instance(self):
        for s in self.iter_active_solvers():
            s.free_instance()

    def get_num_levels(self):
        return 1

    def create_refined_levels(self, engine, lvl):
        '''
        create refined levels and return True if it is created.
        default False (no refined level)
        '''
        return False

    def get_possible_child(self):
        #from solver.solinit_model import SolInit
        from petram.solver.std_solver_model import StdSolver
        from petram.solver.mg_solver_model import MGSolver
        from petram.solver.timedomain_solver_model import TimeDomain
        from petram.solver.set_var import SetVar
        from petram.solver.distance_solver import DistanceSolver

        try:
            from petram.solver.std_meshadapt_solver_model import StdMeshAdaptSolver
            return [StdSolver,
                    TimeDomain,
                    StdMeshAdaptSolver,
                    Break, Continue,
                    DWCCall, SetVar, ]

        except:
            return [StdSolver,
                    TimeDomain,
                    Break, Continue,
                    DWCCall, SetVar]

    def get_possible_child_menu(self):
        #from solver.solinit_model import SolInit
        from petram.solver.std_solver_model import StdSolver
        from petram.solver.mg_solver_model import MGSolver
        from petram.solver.timedomain_solver_model import TimeDomain
        from petram.solver.set_var import SetVar
        from petram.solver.distance_solver import DistanceSolver

        try:
            from petram.solver.std_meshadapt_solver_model import StdMeshAdaptSolver
            return [("", StdSolver),
                    ("", TimeDomain),
                    ("extra", StdMeshAdaptSolver),
                    ("", Break),
                    ("", Continue),
                    ("", DWCCall),
                    ("!", SetVar)]
        except:
            return [("", StdSolver),
                    ("", TimeDomain),
                    ("extra", Break),
                    ("", Continue),
                    ("", DWCCall),
                    ("!", SetVar)]


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
        ret0 = [["Break cond.", self.break_cond, 0, {}, ], ]
        ret1 = [["dwc", self.dwc_name, 0, {}, ],
                ["args.", self.dwc_args, 0, {}, ], ]
        value0 = [self.break_cond]
        value1 = [self.dwc_name, self.dwc_args]
        return [[None, [False, value1, value0], 127, [{'text': 'Use DWC (loopcontrol)'},
                                                      {'elp': ret1},
                                                      {'elp': ret0}]], ]

    def import_panel1_value(self, v):
        self.use_dwc = v[0][0]
        self.dwc_name = v[0][1][0]
        self.dwc_args = v[0][1][1]
        self.break_cond = v[0][2][0]

    def get_panel1_value(self):
        return ([self.use_dwc, [self.dwc_name, self.dwc_args],
                 [self.break_cond], ], )

    def get_all_phys_range(self):
        return self.parent.get_all_phys_range()

    def run(self, engine, count):
        if self.use_dwc:
            return engine.call_dwc(self.get_all_phys_range(),
                                   method="loopcontrol",
                                   callername=self.name(),
                                   dwcname=self.dwc_name,
                                   args=self.dwc_args,
                                   count=count,)
        else:
            if self.break_cond in self._global_ns:
                break_func = self._global_ns[self.break_cond]
            else:
                assert False, self.break_cond + " is not defined"
            g = self._global_ns
            code = "check =" + self.break_cond + '(count)'
            ll = {'count': count}
            exec(code, g, ll)

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
        ret0 = [["Continue cond.", self.continue_cond, 0, {}, ], ]
        ret1 = [["dwc", self.dwc_name, 0, {}, ],
                ["args.", self.dwc_args, 0, {}, ], ]
        value0 = [self.continue_cond]
        value1 = [self.dwc_name, self.dwc_args]
        return [[None, [False, value1, value0], 127, [{'text': 'Use DWC (loopcontrol)'},
                                                      {'elp': ret1},
                                                      {'elp': ret0}]], ]

    def import_panel1_value(self, v):
        self.use_dwc = v[0][0]
        self.dwc_name = v[0][1][0]
        self.dwc_args = v[0][1][1]
        self.continue_cond = v[0][2][0]

    def get_panel1_value(self):
        return ([self.use_dwc, [self.dwc_name, self.dwc_args],
                 [self.continue_cond], ], )

    def get_all_phys_range(self):
        return self.parent.get_all_phys_range()

    def run(self, engine, count):
        if self.use_dwc:
            return engine.call_dwc(self.get_all_phys_range(),
                                   method="loopcontrol",
                                   callername=self.name(),
                                   dwcname=self.dwc_name,
                                   args=self.dwc_args,
                                   count=count)
        else:
            if self.continue_cond in self._global_ns:
                c_func = self._global_ns[self.continue_cond]
            else:
                assert False, self.continue_cond + " is not defined"

            g = self._global_ns
            code = "check=" + self.continue_cond + '(count)'
            ll = {'count': count}
            exec(code, g, ll)

            return ll['check']


class DWCCall(SolveControl):
    '''
    standalone DWC caller
    '''

    def __init__(self, *args, **kwargs):
        SolverBase.__init__(self, *args, **kwargs)

    def attribute_set(self, v):
        v['dwc_args'] = ''
        v['dwc_name'] = ''
        v['dwc_callname'] = 'call'
        super(DWCCall, self).attribute_set(v)
        return v

    def get_possible_child(self):
        return []

    def panel1_param(self):
        panels = [["dwc", self.dwc_name, 0, {}, ],
                  ["method", self.dwc_callname, 0, {}, ],
                  ["args.", self.dwc_args, 0, {}, ], ]

        return panels

    def get_panel1_value(self):
        return [self.dwc_name, self.dwc_callname, self.dwc_args]

    def import_panel1_value(self, v):
        self.dwc_name = v[0]
        self.dwc_callname = v[1]
        self.dwc_args = v[2]

    def get_target_phys(self):
        return []

    def get_child_solver(self):
        return []

    def get_matrix_weight(self, _timestep_config):
        return []

    def get_custom_init(self):
        return []

    def get_all_phys(self):
        phys_root = self.root()['Phys']
        return [x for x in phys_root.iter_enabled()]

    def run(self, engine, is_first=True):
        engine.call_dwc(self.get_all_phys(),
                        method=self.dwc_callname,
                        callername=self.name(),
                        dwcname=self.dwc_name,
                        args=self.dwc_args)
