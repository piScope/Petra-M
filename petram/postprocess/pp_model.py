from petram.model import Model
from petram.helper.variables import var_g
import traceback
from petram.namespace_mixin import NSRef_mixin

import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('PP_Model')

ll = var_g.copy()


class PostProcessBase(Model):
    @property
    def _global_ns(self):
        # used for text box validator
        p = self
        while True:
            if isinstance(p, NSRef_mixin):
                break
            p = p.parent
            if p is None:
                # it should not come here...x
                return {}
        return p.find_ns_by_name()

    def run_postprocess(self, engin):
        raise NotImplemented("Subclass must implement run_postprocess")

    def onItemSelChanged(self, evt):
        '''
        GUI response when model object is selected in
        the dlg_edit_model
        '''
        viewer = evt.GetEventObject().GetTopLevelParent().GetParent()
        viewer.set_view_mode('phys')

    def soldict_to_solvars(self, soldict, variables):
        pass

    def update_dom_selection(self, all_sel=None):
        from petram.model import convert_sel_txt
        try:
            arr = convert_sel_txt(self.sel_index_txt, self._global_ns)
            self.sel_index = arr
        except:
            assert False, "failed to convert "+self.sel_index_txt

        if all_sel is None:
            # clinet GUI panel operation ends here
            return

        allv, alls, alle = all_sel
        if len(self.sel_index) != 0 and self.sel_index[0] == 'all':
            if self.sdim == 3:
                self.sel_index = allv
            if self.sdim == 2:
                self.sel_index = alls
            if self.sdim == 1:
                self.sel_index = alle


class PostProcess(PostProcessBase, NSRef_mixin):
    has_2nd_panel = False

    def __init__(self, *args, **kwargs):
        super(PostProcess, self).__init__(*args, **kwargs)
        NSRef_mixin.__init__(self, *args, **kwargs)

    def attribute_set(self, v):
        v = super(PostProcess, self).attribute_set(v)
        v['use_scanner'] = 0
        v['scanner'] = 'Scan("a", [1,2,3])'
        return v

    def panel1_param(self):
        ret = [self.make_param_panel('scanner',  self.scanner), ]
        value = [self.scanner, ]
        return [[None, [False, value], 27, [{'text': 'Use parametric scan'},
                                            {'elp': ret}]], ]

    def get_panel1_value(self):
        v1 = (self.use_scanner, [self.scanner, ])
        return [v1, ]

    def import_panel1_value(self, v):
        v1 = v[0]
        self.use_scanner = v1[0]
        self.scanner = v1[1][0]

    def get_info_str(self):
        txt = []
        if NSRef_mixin.get_info_str(self) != "":
            txt.append(NSRef_mixin.get_info_str(self))
        return ",".join(txt)

    def get_possible_child(self):
        from petram.postprocess.project_solution import DerivedValue
        from petram.postprocess.discrt_v_integration import (LinearformIntegrator,
                                                             BilinearformIntegrator)
        from petram.postprocess.discrt_v_interpolator import Grad, Curl, Div

        return [DerivedValue, LinearformIntegrator, BilinearformIntegrator, Grad, Curl, Div]

    def get_possible_child_menu(self):
        from petram.postprocess.project_solution import DerivedValue
        from petram.postprocess.discrt_v_integration import (LinearformIntegrator,
                                                             BilinearformIntegrator)
        from petram.postprocess.discrt_v_interpolator import Grad, Curl, Div

        return [("", DerivedValue),
                ("Integrator", LinearformIntegrator),
                ("!", BilinearformIntegrator),
                ("Derivative", Grad),
                ("", Curl),
                ("!", Div),
                ]

    def run_postprocess(self, engine):
        dprint1("running postprocess:" + self.name())

    def get_scanner(self, nosave=False):
        try:
            scanner = self.eval_param_expr(str(self.scanner),
                                           'scanner')[0]
            scanner.set_data_from_model(self.root())
        except:
            traceback.print_exc()
            return
        scanner.set_phys_models([self.find_nsobj_by_name(), ])
        return scanner

    def run(self, engine):

        scanner = self.get_scanner() if self.use_scanner else None
        for mm in self.walk():
            if not mm.enabled:
                continue
            if mm is self:
                continue
            if scanner is not None:
                engine.ppname_postfix = ''
                for kcase, case in enumerate(scanner):
                    engine.ppname_postfix = '_'+str(kcase)
                    mm.run_postprocess(engine)
                engine.ppname_postfix = ''
            else:
                engine.ppname_postfix = ''
                mm.run_postprocess(engine)

    # parameters with validator
    def check_param_expr(self, value, param, ctrl):
        try:
            self.eval_param_expr(str(value), param)
            return True
        except:
            import petram.debug
            import traceback
            if petram.debug.debug_default_level > 2:
                traceback.print_exc()
            return False

    def eval_param_expr(self, value, param):
        x = eval(value, self._global_ns)
        dprint2('Value Evaluation ', param, '=', x)
        return x, None

    # note that physics modules overwrite this with more capablie version
    def make_param_panel(self, base_name, value):
        return [base_name + "(=)",  value, 0,
                {'validator': self.check_param_expr,
                 'validator_param': base_name}]
