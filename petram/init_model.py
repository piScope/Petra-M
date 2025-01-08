import traceback
from petram.model import Model
from petram.phys.vtable import VtableElement, Vtable, Vtable_mixin

data = [("init_value", VtableElement("init_value", type='array',
                                     guilabel="expression", default=0.0, tip="expression",))]


class InitSetting(Model, Vtable_mixin):
    can_rename = True
    has_2nd_panel = False
    vt_coeff = Vtable(data)

    @property
    def _global_ns(self):
        # used for text box validator
        return self.root()['General']._global_ns

    def attribute_set(self, v):
        v = super(InitSetting, self).attribute_set(v)
        v["phys_model"] = ''
        v["init_var"] = ''
        v["init_mode"] = 0
        #v["init_value_txt"]    = '0.0'
        v["init_path"] = ''
        v["init_dwc_name_params"] = ('', '')
        self.vt_coeff.attribute_set(v)
        return v

    def panel1_param(self):
        from petram.pi.widget_init import InitSettingPanel
        return [["physics model",   self.phys_model,  0, {}, ],
                ["variable name (\"\"=all)",   self.init_var,  0, {}, ],
                [None, None, 99, {'UI': InitSettingPanel, 'validator': self.check_phys_expr_array}], ]

    def _init_eval(self, value):
        gg = self.root()['General']._global_ns.copy()

        from petram.helper.variables import var_g
        ll = var_g.copy()

        return eval(value, gg, ll)

    # def init_validator(self, value, param, ctrl):
    #    try:
    #        x = self._init_eval(value)
    #    except:
    #        import traceback
    #        traceback.print_exc()
    #        return False
    #    return True

    def get_panel1_value(self):
        init_value_txt = self.vt_coeff.get_panel_value(self)[0]
        return [self.phys_model, self.init_var,
                (self.init_mode, init_value_txt, self.init_path, self.init_dwc_name_params), ]

    def import_panel1_value(self, v):
        self.phys_model = str(v[0])
        self.init_var = str(v[1])
        self.init_mode = v[2][0]
        #self.init_value_txt = v[1][1]
        self.init_path = v[2][2]
        self.init_dwc_name_params = v[2][3]
        self.vt_coeff.import_panel_value(self, (v[2][1],))

    def preprocess_params(self, engine):
        from petram.helper.init_helper import eval_value
        try:
            self.vt_coeff.preprocess_params(self)
            self.init_value = self.vt_coeff.make_value_or_expression(self)[0]
        except:
            self.init_value = 0.0
            assert False, traceback.format_exc()

    def get_phys(self):
        names = self.phys_model.split(',')
        names = [n.strip() for n in names if n.strip() != '']
        return [self.root()['Phys'][n] for n in names]

    def run(self, engine):
        phys_targets = self.get_phys()
        dwcparams = (self.name(),
                     self.init_dwc_name_params[0],
                     self.init_dwc_name_params[1])
        engine.run_apply_init0(phys_targets, self.init_mode,
                               init_value=self.init_value,
                               init_path=self.init_path,
                               init_var=self.init_var,
                               init_dwc=dwcparams)
        return phys_targets

    def onItemSelChanged(self, evt):
        '''
        GUI response when model object is selected in
        the dlg_edit_model
        '''
        viewer = evt.GetEventObject().GetTopLevelParent().GetParent()
        viewer.set_view_mode('phys')


class CustomInitSetting():
    '''
    initialization w/o GUI
    usually called from Solver::get_custom_init()

    '''

    def __init__(self, phys_targets, var="", mode=1, value=[0, ], path='', dwc=('', ''), name='CustomInit'):
        self.init_name = name
        self.phys_targets = phys_targets
        self.init_mode = mode
        self.init_value = value
        self.init_path = path
        self.init_dwc_name_params = dwc
        self.init_var = var

    def name(self):
        return self.init_name

    def run(self, engine):
        dwcparams = (self.name,
                     self.init_dwc_name_params[0],
                     self.init_dwc_name_params[1])

        engine.run_apply_init0(self.phys_targets,
                               mode=self.init_mode,
                               init_value=self.init_value,
                               init_path=self.init_path,
                               init_var=self.init_var,
                               init_dwc=dwcparams)

        return self.phys_targets
