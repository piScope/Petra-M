import traceback

from petram.model import Model
class InitSetting(Model):
    has_2nd_panel = False            
    def attribute_set(self, v):
        v = super(InitSetting, self).attribute_set(v)
        v["phys_model"] = ''
        v["init_mode"]    = 0
        v["init_value_txt"]    = '0.0'
        v["init_path"]    = ''
        return v

    def panel1_param(self):
        from petram.pi.widget_init import InitSettingPanel
        return [["physics model",   self.phys_model,  0, {},],
                [None, None, 99, {'UI':InitSettingPanel}],]

      
    def get_panel1_value(self):
        return [self.phys_model,
                (self.init_mode, self.init_value_txt, self.init_path)]
    
    def import_panel1_value(self, v):
        self.phys_model = str(v[0])
        self.init_mode = v[1][0]
        self.init_value_txt = v[1][1]
        self.init_path = v[1][2]

    def preprocess_params(self, engine):
        from petram.helper.init_helper import eval_value
        try:
            self.init_value = eval_value(self.init_value_txt)
        except:
            self.init_value = 0.0            
            assert False, traceback.format_exc()

    def get_phys(self):
        names = self.phys_model.split(',')
        names = [n.strip() for n in names if n.strip() != '']        
        return [self.root()['Phys'][n] for n in names]
            
    def run(self, engine):
        phys_targets = self.get_phys()
        engine.run_alloc_sol(phys_targets)
        engine.run_apply_init(phys_targets, self.init_mode,
                             self.init_value, self.init_path)

    def onItemSelChanged(self, evt):
        '''
        GUI response when model object is selected in
        the dlg_edit_model
        '''
        viewer = evt.GetEventObject().GetTopLevelParent().GetParent()
        viewer.set_view_mode('phys')                                        
        
        
    
