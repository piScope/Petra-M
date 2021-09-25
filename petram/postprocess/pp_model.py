import traceback
from petram.namespace_mixin import NSRef_mixin

import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('PP_Mode')

from petram.helper.variables import var_g
ll = var_g.copy()

from petram.model import Model
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

    def run(self, engine):
        for mm in self.walk():
            if not mm.enabled: continue            
            mm.run_postprocess(engine)


