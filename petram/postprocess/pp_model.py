
import traceback

import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('PP_Mode')

from petram.helper.variables import var_g
ll = var_g.copy()

from petram.model import Model
class PostProcessBase(Model):
    @property
    def _global_ns(self):
        # used for text box validator
        return self.root()['General']._global_ns
    
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
    
class PostProcess(PostProcessBase):
    def get_possible_child(self):
        from petram.postprocess.project_solution import DerivedValue
        from petram.postprocess.discrt_v_integration import LinearformIntegrator, BilinearformIntegrator 
        return [DerivedValue, LinearformIntegrator, BilinearformIntegrator]
    
    def get_possible_child_menu(self):
        from petram.postprocess.project_solution import DerivedValue
        from petram.postprocess.discrt_v_integration import LinearformIntegrator, BilinearformIntegrator
        
        return [("", DerivedValue),
                ("Integrator", LinearformIntegrator),
                ("!", BilinearformIntegrator)
                ]
    
    def run_postprocess(self, engine):
        dprint1("running postprocess:" + self.name())

    def run(self, engine):
        for mm in self.walk():
            if not mm.enabled: continue            
            mm.run_postprocess(engine)


