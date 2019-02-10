'''

   discrete variable integration

     1) integration of discrete varialbe on certain domain/boundary
        in such a way natural to the basis function.
    
        3D domain     H1    volume integration
                      L2    volume integration
                      ND    volume integration of dot prodcuts with coefficients
                      RT    volume integration of dot prodcuts with coefficients

           boundary   H1    area integration
                      L2    (undefined)
                      ND/RT area integration of dot prodcuts with coefficients

        2D domain:    H1/L2 area integration
                      ND/RT area integration of dot prodcuts with coefficients

           boundary:  H1    line integration
                      L2    (undefined)
                      ND    line integration of tangentail component
                      RT    line integration of normal component
        1D domain:    H1    line integration
                      L2    line integration
           boundary:  H1    local value
                      L2    (undefined)


'''
import traceback
import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('Integration(PP)')


from petram.postprocess.pp_model import PostProcessBase


from petram.helper.variables import var_g
ll = var_g.copy()

class DiscrtVIntegration(PostProcessBase):
    @classmethod
    def fancy_menu_name(self):
        return 'Integraion(GF)'
    
    def attribute_set(self, v):
        v = super(DiscrtVIntegration, self).attribute_set(v)
        v["integral_name"] = 'intg'
        v["disct_variable"]   = ""
        v['sel_index'] = ['all']
        v['sel_index_txt'] = 'all'
        v['is_boundary_int'] = False
        return v
    
    def panel1_param(self):
        import wx
        panels  =[["name", "", 0, {}],
                  ["variable", "", 0, {}],
                  ["sel.",     "", 0, {}],                  
                  ["boundary", False, 3, {"text":""}],]                  
        return panels

    def panel1_tip(self):
        return ["name", "discrete variable", "sel.", "boundary"]
     
    def get_panel1_value(self):                
        return [self.integral_name,
                self.disct_variable,
                self.sel_index_txt,
                self.is_boundary_int]
    
    def import_panel1_value(self, v):
        self.integral_name = str(v[0])
        self.disct_variable = str(v[1])
        self.sel_index_txt = str(v[2])
        self.is_boundary_int = bool(v[3])

class LinearformIntg(PostProcessBase):
    pass
    
class BilinierformIntg(PostProcessBase):
    pass
