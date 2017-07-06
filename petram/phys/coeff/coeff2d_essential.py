from petram.model import Domain, Bdry, Pair
from petram.phys.phys_model  import Phys

class Coeff2D_Zero(Bdry, Phys):
    is_essential = True
    def __init__(self, **kwargs):
        super(Coeff2D_Zero, self).__init__( **kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(Coeff2D_Zero, self).attribute_set(v)        
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v
        
    def panel1_param(self):
        return [['Zero valued boundary',   "u = 0",  2, {}],]

    def get_panel1_value(self):
        return None

    def import_panel1_value(self, v):
        pass
    
    def apply_essential(self, engine, gf, real = False,
                        **kwargs):
        pass
        
