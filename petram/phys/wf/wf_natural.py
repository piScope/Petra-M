from petram.mfem_config import use_parallel
if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem
    
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('WF_Natrual')

from petram.phys.vtable import VtableElement, Vtable
from petram.model import Domain, Bdry, Edge, Point, Pair
from petram.phys.phys_model import Phys, PhysModule

data =  (('label1', VtableElement(None, 
                                  guilabel = 'Natural BC',
                                  default = "",
                                  tip = "this does not constrain the model" )),)

class WF_Natural(Bdry, Phys):
    has_essential = False
    nlterms = []
    can_timedpendent = False
    has_3rd_panel = True
    vt  = Vtable(data)

    def __init__(self, **kwargs):
        super(WF_Natural, self).__init__(**kwargs)

    def attribute_set(self, v):
        Bdry.attribute_set(self, v)
        Phys.attribute_set(self, v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v

    def get_essential_idx(self, kfes):
        return []



