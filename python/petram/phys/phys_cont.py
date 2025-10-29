'''
  continuity is a condition for internail
  boundary. 
  It has no contribution to weak form. This should be
  fine as far as mesh is conforming.
'''
from petram.model import Domain, Bdry, Pair
from petram.phys.phys_model  import Phys

class PhysContinuity(Bdry, Phys):
    is_essential = False
    def __init__(self, **kwargs):
        super(PhysContinuity, self).__init__( **kwargs)
        Phys.__init__(self)
        self.sel_readonly = False
        self.sel_index = []
