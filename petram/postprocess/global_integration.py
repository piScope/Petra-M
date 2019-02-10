import traceback

import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('Integration(PP)')


from petram.postprocess.pp_model import PostProcessBase


from petram.helper.variables import var_g
ll = var_g.copy()

class Integration(PostProcessBase):
    pass

class LinearformIntg(PostProcessBase):
    pass
    
class BilinierformIntg(PostProcessBase):
    pass
