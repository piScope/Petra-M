#
# utlitiy class to allocate chuck of memory to set of gf
#

import numpy as np
from petram.mfem_config import use_parallel
if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

class GF_allocator():
    def __init__(self):
        self.fes = []
        self.gf = []
        self._memory = None

    def register(self, fes, gf):
        self.fes.append(fes)
        self.gf.append(gf)

    def allocate(self):
        offsets = [0]*(len(self.gf)+1)

        offsets[0] = 0
        for i, fes in enumerate(self.fes):
            offsets[i+1] = fes.GetVSize()

        offsets = np.cumsum(offsets)
        x = mfem.Vector(offsets[-1])
        x.Assign(0.0)

        for i in range(len(self.gf)):
            self.gf[i].MakeRef(self.fes[i], x, offsets[i])

        self._memory = x

