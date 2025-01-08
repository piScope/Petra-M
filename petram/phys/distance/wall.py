from mfem.common.mpi_debug import nicePrint
from petram.phys.vtable import VtableElement, Vtable
from petram.mfem_config import use_parallel
import numpy as np

from petram.model import Domain, Bdry, Edge, Point, Pair
from petram.phys.coefficient import SCoeff, VCoeff
from petram.phys.phys_model import Phys, PhysModule

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('DistanceWall')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem


class Distance_Wall(Bdry, Phys):
    has_essential = True
    nlterms = []
    can_timedpendent = True
    has_3rd_panel = True

    def __init__(self, **kwargs):
        super(Distance_Wall, self).__init__(**kwargs)

    def attribute_set(self, v):
        Bdry.attribute_set(self, v)
        Phys.attribute_set(self, v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v

    @property
    def vt(self):
        root_phys = self.get_root_phys()
        if not isinstance(root_phys, PhysModule):
            data = (('esse_value', VtableElement(None, type="array",
                                                 guilabel="dummy",
                                                 default="0.0",
                                                 tip="dummy")),)
            return Vtable(data)

        dep_vars = self.get_root_phys().dep_vars
        if not hasattr(self, '_dep_var_bk'):
            self._dep_var_bk = ""

        if self._dep_var_bk != dep_vars:
            dep_var = dep_vars[0]
            data = (('esse_value', VtableElement("esse_value", type="array",
                                                 guilabel=dep_var,
                                                 default="0.0",
                                                 tip="Reference for disance measurement")),)
            vt = Vtable(data)
            self._vt1 = vt
            self._dep_var_bk = dep_vars
            self.update_attribute_set()
        else:
            vt = self._vt1
        return vt

    def get_essential_idx(self, kfes):
        if kfes == 0:
            return self._sel_index
        else:
            return []

    def apply_essential(self, engine, gf, real=False, kfes=0):
        if kfes > 0:
            return
        c0 = self.vt.make_value_or_expression(self)[0]

        if real:
            dprint1("Apply Ess.(real)" + str(self._sel_index), 'c0', c0)
        else:
            dprint1("Apply Ess.(imag)" + str(self._sel_index), 'c0', c0)

        name = self.get_root_phys().dep_vars[0]
        fes = engine.get_fes(self.get_root_phys(), name=name)

        fec_name = fes.FEColl().Name()

        mesh = engine.get_emesh(mm=self)
        ibdr = mesh.bdr_attributes.ToList()

        bdr_attr = [0]*mesh.bdr_attributes.Max()
        for idx in self._sel_index:
            bdr_attr[idx-1] = 1

        ess_vdofs = mfem.intArray()
        fes.GetEssentialVDofs(mfem.intArray(bdr_attr), ess_vdofs, 0)
        vdofs = np.where(np.array(ess_vdofs.ToList()) == -1)[0]
        dofs = mfem.intArray([fes.VDofToDof(i) for i in vdofs])
        fes.BuildDofToArrays()

        coeff1 = SCoeff(c0, self.get_root_phys().ind_vars,
                        self._local_ns, self._global_ns,
                        real=real)
        gf.ProjectCoefficient(coeff1, dofs, 0)
