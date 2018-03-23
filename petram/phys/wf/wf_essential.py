from petram.model import Domain, Bdry, Edge, Point, Pair
from petram.phys.weakform import SCoeff, VCoeff
from petram.phys.phys_model import Phys, PhysModule

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('WF_Essential')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem

from petram.phys.vtable import VtableElement, Vtable

class WF_Essential(Bdry, Phys):
    has_essential = True
    nlterms = []
    def __init__(self, **kwargs):
        super(WF_Essential, self).__init__( **kwargs)

    @property
    def vt(self):
        root_phys = self.get_root_phys()
        if not isinstance(root_phys, PhysModule):
             data =  (('esse_value', VtableElement(None, type="array",
                                                   guilabel = "dummy",
                                                   default = "0.0",
                                                   tip = "dummy" )),)
             return Vtable(data)

        dep_vars = self.get_root_phys().dep_vars
        if not hasattr(self, '_dep_var_bk'):
            self._dep_var_bk = ""

        if self._dep_var_bk != dep_vars:
            dep_var = dep_vars[0]
            data =  (('esse_value', VtableElement("esse_value", type="array",
                                                   guilabel = dep_var,
                                                   default =   "0.0",
                                                   tip = "Essentail BC" )),
                      ('esse_vdim', VtableElement("esse_vdim", type="string",
                                                   guilabel = "vdim (0-base)",
                                                   default =   "all",
                                                   readonly = True, 
                                                   tip = "vdim (not supported)")),)
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
        
    def apply_essential(self, engine, gf, real = False, kfes = 0):
        if kfes > 0: return
        if real:       
            dprint1("Apply Ess.(real)" + str(self._sel_index))
        else:
            dprint1("Apply Ess.(imag)" + str(self._sel_index))
            
        c0, vdim0 = self.vt.make_value_or_expression(self)
        dprint1('c0, v0', c0, vdim0)
        name =  self.get_root_phys().dep_vars[0]
        fes = engine.get_fes(self.get_root_phys(), name=name)

        vdim = fes.GetVDim()
        vvdim = -1
        
        if vdim0 != 'all':
            lvdim = len(self.esse_vdim.split(","))
            assert lvdim == 1, "Specify only one vdim"            
            vvdim = int(self.esse_vdim.split(",")[0])
        fec_name = fes.FEColl().Name()
        
        mesh = engine.get_mesh(mm = self)        
        ibdr = mesh.bdr_attributes.ToList()
        bdr_attr = [0]*mesh.bdr_attributes.Max()
        for idx in self._sel_index:
            bdr_attr[idx-1] = 1
            
        method = gf.ProjectBdrCoefficient
        
        if fec_name.startswith("ND"):
            assert vdim == 1, "ND element vdim must be one"
            vdim = mesh.Dimension()
            method = gf.ProjectBdrCoefficientTangent         
        if fec_name.startswith("RT"):
            assert vdim == 1, "RT element vdim must be one"            
            vdim = mesh.Dimension()
            method = gf.ProjectBdrCoefficientNormal                     

        
        if vdim == 1:
           coeff1 = SCoeff(c0[0], self.get_root_phys().ind_vars,
                           self._local_ns, self._global_ns,
                           real = real)
        else:
           coeff1 = VCoeff(vdim, c0, self.get_root_phys().ind_vars,
                           self._local_ns, self._global_ns,
                           real = real)

        assert not (vvdim != -1 and vdim > 1), "Wrong setting...(vvdim != -1 and vdim > 1)"
        if vvdim == -1:
            method(coeff1, mfem.intArray(bdr_attr))
        else:
            method(coeff1, mfem.intArray(bdr_attr), vdim)            
