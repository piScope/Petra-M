'''

 Direct Wrappwer Call provides the access to low level (PyMFEM)
 functionality during a Petra-M simulation

'''
import numpy as np

from petram.mfem_config import use_parallel
if use_parallel:
   from petram.helper.mpi_recipes import *
   import mfem.par as mfem   
else:
   import mfem.ser as mfem

class DWC(object):
    def __init__(self):
        pass

    def postprocess(self, caller,  *args, **kwargs):
        ''' 
        postprocess is called from solvestep after store_sol
        '''
        raise NotImplementedError("postprocess must be implemented by a user")

### sample DWC class (see em3d_TE8.pfz)    
class Eval_E_para(DWC):
    def __init__(self, faces):
        DWC.__init__(self)
        self.faces = faces

    def postprocess(self, caller, gf=None, edges = None):

        from petram.helper.mpi_recipes import safe_flatstack
        from mfem.common.mpi_debug import nicePrint
        if edges is None: return

        print("postprocess is called")
        gfr, gfi = gf
        print(caller, gfr)
        try:
            fes = gfr.ParFESpace()
            mesh = fes.GetParMesh()
        except:
            fes = gfr.FESpace()
            mesh = fes.GetMesh()
        from petram.mesh.mesh_utils import get_extended_connectivity
        if not hasattr(mesh, 'extended_connectivity'):
           get_extended_connectivity(mesh)
        l2e = mesh.extended_connectivity['line2edge']
        idx = safe_flatstack([l2e[e] for e in edges])
        dofs = safe_flatstack([fes.GetEdgeDofs(i) for i in idx])
        size = dofs.size/idx.size

        w = []
        for i in idx:
            # don't put this Tr outside the loop....
            Tr = mfem.IsoparametricTransformation()            
            mesh.GetEdgeTransformation(i, Tr)
            w.extend([Tr.Weight()]*size)
        w = np.array(w)    
        data = gfr.GetDataArray()[dofs] + 1j*gfi.GetDataArray()[dofs]
        nicePrint(w)        
        nicePrint(data/w)


