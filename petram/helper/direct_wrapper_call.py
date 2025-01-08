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

    def make_args(self, mode, kwargs):
        if mode == 'postprocess':
            return tuple(), kwargs
        elif mode == 'timestep':
            t = kwargs.pop('time')
            return (t, ), kwargs
        elif mode == 'checkpoint':
            t = kwargs.pop('time')
            icp = kwargs.pop('icheckpoint')
            return (t, icp), kwargs
        elif mode == 'loopcontrol':
            ct = kwargs.pop('count')
            return (ct,), kwargs
        elif mode == 'nlcheckpoint':
            ct = kwargs.pop('count')
            return (ct,), kwargs
        elif mode == 'init':
            return tuple(), kwargs
        else:
            # default no argument
            return tuple(), kwargs

    def call(self, caller, *args, **kwargs):
        '''
        general call
        '''
        raise NotImplementedError("call must be implemented by a user")

    def loopcontrol(self, caller, count, *args, **kwargs):
        raise NotImplementedError("loopcontrol must be implemented by a user")

    def postprocess(self, caller,  *args, **kwargs):
        '''
        postprocess is called from solvestep after store_sol
        '''
        raise NotImplementedError("postprocess must be implemented by a user")

    def timestep(self, caller, t, *args, **kwargs):
        '''
        timestep is called from time-dependent solver at every time step
        t = current time
        '''
        raise NotImplementedError("timestep must be implemented by a user")

    def checkpoint(self, caller, t, cp, *args, **kwargs):
        '''
        timestep is called from time-dependent solver at checkpoint
        t = current time
        cp = check pioint index
        '''
        raise NotImplementedError("checkpoint must be implemented by a user")

    def nl_checkpoint(self, caller, count, *args, **kwargs):
        '''
        called during nonlinear iteration loop
        return value: 
             False: normal
             True: emergency stop of iteration
        '''
        raise NotImplementedError(
            "nl_checkpoint must be implemented by a user")

    def nl_start(self, caller, *args, **kwargs):
        '''
        called at the begining of non-linear iteration
        '''
        raise NotImplementedError("nl_start must be implemented by a user")

    def nl_end(self, caller, *args, **kwargs):
        '''
        called at the end of non-linear iteration
        '''
        raise NotImplementedError("nl_end must be implemented by a user")

    def init(self, caller, *args, **kwargs):
        '''
        init is called from init_setting
        '''
        raise NotImplementedError("init must be implemented by a user")


# sample DWC class (see em3d_TE8.pfz)
class Eval_E_para(DWC):
    def __init__(self, faces):
        DWC.__init__(self)
        self.faces = faces

    def postprocess(self, caller, gf=None, edges=None):

        from petram.helper.mpi_recipes import safe_flatstack
        from mfem.common.mpi_debug import nicePrint
        if edges is None:
            return

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
        if len(idx) > 0:
            dofs = safe_flatstack([fes.GetEdgeDofs(i) for i in idx])
            size = dofs.size//idx.size

            w = []
            for i in idx:
                # don't put this Tr outside the loop....
                Tr = mesh.GetEdgeTransformation(i)
                w.extend([Tr.Weight()]*size)
            w = np.array(w)
            data = gfr.GetDataArray()[dofs] + 1j*gfi.GetDataArray()[dofs]
            field = data/w
        else:
            w = np.array([])
            field = np.array([])
        nicePrint(w)
        nicePrint(field)
