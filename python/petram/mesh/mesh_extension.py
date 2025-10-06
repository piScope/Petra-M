import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('MeshExtension')

from petram.mfem_config import use_parallel

if use_parallel:
   import mfem.par as mfem
   from mfem.common.mpi_debug import nicePrint
else:
   def nicePrint(x):
      print(x)

class MeshExtInfo(dict):
    

    def __init__(self, *args, **kwargs):
        cut = set(kwargs.pop('cut', []))
        sel = set(kwargs.pop('sel', []))
        dim = kwargs.pop('dim', 1)
        base = kwargs.pop('base', 0)        
        self.set_selection(sel)
        self.set_cut(cut)
        self['dim'] = dim
        self['base'] = base

        dict.__init__(self, *args, **kwargs)

    def set_dim(self, dim):
        self['dim'] = dim
        
    def set_selection(self, sel):
        self['sel']  = set([int(x) for x in sel])

    def set_cut(self, sel):
        self['cut']  = set([int(x) for x in sel])

    def __eq__(self, x):
        for key in x:
            if not key in self: return False
            if x[key] != self[key]: return False
        return True
    
class MeshExt(object):
    '''
    Mesh Manipulation 

    '''
    def __init__(self):
        from petram.model import ModelDict
        self.info = []

    def do_add_info(self, info):
        '''
        add info and return mesh index
        '''
        for k, x in enumerate(self.info):
            if x == info: return k
        self.info.append(info)
        return len(self.info)-1
    
    def add_default_info(self, j, dim, sel):
        info = MeshExtInfo(base = j, dim = dim, sel = sel)
        self.do_add_info(info)
        
    def get_info(self, idx):
        return self.info[idx]
     
    def add_info(self, info):
        '''
        add info and return mesh index
            if cut is on, make sure that there is a common no-cut mesh
        '''
        if len(info['cut']) != 0:
            info2 = MeshExtInfo(info)
            info2['cut'] = set(list())
            base = self.do_add_info(info2)
            info['base'] = base
        return self.do_add_info(info)
        
            

def generate_emesh(emeshes, info):
    '''
    apply menh manipulation defined by info to mesh

    '''
    from petram.mesh.partial_mesh import surface, volume, edge
    
    base_mesh = emeshes[info['base']]
    dprint1(info)
    if info['dim'] == 3:
        if base_mesh.Dimension() == 3:         
            alldom = set(base_mesh.extended_connectivity['vol2surf'].keys())
            if alldom == info['sel']: return base_mesh
            if len(info['sel']) == 0: return base_mesh
        dprint1("calling volume")
        m = volume(base_mesh, list(info['sel']))#, filename = 'par_part.mesh')
    elif info['dim'] == 2:
        if base_mesh.Dimension() == 2: 
            alldom = set(base_mesh.extended_connectivity['surf2line'].keys())
            if alldom == info['sel']: return base_mesh
            if len(info['sel']) == 0: return base_mesh
        dprint1("calling surface")            
        m = surface(base_mesh, list(info['sel']))#, filename = 'par_part.mesh')            
    elif info['dim'] == 1:
        if base_mesh.Dimension() == 1: 
            alldom = set(base_mesh.extended_connectivity['line2vert'].keys())
            if alldom == info['sel']: return base_mesh
            if len(info['sel']) == 0: return base_mesh
        dprint1("calling edge")                        
        m = edge(base_mesh, list(info['sel']))#, filename = 'par_part.mesh')            
    else:
        raise NotImplementedError("emesh with " +str(info["dim"]))
    return m
