'''

    Model Tree to stroe MFEM model parameters

'''
from petram.model import Model

from petram.namespace_mixin import NS_mixin
class MFEM_GeneralRoot(Model, NS_mixin):
    can_delete = False
    has_2nd_panel = False

    def __init__(self, *args, **kwargs):
        Model.__init__(self, *args, **kwargs)        
        NS_mixin.__init__(self, *args, **kwargs)
        
    def attribute_set(self, v):
        v['debug_level'] = 1
        super(MFEM_GeneralRoot, self).attribute_set(v)
        return v
        
    def panel1_param(self):
        txt = ["0-3: Larger number prints more info.",
               "Negative number: print from all nodes",
               "Speical debug bits",
               " 4: write essentail BC vector",
               " 8: memory check"]
        return [["debug level",   self.debug_level,  400, {}],
                ["", "\n".join(txt) ,2, None],]

    def get_panel1_value(self):
        return (self.debug_level, None)

    def import_panel1_value(self, v):
        self.debug_level = v[0]
        import petram.debug        
        petram.debug.debug_default_level = int(self.debug_level)        
        
    def run(self):
        import petram.debug
        if petram.debug.debug_default_level == 0:
            petram.debug.debug_default_level = int(self.debug_level)
            
    def get_defualt_local_ns(self):
        '''
        GeneralRoot Namelist knows basic functions
        '''
        import petram.helper.functions
        return petram.helper.functions.f.copy()

class MFEM_PhysRoot(Model):
    can_delete = False
    has_2nd_panel = False        
    def get_possible_child(self):
        ans = []
        try:
            from phys.em3d.em3d_model import EM3D
            ans.append(EM3D)
        except:
            pass
        try:
            from phys.th3ds.th3ds_model import TH3Ds
            ans.append(TH3Ds)
        except:
            pass
        return ans
    
    def make_solvars(self, solsets, g=None):
        solvars = [None]*len(solsets)
        if g is None: g = {}
        for k, v in enumerate(solsets):
            mesh, soldict = v
            solvar = g.copy()
            for phys in self.iter_enabled():
                phys.soldict_to_solvars(soldict, solvar)
            solvars[k] = solvar
        return solvars

class MFEM_InitRoot(Model):    
    can_delete = False
    has_2nd_panel = False        
    def get_possible_child(self):
        from init_model import InitSetting
        return [InitSetting]
    
class MFEM_MeshRoot(Model):
    can_delete = False
    has_2nd_panel = False    
    def get_possible_child(self):
        from mesh.mesh_model import MeshFile, UniformRefinement, MeshGroup
        return [MeshGroup,]

class MFEM_SolverRoot(Model):
    can_delete = False
    has_2nd_panel = False    
    def get_possible_child(self):
        from solver.solinit_model import SolInit
        from solver.std_solver_model import StdSolver
        from solver.parametric import Parametric
        return [StdSolver, Parametric]
    
    def get_active_solvers(self, mm = None):
        return [x for x in self.iter_enabled()]
    
try:    
   from petram.geom.geom_model import MFEM_GeomRoot
   has_geom = True
except:
   has_geom = False

class MFEM_ModelRoot(Model):
    def __init__(self, **kwargs):
        super(MFEM_ModelRoot, self).__init__(**kwargs)
        self['General'] = MFEM_GeneralRoot()

        if has_geom:
            self['Geometry'] = MFEM_GeomRoot()        
        self['Mesh'] = MFEM_MeshRoot()
        self['Phys'] = MFEM_PhysRoot()
        self['InitialValue'] = MFEM_InitRoot()        
        self['Solver'] = MFEM_SolverRoot()

        from petram.helper.variables import Variables
        self._variables = Variables()
        
    def attribute_set(self, v):
        from petram.helper.variables import Variables
        v['_variables'] = Variables()
        v['enabled'] = True        
        return v
        
    def save_setting(self, filename = ''):
        fid = open(fiilename, 'w')
        for od in self.walk():
            od.write_setting(fid)
        fid.close()

    def save_to_file(self, path, meshfile_relativepath=False):
        import cPickle as pickle
        if meshfile_relativepath:
            for od in self.walk():
                if hasattr(od, 'use_relative_path'):
                    od.use_relative_path()
        pickle.dump(self, open(path, 'wb'))
        if meshfile_relativepath:
            for od in self.walk():
                if hasattr(od, 'use_relative_path'):
                    od.restore_fullpath()

               

        
