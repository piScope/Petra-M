'''

    Model Tree to stroe MFEM model parameters

'''
import numpy as np
from petram.model import Model

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('MFEMModel')

from petram.namespace_mixin import NS_mixin
class MFEM_GeneralRoot(Model, NS_mixin):
    can_delete = False
    has_2nd_panel = False

    def __init__(self, *args, **kwargs):
        Model.__init__(self, *args, **kwargs)        
        NS_mixin.__init__(self, *args, **kwargs)
        
    def get_info_str(self):
        return NS_mixin.get_info_str(self)
        
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
        from petram.helper.phys_module_util import all_phys_models
        models, classes = all_phys_models()
        return classes
    
    def make_solvars(self, solsets, g=None):
        solvars = [None]*len(solsets)
        if g is None: g = {}
        for k, v in enumerate(solsets):
            mesh, soldict = v
            solvar = g.copy()
            for phys in self.iter_enabled():
                phys.soldict_to_solvars(soldict, solvar)
            solvars[k] = solvar
        print "solvars", solvars
        return solvars

    def onItemSelChanged(self, evt):
        '''
        GUI response when model object is selected in
        the dlg_edit_model
        '''
        viewer = evt.GetEventObject().GetTopLevelParent().GetParent()
        viewer.set_view_mode('phys')

    def dependent_values(self):
        '''
        return dependent_values
           names: name of values
           pnames: list of physics module
           pindex: index of dependent value in the physics module 
        '''
        names =  sum([c.dep_vars for c in self.iter_enabled()], [])
        pnames = sum([[c.name()]*len(c.dep_vars) for c in self.iter_enabled()], [])
        pindex = sum([range(len(c.dep_vars)) for c in self.iter_enabled()], [])

        return names, pnames, pindex
    
    def get_num_matrix(self, get_matrix_weight, phys_target = None):
        # get_matrix_weight: solver method to evaulate matrix weight
        if phys_target is None:
             phys_target = [self[k] for k in self]
             
        num_matrix = 0
        for phys in phys_target:
            for mm in phys.walk():
                if not mm.enabled: continue
                mm.set_matrix_weight(get_matrix_weight)

                wt = np.array(mm.get_matrix_weight())
                tmp = int(np.max((wt != 0)*(np.arange(len(wt))+1)))
                num_matrix = max(tmp, num_matrix)
        dprint1("number of matrix", num_matrix)
        return num_matrix
            

    def all_dependent_vars(self, num_matrix):
        '''
        FES variable + extra variable
        '''
        dep_vars  = []
        isFesvars_g = []
        
        phys_target = [self[k] for k in self]
        
        for phys in phys_target:
            if not phys.enabled: continue            
            dep_vars  = []
            dv = phys.dep_vars
            dep_vars.extend(dv)
            extra_vars = []
            for mm in phys.walk():
                if not mm.enabled: continue                
                for j in range(num_matrix):
                      for phys2 in phys_target:
                          if not phys2.enabled: continue
                          if not mm.has_extra_DoF2(k, phys2, j): continue
                          print "passed", k, phys2, j
                          name = mm.extra_DoF_name()
                          if not name in extra_vars:
                              extra_vars.append(name)
                    
            dep_vars.extend(extra_vars)
        return dep_vars
    

class MFEM_InitRoot(Model):    
    can_delete = False
    has_2nd_panel = False        
    def get_possible_child(self):
        from init_model import InitSetting
        return [InitSetting]
    
    def onItemSelChanged(self, evt):
        '''
        GUI response when model object is selected in
        the dlg_edit_model
        '''
        viewer = evt.GetEventObject().GetTopLevelParent().GetParent()
        viewer.set_view_mode('phys', self)
        
    def is_viewmode_grouphead(self):
        return True

class MFEM_GeomRoot(Model):
    can_delete = False
    has_2nd_panel = False
    def get_possible_child(self):
        try:
            from petram.geom.gmsh_geom_model import GmshGeom
            return [GmshGeom]
        except:
            return []
        
class MFEM_MeshRoot(Model):
    can_delete = False
    has_2nd_panel = False    
    def get_possible_child(self):
        from mesh.mesh_model import MFEMMesh, MeshGroup
        try:
            from petram.mesh.gmsh_mesh_model import GmshMesh
            return [MFEMMesh, GmshMesh]
        except:
            return [MFEMMesh]

    def onItemSelChanged(self, evt):
        '''
        GUI response when model object is selected in
        the dlg_edit_model
        '''
        viewer = evt.GetEventObject().GetTopLevelParent().GetParent()
        viewer.set_view_mode('mesh')                                
        
class MFEM_SolverRoot(Model):
    can_delete = False
    has_2nd_panel = False    
    def get_possible_child(self):
        from solver.solinit_model import SolInit
        from solver.std_solver_model import StdSolver
        from solver.timedomain_solver_model import TimeDomain
        from solver.parametric import Parametric
        return [StdSolver, Parametric, TimeDomain]
    
    def get_active_solvers(self, mm = None):
        return [x for x in self.iter_enabled()]
    
    def onItemSelChanged(self, evt):
        '''
        GUI response when model object is selected in
        the dlg_edit_model
        '''
        viewer = evt.GetEventObject().GetTopLevelParent().GetParent()
        viewer.set_view_mode('phys', self)                                        
    
    def is_viewmode_grouphead(self):
        return True
    
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
        v['root_path'] = ''
        return v
    
    def set_root_path(self, path):
        self.root_path = path
        
    def get_root_path(self):
        return self.root_path
        
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

               

        
