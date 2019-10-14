import os
from petram.mesh.mesh_model import Mesh

is_licenses_initialized = False

class PumiMesh(Mesh):
    isMeshGenerator = True   
    isRefinement = False   
    has_2nd_panel = False        
    def __init__(self, parent = None, **kwargs):
        self.path = kwargs.pop("path", "")
        self.generate_edges = kwargs.pop("generate_edges", 1)
        self.refine = kwargs.pop("refien", 1)
        self.fix_orientation = kwargs.pop("fix_orientation", True)        
        super(PumiMesh, self).__init__(parent = parent, **kwargs)

    def __repr__(self):
        try:
           return 'PumiMesh('+self.mesh_path+')'
        except:
           return 'PumiMesh(!!!Error!!!)'
        
    def attribute_set(self, v):
        v = super(PumiMesh, self).attribute_set(v)
        v['mesh_path'] = ''
        v['model_path'] = ''        
        v['generate_edges'] = 1
        v['refine'] = True
        v['fix_orientation'] = True

        return v
        
    def panel1_param(self):
        return [["Model Path",   self.model_path,  200, {}],
                ["Mesh Path",   self.mesh_path,  200, {}],                
                ["", "rule: {petram}=$PetraM, {mfem}=PyMFEM, \n     {home}=~ ,{model}=project file dir."  ,2, None],
                ["Generate edges",    self.generate_edges == 1,  3, {"text":""}],
                ["Refine",    self.refine==1 ,  3, {"text":""}],
                ["FixOrientation",    self.fix_orientation ,  3, {"text":""}]]
    def get_panel1_value(self):
        return (self.mesh_path, self.model_path, None, self.generate_edges, self.refine, self.fix_orientation)
    
    def import_panel1_value(self, v):
        self.mesh_path = str(v[0])
        self.model_path = str(v[1])        
        self.generate_edges = 1 if v[3] else 0
        self.refine = 1 if v[4] else 0
        self.fix_orientation = v[5]
        
    def use_relative_path(self):
        self._path_bk  = self.path
        self.path = os.path.basename(self.get_real_path())

        
    def restore_fullpath(self):       
        self.path = self._path_bk
        self._path_bk = ''


    def get_real_path(self, path):
        if path == '':
           # if path is empty, file is given by internal mesh generator.
           parent = self.get_mesh_root()
           for key in parent.keys():
              if not parent[key].is_enabled(): continue
              if hasattr(parent[key], 'get_meshfile_path'):
                 return parent[key].get_meshfile_path()
        if path.find('{mfem}') != -1:
            path = path.replace('{mfem}', PyMFEM_PATH)
        if path.find('{petram}') != -1:
            path = path.replace('{petram}', PetraM_PATH)
        if path.find('{home}') != -1:
            path = path.replace('{home}', HOME)
        if path.find('{model}') != -1:
            path = path.replace('{model}', str(self.root().model_path))

        if not os.path.isabs(path):
            dprint2("meshfile relative path mode")
            path1 = os.path.join(os.getcwd(), path)
            dprint2("trying :", path1)
            if not os.path.exists(path1):
                path1 = os.path.join(os.path.dirname(os.getcwd()), path)
                dprint2("trying :", path1)
                if (not os.path.exists(path1) and "__main__" in globals() and hasattr(__main__, '__file__')):
                    from __main__ import __file__ as mainfile        
                    path1 = os.path.join(os.path.dirname(os.path.realpath(mainfile)), path)   
                    dprint1("trying :", path1)
                if not os.path.exists(path1) and os.getenv('PetraM_MeshDir') is not None:
                    path1 = os.path.join(os.getenv('PetraM_MeshDir'), path)
                    dprint1("trying :", path1)                    
            if os.path.exists(path1):
                path = path1
            else:
                assert False, "can not find mesh file from relative path: "+path
        return path

    def run(self, mesh = None):
        model_path = self.get_real_path(self.model_path)
        mesh_path = self.get_real_path(self.mesh_path)       

        if not os.path.exists(mesh_path):
            print("mesh file does not exists : " + path + " in " + os.getcwd())
            return None
        if not os.path.exists(model_path):
            print("model file does not exists : " + path + " in " + os.getcwd())
            return None

        ####

        if not globals()['is_licenses_initialized']:
            print("do license etc here ...once")
            globals()['is_licenses_initialized'] = True
            
        assert False, "not implemented : pumi_mesh must return mfem mesh"


        '''
        args = (path,  self.generate_edges, self.refine, self.fix_orientation)
        mesh =  mfem.Mesh(*args)
        self.parent.sdim = mesh.SpaceDimension()
        try:
           mesh.GetNBE()
           return mesh
        except:
           return None
        '''
        
