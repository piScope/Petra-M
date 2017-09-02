import os
import mfem

PyMFEM_PATH =os.path.dirname(os.path.dirname(mfem.__file__))
PetraM_PATH =os.getenv("PetraM")
HOME = os.path.expanduser("~")

from petram.model import Model
from petram.namespace_mixin import NS_mixin
from petram.mfem_config import use_parallel

if use_parallel:
   import mfem.par as mfem
   from mpi4py import MPI
   num_proc = MPI.COMM_WORLD.size
   myid     = MPI.COMM_WORLD.rank
else:
   import mfem.ser as mfem

class Mesh(Model, NS_mixin):
    def __init__(self, *args, **kwargs):
        super(Mesh, self).__init__(*args, **kwargs)
        NS_mixin.__init__(self, *args, **kwargs)
   
    def onItemSelChanged(self, evt):
        '''
        GUI response when model object is selected in
        the dlg_edit_model
        '''
        viewer = evt.GetEventObject().GetTopLevelParent().GetParent()
        viewer.set_view_mode('mesh', self)                        

class MeshGroup(Model):
    can_delete = True
    has_2nd_panel = False
    isMeshGroup = True    
    def get_possible_child(self):
        return [MeshFile, UniformRefinement]
     
    def onItemSelChanged(self, evt):
        '''
        GUI response when model object is selected in
        the dlg_edit_model
        '''
        viewer = evt.GetEventObject().GetTopLevelParent().GetParent()
        viewer.set_view_mode('mesh', self)
        
    def is_viewmode_grouphead(self):
        return True
     
    def figure_data_name(self):
        return 'mfem'
        
        
MFEMMesh = MeshGroup

class MeshFile(Mesh):
    has_2nd_panel = False        
    def __init__(self, parent = None, **kwargs):
        self.path = kwargs.pop("path", "")
        self.generate_edges = kwargs.pop("generate_edges", 1)
        self.refine = kwargs.pop("refien", 1)
        self.fix_orientation = kwargs.pop("fix_orientation", True)        
        super(MeshFile, self).__init__(parent = parent, **kwargs)

    def __repr__(self):
        try:
           return 'MeshFile('+self.path+')'
        except:
           return 'MeshFile(!!!Error!!!)'
        
    def attribute_set(self, v):
        v = super(MeshFile, self).attribute_set(v)
        v['path'] = ''
        v['generate_edges'] = 1
        v['refine'] = 1
        v['fix_orientation'] = True

        return v
        
    def panel1_param(self):
        return [["Path",   self.path,  200, {}],
                ["", "replacement rule: {petram}=$PetraM, {mfem}=PyMFEM, {home}=~"  ,2, None],
                ["Generate edges",    self.generate_edges == 1,  3, {"text":""}],
                ["Refine",    self.refine==1 ,  3, {"text":""}],
                ["FixOrientatijon",    self.fix_orientation ,  3, {"text":""}]]
    def get_panel1_value(self):
        return (self.path, None, self.generate_edges, self.refine, self.fix_orientation)
    
    def import_panel1_value(self, v):
        self.path = str(v[0])
        self.generate_edges = 1 if v[2] else 0
        self.refine = 1 if v[3] else 0
        self.fix_orientation = v[4]
        
    def use_relative_path(self):
        self._path_dir = os.path.dirname(self.path)
        self.path = os.path.basename(self.path)

        
    def restore_fullpath(self):       
        self.path = os.path.join(self._path_dir, self.path)
        self._path_dir = ''


    def get_real_path(self):
        path = str(self.path)
        if path.find('{mfem}') != -1:
            path = path.replace('{mfem}', PyMFEM_PATH)
        if path.find('{petram}') != -1:
            path = path.replace('{petram}', PetraM_PATH)
        if path.find('{home}') != -1:
            path = path.replace('{home}', HOME)
        return path

    def run(self, mesh = None):
        path = self.get_real_path()
        if not os.path.exists(path):
            print("mesh file does not exists : " + path + " in " + os.getcwd())
            return None
        args = (path,  self.generate_edges, self.refine, self.fix_orientation)
        mesh =  mfem.Mesh(*args)
        try:
           mesh.GetNBE()
           return mesh
        except:
           return None
        
class UniformRefinement(Mesh):
    has_2nd_panel = False           
    def __init__(self, parent = None, **kwargs):
        self.num_refine = kwargs.pop("num_refine", "0")
        super(UniformRefinement, self).__init__(parent = parent, **kwargs)        
    def __repr__(self):
        try:
           return 'MeshUniformRefinement('+self.num_refine+')'
        except:
           return 'MeshUniformRefinement(!!!Error!!!)'
        
    def attribute_set(self, v):
        v = super(UniformRefinement, self).attribute_set(v)       
        v['num_refine'] = '0'
        return v
        
    def panel1_param(self):
        return [["Number",   str(self.num_refine),  0, {}],]
    def import_panel1_value(self, v):
        self.num_refine = str(v[0])
    def get_panel1_value(self):
        return (str(self.num_refine))
     
    def run(self, mesh):
        if use_parallel:
           for i in range(int(self.num_refine)):           
               mesh.UniformRefinement() # this is parallel refinement
        else:
           for i in range(int(self.num_refine)):
               mesh.UniformRefinement()
        return mesh

   

