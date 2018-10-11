import os
import numpy as np
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
   from petram.helper.mpi_recipes import *   
else:
   import mfem.ser as mfem
   
import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('MeshModel')

class Mesh(Model, NS_mixin):
    isMeshGenerator = False
    isRefinement = False      
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
        
    def get_mesh_root(self):
        from petram.mfem_model import MFEM_MeshRoot
        
        p = self.parent
        while p is not None:
            if isinstance(p, MFEM_MeshRoot): return p           
            p = p.parent
            
class MeshGroup(Model):  
    can_delete = True
    has_2nd_panel = False
    isMeshGroup = True    
    def get_possible_child(self):
        return [MeshFile, Mesh1D, UniformRefinement, DomainRefinement]
     
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
    isMeshGenerator = True   
    isRefinement = False   
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
        v['refine'] = True
        v['fix_orientation'] = True

        return v
        
    def panel1_param(self):
        return [["Path",   self.path,  200, {}],
                ["", "rule: {petram}=$PetraM, {mfem}=PyMFEM, \n     {home}=~ ,{model}=project file dir."  ,2, None],
                ["Generate edges",    self.generate_edges == 1,  3, {"text":""}],
                ["Refine",    self.refine==1 ,  3, {"text":""}],
                ["FixOrientation",    self.fix_orientation ,  3, {"text":""}]]
    def get_panel1_value(self):
        return (self.path, None, self.generate_edges, self.refine, self.fix_orientation)
    
    def import_panel1_value(self, v):
        self.path = str(v[0])
        self.generate_edges = 1 if v[2] else 0
        self.refine = 1 if v[3] else 0
        self.fix_orientation = v[4]
        
    def use_relative_path(self):
        self._path_bk  = self.path
        self.path = os.path.basename(self.get_real_path())

        
    def restore_fullpath(self):       
        self.path = self._path_bk
        self._path_bk = ''


    def get_real_path(self):
        path = str(self.path)
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
            path = path.replace('{model}', self.root().model_path)

        if not os.path.isabs(path):
            dprint2("meshfile relative path mode")
            path1 = os.path.join(os.getcwd(), path)
            dprint2("trying :", path1)
            if not os.path.exists(path1):
                print(os.path.dirname(os.getcwd()))
                path1 = os.path.join(os.path.dirname(os.getcwd()), path)
                dprint1("trying :", path1)
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
        
class Mesh1D(Mesh):
    isMeshGenerator = True      
    isRefinement = False   
    has_2nd_panel = False        

    def attribute_set(self, v):
        v = super(Mesh1D, self).attribute_set(v)
        v['length'] = 1
        v['nsegs'] = 100
        v['length_txt'] = "1"
        v['nsegs_txt'] = "100"
        v['refine'] = 1
        v['fix_orientation'] = True
        return v
        
    def panel1_param(self):
        def check_int_array(txt, param, w):
            try:
               val  = [int(x) for x in txt.split(',')]
               return True
            except:
               return False
            
        return [["Length",   self.length_txt,  0, {"validator":check_int_array}],
                ["N segments",   self.nsegs_txt,  0, {"validator":check_int_array}],
                [None, "Note: use comma separated integer to generate a multisegments mesh",   2, {}],]

    def get_panel1_value(self):
        return (self.length_txt, self.nsegs_txt, None)
    
    def import_panel1_value(self, v):
        self.length_txt = str(v[0])
        self.nsegs_txt = str(v[1])
        
        try:
            self.length = [int(x) for x in self.length_txt.split(',')]
            self.nsegs= [int(x) for x in self.nsegs_txt.split(',')]
        except:
            pass

    def run(self, mesh = None):

        from petram.mesh.make_mesh1d import straight_line_mesh

        m = straight_line_mesh(self.length, self.nsegs,
                              filename='',
                              refine = self.refine == 1,
                              fix_orientation = self.fix_orientation)
        try:
           m.GetNBE()
           return m
        except:
           return None
        
class UniformRefinement(Mesh):
    isRefinement = True
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
        gtype = np.unique([mesh.GetElementBaseGeometry(i) for i in range(mesh.GetNE())])
        if use_parallel:
            from mpi4py import MPI
            gtype = gtype.astype(np.int32)
            gtype = np.unique(allgather_vector(gtype, MPI.INT))

        if len(gtype) > 1:
           dprint1("(Warning) Element Geometry Type is mixed. Cannot perform UniformRefinement")
           return mesh
        for i in range(int(self.num_refine)):           
            mesh.UniformRefinement() # this is parallel refinement
        return mesh
     
class DomainRefinement(Mesh):
    isRefinement = True
    has_2nd_panel = False 
    def __init__(self, parent = None, **kwargs):
        self.num_refine = kwargs.pop("num_refine", "0")
        self.domain_txt = kwargs.pop("domain_txt", "")
        super(DomainRefinement, self).__init__(parent = parent, **kwargs)        
    def __repr__(self):
        try:
           return 'MeshUniformRefinement('+self.num_refine+')'
        except:
           return 'MeshUniformRefinement(!!!Error!!!)'
        
    def attribute_set(self, v):
        v = super(DomainRefinement, self).attribute_set(v)       
        v['num_refine'] = '0'
        v['domain_txt'] = ''
        return v
        
    def panel1_param(self):
        return [["Number",    str(self.num_refine),  0, {}],
                ["Domains",   self.domain_txt,  0, {}],]
     
    def import_panel1_value(self, v):
        self.num_refine = str(v[0])
        self.domain_txt = str(v[1])
        
    def get_panel1_value(self):
        return (str(self.num_refine), str(self.domain_txt))
     
    def run(self, mesh):
        gtype = np.unique([mesh.GetElementBaseGeometry(i) for i in range(mesh.GetNE())])
        if use_parallel:
            from mpi4py import MPI
            gtype = gtype.astype(np.int32)
            gtype = np.unique(allgather_vector(gtype, MPI.INT))

        if len(gtype) > 1:
           dprint1("(Warning) Element Geometry Type is mixed. Cannot perform UniformRefinement")
           return mesh
        domains = [int(x) for x in self.domain_txt.split(',')]
        if len(domains) == 0: return mesh

        attr = mesh.GetAttributeArray()
        idx = mfem.intArray(list(np.where(np.in1d(attr, domains))[0]))

        for i in range(int(self.num_refine)):           
            mesh.GeneralRefinement(idx) # this is parallel refinement
        return mesh

   

