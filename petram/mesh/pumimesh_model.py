import pyCore

# pyCore.start_sim('simlog.txt')
# pyCore.gmi_register_mesh()
# pyCore.gmi_sim_start()
# pyCore.gmi_register_sim()


import os
from petram.mesh.mesh_model import Mesh
from mfem._par import pumi
from mfem.par import intArray

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
        return [["Mesh Path",   self.mesh_path,  200, {}],
                ["Model Path",  self.model_path,  200, {}],
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

        print("Model Path is " + model_path)
        print("Mesh Path is " + mesh_path)

        if not os.path.exists(mesh_path):
            print("mesh file does not exists : " + mesh_path + " in " + os.getcwd())
            return None
        if not os.path.exists(model_path):
            print("model file does not exists : " + model_path + " in " + os.getcwd())
            return None

        ####
        # import pyCore
	pyCore.lion_set_verbosity(1)


        from mpi4py import MPI
	if not pyCore.PCU_Comm_Initialized():
	  pyCore.PCU_Comm_Init()

	pyCore.PCU_Comm_Peers()
	pyCore.PCU_Comm_Self()

	pyCore.start_sim('simlog.txt')
	pyCore.gmi_register_mesh()
	pyCore.gmi_sim_start()
	pyCore.gmi_register_sim()

	pumi_mesh_path = mesh_path[0:-5] + ".smb"

	pumi_mesh = pyCore.loadMdsMesh(model_path, pumi_mesh_path)

	self.root()._pumi_mesh = pumi_mesh # hack to be able to access pumi_mesh later!

        if not globals()['is_licenses_initialized']:
            print("do license etc here ...once")

            globals()['is_licenses_initialized'] = True

	# we need to have a numbering to be able to track changes
	# in the orientations of tets caused by calling ReorientTet
	# on MFEM meshes
	# The numbering must be consistent with how the MFEM mesh
	# is created from the PUMI mesh
	# 1) make a local numbering
	# 2) convert it to global
	pumi_field_shape = pyCore.getConstant(0)
	aux_numbering = pyCore.createNumbering(pumi_mesh, "aux_vertex_numbering", pumi_field_shape, 1)


	it = pumi_mesh.begin(0)
	all_count = 0
	shared_count = 0
	owned_count = 0
	while True:
	  ent = pumi_mesh.iterate(it)
	  if not ent:
	    break
	  all_count = all_count + 1
	  if pumi_mesh.isOwned(ent):
	    pyCore.number(aux_numbering, ent, 0, 0, owned_count)
	    owned_count = owned_count + 1
	  if pumi_mesh.isShared(ent):
	    shared_count = shared_count + 1
	pumi_mesh.end(it)

	global_numbering = pyCore.makeGlobal(aux_numbering, True)
	pyCore.synchronize(global_numbering)

	this_ids = intArray(all_count)
        it = pumi_mesh.begin(0)
        all_count = 0
	while True:
	  ent = pumi_mesh.iterate(it)
	  if not ent:
	    break
	  idd = pyCore.getNumber(global_numbering, ent, 0, 0)
	  this_ids[all_count] = idd
	  all_count = all_count + 1
	pumi_mesh.end(it)
	this_ids.Sort()

	pumi_coord_field = pumi_mesh.getCoordinateField()
	pumi_field_shape = pyCore.getShape(pumi_coord_field)
	local_numbering = pyCore.createNumbering(pumi_mesh, "local_vert_numbering", pumi_field_shape, 1)

	it = pumi_mesh.begin(0)
	while True:
	  ent = pumi_mesh.iterate(it)
	  if not ent:
	    break
	  idd = pyCore.getNumber(global_numbering, ent, 0, 0)
	  ordered_idd = this_ids.FindSorted(idd)
	  pyCore.number(local_numbering, ent, 0, 0, ordered_idd)
	pumi_mesh.end(it)

	pyCore.destroyGlobalNumbering(global_numbering)

	# convert pumi_mesh to mfem mesh
	mesh = pumi.ParPumiMesh(MPI.COMM_WORLD, pumi_mesh)

    	# reverse classifications based on model tags
	dim = pumi_mesh.getDimension()
	it = pumi_mesh.begin(dim-1)
	bdr_cnt = 0
	while True:
	  e = pumi_mesh.iterate(it)
	  if not e: break
	  model_tag  = pumi_mesh.getModelTag(pumi_mesh.toModel(e))
	  model_type = pumi_mesh.getModelType(pumi_mesh.toModel(e))
	  if model_type == (dim-1):
	    mesh.GetBdrElement(bdr_cnt).SetAttribute(model_tag)
	  bdr_cnt += 1
	pumi_mesh.end(it)

	it = pumi_mesh.begin(dim)
	elem_cnt = 0
	while True:
	  e = pumi_mesh.iterate(it)
	  if not e: break
	  model_tag  = pumi_mesh.getModelTag(pumi_mesh.toModel(e))
	  model_type = pumi_mesh.getModelType(pumi_mesh.toModel(e))
	  if model_type == (dim):
	    mesh.SetAttribute(elem_cnt, model_tag)
	  elem_cnt += 1
	pumi_mesh.end(it)

	print(type(mesh))
	print(id(mesh))

	mesh.SetAttributes();

	self.root()._par_pumi_mesh = mesh # hack to be able to access par_pumi_mesh later!

	try:
	  mesh.GetNBE()
	  return mesh
	except:
	  return None

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
      
