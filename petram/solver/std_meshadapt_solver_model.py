import pyCore
import mfem.par as mfem
from mfem.par import intArray
from mfem.par import Vector
from mfem.par import DenseMatrix
from mfem._par.pumi import ParPumiMesh
from mfem._par.pumi import ParMesh2ParPumiMesh
import os
import numpy as np
import math

from petram.model import Model
from petram.solver.solver_model import Solver
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('StdMeshAdaptSolver')
rprint = debug.regular_print('StdMeshAdaptSolver')

from petram.solver.std_solver_model import StdSolver, StandardSolver

def get_field_z_averaged(pumi_mesh, field_name, field_type, grid):
  # get the mfem mesh and fespace
  fes = grid.ParFESpace()
  mesh = fes.GetParMesh()

  # we expect the pumi mesh to have a numbering with the name
  # "local_vert_numbering"
  numbering = pumi_mesh.findNumbering("local_vert_numbering")
  if numbering == None:
    assert False, "numbering \"local_vert_numbering\" was not found"


  # create the necessary fields
  field_z = pyCore.createFieldOn(pumi_mesh, field_name, field_type)
  dim = pumi_mesh.getDimension()
  count_field = pyCore.createFieldOn(pumi_mesh, "count_field", pyCore.SCALAR)
  sol_field = pyCore.createFieldOn(pumi_mesh, "sol_field_averaged", pyCore.VECTOR)

  # initialize the count and sol field to zero
  it = pumi_mesh.begin(0)
  while True:
    ent = pumi_mesh.iterate(it)
    if not ent:
      break
    pyCore.setScalar(count_field, ent, 0, 0.0)
    p = pyCore.Vector3(0.0, 0.0, 0.0)
    pyCore.setVector(sol_field, ent, 0, p)
  pumi_mesh.end(it)

  it = pumi_mesh.begin(dim)
  eid = 0
  while True:
    ent = pumi_mesh.iterate(it)
    if not ent:
      break

    felem = fes.GetFE(eid)
    elem_vert = mfem.geom.Geometry().GetVertices(felem.GetGeomType())

    vval = DenseMatrix()
    pmat = DenseMatrix()
    grid.GetVectorValues(eid, elem_vert, vval, pmat)
    pyCore.PCU_ALWAYS_ASSERT(vval.Width() == 4)
    pyCore.PCU_ALWAYS_ASSERT(vval.Height() == 3)

    mfem_vids = mesh.GetElementVertices(eid)

    for i in range(4):
      current_count = pumi_mesh.getVertScalarField(count_field, ent, i, 0)
      current_sol = pumi_mesh.getVertVectorField(sol_field, ent, i, 0)
      pumi_vid = pumi_mesh.getVertNumbering(numbering, ent, i, 0, 0)
      # this is the index of pumi_vid in mfem_vids list
      # (note that we need to do this because of ReorientTet call)
      j = mfem_vids.index(pumi_vid)
      pumi_mesh.setVertScalarField(count_field, ent, i, 0, current_count + 1.0)
      # pumi_mesh.setVertVectorField(sol_field, ent, i, 0, current_sol.x()+vval[0,j],
      #                                                    current_sol.y()+vval[1,j],
      #                                                    current_sol.z()+vval[2,j])
      pumi_mesh.setVertVectorField(sol_field, ent, i, 0, vval[0,j],
                                                         vval[1,j],
                                                         vval[2,j])

    eid = eid + 1

  pumi_mesh.end(it)

  # do the average since each vertex gets a value from multiple tets
  it = pumi_mesh.begin(0)
  while True:
    ent = pumi_mesh.iterate(it)
    if not ent:
      break
    current_count = pyCore.getScalar(count_field, ent, 0)
    current_sol = pyCore.Vector3()
    pyCore.getVector(sol_field, ent, 0, current_sol)
    # avg_sol = pyCore.Vector3(current_sol.x() / current_count,
    #                          current_sol.y() / current_count,
    #                          current_sol.z() / current_count)
    sol_limit = 5000.0
    avg_sol_x = sol_limit
    avg_sol_y = sol_limit
    avg_sol_z = sol_limit
    if abs(current_sol.x()) < sol_limit:
      avg_sol_x = current_sol.x()
    if abs(current_sol.y()) < sol_limit:
      avg_sol_y = current_sol.y()
    if abs(current_sol.z()) < sol_limit:
      avg_sol_z = current_sol.z()
    avg_sol = pyCore.Vector3(avg_sol_x,
                             avg_sol_y,
                             avg_sol_z)
    # mag = sqrt(avg_sol.x() * avg_sol.x() +
    #          avg_sol.y() * avg_sol.y() +
    #          avg_sol.z() * avg_sol.z())
    mag = math.sqrt(avg_sol.x() * avg_sol.x()+
    	            avg_sol.y() * avg_sol.y()+
    	            avg_sol.z() * avg_sol.z())
    pyCore.setScalar(field_z, ent, 0, mag)
    pyCore.setVector(sol_field, ent, 0, avg_sol)



  pumi_mesh.end(it)
  pumi_mesh.removeField(count_field)
  pyCore.destroyField(count_field)
  # pumi_mesh.removeField(sol_field)
  # pyCore.destroyField(sol_field)
  return field_z


def get_curl_ip_field(pumi_mesh, field_name, field_type, field_order, grid):
  # get the mfem mesh and fespace
  fes = grid.ParFESpace()
  mesh = fes.GetParMesh()


  # create the necessary fields
  curl_field = pyCore.createIPField(pumi_mesh, field_name, field_type, field_order)
  # curl_field_shape = pyCore.getShape(curl_field)

  dim = pumi_mesh.getDimension()

  it = pumi_mesh.begin(dim)
  eid = 0
  while True:
    ent = pumi_mesh.iterate(it)
    if not ent:
      break

    elem_transformation = fes.GetElementTransformation(eid)
    curl_vec = mfem.Vector()
    grid.GetCurl(elem_transformation, curl_vec)

    # for i in range(4):
    #   current_count = pumi_mesh.getVertScalarField(count_field, ent, i, 0)
    #   current_sol = pumi_mesh.getVertVectorField(sol_field, ent, i, 0)
    #   pumi_vid = pumi_mesh.getVertNumbering(numbering, ent, i, 0, 0)
    #   # this is the index of pumi_vid in mfem_vids list
    #   # (note that we need to do this because of ReorientTet call)
    #   j = mfem_vids.index(pumi_vid)
    #   pumi_mesh.setVertScalarField(count_field, ent, i, 0, current_count + 1.0)
    #   pumi_mesh.setVertVectorField(sol_field, ent, i, 0, current_sol.x()+vval[0,j],
    #                                                      current_sol.y()+vval[1,j],
    #                                                      current_sol.z()+vval[2,j])
    # pyCore.setComponents(curl_field, ent, 0, curl_vec.GetData())
    pumi_mesh.setIPxyz(curl_field, ent, 0, curl_vec[0], curl_vec[1], curl_vec[2])

    eid = eid + 1

  pumi_mesh.end(it)

  return curl_field

def limit_refine_level(pumi_mesh, sizefield, level):
  it = pumi_mesh.begin(0)
  while True:
    ent = pumi_mesh.iterate(it)
    if not ent:
      break
    current_size = pumi_mesh.measureSize(ent)
    computed_size = pyCore.getScalar(sizefield, ent, 0)
    if computed_size < current_size / (2**level):
      computed_size = current_size / (2**level)
    if computed_size > current_size:
      computed_size = current_size;
    pyCore.setScalar(sizefield, ent, 0, computed_size)
  pumi_mesh.end(it)


class StdMeshAdaptSolver(StdSolver):
    def panel1_param(self):
        return [#["Initial value setting",   self.init_setting,  0, {},],
                ["physics model",   self.phys_model,  0, {},],
                ["initialize solution only", self.init_only,  3, {"text":""}], 
                ["clear working directory",
                 self.clear_wdir,  3, {"text":""}],
                ["convert to real matrix (complex prob.)",
                 self.assemble_real,  3, {"text":""}],
                ["save parallel mesh",
                 self.save_parmesh,  3, {"text":""}],
                ["use cProfiler",
                 self.use_profiler,  3, {"text":""}],
                ["indicator",   self.mesh_adapt_indicator,  0, {}],
                ["#mesh adapt",   self.mesh_adapt_num,  0, {},],]                

    def attribute_set(self, v):
        super(StdMeshAdaptSolver, self).attribute_set(v)
        v["mesh_adapt_indicator"] = ""
        v["mesh_adapt_num"] = 0
        return v
                 
    def get_panel1_value(self):
        return (#self.init_setting,
                self.phys_model,
                self.init_only, 
                self.clear_wdir,
                self.assemble_real,
                self.save_parmesh,
                self.use_profiler,
                self.mesh_adapt_indicator,
                self.mesh_adapt_num)
    
    def import_panel1_value(self, v):
        #self.init_setting = str(v[0])        
        self.phys_model = str(v[0])
        self.init_only = v[1]                
        self.clear_wdir = v[2]
        self.assemble_real = v[3]
        self.save_parmesh = v[4]
        self.use_profiler = v[5]
        self.mesh_adapt_indicator = v[6]
        self.mesh_adapt_num = int(v[7])
                 
    @debug.use_profiler
    def run(self, engine, is_first = True, return_instance=False):
        dprint1("Entering run", is_first, self.fullpath())
        if self.clear_wdir:
            engine.remove_solfiles()

        instance = StandardMeshAdaptSolver(self, engine)
        instance.set_blk_mask()
        if return_instance: return instance                    
        # We dont use probe..(no need...)
        #instance.configure_probes(self.probe)

        if self.init_only:
            engine.sol = engine.assembled_blocks[1][0]
            instance.sol = engine.sol
        else:
            if is_first:
                instance.assemble()            
                is_first=False
            instance.solve()

        instance.save_solution(ksol = 0,
                               skip_mesh = False, 
                               mesh_only = False,
                               save_parmesh=self.save_parmesh)
        engine.sol = instance.sol        
        dprint1(debug.format_memory_usage())


	# pmesh2 = ParMesh2ParPumiMesh(engine.meshes[0])
	# # pmesh2 = engine.meshes[0]
	# print(pmesh2)
	x = engine.r_x[0]

	par_pumi_mesh = self.root()._par_pumi_mesh

	pumi_mesh = self.root()._pumi_mesh
	pyCore.writeASCIIVtkFiles('before_size_computation', pumi_mesh);

	## projecting the field back to the pumi mesh and then using standard
	## spr which used gradients of the field
	# field_z = get_field_z_averaged(pumi_mesh, "field_z_averaged", pyCore.SCALAR, x)
	# ip_field = pyCore.getGradIPField(field_z, "mfem_grad_ip", 2)

	## transferring the curl of E and using it in spr
	ip_field = get_curl_ip_field(pumi_mesh, "curl_ip_field", pyCore.VECTOR, 1, x)
        size_field = pyCore.getSPRSizeField(ip_field, 0.05)

        limit_refine_level(pumi_mesh, size_field, 5)

	pyCore.writeASCIIVtkFiles('before_adapt', pumi_mesh);


	pumi_mesh.removeField(ip_field)
        pyCore.destroyField(ip_field)
	pyCore.destroyNumbering(pumi_mesh.findNumbering("local_vert_numbering"))

	adapt_input = pyCore.configure(pumi_mesh, size_field)
        adapt_input.shouldFixShape = True
        adapt_input.shouldCoarsen = False
        adapt_input.maximumIterations = 2
        adapt_input.goodQuality = 0.008

        pyCore.adapt(adapt_input)


	pyCore.writeASCIIVtkFiles('after_adapt', pumi_mesh);
	pumi_mesh.writeNative("adapted_mesh.smb")

	## TODO: Loop needs to be closed here
	## some of the functions seem to not working mainly because
	## we can not get a correct ParPumiMesh from a ParMesh object
	## the same way we do in C++
	adapted_mesh = ParPumiMesh(pyCore.PCU_Get_Comm(), pumi_mesh)
	print(type(adapted_mesh))
	print(type(par_pumi_mesh))
	print(id(par_pumi_mesh))
        par_pumi_mesh.UpdateMesh(adapted_mesh)
        par_pumi_mesh.PrintVTK("after_update")


        return is_first

class StandardMeshAdaptSolver(StandardSolver):
    pass

