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

def vector3_to_nparray(vec):
  npa = np.array([vec.x(), vec.y(), vec.z()])
  return npa

def project_np_array_onto_plane(e, normal):
  e_dot_normal = np.dot(e, normal)
  return e - e_dot_normal * normal


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
    # if computed_size > current_size:
    #   computed_size = current_size;
    pyCore.setScalar(sizefield, ent, 0, computed_size)
  pumi_mesh.end(it)

def limit_coarsen(pumi_mesh, sizefield, ratio):
  it = pumi_mesh.begin(0)
  while True:
    ent = pumi_mesh.iterate(it)
    if not ent:
      break
    current_size = pumi_mesh.measureSize(ent)
    computed_size = pyCore.getScalar(sizefield, ent, 0)
    if computed_size > ratio * current_size:
      computed_size = ratio * current_size
    pyCore.setScalar(sizefield, ent, 0, computed_size)
  pumi_mesh.end(it)


def clean_e_field(pumi_mesh,
                  in_field,
                  max_val):

  name = pyCore.getName(in_field)
  clean_name = name+"_clean"
  out_field = pyCore.createField(pumi_mesh,
                                 clean_name,
                                 pyCore.VECTOR,
                                 pyCore.getLagrange(1))

  it = pumi_mesh.begin(0)
  while True:
    ent = pumi_mesh.iterate(it)
    if not ent:
      break
    vec = pyCore.Vector3()
    pyCore.getVector(in_field, ent, 0, vec)

    npvec = vector3_to_nparray(vec)

    for i in range(3):
      if npvec[i] < -max_val:
        npvec[i] = -max_val
      if npvec[i] > max_val:
        npvec[i] = max_val

    new_vec = pyCore.Vector3(npvec[0], npvec[1], npvec[2])

    pyCore.setVector(out_field, ent, 0, new_vec)

  pumi_mesh.end(it)
  return out_field

def compute_relative_size(pumi_mesh,
                          size):

  name = pyCore.getName(size)
  relative_name = "relative_"+name
  out_field = pyCore.createField(pumi_mesh,
                                 relative_name,
                                 pyCore.SCALAR,
                                 pyCore.getLagrange(1))

  it = pumi_mesh.begin(0)
  while True:
    ent = pumi_mesh.iterate(it)
    if not ent:
      break
    current_size = pumi_mesh.measureSize(ent)
    s = pyCore.getScalar(size, ent, 0)
    pyCore.setScalar(out_field, ent, 0, s/current_size)

  pumi_mesh.end(it)
  return out_field

def compute_phase_amplitude_fields(pumi_mesh,
                                   field_real,
                                   field_imag,
                                   phase_field,
                                   amplitude_field,
                                   plane):
  it = pumi_mesh.begin(0)
  while True:
    ent = pumi_mesh.iterate(it)
    if not ent:
      break
    real_e = pyCore.Vector3()
    imag_e = pyCore.Vector3()
    pyCore.getVector(field_real, ent, 0, real_e)
    pyCore.getVector(field_imag, ent, 0, imag_e)

    npreal_e = vector3_to_nparray(real_e)
    npimag_e = vector3_to_nparray(imag_e)

    if (plane != -1):
      npreal_e[plane] = 0.0
      npimag_e[plane] = 0.0

    real_e_mag = np.linalg.norm(npreal_e);
    imag_e_mag = np.linalg.norm(npimag_e);

    emag = math.sqrt(real_e_mag*real_e_mag + imag_e_mag*imag_e_mag)
    phase = math.atan2(imag_e_mag, real_e_mag)
    pyCore.setScalar(phase_field, ent, 0, phase)
    pyCore.setScalar(amplitude_field, ent, 0, emag)

  pumi_mesh.end(it)


def compute_phase_amplitude_fields_radial_x(pumi_mesh,
                                            field_real,
                                            field_imag,
                                            phase_field,
                                            amplitude_field):

  it = pumi_mesh.begin(0)
  while True:
    ent = pumi_mesh.iterate(it)
    if not ent:
      break
    loc = pyCore.Vector3()
    pumi_mesh.getPoint(ent, 0, loc)
    p = vector3_to_nparray(loc)
    # adjust the center
    p[2] = p[2] - 0.2
    # radial and tangential vectors
    r_vec = np.array([0.,  p[1], p[2]])
    t_vec = np.array([0., -p[2], p[1]])

    r_norm = np.linalg.norm(r_vec)
    t_norm = np.linalg.norm(t_vec)

    if r_norm < 0.000001:
      r_vec = np.array([0.,1.,0.])
      t_vec = np.array([0.,0.,1.])
    else:
      r_vec = r_vec / r_norm
      t_vec = t_vec / t_norm


    real_e = pyCore.Vector3()
    imag_e = pyCore.Vector3()
    pyCore.getVector(field_real, ent, 0, real_e)
    pyCore.getVector(field_imag, ent, 0, imag_e)

    real_e_projected = project_np_array_onto_plane(vector3_to_nparray(real_e), t_vec)
    imag_e_projected = project_np_array_onto_plane(vector3_to_nparray(imag_e), t_vec)


    real_e_mag = np.linalg.norm(real_e_projected);
    imag_e_mag = np.linalg.norm(imag_e_projected);

    emag = math.sqrt(real_e_mag*real_e_mag + imag_e_mag*imag_e_mag)
    phase = math.atan2(imag_e_mag, real_e_mag)
    pyCore.setScalar(phase_field, ent, 0, phase)
    pyCore.setScalar(amplitude_field, ent, 0, emag)

  pumi_mesh.end(it)

def get_field_component(pumi_mesh, in_field, component):
  name = ""
  if component == 0:
    name = "emag"
  elif component == 1:
    name = "ex"
  elif component == 2:
    name = "ey"
  elif component == 3:
    name = "ez"
  else:
    assert False, "wrong component: " + component

  field_component = pyCore.createFieldOn(pumi_mesh,
					name,
					pyCore.SCALAR)
  it = pumi_mesh.begin(0)
  while True:
    ent = pumi_mesh.iterate(it)
    if not ent:
      break
    vec = pyCore.Vector3()
    pyCore.getVector(in_field, ent, 0, vec)
    if component == 0:
      outvalue = vec.x() * vec.x() + vec.y() * vec.y() + vec.z() * vec.z()
      outvalue = math.sqrt(outvalue)
    elif component == 1:
      outvalue = vec.x()
    elif component == 2:
      outvalue = vec.y()
    elif component == 3:
      outvalue = vec.z()
    else:
      assert False, ""
    pyCore.setScalar(field_component, ent, 0, outvalue)
  pumi_mesh.end(it)

  return field_component

def ignore_refine_in_model_region(pumi_mesh, sizefield, region_tag):
  it = pumi_mesh.begin(0)
  while True:
    ent = pumi_mesh.iterate(it)
    if not ent:
      break
    mtag = pumi_mesh.getModelTag(pumi_mesh.toModel(ent))
    mdim = pumi_mesh.getModelType(pumi_mesh.toModel(ent))
    if region_tag == mtag or pumi_mesh.isBoundingModelRegion(region_tag, mdim, mtag):
      current_size = pumi_mesh.measureSize(ent)
      pyCore.setScalar(sizefield, ent, 0, current_size)
  pumi_mesh.end(it)


class StdMeshAdaptSolver(StdSolver):
    def panel1_param(self):
        return [#["Initial value setting",   self.init_setting,  0, {},],
                ["physics model",   self.phys_model,  0, {},],
                ["initialize solution only", self.init_only,  3, {"text":""}], 
                [None,
                 self.clear_wdir,  3, {"text":"clear working directory"}],
                [None,
                 self.assemble_real,  3, {"text":"convert to real matrix (complex prob.)"}],
                [None,
                 self.save_parmesh,  3, {"text":"save parallel mesh"}],
                [None,
                 self.use_profiler,  3, {"text":"use profiler"}],
                ["indicator",   self.mesh_adapt_indicator,  0, {},],
                ["#mesh adapt",   self.mesh_adapt_num,  0, {},],
                ["adapt ratio",   self.mesh_adapt_ar,  0, {},]]

    def attribute_set(self, v):
        super(StdMeshAdaptSolver, self).attribute_set(v)
        v["mesh_adapt_indicator"] = "E"
        v["mesh_adapt_num"] = 0
        v["mesh_adapt_ar"] = 0.1
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
                self.mesh_adapt_num,
                self.mesh_adapt_ar)
    
    def import_panel1_value(self, v):
        #self.init_setting = str(v[0])        
        self.phys_model = str(v[0])
        self.init_only = v[1]                
        self.clear_wdir = v[2]
        self.assemble_real = v[3]
        self.save_parmesh = v[4]
        self.use_profiler = v[5]
        self.mesh_adapt_indicator = v[6]
        self.mesh_adapt_num = v[7]
        self.mesh_adapt_ar = v[8]

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

        # get the fespaces order
        # TODO: there must be a better way to do this
        assert len(engine.model['Phys'].keys())==1, "only can handle 1 physics model"
        order = 0
        for k in engine.model['Phys'].keys():
            order = engine.model['Phys'][k].order

        # first solve outside of the loop
        if self.init_only:
            engine.sol = engine.assembled_blocks[1][0]
            instance.sol = engine.sol
        else:
            if is_first:
                instance.assemble()
                is_first=False
            instance.solve()

        adapt_loop_no = 0;
        while adapt_loop_no < int(self.mesh_adapt_num):
            instance.save_solution(ksol = 0,
                                skip_mesh = False,
                                mesh_only = False,
                                save_parmesh=True)
            engine.sol = instance.sol
            dprint1(debug.format_memory_usage())


            x = engine.r_x[0]
            y = engine.i_x[0]
            # par_pumi_mesh = self.root()._par_pumi_mesh

            par_pumi_mesh = ParMesh2ParPumiMesh(engine.meshes[0])
            # par_pumi_emesh = ParMesh2ParPumiMesh(engine.emeshes[0])
            pumi_mesh = self.root()._pumi_mesh

            # transfer the e field to a nedelec field in pumi
            e_real = pyCore.createField(pumi_mesh,
                                        "e_real_nd",
                                        pyCore.SCALAR,
                                        pyCore.getNedelec(order))
            e_imag = pyCore.createField(pumi_mesh,
                                        "e_imag_nd",
                                        pyCore.SCALAR,
                                        pyCore.getNedelec(order))

            e_real_projected = pyCore.createField(pumi_mesh,
                                                  "e_real_projected",
                                                   pyCore.VECTOR,
                                                   pyCore.getLagrange(1))
            e_imag_projected = pyCore.createField(pumi_mesh,
                                                  "e_imag_projected",
                                                   pyCore.VECTOR,
                                                   pyCore.getLagrange(1))

            par_pumi_mesh.NedelecFieldMFEMtoPUMI(pumi_mesh, x, e_real)
            par_pumi_mesh.NedelecFieldMFEMtoPUMI(pumi_mesh, y, e_imag)
            pyCore.projectNedelecField(e_real_projected, e_real)
            pyCore.projectNedelecField(e_imag_projected, e_imag)

            e_real_projected_clean = clean_e_field(pumi_mesh, e_real_projected, 1.25)
            e_imag_projected_clean = clean_e_field(pumi_mesh, e_imag_projected, 1.25)

            phase_r = pyCore.createField(pumi_mesh,
                                       "phase_radial",
                                       pyCore.SCALAR,
                                       pyCore.getLagrange(1))
            amplitude_r = pyCore.createField(pumi_mesh,
                                           "amplitude_radial",
                                           pyCore.SCALAR,
                                           pyCore.getLagrange(1))

            compute_phase_amplitude_fields_radial_x(pumi_mesh,
                                                    e_real_projected_clean,
                                                    e_imag_projected_clean,
                                                    phase_r,
                                                    amplitude_r)


            pumi_mesh.removeField(e_real)
            pumi_mesh.removeField(e_imag)
            pyCore.destroyField(e_real)
            pyCore.destroyField(e_imag)


            # pumi_projected_nedelec_field = pumi_projected_nedelec_field_real
            # print("user choose the indicator ", self.mesh_adapt_indicator)
            # indicator_field = 0;
            # if self.mesh_adapt_indicator == "E":
            #   indicator_field = pumi_projected_nedelec_field
            # elif self.mesh_adapt_indicator == "Emag":
            #   indicator_field = get_field_component(pumi_mesh, pumi_projected_nedelec_field, 0)
            # elif self.mesh_adapt_indicator == "Ex":
            #   indicator_field = get_field_component(pumi_mesh, pumi_projected_nedelec_field, 1)
            # elif self.mesh_adapt_indicator == "Ey":
            #   indicator_field = get_field_component(pumi_mesh, pumi_projected_nedelec_field, 2)
            # elif self.mesh_adapt_indicator == "Ez":
            #   indicator_field = get_field_component(pumi_mesh, pumi_projected_nedelec_field, 3)
            # else:
            #   assert False, "wrong indicator selected"

            ip_field = pyCore.getGradIPField(amplitude_r, "mfem_grad_ip", 2)
            size_field = pyCore.getTargetSPRSizeField(ip_field, int(pumi_mesh.count(3)*2) , 0.125, 2.)
            limit_refine_level(pumi_mesh, size_field, 3)
            limit_coarsen(pumi_mesh, size_field, 1.5)
            relative_size_field = compute_relative_size(pumi_mesh, size_field)

            native_name = "pumi_mesh_before_adapt_"+str(adapt_loop_no)+"_.smb";
            pumi_mesh.writeNative(native_name)

            before_prefix = "before_adapt_"+str(adapt_loop_no);
            pyCore.writeASCIIVtkFiles(before_prefix, pumi_mesh);

            pumi_mesh.removeField(ip_field)
            pyCore.destroyField(ip_field)

            adapt_input = pyCore.configure(pumi_mesh, size_field)
            adapt_input.shouldFixShape = True
            adapt_input.shouldCoarsen = True
            adapt_input.maximumIterations = 3
            adapt_input.goodQuality = 0.35 * 0.35 * 0.35 # mean-ratio cubed

            pyCore.adaptVerbose(adapt_input)

            after_prefix = "after_adapt_"+str(adapt_loop_no);
            pyCore.writeASCIIVtkFiles(after_prefix, pumi_mesh);

            native_name = "pumi_mesh_after_adapt_"+str(adapt_loop_no)+".smb";
            pumi_mesh.writeNative(native_name)

            # clean up rest of the fields
            pumi_mesh.removeField(e_real_projected)
            pyCore.destroyField(e_real_projected)
            pumi_mesh.removeField(e_real_projected_clean)
            pyCore.destroyField(e_real_projected_clean)

            pumi_mesh.removeField(e_imag_projected)
            pyCore.destroyField(e_imag_projected)
            pumi_mesh.removeField(e_imag_projected_clean)
            pyCore.destroyField(e_imag_projected_clean)

            pumi_mesh.removeField(phase_r)
            pyCore.destroyField(phase_r)

            pumi_mesh.removeField(size_field)
            pyCore.destroyField(size_field)

            pumi_mesh.removeField(relative_size_field)
            pyCore.destroyField(relative_size_field)

            pumi_mesh.removeField(amplitude_r)

            adapted_mesh = mfem.ParMesh(pyCore.PCU_Get_Comm(), pumi_mesh)
            # add the boundary attributes
            dim = pumi_mesh.getDimension()
            it = pumi_mesh.begin(dim-1)
            bdr_cnt = 0
            while True:
                e = pumi_mesh.iterate(it)
                if not e: break
                model_tag  = pumi_mesh.getModelTag(pumi_mesh.toModel(e))
                model_type = pumi_mesh.getModelType(pumi_mesh.toModel(e))
                if model_type == (dim-1):
                    adapted_mesh.GetBdrElement(bdr_cnt).SetAttribute(model_tag)
                    bdr_cnt += 1
            pumi_mesh.end(it)
            it = pumi_mesh.begin(dim)
            elem_cnt = 0
            while True:
                e = pumi_mesh.iterate(it)
                if not e: break
                model_tag  = pumi_mesh.getModelTag(pumi_mesh.toModel(e))
                model_type = pumi_mesh.getModelType(pumi_mesh.toModel(e))
                if model_type == dim:
                    adapted_mesh.SetAttribute(elem_cnt, model_tag)
                    elem_cnt += 1
            pumi_mesh.end(it)
            adapted_mesh.SetAttributes()

            par_pumi_mesh.UpdateMesh(adapted_mesh)
            # update the _par_pumi_mesh and _pumi_mesh as well

            # self.root()._par_pumi_mesh = par_pumi_mesh
            # self.root()._pumi_mesh = pumi_mesh

            # engine.meshes[0] = mfem.ParMesh(pyCore.PCU_Get_Comm(), par_pumi_mesh)
            engine.emeshes[0] = engine.meshes[0]
            mfem_prefix_1 = "after_update_using_par_pmesh_"+str(adapt_loop_no)+".vtk"
            mfem_prefix_2 = "after_update_using_meshes0_"+str(adapt_loop_no)+".vtk"
            par_pumi_mesh.PrintVTK(mfem_prefix_1)
            engine.meshes[0].PrintVTK(mfem_prefix_2)
            engine.meshes[0].Print("test_mesh_after_update.mesh")
            engine.emeshes[0].PrintVTK("emesh.vtk")
            engine.emeshes[0].Print("emesh.mesh")

            # reorient the new mesh
            engine.meshes[0].ReorientTetMesh()

            # the rest of the updates happen here
            instance.ma_update_form_sol_variables()
            instance.ma_init()
            instance.set_blk_mask()
            instance.ma_update_assemble()
            instance.solve()
            adapt_loop_no = adapt_loop_no + 1
        return is_first

class StandardMeshAdaptSolver(StandardSolver):
    def ma_update_assemble(self, inplace=True):
        engine = self.engine
        phys_target = self.get_phys()
        phys_range = self.get_phys_range()

        engine.run_verify_setting(phys_target, self.gui)
        engine.run_assemble_mat(phys_target, phys_range, update=False)
        engine.run_assemble_b(phys_target, update=False)
        self.engine.run_assemble_blocks(self.compute_A,self.compute_rhs,inplace=True,update=False)
        self.assembled = True
    def ma_update_form_sol_variables(self):
        engine = self.engine
        phys_target = self.get_phys()
        phys_range = self.get_phys_range()
        num_matrix = engine.n_matrix
        engine.set_formblocks(phys_target, phys_range, num_matrix)
        # mesh should be already updated
        # not all of run_alloc_sol needs to be called here
        # so call the ones that are necessary
        for phys in phys_range:
            engine.initialize_phys(phys, update=True)
        for j in range(engine.n_matrix):
            engine.accept_idx = j
            engine.r_x.set_no_allocator()
            engine.i_x.set_no_allocator()
    def ma_init(self):
        engine = self.engine
        phys_target = self.get_phys()
        phys_range = self.get_phys_range()
        engine.run_apply_essential(phys_target, phys_range, update=False)
        engine.run_fill_X_block(update=False)
