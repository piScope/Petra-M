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
        while adapt_loop_no < self.mesh_adapt_num:
            instance.save_solution(ksol = 0,
                                skip_mesh = False,
                                mesh_only = False,
                                save_parmesh=self.save_parmesh)
            engine.sol = instance.sol
            dprint1(debug.format_memory_usage())


            x = engine.r_x[0]
            # par_pumi_mesh = self.root()._par_pumi_mesh

            par_pumi_mesh = ParMesh2ParPumiMesh(engine.meshes[0])
            # par_pumi_emesh = ParMesh2ParPumiMesh(engine.emeshes[0])
            pumi_mesh = self.root()._pumi_mesh

            # transfer the e field to a nedelec field in pumi
            pumi_nedelec_field = pyCore.createField(pumi_mesh,
                                                    "e_field",
                                                    pyCore.SCALAR,
                                                    pyCore.getNedelec(order))

            pumi_projected_nedelec_field = pyCore.createField(pumi_mesh,
                                                            "projected_e_field",
                                                            pyCore.VECTOR,
                                                            pyCore.getLagrange(1))

            par_pumi_mesh.NedelecFieldMFEMtoPUMI(pumi_mesh, x, pumi_nedelec_field)
            pyCore.projectNedelecField(pumi_projected_nedelec_field, pumi_nedelec_field)

            # native file output for debug
            native_name = "pumi_mesh_e_field_ma_iter_" + str(adapt_loop_no) + ".smb"
            print("native_name is ", native_name)
            pumi_mesh.writeNative(native_name)


            pumi_mesh.removeField(pumi_nedelec_field)
            pyCore.destroyField(pumi_nedelec_field)


            print("user choose the indicator ", self.mesh_adapt_indicator)
            indicator_field = 0;
            if self.mesh_adapt_indicator == "E":
              indicator_field = pumi_projected_nedelec_field
            elif self.mesh_adapt_indicator == "Emag":
              indicator_field = get_field_component(pumi_mesh, pumi_projected_nedelec_field, 0)
            elif self.mesh_adapt_indicator == "Ex":
              indicator_field = get_field_component(pumi_mesh, pumi_projected_nedelec_field, 1)
            elif self.mesh_adapt_indicator == "Ey":
              indicator_field = get_field_component(pumi_mesh, pumi_projected_nedelec_field, 2)
            elif self.mesh_adapt_indicator == "Ez":
              indicator_field = get_field_component(pumi_mesh, pumi_projected_nedelec_field, 3)
            else:
              assert False, "wrong indicator selected"

            ip_field = pyCore.getGradIPField(indicator_field, "mfem_grad_ip", 2)

            ## transferring the curl of E and using it in spr
            size_field = pyCore.getSPRSizeField(ip_field, float(self.mesh_adapt_ar))


            limit_refine_level(pumi_mesh, size_field, 5)
            before_prefix = "before_adapt_"+str(adapt_loop_no);
            pyCore.writeASCIIVtkFiles(before_prefix, pumi_mesh);


            pumi_mesh.removeField(ip_field)
            pyCore.destroyField(ip_field)

            adapt_input = pyCore.configure(pumi_mesh, size_field)
            adapt_input.shouldFixShape = True
            adapt_input.shouldCoarsen = True
            adapt_input.maximumIterations = 6
            adapt_input.goodQuality = 0.35 * 0.35 * 0.35 # mean-ratio cubed

            pyCore.adaptVerbose(adapt_input)

            after_prefix = "after_adapt_"+str(adapt_loop_no);
            pyCore.writeASCIIVtkFiles(after_prefix, pumi_mesh);

            # clean up rest of the fields
            pumi_mesh.removeField(pumi_projected_nedelec_field)
            pyCore.destroyField(pumi_projected_nedelec_field)
            pumi_mesh.removeField(size_field)
            pyCore.destroyField(size_field)
            pumi_mesh.removeField(indicator_field)
            pyCore.destroyField(indicator_field)

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
            instance.ma_update_form_sol_variables(engine)
            instance.ma_init(engine)
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
    def ma_update_form_sol_variables(self, engine):
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
    def ma_init(self, engine):
        phys_target = self.get_phys()
        phys_range = self.get_phys_range()
        engine.run_apply_essential(phys_target, phys_range, update=False)
        engine.run_fill_X_block(update=False)
