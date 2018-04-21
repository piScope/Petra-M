#!/bin/env python
import sys
import os
import numpy as np
import scipy.sparse
from warnings import warn

from petram.mfem_config import use_parallel
if use_parallel:
   from petram.helper.mpi_recipes import *
   import mfem.par as mfem   
else:
   import mfem.ser as mfem
import mfem.common.chypre as chypre

#these are only for debuging
from mfem.common.parcsr_extra import ToScipyCoo
from mfem.common.mpi_debug import nicePrint

from petram.model import Domain, Bdry, ModelDict
import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('Engine')
from petram.helper.matrix_file import write_coo_matrix, write_vector


#groups = ['Domain', 'Boundary', 'Edge', 'Point', 'Pair']
groups = ['Domain', 'Boundary', 'Pair']


def iter_phys(phys_targets, *args):
    for phys in phys_targets:
        yield [phys] + [a[phys] for a in args]
        
def enum_fes(phys, *args):
    '''
    enumerate fespaces under a physics module
    '''
    for k in range(len(args[0][phys])):
        yield ([k] + [(None if a[phys] is None else a[phys][k][1])
                     for a in args])
class Engine(object):
    def __init__(self, modelfile='', model = None):
        if modelfile != '':
           import cPickle as pickle
           model = pickle.load(open(modelfile, 'rb'))
           
        self.set_model(model)
        if not 'InitialValue' in model:
           idx = model.keys().index('Phys')+1
           from petram.mfem_model import MFEM_InitRoot
           model.insert_item(idx, 'InitialValue', MFEM_InitRoot())
        from petram.mfem_model import has_geom
        if not 'Geom' in model and has_geom:
           from petram.geom.geom_model import MFEM_GeomRoot
           model.insert_item(1, 'Geometry', MFEM_GeomRoot())
           
        
        self.is_assembled = False
        self.is_initialized = False
        #
        # I am not sure if there is any meaing to make mesh array at
        # this point...
        self.meshes = []
        self.emeshes = []

        ## number of matrices to be filled
        ##  
        ##  M0 * x_n = M1 * x_n-1 + M2 * x_n-2 + M3 * x_n-3... Mn x_0 + rhs_vector
        self.fec = {}        
        self.fespaces = {}
        self._num_matrix= 1
        self._access_idx= -1
        self._dep_vars = []
        self._isFESvar = []

        self.alloc_flag = {}
        self.max_bdrattr = -1
        self.max_attr = -1        
        
        # place holder : key is base physics modules, such as EM3D1...
        #
        # for example : self.r_b['EM3D1'] = [LF for E, LF for psi]
        #               physics moduel provides a map form variable name to index.

        self.case_base = 0
        
    @property
    def n_matrix(self):
        return self._num_matrix
    @n_matrix.setter
    def n_matrix(self, i):
        self._num_matrix= i
        
    def set_formblocks(self, phys_target, n_matrix):
        '''
        This version assembles a linear system as follows

            M_0 y_n = M_1 y_1 + M_2 y_2 + ... + M_n-1 y_n-1 + b
  
        y_n   (y[0])  is unknonw (steady-state solution or next time step value)
        y_n-1 (y[-1]) is the last time step value.


        solver (=MUMPS caller) should perform back interpolation

        final matrix looks like as follows + off-diagomal (betwween-physics
        coupling)
           [phys1_space1       A                                          ]
           [     B        phys1_space2                                    ]    
           [                            phys2_space2       C              ]    
           [                                 D        phys2_space2        ]    
                                     ......
                                     ......
           [                                                              ],
        where A, B, C and D are in-physics coupling    
        '''
        from petram.helper.formholder import FormBlock
        
        self.n_matrix = n_matrix
        self.collect_dependent_vars(phys_target)
        
        n_fes = len(self.fes_vars)
        n_mat = self.n_matrix
        
        self._matrix_blocks = [None for k in range(n_mat)]
        
        self._r_a = [FormBlock((n_fes, n_fes), new = self.alloc_bf, mixed_new=self.alloc_mbf)
                     for k in range(n_mat)]
        self._i_a = [FormBlock((n_fes, n_fes),new = self.alloc_bf, mixed_new=self.alloc_mbf)
                     for k in range(n_mat)]
        self._r_x = [FormBlock(n_fes, new = self.alloc_gf) for k in range(n_mat)]
        self._i_x = [FormBlock(n_fes, new = self.alloc_gf) for k in range(n_mat)]
        
        self.r_b = FormBlock(n_fes, new = self.alloc_lf)
        self.i_b = FormBlock(n_fes, new = self.alloc_lf)
        
        self.interps = {}
        self.projections = {}        
        self.gl_ess_tdofs = {n:[] for n in self.fes_vars}
        self.ess_tdofs = {n:[] for n in self.fes_vars}        
        
        self._extras = [None for i in range(n_mat)]
        
    @property
    def access_idx(self):
        return self._access_idx
    @access_idx.setter
    def access_idx(self, i):
        self._access_idx= i

    # bilinearforms
    @property
    def r_a(self):
        return self._r_a[self._access_idx]
    @r_a.setter
    def r_a(self, v):
        self._r_a[self._access_idx] = v
    @property
    def i_a(self):
        return self._i_a[self._access_idx]
    @i_a.setter
    def i_a(self, v):
        self._i_a[self._access_idx] = v

    # grid functions 
    @property
    def r_x(self):
        return self._r_x[self._access_idx]
    @r_x.setter
    def r_x(self, v):
        self._r_x[self._access_idx] = v
    @property
    def i_x(self):
        return self._i_x[self._access_idx]
    @i_x.setter
    def i_x(self, v):
        self._i_x[self._access_idx] = v
        
    @property
    def extras(self):
        return self._extras[self._access_idx]
    @extras.setter     
    def extras(self, v):
        self._extras[self._access_idx] = v

    @property
    def matvecs(self):
        return self._matrix_block[self._access_idx]
    @matvecs.setter     
    def matvecs(self, v):
        self._matrix_block[self._access_idx] = v
        
     
    def set_model(self, model):
        self.model = model
        self.is_assembled = False
        self.is_initialized = False        
        self.meshes = []
        self.emeshes = []
        if model is None: return
            
        self.alloc_flag  = {}
        # below is to support old version
        from mesh.mesh_model import MeshGroup
        g = None
        items = []

        for k in model['Mesh'].keys():
            if  not hasattr(model['Mesh'][k], 'isMeshGroup'):
                if g is None:
                    name = model['Mesh'].add_item('MeshGroup', MeshGroup)
                    g = model['Mesh'][name]
                items.append((k, model['Mesh'][k]))

        for name, obj in items:
            del model['Mesh'][name]
            model['Mesh']['MeshGroup1'][name] = obj


    def get_mesh(self, idx = 0, mm = None):
        if len(self.meshes) == 0: return None
        if mm is not None:
           idx = mm.get_root_phys().mesh_idx
        return self.meshes[idx]
     
    def get_smesh(self, idx = 0, mm = None):
        if len(self.smeshes) == 0: return None
        if mm is not None:
           idx = mm.get_root_phys().mesh_idx
        return self.smeshes[idx]
     
    def get_emesh(self, idx = 0, mm = None):
        if len(self.emeshes) == 0: return None
        if mm is not None:
           idx = mm.get_root_phys().emesh_idx
        return self.emeshes[idx]

    def get_emesh_idx(self, mm = None, name=None):
        if len(self.emeshes) == 0: return -1
        if mm is not None:
           return mm.get_root_phys().emesh_idx

        if name is None:
           for item in self.model['Phys']:
              mm = self.model['Phys'][item]
              if not mm.enabled: continue
              if name in mm.dep_vars():
                  return mm.emesh_idx
        return -1
     
    '''    
    def generate_fespace(self, phys):    
        fecs = phys.get_fecs(self)
        if (phys.element == 'ND_FECollection'):
            self.mesh.ReorientTetMesh()
        self.fespaces[phys] = [self.new_fespace(fec) for fec in fecs]
    ''' 
    def preprocess_modeldata(self, dir = None):
        '''
        do everything it takes to run a newly built
        model data strucutre.
        used from text script execution
        '''
        import os
        from __main__ import __file__ as mainfile        
        model = self.model
        model['General'].run()
        self.run_mesh_serial()
        self.assign_sel_index()

        if dir is None:
            dir = os.path.dirname(os.path.realpath(mainfile))           
        for node in model.walk():
            if node.has_ns() and node.ns_name is not None:
                node.read_ns_script_data(dir = dir)
        self.build_ns()

        self.run_preprocess()  # this must run when mesh is serial
        
        self.run_mesh() # make ParMesh and Par-Extended-Mesh

        from petram.mfem_config import use_parallel
        if use_parallel:        
            self.emeshes = []
            for k in self.model['Phys'].keys():
                phys = self.model['Phys'][k]
                self.run_mesh_extension(phys)
                self.allocate_fespace(phys)
                
        solver = model["Solver"].get_active_solvers()
        return solver
     
    def run_config(self):
        '''
        this runs model['General'] and
        fill namespace dict
        '''
        self.model['General'].run()
        self.build_ns()
      
    def run_preprocess(self, ns_folder = None, data_folder = None):
        if ns_folder is not None:
           self.preprocess_ns(ns_folder, data_folder)

        from .model import Domain, Bdry               
        for k in self.model['Phys'].keys():
            phys = self.model['Phys'][k]
            self.run_mesh_extension(phys)
            self.allocate_fespace(phys)
            self.assign_sel_index(phys)
            for node in phys.walk():
                if not node.enabled: continue
                node.preprocess_params(self)
        for k in self.model['InitialValue'].keys():
            init = self.model['InitialValue'][k]
            init.preprocess_params(self)

    def run_verify_setting(self, phys_target, solver):
        for phys in phys_target:
            for mm in phys.walk():
                if not mm.enabled: continue
                error, txt, long_txt = mm.verify_setting()
                assert error, mm.fullname() + ":" + long_txt

        for mm in solver.walk():
                if not mm.enabled: continue
                error, txt, long_txt = mm.verify_setting()           
                assert error, mm.fullname() + ":" + long_txt
                

    #  mesh manipulation
    #
    def run_mesh_extension(self, phys):
        from petram.mesh.mesh_extension import MeshExt, generate_emesh
        from petram.mesh.mesh_model import MFEMMesh

        if len(self.emeshes) == 0:
            self.emeshes = self.meshes[:]
            for j in range(len(self.emeshes)):
                self.emesh_data.add_default_info(j)
        info = phys.get_mesh_ext_info()
        idx = self.emesh_data.add_info(info)

        phys.emesh_idx = idx
        if len(self.emeshes) <= idx: 
            m = generate_emesh(self.emeshes, info)
            self.emeshes.extend([None]*(1+idx-len(self.emeshes)))
            self.emeshes[idx] = m
        
    #
    #  assembly 
    #
    def run_alloc_sol(self, phys_target = None):
        '''
        allocate fespace and gridfunction (unknowns)
        apply essentials
        define model variables

        alloc_flag is used to avoid repeated allocation.
        '''

        allocated_phys = []
        for phys in phys_target:
           try:
              if self.alloc_flag[phys.name()]: alloced_phys.append[phys.name()]
           except:
              pass
        phys_target = [phys for phys in phys_target
                       if not phys.name() in allocated_phys]
        dprint1("allocating fespace/sol vector for " + str(phys_target))
        
        for phys in phys_target:
            self.run_update_param(phys)
        for phys in phys_target:
            self.initialize_phys(phys)
            
        for j in range(self.n_matrix):
            self.access_idx = j
            self.r_x.set_no_allocator()
            self.i_x.set_no_allocator()            
            

        from petram.helper.variables import Variables
        variables = Variables()

        self.access_idx = -1        
        for phys in phys_target:
            for name in phys.dep_vars:
                ifes = self.ifes(name)
                rgf = self.r_x[ifes]
                igf = self.i_x[ifes]
                phys.add_variables(variables, name, rgf, igf)

        keys = self.model._variables.keys()
        self.model._variables.clear()
        dprint1("===  List of variables ===")
        dprint1(variables)
        for k in variables.keys():
           self.model._variables[k] = variables[k]
        self.is_initialized = True
        
        for phys in phys_target:        
            self.alloc_flag[phys.name()] = True

    @property
    def isInitialized(self):
        return  self.is_initialized

           
    def run_apply_init(self, phys_target, mode,
                       init_value=0.0, init_path=''):
        # mode
        #  0: zero
        #  1: init to constant
        #  2: use init panel values
        #  3: load file
        #  4: do nothing
        for j in range(self.n_matrix):
           self.access_idx = j                      
           for phys in phys_target:
              names = phys.dep_vars
              if mode == 0:
                  for name in names:
                      ifes = self.ifes(name)
                      rgf = self.r_x[ifes]
                      igf = self.i_x[ifes]
                      rgf.Assign(0.0)
                      if igf is not None: igf.Assign(0.0)
              elif mode == 1:
                  for name in names:
                      ifes = self.ifes(name)
                      rgf = self.r_x[ifes]
                      igf = self.i_x[ifes]
                      rgf.Assign(init_value)                      
                      if igf is not None: igf.Assign(init_value)
              elif mode == 2: # apply Einit
                  self.apply_init_from_init_panel(phys)
              elif mode == 3:
                  self.apply_init_from_file(phys, init_path)              
              elif mode == 4:
                  pass
              else: #
                  raise NotImplementedError(
                            "unknown init mode")
            
    def run_apply_essential(self, phys_target):
        for phys in phys_target:
            self.gather_essential_tdof(phys)
        self.collect_all_ess_tdof()
        
        for j in range(self.n_matrix):
            self.access_idx = j
            for phys in phys_target:
                self.apply_essential(phys)

    def run_assemble_mat(self, phys_target=None,):
        #for phys in phys_target:
        #    self.gather_essential_tdof(phys)

        for phys in phys_target:       
            self.assemble_interp(phys)     ## global interpolation (periodic BC)
            self.assemble_projection(phys) ## global interpolation (mesh coupling)


        self.extras_mm = {}        
        for j in range(self.n_matrix):
            self.access_idx = j
            
            for phys in phys_target:
                self.fill_bf(phys)
                self.fill_mixed(phys)
                #self.fill_coupling(phys)                        
            self.r_a.set_no_allocator()
            self.i_a.set_no_allocator()
            for form in self.r_a: form.Assemble()
            for form in self.i_a: form.Assemble()
            
            self.extras = {}        
            for phys in phys_target:               
                self.assemble_extra(phys, phys_target)
            self.aux_ops = {}
            #for phys in phys_target:               
            self.assemble_aux_ops(phys_target)
                
        return

    def run_assemble_rhs(self, phys_target = None):
        '''
        assemble only RHS
 
        bilinearform should be assmelbed before-hand
        note that self.r_b, self.r_x, self.i_b, self.i_x 
        are reused. And, since r_b and r_x shares the
        data, and i_b and i_x do too, we need to be careful
        to copy the result (b arrays) to other place to call 
        this. When MUMPS is used, the data ia gathered to
        root node at the end of each assembly process. When
        other solve is added, this must be taken care. 
        '''
        for phys in phys_target:
            self.run_update_param(phys)

        for phys in phys_target:
            self.gather_essential_tdof(phys)
        self.collect_all_ess_tdof()            

        self.access_idx = 0
        for phys in phys_target:
            self.fill_lf(phys)

        self.r_b.set_no_allocator()
        self.i_b.set_no_allocator() 
        for form in self.r_b:
            form.Assemble()
        for form in self.i_b:
            form.Assemble()
           
        return

    def run_assemble_blocks(self, compute_A, compute_rhs,
                            inplace = True):
        '''
        assemble M, B, X blockmatrices.

        in parallel, inplace = False makes sure that blocks in A and RHS  
        are not shared by M, B, X
        '''
        M, B, X = self.prepare_blocks()

        self.fill_M_B_X_blocks(M, B, X)
        #B.save_to_file("B")
        #M[0].save_to_file("M0")        
        #M[1].save_to_file("M1")
        #X[0].save_to_file("X0")
        
        A = compute_A(M, B, X)          # solver determins A
        RHS = compute_rhs(M, B, X)      # solver determins RHS
                      
        #for m in M: 
        #    A.check_shared_id(m)
        #for x in X: 
        #    B.check_shared_id(x)
        
        #RHS.save_to_file("RHSbefore")                
        A, Ae = self.fill_BCeliminate_matrix(A, inplace=inplace)     # generate Ae
        #M[0].save_to_file("M0there")                        
        #Ae.save_to_file("Ae")
        RHS = self.eliminateBC(Ae, X[0], RHS)       # modify RHS and
        A, RHS = self.apply_interp(A, RHS)  # A and RHS is modifedy by global DoF coupling P
        #M[0].save_to_file("M0there2")                        
        #M[1].save_to_file("M1")
        #X[0].save_to_file("X0")
        #RHS.save_to_file("RHS")                
        return A, X, RHS, Ae,  B,  M, self.dep_vars[:]
     
    def run_update_B_blocks(self):
        '''
        assemble M, B, X blockmatrices.

        in parallel, inplace = False makes sure that blocks in A and RHS  
        are not shared by M, B, X
        '''
        B = self.prepare_B_blocks()
        self.fill_B_blocks(B)

        return B
    #
    #  step 0: update mode param
    #
    def run_update_param(self, phys):
        for mm in phys.walk():
            if not mm.enabled: continue
            mm.update_param()

    def initialize_phys(self, phys):
        is_complex = phys.is_complex()        
        self.assign_sel_index(phys)
        
        self.allocate_fespace(phys)
        true_v_sizes = self.get_true_v_sizes(phys)
        
        flags = self.get_essential_bdr_flag(phys)
        self.get_essential_bdr_tofs(phys, flags)

        # this loop alloates GridFunctions
        for j in range(self.n_matrix):
            self.access_idx = j
            is_complex = phys.is_complex()
            for n in phys.dep_vars:
                ifes = self.ifes(n)
                void = self.r_x[ifes]
                if is_complex:
                   void = self.i_x[ifes]
    '''     
    def assemble_mat(self, phys, phys_target):
        
        #assemble matrix made from block matrices
        #each block can be either PyMatrix or HypreMatrix
        
        #is_complex = phys.is_complex()        
        #self.allocate_lf(phys)
        #self.allocate_bf(phys)

        #matvec = self.allocate_matvec(phys)
                
        #self.assemble_lf(phys)
        self.fill_bf(phys)
        self.fill_mixed(phys)        

        #self.call_FormLinearSystem(phys, ess_tdofs, matvec)

        # collect data for essential elimination
        self.assemble_extra(phys, phys_target)


        return matvec #r_X, r_B, r_A, i_X, i_B, i_A 
    '''   
    '''
    def assemble_rhs(self, phys, phys_target):
        is_complex = phys.is_complex()

        flags = self.get_essential_bdr_flag(phys)
        ess_tdofs = self.get_essential_bdr_tofs(phys, flags)
        self.collect_all_ess_tdof(phys, ess_tdofs)
        
        self.allocate_lf(phys)
        
        self.apply_essential(phys)
        self.fill_lf(phys)        

        matvec = self.allocate_matvec(phys)
        self.call_FormLinearSystem(phys, ess_tdofs, matvec)
        
        self.assemble_extra(phys, phys_target)
        
        return matvec
    '''
    #
    #  Step 1  set essential and initial values to the solution vector.
    #
    def apply_essential(self, phys):       
        is_complex = phys.is_complex()       
        for kfes, name in enumerate(phys.dep_vars):
            #rgf.Assign(0.0)
            #if igf is not None:
            #   igf.Assign(0.0)
            ifes = self.ifes(name)
            rgf = self.r_x[ifes]
            igf = None if not is_complex else self.i_x[ifes]
            for mm in phys.walk():
                if not mm.enabled: continue
                if not mm.has_essential: continue
                if len(mm.get_essential_idx(kfes)) == 0: continue
                mm.apply_essential(self, rgf, real = True, kfes = kfes)
                if igf is not None:
                    mm.apply_essential(self, igf, real = False, kfes = kfes)

    def apply_init_from_init_panel(self, phys):
        is_complex = phys.is_complex()
        
        for kfes, name in enumerate(phys.dep_vars):
            ifes = self.ifes(name)
            rfg = self.r_x[ifes]
            for mm in phys.walk():
                if not mm.enabled: continue
                c = mm.get_init_coeff(self, real = True, kfes = kfes)
                if c is None: continue
                rfg.ProjectCoefficient(c)                
                #rgf += tmp
            if not is_complex: continue
            ifg = self.i_x[ifes]            
            for mm in phys.walk():
                if not mm.enabled: continue
                c = mm.get_init_coeff(self, real = False, kfes = kfes)
                if c is None: continue
                ifg.ProjectCoefficient(c)
                #igf += tmp

    def apply_init_from_file(self, phys, init_path):
        '''
        read initial gridfunction from solution
        if init_path is "", then file is read from cwd.
        if file is not found, then it zeroes the gf
        '''
        mesh_idx = phys.emesh_idx
        names = phys.dep_vars
        for kfes, name in enumerate(phys.dep_vars):
            ifes = self.ifes(name)
            rfg = self.r_x[ifes]
            if phys.is_complex():
                ifg = self.i_x[ifes]
            else:
                ifg = None
            fr, fi, meshname = self.solfile_name(names[kfes],
                                                         emesh_idx)
            path = os.path.expanduser(init_path)
            if path == '': path = os.getcwd()
            fr = os.path.join(path, fr)
            fi = os.path.join(path, fi)
            meshname = os.path.join(path, meshname)

            rgf.Assign(0.0)
            if igf is not None: igf.Assign(0.0)
            if not os.path.exists(meshname):
               assert False, "Meshfile for sol does not exist."
            if not os.path.exists(fr):
               assert False, "Solution (real) does not exist."
            if igf is not None and not os.path.exists(fi):
               assert False, "Solution (imag) does not exist."

            m = mfem.Mesh(str(meshname), 1, 1)
            m.ReorientTetMesh()            
            solr = mfem.GridFunction(m, str(fr))
            if solr.Size() != rgf.Size():
               assert False, "Solution file (real) has different length!!!"
            rgf += solr
            if igf is not None:
               soli = mfem.GridFunction(m, str(fi))
               if soli.Size() != igf.Size():
                   assert False, "Solution file (imag) has different length!!!"
               igf += soli               
    #
    #  Step 2  fill matrix/rhs elements
    #
    def fill_bf(self, phys):
        is_complex = phys.is_complex()
        for kfes, name in enumerate(phys.dep_vars):
            ifes = self.ifes(name)

            for mm in phys.walk():
                if not mm.enabled: continue
                if not mm.has_bf_contribution2(kfes, self.access_idx):continue
                if len(mm._sel_index) == 0: continue
                proj = mm.get_projection()
                ra = self.r_a[ifes, ifes, proj]                
                mm.add_bf_contribution(self, ra, real = True, kfes = kfes)
            
        if not is_complex: return
        for kfes, name in enumerate(phys.dep_vars):
            ifes = self.ifes(name)
            for mm in phys.walk():
                if not mm.enabled: continue
                if not mm.has_bf_contribution2(kfes, self.access_idx):continue
                if len(mm._sel_index) == 0: continue
                proj = mm.get_projection()                
                ia = self.i_a[ifes, ifes, proj]                                
                mm.add_bf_contribution(self, ia, real = False, kfes = kfes)
            
    def fill_lf(self, phys): 
        is_complex = phys.is_complex()
        
        for kfes, name in enumerate(phys.dep_vars):
            ifes = self.ifes(name)
            rb = self.r_b[ifes]
            rb.Assign(0.0)
            for mm in phys.walk():
               if not mm.enabled: continue
               if not mm.has_lf_contribution2(kfes, self.access_idx): continue
               if len(mm._sel_index) == 0: continue                          
               mm.add_lf_contribution(self, rb, real=True, kfes=kfes)
            
        if not is_complex: return

        for kfes, name in enumerate(phys.dep_vars):        
            ifes = self.ifes(name)
            ib = self.i_b[ifes]
            ib.Assign(0.0)
            for mm in phys.walk():
               if not mm.enabled: continue
               if not mm.has_lf_contribution2(kfes, self.access_idx): continue
               if len(mm._sel_index) == 0: continue                          
               mm.add_lf_contribution(self, ib, real=False, kfes=kfes)
           
    def fill_mixed(self, phys):
        is_complex = phys.is_complex()
        mixed_bf = {}
        tasks = {}
        phys_offset = self.phys_offsets(phys)[0]
        
        for mm in phys.walk():
            if not mm.enabled: continue
            if not mm.has_mixed_contribution2(self.access_idx):continue
            if len(mm._sel_index) == 0: continue
                          
            loc_list = mm.get_mixedbf_loc()

            for loc in loc_list:
                r, c, is_trans, is_conj= loc
                if isinstance(r, int):
                    idx1 = phys_offset + r
                    idx2 = phys_offset + c
                else:
                    idx1 = self.ifes(r)
                    idx2 = self.ifes(c)                                      
                if loc[2] < 0:
                    idx1, idx2 = idx2, idx1

                bf =  self.r_a[idx1, idx2]

                ## ToDo fix this bool logic...;D
                is_trans = (is_trans == -1)
                is_conj  = (is_conj == -1)
                mm.add_mix_contribution2(self, bf, r, c, is_trans, is_conj, real=True)

                if is_complex:
                    bf =  self.i_a[idx1, idx2]
                    mm.add_mix_contribution2(self, bf, r, c, is_trans, is_conj, real=False)
                    
    def update_bf(self):
        fes_vars = self.fes_vars       
        for j in range(self.n_matrix):
            self.access_idx = j
            for name in self.fes_vars:
                ifes = self.ifes(name)
                projs = self.r_a.get_projections(ifes, ifes)
                fes = self.fespaces[name]
                for p in projs:
                    ra = self.r_a[ifes, ifes, p]
                    ra.Update(fes)
                projs = self.i_a.get_projections(ifes, ifes)
                for p in projs:
                    ia = self.i_a[ifes, ifes, p]
                    ia.Update(fes)
       
    def fill_coupling(self, coupling, phys_target):
        raise NotImplementedError("Coupling is not supported")
  
    def assemble_extra(self, phys, phys_target):
        for mm in phys.walk():
            if not mm.enabled: continue
            for phys2 in phys_target:
                names = phys2.dep_vars      
                for kfes, name in enumerate(names):
                    if not mm.has_extra_DoF2(kfes, phys2, self.access_idx): continue
                    
                    gl_ess_tdof = self.gl_ess_tdofs[name]                    
                    tmp  = mm.add_extra_contribution(self,
                                                     ess_tdof=gl_ess_tdof, 
                                                     kfes = kfes,
                                                     phys = phys2)
                    if tmp is None: continue

                    dep_var = names[kfes]
                    extra_name = mm.extra_DoF_name()
                    key = (extra_name, dep_var)
                    if key in self.extras:
                        assert False, "extra with key= " + str(key) + " already exists."
                    self.extras[key] = tmp
                    self.extras_mm[key] = mm.fullpath()
                    
    def assemble_aux_ops(self, phys_target):
        allmm = [mm for phys in phys_target for mm in phys.walk() if mm.is_enabled()]
        for phys1 in phys_target:
            names = phys1.dep_vars           
            for kfes1, name1 in enumerate(names):
                for phys2 in phys_target:
                    names2 = phys2.dep_vars
                    for kfes2, name2 in enumerate(names2):
                      for mm in allmm:
                        if not mm.has_aux_op2(phys1, kfes1,
                                              phys2, kfes2, self.access_idx): continue
                        gl_ess_tdof1 = self.gl_ess_tdofs[name1]
                        gl_ess_tdof2 = self.gl_ess_tdofs[name2]                    
                        op = mm.get_aux_op(self, phys1, kfes1, phys2, kfes2,
                                           test_ess_tdof=gl_ess_tdof1,
                                           trial_ess_tdof=gl_ess_tdof2)
                        self.aux_ops[(name1, name2, mm.fullpath())] = op

    def assemble_interp(self, phys):
        names = phys.dep_vars
        for name in names:
            gl_ess_tdof = self.gl_ess_tdofs[name]
            kfes = names.index(name)
            interp = []
            for mm in phys.walk():
                if not mm.enabled: continue           
                if not mm.has_interpolation_contribution(kfes):continue
                interp.append(mm.add_interpolation_contribution(self,
                                                       ess_tdof=gl_ess_tdof,
                                                       kfes = kfes))
            # merge all interpolation constraints
            P = None
            nonzeros=[]
            zeros=[]        
            for P0, nonzeros0, zeros0 in interp:
                if P is None:
                    P = P0
                    zeros = zeros0
                    noneros = nonzeros0
                else:
                    P = P.dot(P0)
                    zeros = np.hstack((zeros, zeros0))
                    nonzeros = np.hstack((nonzeros, nonzeros0)) 

            self.interps[name] = (P, nonzeros, zeros)

    def assemble_projection(self, phys):
        pass

    #
    #  step3 : generate block matrices/vectors
    #
    def prepare_blocks(self):
        size = len(self.dep_vars)                         
        M_block = [self.new_blockmatrix((size, size)) for i in range(self.n_matrix)]
        B_block = self.new_blockmatrix((size, 1))
        X_block = [self.new_blockmatrix((size, 1))  for i in range(self.n_matrix)]
        return (M_block, B_block, X_block)

    def prepare_B_blocks(self):
        size = len(self.dep_vars)                         
        B_block = self.new_blockmatrix((size, 1))
        return B_block
    
    def fill_M_B_X_blocks(self, M, B, X):
        from petram.helper.formholder import convertElement
        from mfem.common.chypre import BF2PyMat, LF2PyVec
        from mfem.common.chypre import MfemVec2PyVec, MfemMat2PyMat    
        from itertools import product

        nfes = len(self.fes_vars)
        for k in range(self.n_matrix):
            self.access_idx = k
            
            self.r_a.generateMatVec(self.a2A, self.a2Am)
            self.i_a.generateMatVec(self.a2A, self.a2Am)
            
            for i, j in product(range(nfes),range(nfes)):
                m = convertElement(self.r_a, self.i_a,
                                   i, j, MfemMat2PyMat)
                r = self.dep_var_offset(self.fes_vars[i])
                c = self.dep_var_offset(self.fes_vars[j])

                M[k][r,c] = m if M[k][r,c] is None else M[k][r,c] + m
                
            for extra_name, dep_name in self.extras.keys():
                r = self.dep_var_offset(extra_name)
                c = self.dep_var_offset(dep_name)

                # t1,t2,t3,t4 = (horizontal, vertical, diag, rhs). 
                t1, t2, t3, t4, t5 = self.extras[(extra_name, dep_name)]
                
                M[k][c,r] = t1 if M[k][c,r] is None else M[k][c,r]+t1
                M[k][r,c] = t2 if M[k][r,c] is None else M[k][r,c]+t2
                M[k][r,r] = t3 if M[k][r,r] is None else M[k][r,r]+t3
                                     
            for key in self.aux_ops.keys():
                testname, trialname, mm_fullpath = key
                r = self.dep_var_offset(testname)
                c = self.dep_var_offset(trialname)
                m = self.aux_ops[key]
                M[k][r,c] = m if M[k][r,c] is None else M[k][r,c]+m
                

        self.access_idx = 0
        self.r_b.generateMatVec(self.b2B)
        self.i_b.generateMatVec(self.b2B)            
        for i in range(nfes):
            v = convertElement(self.r_b, self.i_b,
                                      i, 0, MfemVec2PyVec)
            r = self.dep_var_offset(self.fes_vars[i])
            B[r] = v
            
        self.access_idx = 0            
        for extra_name, dep_name in self.extras.keys():
            r = self.dep_var_offset(extra_name)
            t1, t2, t3, t4, t5 = self.extras[(extra_name, dep_name)]            
            B[r] = t4

        for k in range(self.n_matrix):
            self.access_idx = k
            self.r_x.generateMatVec(self.x2X)
            self.i_x.generateMatVec(self.x2X)            
            for dep_var in self.dep_vars:
                if self.isFESvar(dep_var):
                    i = self.ifes(dep_var)
                    v = convertElement(self.r_x, self.i_x,
                                       i, 0, MfemVec2PyVec)
                    r = self.dep_var_offset(dep_var)
                    X[k][r] = v
                else:
                    pass 
                    # For now, it leaves as None for Lagrange Multipler?
                    # May need to allocate zeros...
        return M, B, X
     
    def fill_B_blocks(self, B):
        from petram.helper.formholder import convertElement
        from mfem.common.chypre import MfemVec2PyVec
       
        nfes = len(self.fes_vars)
        
        self.access_idx = 0
        self.r_b.generateMatVec(self.b2B)
        self.i_b.generateMatVec(self.b2B)            
        for i in range(nfes):
            v = convertElement(self.r_b, self.i_b,
                                      i, 0, MfemVec2PyVec)
            r = self.dep_var_offset(self.fes_vars[i])
            B[r] = v
            
        self.access_idx = 0            
        for extra_name, dep_name in self.extras.keys():
            r = self.dep_var_offset(extra_name)
            t1, t2, t3, t4, t5 = self.extras[(extra_name, dep_name)]            
            B[r] = t4
       
    def fill_BCeliminate_matrix(self, A, inplace=True):
        nblock = A.shape[0]
        Ae = self.new_blockmatrix(A.shape)
        for name in self.gl_ess_tdofs:
           gl_ess_tdof = self.gl_ess_tdofs[name]
           ess_tdof = self.ess_tdofs[name]
           idx = self.dep_var_offset(name)
           if A[idx, idx] is not None:
              Ae[idx, idx], A[idx,idx] = A[idx, idx].eliminate_RowsCols(ess_tdof,
                                                                        inplace=inplace)
              
              #print "index", idx, idx, name, len(ess_tdof)

           for j in range(nblock):
              if j == idx: continue
              if A[idx, j] is None: continue
              A[idx, j].resetRow(gl_ess_tdof)
                  
           for j in range(nblock):            
              if j == idx: continue
              if A[j, idx] is None: continue
              SM = A.get_squaremat_from_right(j, idx)
              SM.setDiag(gl_ess_tdof)              
              Ae[j, idx] = A[j, idx].dot(SM)
              A[j, idx].resetCol(gl_ess_tdof)
        return A, Ae

    def eliminateBC(self, Ae, X, RHS):
        try:
            AeX = Ae.dot(X)
            for name in self.gl_ess_tdofs:
               gl_ess_tdof = self.gl_ess_tdofs[name]
               idx = self.dep_var_offset(name)
               if AeX[idx, 0] is not None:
                    AeX[idx, 0].resetRow(gl_ess_tdof)                  
            RHS = RHS - AeX
        except:
            print "RHS", RHS
            print "Ae", Ae
            print "X", X
            raise
        for name in self.gl_ess_tdofs:
            gl_ess_tdof = self.gl_ess_tdofs[name]
            ess_tdof = self.ess_tdofs[name]
            idx = self.dep_var_offset(name)
            RHS[idx].copy_element(gl_ess_tdof, X[idx])
            
        return RHS
              
    def apply_interp(self, A=None, RHS=None):
        ''''
        without interpolation, matrix become
              [ A    B ][x]   [b]
              [        ][ ] = [ ]
              [ C    D ][l]   [c], 
        where B, C, D is filled as extra
        if P is not None: 
              [ P A P^t  P B ][y]   [P b]
              [              ][ ] = [   ]
              [ C P^t     D  ][l]   [ c ]
        and 
             x  = P^t y
        '''
        for name in self.interps:
            idx = self.dep_var_offset(name)
            P, nonzeros, zeros = self.interps[name]
            if P is None: continue
            
            if A is not None:
               shape = A.shape               
               A1 = A[idx,idx]
               A1 = A1.rap(P.transpose())
               A1.setDiag(zeros, 1.0)
               A[idx, idx] = A1

               PP = P.conj(inplace=True)
               for i in range(shape[1]):
                   if idx == i: continue
                   if A[idx,i] is not None:
                       A[idx,i] = PP.dot(A[idx,i])

               P = PP.conj(inplace=True)                        
               for i in range(shape[0]):
                   if idx == i: continue
                   if A[i,idx] is not None:
                       A[i, idx] = A[i, idx].dot(t2)
            if RHS is not None:
                RHS[idx] = P.conj(inplace=True).dot(RHS[idx])
                P.conj(inplace=True)

        
        if A is not None and RHS is not None: return A, RHS
        if A is not None : return A, 
        if RHS is not None: return RHS

    '''        
    def call_FormLinearSystem(self, phys, ess_tdofs, matvec):
        is_complex = phys.is_complex()
        
        if not is_complex:
            for kfes, r_a, r_x, r_b in enum_fes(phys, self.r_a, self.r_x, self.r_b):
                r_X, r_B, r_A  = [matvec[x][kfes] for x in range(3)]
                r_a.FormLinearSystem(ess_tdofs[kfes][1], r_x, r_b, r_A, r_X, r_B)

        else:   
            for kfes, r_a, r_x, r_b, i_a, i_x, i_b in enum_fes(phys,  
                                               self.r_a, self.r_x, self.r_b,
                                               self.i_a, self.i_x, self.i_b):

                r_X, r_B, r_A, i_X, i_B, i_A  = [matvec[x][kfes] for x in range(6)]

                # Re(b) = - (Re(Me)*Re(x) - Im(Me)*Im(x))
                # Im(b) = - (Im(Me)*Re(x) + Re(Me)*Im(x))
                # note that we don't care at this point what is in r_X and i_X,
                # since we are not using iterative solver.

                # key idea is to reset (zeroing) GridFunction and Desntiation
                # vector. Serial and parallel handles EssentailBC differently.
                # This approach seems working in both mode.
                
                ess = ess_tdofs[kfes][1].ToList()

                i_a.FormLinearSystem(ess_tdofs[kfes][1], i_x, r_b, i_A, i_X, r_B)
                rb = r_B.GetDataArray().astype(float) # astype will allocate
                                                      # new memory
                r_B *= 0.0; r_b.Assign(0.0)                
                r_a.FormLinearSystem(ess_tdofs[kfes][1], r_x, r_b, r_A, r_X, r_B)
                rb2 = r_B.GetDataArray().astype(float)
                r_B -= mfem.Vector(rb)
                for k in ess: r_B[k] = rb2[k]
                
                r_a.FormLinearSystem(ess_tdofs[kfes][1], i_x, i_b, r_A, i_X, i_B)
                #fid = open('debug_matrix', 'w')
                #r_A.Print(fid)
                #fid.close()
                ib = i_B.GetDataArray().astype(float) # astype will allocate
                                                      # new memory
                i_B *= 0.0; i_b.Assign(0.0)
                i_a.FormLinearSystem(ess_tdofs[kfes][1], r_x, i_b, i_A, r_X, i_B)
                i_B += mfem.Vector(ib)
                for k in ess: i_B[k] = ib[k]

        import petram.debug as debug
        if debug.debug_essential_bc:
            name = self.fespaces[phys][kfes][0]
            mesh_idx = phys.emesh_idx 
            self.save_solfile_fespace(name, mesh_idx, r_x, i_x,
                                          namer = 'x_r',
                                          namei = 'x_i')
    '''
    #
    #  build linear system construction
    #
    '''    
    def generate_linear_system(self, phys_target, matvecs, matvecs_c):
        dprint2('matrix format', format)
        blocks = self.prepare_blocks(phys_target)
        self.fill_block_matrix(phys_target, blocks,
                                      matvecs, matvecs_c)
        self.fill_block_rhs(phys_target, blocks, matvecs, matvecs_c)

        # elimination of
        self.fill_elimination_block(phys_target, blocks)
        return blocks

    def generate_rhs(self, phys_target, vecs, vecs_c):
        blocks = self.prepare_blocks()
        self.fill_block_rhs(phys_target, blocks, vecs, vecs_c)
        return blocks

    def fill_block_matrix(self, phys_target, blocks, matvecs, matvecs_c):
        M_block, B_block,  Me_block = blocks
        # (step1) each FESpace is handled 
        #         apply projection

        for phys in iter_phys(phys_target):
            offsets = self.phys_offsets[phys]           
            for kfes, interp, gl_ess_tdof in enum_fes(phys,
                                  self.interps, self.gl_ess_tdofs):
                offset = offsets[kfes]              
                mvs = matvecs[phys]
                mv = [mm[kfes] for mm in mvs]                
                #matvec[k]  =  r_X, r_B, r_A, i_X, i_B, i_A or r only
                self.fill_block_matrix_fespace(blocks, mv,
                                               gl_ess_tdof,
                                               interp, 
                                               offset)
        # (step2) in-physics coupling (MixedForm)
        for phys, mixed_bf, interps in iter_phys(phys_target, 
                                          self.mixed_bf, self.interps):
            offsets = self.phys_offsets[phys]           
            for loc in mixed_bf:
                r = offsets[loc[0]]
                c = offsets[loc[1]]
                elem = chypre.BF2PyMat(mixed_bf[loc][0], mixed_bf[loc][1],
                                finalize = True)
                #elem = M_block.mixed_bf_to_element(mixed_bf[loc])
                M_block.add_to_element(r, c,
                                       self.fill_block_from_mixed(loc, elem,
                                                                 interps[loc[0]],
                                                                 interps[loc[1]]))
        # (step3) mulit-physics coupling (???)
        #         apply projection + place in a block format
        return
    '''
    '''
    def fill_block_rhs(self, phys_target, blocks, matvecs, matvecs_c):
        # (step1) each FESpace is handled 
        #         apply projection
        dprint1("Entering Filling Block RHS")
        dprint2("\n", blocks[1])
        
        for phys, mvs in iter_phys(phys_target, matvecs):
            offsets = self.phys_offsets[phys]
            for kfes, interp, gl_ess_tdof in enum_fes(phys,
                          self.interps, self.gl_ess_tdofs):
                offset = offsets[kfes]              
                mv = [mm[kfes] for mm in mvs]
                self.fill_block_rhs_fespace(blocks, mv, interp,
                                            offset)

        # (step3) mulit-physics coupling (???)
        #         apply projection + place in a block format
        dprint2("\n", blocks[1])
        dprint1("Exiting Filling Block RHS")        
        return
    '''
    '''
    def fill_elimination_block(self, phys_target, blocks):
        #make block eliminaition matrix for off-diagonal blocks
        #for essential DoF
        dprint1("Entering Filling Elimination Block")

        M = blocks[0]
        Me = blocks[2]
        size = Me.shape[0]
        dprint2("\n", M)

        # eliminate horizontal.
        dprint2("Filling Elimination Block (step1)")    
        for phys, gl_ess_tdofs in iter_phys(phys_target, self.gl_ess_tdofs):
            offsets = self.phys_offsets[phys]
            for offset, gl_ess_tdof in zip(offsets, gl_ess_tdofs):
                for ib in range(size):
                    if ib == offset: continue
                    if M[offset, ib] is None: continue
                    M[offset, ib].resetRow(gl_ess_tdof[1])

        # store vertical to Me (choose only essential col)
        dprint2("Filling Elimination Block (step2)")
        for phys, gl_ess_tdofs in iter_phys(phys_target, self.gl_ess_tdofs):
            offsets = self.phys_offsets[phys]           
            for offset, gl_ess_tdof in zip(offsets, gl_ess_tdofs):
                for ib in range(size):
                    if ib == offset: continue
                    if M[ib, offset] is None: continue
                    SM = M.get_squaremat_from_right(ib, offset)
                    SM.setDiag(gl_ess_tdof[1])
                    dprint3("calling dot", ib, offset, M[ib, offset], SM)
                    Me[ib, offset] = M[ib, offset].dot(SM)

        dprint1("Exiting fill_elimination_block\n", M)
    '''    

    '''
    #
    #  finalize linear system construction
    #
    def eliminate_and_shrink(self,  M_block, B_blocks, Me):
        # eliminate dof
        dprint1("Eliminate and Shrink")
        dprint1("Me\n", Me)

        # essentailBC is stored in b
        #for b in B_blocks:
        #    print b, Me.dot(b)
        B_blocks = [b - Me.dot(b) for b in B_blocks]

        
        # shrink size
        dprint2("M (before shrink)\n", M_block)        
        M_block2, P2 = M_block.eliminate_empty_rowcol()
        dprint1("P2\n", P2)
        
        B_blocks = [P2.dot(b) for b in B_blocks]

        dprint2("M (after shrink)\n", M_block2)                
        dprint2("B (after shrink)\n", B_blocks[0])

        return M_block2, B_blocks, P2
    '''     
    def finalize_matrix(self, M_block, is_complex,format = 'coo', verbose=True):
        if format == 'coo': # coo either real or complex
            M = self.finalize_coo_matrix(M_block, is_complex, verbose=verbose)
            
        elif format == 'coo_real': # real coo converted from complex
            M = self.finalize_coo_matrix(M_block, is_complex,
                                            convert_real = True, verbose=verbose)

        elif format == 'blk_interleave': # real coo converted from complex
            M = M_block.get_global_blkmat_interleave()
            
        dprint2('exiting finalize_matrix')
        self.is_assembled = True
        return M
     
    def finalize_rhs(self,  B_blocks, is_complex, format = 'coo', verbose=True):
        if format == 'coo': # coo either real or complex
            B = [self.finalize_coo_rhs(b, is_complex, verbose=verbose) for b in B_blocks]
            B = np.hstack(B)
            
        elif format == 'coo_real': # real coo converted from complex
            B = [self.finalize_coo_rhs(b, is_complex,
                                       convert_real = True, verbose=verbose)
                    for b in B_blocks]
            B = np.hstack(B)
        elif format == 'blk_interleave': # real coo converted from complex
            B = [b.gather_blkvec_interleave() for b in B_blocks]
            
        return B
     
    def finalize_x(self,  X_block, RHS, is_complex, format = 'coo', verbose=True):
        if format == 'blk_interleave': # real coo converted from complex
            X = X_block.gather_blkvec_interleave(size_hint=RHS)
        else:
            assert False, "unsupported format for X"
        
        return X
     
    def finalize_coo_matrix(self, M_block, is_complex, convert_real = False,
                            verbose=True):
        if verbose: dprint1("A (in finalizie_coo_matrix) \n",  M_block)       
        if not convert_real:
            if is_complex:
                M = M_block.get_global_coo(dtype='complex')           
            else:
                M = M_block.get_global_coo(dtype='float')                          
        else:
            M = M_block.get_global_coo(dtype='complex')                      
            M = scipy.sparse.bmat([[M.real, -M.imag], [M.imag, M.real]], format='coo')
            # (this one make matrix symmetric, for now it is off to do the samething
            #  as GMRES case)
            # M = scipy.sparse.bmat([[M.real, -M.imag], [-M.imag, -M.real]], format='coo')
        return M

    def finalize_coo_rhs(self, b, is_complex,
                         convert_real = False,
                         verbose=True):
        if verbose: dprint1("b (in finalizie_coo_rhs) \n",  b)
        B = b.gather_densevec()
        if convert_real:
             B = np.vstack((B.real, B.imag))
             # (this one make matrix symmetric)           
             # B = np.vstack((B.real, -B.imag))
        else:
           if not is_complex:
              pass
             # B = B.astype(float)
        return B
    #
    #  processing solution
    #
    def split_sol_array(self, sol):
        s = [None]*len(self.fes_vars)
        for name in self.fes_vars:
            j = self.dep_var_offset(name)
            sol_section = sol[j, 0]

            if name in self.interps:
               P, nonzeros, zeros = self.interps[name]
               if P is not None:
                   sol_section = (P.transpose()).dot(sol_section)

            ifes = self.ifes(name)
            s[ifes] = sol_section

        e = []    
        for name in self.dep_vars:
           if not self.isFESvar(name):
              e.append(sol[self.dep_var_offset(name)])
        return s, e
     
    def recover_sol(self, sol):

        self.access_idx=0
        for k, s in enumerate(sol):
           name = self.fes_vars[k]
           ifes = self.ifes(name)
           idx  = self.dep_var_offset(name)
           s = s.toarray()
           X = self.r_x.get_matvec(ifes)
           X.Assign(s.flatten().real)
           self.X2x(X, self.r_x[ifes])
           if self.i_x[ifes] is not None:
               X = self.i_x.get_matvec(ifes)              
               X.Assign(s.flatten().imag)              
               self.X2x(X, self.i_x[ifes])
           else:
               dprint2("real value problem skipping i_x")
               
    def process_extra(self, sol_extra):
        ret = {}
        k = 0
        extra_names = [name for name in self.dep_vars
                       if not self.isFESvar(name)]

        for extra_name, dep_name in self.extras.keys():
            data = sol_extra[extra_names.index(extra_name)]
            t1, t2, t3, t4, t5 = self.extras[(extra_name, dep_name)]
            mm_path = self.extras_mm[(extra_name, dep_name)]
            mm = self.model[mm_path]
            ret[extra_name] = {}
            
            if not t5: continue
            if data is not None:            
                ret[extra_name][mm.extra_DoF_name()] = data.toarray()
            else:
                pass
            '''
            if data is None:
                # extra can be none in MPI child nodes
                # this is called so that we can use MPI
                # in postprocess_extra in future
                mm.postprocess_extra(None, t5, ret[extra_name])
            else:
                mm.postprocess_extra(data, t5, ret[extra_name])
            '''
        dprint1("extra", ret)
        return ret

    #
    #  save to file
    #
    
    def save_sol_to_file(self, phys_target, skip_mesh = False,
                               mesh_only = False,
                               save_parmesh = False):
        if not skip_mesh:
            mesh_filenames =self.save_mesh()
        if save_parmesh:
            self.save_parmesh()
        if mesh_only: return mesh_filenames

        self.access_idx = 0
        for phys in phys_target:
            emesh_idx = phys.emesh_idx
            for name in phys.dep_vars:
                ifes = self.ifes(name)
                r_x = self.r_x[ifes]
                i_x = self.i_x[ifes]
                self.save_solfile_fespace(name, emesh_idx, r_x, i_x)
     
    def save_extra_to_file(self, sol_extra):
        if sol_extra is None: return
        fid = open(self.extrafile_name(), 'w')
        for name in sol_extra.keys():
            for k in sol_extra[name].keys():
                data = sol_extra[name][k]
                #  data must be NdArray
                #  dataname : "E1.E_out"
                fid.write('name : ' + name + '.' + str(k) +'\n')
                fid.write('size : ' + str(data.size) +'\n')
                fid.write('dim : ' + str(data.ndim) +'\n')            
                if data.ndim == 0:
                    fid.write(str(0) + ' ' + str(data) +'\n')
                else:
                    for kk, d in enumerate(data):
                         fid.write(str(kk) + ' ' + str(d) +'\n')
        fid.close()

    #
    #  helper methods
    #
    def assign_sel_index(self, phys = None):
        if len(self.meshes) == 0:
           dprint1('!!!! mesh is None !!!!')
           return
        if phys is None:
            all_phys = [self.model['Phys'][k] for
                        k in self.model['Phys'].keys()]
        else:
            all_phys = [phys]
        for p in all_phys:
            if p.mesh_idx < 0: continue
            mesh = self.meshes[p.mesh_idx]
            if mesh is None: continue
            if len(p.sel_index) == 0: continue

            dom_choice, bdr_choice = p.get_dom_bdr_choice(self.meshes[p.mesh_idx])

            p._phys_sel_index = dom_choice
            self.do_assign_sel_index(p, dom_choice, Domain)
            self.do_assign_sel_index(p, bdr_choice, Bdry)
     
    def do_assign_sel_index(self, m, choice, cls):
        dprint1("## setting _sel_index (1-based number): "+m.fullname())
        #_sel_index is 0-base array
        def _walk_physics(node):
            yield node
            for k in node.keys():
                yield node[k]
        rem = None
        checklist = [True]*len(choice)
        for node in m.walk():
           if not isinstance(node, cls): continue
           if not node.enabled: continue
           ret = node.process_sel_index(choice)
           if ret is None:
              if rem is not None: rem._sel_index = []
              rem = node
           elif ret == -1:
              node._sel_index = choice
           else:
              dprint1(node.fullname(), ret)
              for k in ret:
                 idx = list(choice).index(k)
                 if node.is_secondary_condition: continue
                 checklist[idx] = False
        if rem is not None:
           rem._sel_index = list(np.array(choice)[checklist])
           dprint1(rem.fullname() + ':' + rem._sel_index.__repr__())

    def find_domain_by_index(self, phys, idx,  check_enabled = False):
        return self._do_find_by_index(phys, idx, Domain,
                                      check_enabled = check_enabled)       

    def find_bdry_by_index(self, phys, idx, check_enabled = False):
        return self._do_find_by_index(phys, idx, Bdry,
                                      check_enabled = check_enabled)
        
    def _do_find_by_index(self, phys, idx, cls, ignore_secondary=True,
                          check_enabled = False):
        for node in phys.walk():
            if (check_enabled and (not node.enabled)): continue
            if not isinstance(node, cls): continue
            if idx in node._sel_index:
                if ((ignore_secondary and not node.is_secondary_condition)
                    or not ignore_secondary):
                    return node

    def gather_essential_tdof(self, phys):
        flags = self.get_essential_bdr_flag(phys)
        self.get_essential_bdr_tofs(phys, flags)

                 
    def get_essential_bdr_flag(self, phys):
        flag = []
        for k,  name in enumerate(phys.dep_vars):
            fes = self.fespaces[name]
            index = []
            for node in phys.walk():
                #if not isinstance(node, Bdry): continue           
                if not node.enabled: continue
                if node.has_essential:
                    index = index + node.get_essential_idx(k)

            ess_bdr = [0]*self.emeshes[phys.emesh_idx].bdr_attributes.Max()
            for kk in index: ess_bdr[kk-1] = 1
            flag.append((name, ess_bdr))
        dprint1("esse flag", flag)
        return flag

    def get_essential_bdr_tofs(self, phys, flags):
        for name, ess_bdr in flags:
            ess_tdof_list = mfem.intArray()
            ess_bdr = mfem.intArray(ess_bdr)
            fespace = self.fespaces[name]
            fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)
            self.ess_tdofs[name] = ess_tdof_list.ToList()
        return

    def allocate_fespace(self, phys):
        #
        for name, elem in phys.get_fec():
            vdim = phys.vdim
            
            dprint1("allocate_fespace: " + name)
            mesh = self.emeshes[phys.emesh_idx]
            fec = getattr(mfem, elem)

            #if fec is mfem.ND_FECollection:
            #   mesh.ReorientTetMesh()

            dim = mesh.Dimension()
            sdim= mesh.SpaceDimension()
            f = fec(phys.order, sdim)
            self.fec[name] = f

            if name.startswith('RT'): vdim = 1
            if name.startswith('ND'): vdim = 1
            
            fes = self.new_fespace(mesh, f, vdim)
            mesh.GetEdgeVertexTable()
            self.fespaces[name] = fes


    def get_fes(self, phys, kfes = 0, name = None):
        if name is None:
            name = phys.dep_vars[kfes]
            return self.fespaces[name]
        else:
            return self.fespaces[name]

    def alloc_gf(self, idx, idx2=0):
        fes = self.fespaces[self.fes_vars[idx]]
        return self.new_gf(fes)
     
    def alloc_lf(self, idx, idx2=0):
        fes = self.fespaces[self.fes_vars[idx]]
        return self.new_lf(fes)

    def alloc_bf(self, idx, idx2=None):
        fes = self.fespaces[self.fes_vars[idx]]
        return self.new_bf(fes)

    def alloc_mbf(self, idx1, idx2): #row col
        fes1 = self.fespaces[self.fes_vars[idx1]]
        fes2 = self.fespaces[self.fes_vars[idx2]]
        return self.new_mixed_bf(fes2, fes1) # argument = trial, test
    '''     
    def allocate_gf(self, phys):
        #print("allocate_gf")
        
        is_complex = phys.is_complex()
        r_x = [(name, self.new_gf(fes))  for name, fes in self.fespaces[phys]]
        self.r_x[phys] = r_x
        if is_complex:
            i_x = [(name, self.new_gf(fes))  for name, fes
                   in self.fespaces[phys]]
        else:
            i_x = [(name, None)  for name, fes
                   in self.fespaces[phys]]
        self.i_x[phys] = i_x
        
    def allocate_bf(self, phys):
        #print("allocate_bf")
        
        is_complex = phys.is_complex()
        r_a = [(name, self.new_bf(fes))  for name, fes in self.fespaces[phys]]
        self.r_a[phys] = r_a
 
        if is_complex:
            i_a = [(name, self.new_bf(fes))  for name, fes
                   in self.fespaces[phys]]
        else:
            i_a = [(name, None)  for name, fes
                   in self.fespaces[phys]]
        self.i_a[phys] = i_a            
    def allocate_lf(self, phys):
        #print("allocate_lf")
        
        is_complex = phys.is_complex()
        r_b = [(name, self.new_lf(fes))  for name, fes in self.fespaces[phys]]
        self.r_b[phys] = r_b
 
        if is_complex:
            i_b = [(name, self.new_lf(fes))  for name, fes in self.fespaces[phys]]
        else:
            i_b = [(name, None)  for name, fes in self.fespaces[phys]]
        self.i_b[phys] = i_b

    def allocate_matvec(self, phys):
        #print("allocate_matvec")       
        is_complex = phys.is_complex()
        ret = []
        ret.append([mfem.Vector()  for name, fes in self.fespaces[phys]])
        ret.append([mfem.Vector()  for name, fes in self.fespaces[phys]])
        ret.append([self.new_matrix()  for name, fes in self.fespaces[phys]])
        if not is_complex: return ret  # r_X, r_B, r_A

        ret.append([mfem.Vector()  for name, fes in self.fespaces[phys]])
        ret.append([mfem.Vector()  for name, fes in self.fespaces[phys]])
        ret.append([self.new_matrix()  for name, fes in self.fespaces[phys]])
        return ret # r_X, r_B, r_A, i_X, i_B, i_A
    '''
    def build_ns(self):
        for node in self.model.walk():
           if node.has_ns():
              try:
                  node.eval_ns()
              except Exception as e:
                  assert False, "Failed to build name space: " + e.message
           else:
              node._local_ns = self.model.root()._variables

    def preprocess_ns(self, ns_folder, data_folder):
        '''
        folders are tree object
        '''
        for od in self.model.walk():
            if od.has_ns():
               od.preprocess_ns(ns_folder, data_folder)

    def form_linear_system(self, ess_tdof_list, extra, interp, r_A, r_B, i_A, i_B):
        raise NotImplementedError(
             "you must specify this method in subclass")

    def run_mesh_serial(self, meshmodel = None,
                        skip_refine = False):
        from petram.mesh.mesh_model import MeshFile, MFEMMesh
        from petram.mesh.mesh_extension import MeshExt
        from petram.mesh.mesh_utils import  get_extended_connectivity
    
        self.meshes = []
        self.emeshes = []
        self.emesh_data = MeshExt()
        if meshmodel is None:
            parent = self.model['Mesh']
            children =  [parent[g] for g in parent.keys()
                         if isinstance(parent[g], MFEMMesh) and parent[g].enabled]
            for idx, child in enumerate(children):
                self.meshes.append(None)                
                #if not child.enabled: continue
                target = None
                for k in child.keys():
                    o = child[k]
                    if not o.enabled: continue
                    if isinstance(o, MeshFile):
                        self.meshes[idx] = o.run()
                        target = self.meshes[idx]
                    else:
                        if o.isRefinement and skip_refine: continue
                        if hasattr(o, 'run') and target is not None:
                            self.meshes[idx] = o.run(target)
        self.max_bdrattr = -1
        self.max_attr = -1        
        for m in self.meshes:
            self.max_bdrattr = np.max([self.max_bdrattr, max(m.GetBdrAttributeArray())])
            self.max_attr = np.max([self.max_attr, max(m.GetAttributeArray())])
            m.ReorientTetMesh()
            m.GetEdgeVertexTable()                                   
            get_extended_connectivity(m)           
           
    def run_mesh(self):
        raise NotImplementedError(
             "you must specify this method in subclass")

    def new_lf(self, fes):
        raise NotImplementedError(
             "you must specify this method in subclass")
     
    def new_bf(self, fes):
        raise NotImplementedError(
             "you must specify this method in subclass")
     
    def new_mixed_bf(self, fes1, fes2):
        raise NotImplementedError(
             "you must specify this method in subclass")
       
    def new_gf(self, fes):
        raise NotImplementedError(
             "you must specify this method in subclass")

    def new_fespace(self, mesh, fec):
        raise NotImplementedError(
             "you must specify this method in subclass")

     
    def eliminate_ess_dof(self, ess_tdof_list, M, B):     
        raise NotImplementedError(
             "you must specify this method in subclass")
     
    def solfile_name(self, name, mesh_idx):
        raise NotImplementedError(
             "you must specify this method in subclass")
     
    def save_solfile_fespace(self, name, mesh_idx, r_x, i_x):
        fnamer, fnamei, meshname = self.solfile_name(name, mesh_idx)
        r_x.SaveToFile(fnamer, 8)
        if i_x is not None:
            i_x.SaveToFile(fnamei, 8)

    @property   ### ALL dependent variables including Lagrange multipliers
    def dep_vars(self):
        return self._dep_vars
    @property   ### ALL finite element space variables
    def fes_vars(self):
        return self._fes_vars
               
    def ifes(self, name):
        return self._fes_vars.index(name)
           
    def phys_offsets(self, phys):
        name = phys.dep_vars[0]
        idx0 = self._dep_vars.index(name)
        for names in self._dep_var_grouped:
           if name in names: l = len(names)
        return range(idx0, idx0+l)

    def dep_var_offset(self, name):
        return self._dep_vars.index(name)       
     
    def isFESvar(self, name):
        if not name in self._dep_vars:
           assert False, "Variable " + name + " not used in the model"
        idx = self._dep_vars.index(name)
        return self._isFESvar[idx]
        
    def collect_dependent_vars(self, phys_target=None):
        if phys_target is None:
           phys_target = [self.model['Phys'][k] for k in self.model['Phys']
                          if self.model['Phys'].enabled]

        dep_vars_g  = []
        isFesvars_g = []
        
        for phys in phys_target:
            dep_vars  = []
            isFesvars = []            
            if not phys.enabled: continue
            
            dv = phys.dep_vars
            dep_vars.extend(dv)
            isFesvars.extend([True]*len(dv))

            extra_vars = []
            for mm in phys.walk():
               if not mm.enabled: continue

               for j in range(self.n_matrix):
                  for k in range(len(dv)):
                      for phys2 in phys_target:
                          if not phys2.enabled: continue                         
                          if not mm.has_extra_DoF2(k, phys2, j): continue
                      
                          name = mm.extra_DoF_name()
                          if not name in extra_vars:
                              extra_vars.append(name)
            dep_vars.extend(extra_vars)
            isFesvars.extend([False]*len(extra_vars))
            
            dep_vars_g.append(dep_vars)
            isFesvars_g.append(isFesvars)
            
        self._dep_vars = sum(dep_vars_g, [])
        self._dep_var_grouped = dep_vars_g
        self._isFESvar = sum(isFesvars_g, [])
        self._isFESvar_grouped = isFesvars_g
        self._fes_vars = [x for x, flag in zip(self._dep_vars, self._isFESvar) if flag]

        dprint1("dependent variables", self._dep_vars)
        dprint1("is FEspace variable?", self._isFESvar)
     
class SerialEngine(Engine):
    def __init__(self, modelfile='', model=None):
        super(SerialEngine, self).__init__(modelfile = modelfile, model=model)

    def run_mesh(self, meshmodel = None, skip_refine=False):
        '''
        skip_refine is for mfem_viewer
        '''
        return self.run_mesh_serial(meshmodel = meshmodel,
                                    skip_refine=skip_refine)

    def run_assemble_mat(self, phys_target=None):
        self.is_matrix_distributed = False       
        return super(SerialEngine, self).run_assemble_mat(phys_target=phys_target)

    def new_lf(self, fes):
        return  mfem.LinearForm(fes)

    def new_bf(self, fes):
        return  mfem.BilinearForm(fes)

    def new_mixed_bf(self, fes1, fes2):
        return  mfem.MixedBilinearForm(fes1, fes2)
     
    def new_gf(self, fes, init = True, gf = None):
        if gf is None:
           gf = mfem.GridFunction(fes)
        else:
           gf = mfem.GridFunction(gf.FESpace())               
        if init: gf.Assign(0.0)
        return gf

    def new_matrix(self, init = True):                                 
        return  mfem.SparseMatrix()

    def new_blockmatrix(self, shape):
        from petram.helper.block_matrix import BlockMatrix
        return BlockMatrix(shape, kind = 'scipy')

    def new_fespace(self, mesh, fec, vdim):
        return  mfem.FiniteElementSpace(mesh, fec, vdim)
    ''' 
    def fill_block_matrix_fespace(self, blocks, mv,
                                        gl_ess_tdof, interp,
                                        offset, convert_real = False):
        M, B, Me = blocks

        if len(mv) == 6:
            r_X, r_B, r_A, i_X, i_B, i_A = mv 
            is_complex = True
        else:
            r_X, r_B, r_A = mv ; i_A = None
            is_complex = False

        A1 = chypre.MfemMat2PyMat(r_A, i_A)
        M[offset, offset] = A1;  A1 = M[offset, offset]
        # this looks silly.. it actually convert A1 to ScipyCoo
        A1.resetDiagImag(gl_ess_tdof)
        # fix diagonal since they are set 1+1j. Serial version does not set
        # diagnal one. Here, it only set imaringary part to zero.
        
        P, nonzeros, zeros = interp
        if P is not None:
           PP = P.conj()           
           A1 = A1.rap(P.transpose())
           A1.setDiag(zeros, 1.0)
        
        M[offset, offset] = A1

        all_extras = [(key, self.extras[phys][key])  for phys in self.extras
                       for key in self.extras[phys]]
                      
        for key, v in all_extras:
            dep_var, extra_name = key
            idx0 = self.dep_var_offset(dep_var)
            idx1 = self.dep_var_offset(extra_name)                      
            t1, t2, t3, t4, t5 = v[0]
            mm = v[1]
            kk = mm_list.index(mm.fullname())
            if t1 is not None:
               dprint2("extra matrix nnz before elimination (t1), kfes="+str(k),
                    len(t1.nonzero()[0]))
            if t2 is not None:
               dprint2("extra matrix nnz before elimination (t2), kfes="+str(k),
                    len(t2.nonzero()[0]))

            if isinstance(t1, np.ndarray) or isinstance(t2, np.ndarray):
                if P is not None:               
                   if t1 is not None: t1 = PP.dot(t1)
                   #if t2 is not None: t2 = P.dot(t2.transpose()).transpose()
                   if t2 is not None: t2 = P.dot(t2)
            else:
                if t1 is not None:
                   #t1 = t1.tolil()
                   if P is not None:
                      #for i in  zeros:
                      #    t1[:, i] = 0.0
                      t1 = PP.dot(t1)#.tolil()                      
                if t2 is not None:
                   #t2 = t2.tolil()
                   if P is not None:
                       #for i in  zeros:
                       #    t2[i, :] = 0.0
                       #t2 = P.dot(t2.transpose()).transpose().tolil()
                       t2 = P.dot(t2)
            if t1 is not None: M[offset,   kk+offsete] = t1
            if t2 is not None: M[kk+offsete,   offset] = t2.transpose()
            if t3 is not None: M[kk+offsete, kk+offsete] = t3                

            #M[k+1+offset, offset] = t2
            #M[offset, k+1+offset] = t1
            #M[k+1+offset, k+1+offset] = t3

            #t4 = np.zeros(t2.shape[0])+t4 (t4 should be vector...)
            #t5 = [t5]*(t2.shape[0])

        return 

    def fill_block_rhs_fespace(self, blocks, mv, interp, offset):

        M, B, Me = blocks
        if len(mv) == 6:
            r_X, r_B, r_A, i_X, i_B, i_A = mv 
            is_complex = True
        else:
            r_X, r_B, r_A = mv ; i_B = None
            is_complex = False

        P, nonzeros, zeros = interp

        b1 = chypre.MfemVec2PyVec(r_B, i_B)
        
        if P is not None:
           PP = P.conj()
           b1 = PP.dot(b1)
           
        B[offset] = b1
        
        all_extras = [(key, self.extras[phys][key])  for phys in self.extras
                       for key in self.extras[phys]]
                      
        for key, v in all_extras:
            dep_var, extra_name = key
            idx0 = self.dep_var_offset(dep_var)
            idx1 = self.dep_var_offset(extra_name)                      
            t1, t2, t3, t4, t5 = v[0]
            mm = v[1]
            kk = mm_list.index(mm.fullname())
            if t4 is None: continue
            try:
                void = len(t4)
                t4 = t4
            except:
                raise ValueError("This is not supported")                
                t4 = np.zeros(t2.shape[0])+t4
            B[idx1] = t4

    def fill_block_from_mixed(self, loc,  m, interp1, interp2):
        if loc[2]  == -1:
           m = m.transpose()
        if loc[3]  == -1:
           m = m.conj()

        P1, nonzeros, zeros = interp1[1]
        P2, nonzeros, zeros = interp2[1]

        if P1 is not None:
           m = P1.dot(m)
        if P2 is not None:
           m = m.dot(P2.conj().transpose())
        return m
    '''
    ''' 
    def finalize_coo_matrix(self, M_block, is_complex, convert_real = False):
        if not convert_real:
            if is_complex:
                M = M_block.get_global_coo(dtype='complex')           
            else:
                M = M_block.get_global_coo(dtype='float')                          
        else:
            M = M_block.get_global_coo(dtype='complex')                      
            M = scipy.sparse.bmat([[M.real, -M.imag], [-M.imag, -M.real]], format='coo')
        return M
    '''
    def collect_all_ess_tdof(self):
        self.gl_ess_tdofs = self.ess_tdofs

    def save_mesh(self):
        mesh_names = []
        for k, mesh in enumerate(self.emeshes):
            if mesh is None: continue
            name = 'solmesh_' + str(k)           
            mesh.PrintToFile(name, 8)
            mesh_names.append(name)
        return mesh_names

    def save_parmesh(self):
        # serial engine does not do anything
        return

    def solfile_name(self, name, mesh_idx,
                     namer = 'solr', namei = 'soli' ):
        fnamer = '_'.join((namer, name, str(mesh_idx)))
        fnamei = '_'.join((namei, name, str(mesh_idx)))
        mesh_name  =  "solmesh_"+str(mesh_idx)              
        return fnamer, fnamei, mesh_name

    def extrafile_name(self):
        return 'sol_extended.data'

    def get_true_v_sizes(self, phys):
        fe_sizes = [self.fespaces[name].GetTrueVSize() for name in phys.dep_vars]
        dprint1('Number of finite element unknowns: '+  str(fe_sizes))
        return fe_sizes

    def split_sol_array_fespace(self, sol, P):
        sol0 = sol[0, 0]
        if P is not None:
           sol0 = P.transpose().dot(sol0)
        return sol0
     

    def mkdir(self, path):
        if not os.path.exists(path):  os.mkdir(path)
    def cleancwd(self):
        for f in os.listdir("."): os.remove(f)
    def remove_solfiles(self):       
        dprint1("clear sol: ", os.getcwd())                  
        d = os.getcwd()
        files = os.listdir(d)
        for file in files:
            if file.startswith('solmesh'): os.remove(os.path.join(d, file))
            if file.startswith('solr'): os.remove(os.path.join(d, file))
            if file.startswith('soli'): os.remove(os.path.join(d, file))


    def a2A(self, a):  # BilinearSystem to matrix
        # we dont eliminate essentiaal at this level...                 
        inta = mfem.intArray()
        m = self.new_matrix()
        a.FormSystemMatrix(inta, m)
        return m

    def a2Am(self, a):  # MixedBilinearSystem to matrix
        a.ConformingAssemble()
        return a.SpMat()
    
    def b2B(self, b):  # FormLinearSystem w/o elimination
        fes = b.FESpace()
        B = mfem.Vector()
        if not fes.Conforming():
            P = fes.GetConformingProlongation()
            R = fes.GetConformingRestriction()
            B.SetSize(P.Width())
            P.MultTranspose(b, B)
        else:
            B.NewDataAndSize(b.GetData(), b.Size())
        return B
     
    def x2X(self, x):  # gridfunction to vector
        fes = x.FESpace()
        X = mfem.Vector()
        if not fes.Conforming():
            P = fes.GetConformingProlongation()
            R = fes.GetConformingRestriction()
            X.SetSize(R.Height())
            R.Mult(x, X)
        else:
            X.NewDataAndSize(x.GetData(), x.Size())
        return X
     
    def X2x(self, X, x): # RecoverFEMSolution
        fes = x.FESpace()
        if fes.Conforming():
            pass
        else:
            P = fes.GetConformingProlongation()
            x.SetSize(P.Height())
            P.Mult(X, x)

            
class ParallelEngine(Engine):
    def __init__(self, modelfile='', model=None):
        super(ParallelEngine, self).__init__(modelfile = modelfile, model=model)


    def run_mesh(self, meshmodel = None):
        from mpi4py import MPI
        from petram.mesh.mesh_model import MeshFile, MFEMMesh
        from petram.mesh.mesh_extension import MeshExt
        from petram.mesh.mesh_utils import  get_extended_connectivity


        self.meshes = []
        self.emeshes = []
        self.emesh_data = MeshExt()
        
        if meshmodel is None:
            parent = self.model['Mesh']
            children =  [parent[g] for g in parent.keys()
                         if isinstance(parent[g], MFEMMesh)]
            for idx, child in enumerate(children):
                self.meshes.append(None)
                if not child.enabled: continue
                target = None
                for k in child.keys():
                    o = child[k]
                    if not o.enabled: continue
                    if isinstance(o, MeshFile):
                        smesh = o.run()
                        self.max_bdrattr = np.max([self.max_bdrattr,
                                                   max(smesh.GetBdrAttributeArray())])
                        self.max_attr = np.max([self.max_attr,
                                                max(smesh.GetAttributeArray())])
                        self.meshes[idx] = mfem.ParMesh(MPI.COMM_WORLD, smesh)
                        target = self.meshes[idx]
                    else:
                        if hasattr(o, 'run') and target is not None:
                            self.meshes[idx] = o.run(target)
                            
        for m in self.meshes:
            m.ReorientTetMesh()
            m.GetEdgeVertexTable()                                   
            get_extended_connectivity(m)           

    def run_assemble_mat(self, phys_target=None):
        self.is_matrix_distributed = True       
        return super(ParallelEngine, self).run_assemble_mat(phys_target=phys_target)
     
    def new_lf(self, fes):
        return  mfem.ParLinearForm(fes)

    def new_bf(self, fes):
        return  mfem.ParBilinearForm(fes)
     
    def new_mixed_bf(self, fes1, fes2):
        return  mfem.ParMixedBilinearForm(fes1, fes2)

    def new_gf(self, fes, init = True, gf = None):
        if gf is None:
           gf = mfem.ParGridFunction(fes)
        else:
           gf = mfem.ParGridFunction(gf.ParFESpace())               
        if init: gf.Assign(0.0)
        return gf
               
    def new_fespace(self,mesh, fec, vdim):
        if mesh.__class__.__name__ == 'ParMesh':
            return  mfem.ParFiniteElementSpace(mesh, fec, vdim)
        else:
            return  mfem.FiniteElementSpace(mesh, fec, vdim)
         
    def new_matrix(self, init = True):
        return  mfem.HypreParMatrix()

    def new_blockmatrix(self, shape):
        from petram.helper.block_matrix import BlockMatrix
        return BlockMatrix(shape, kind = 'hypre')

    def get_true_v_sizes(self, phys):
        fe_sizes = [self.fespaces[name].GlobalTrueVSize() for name in phys.dep_vars]       
        from mpi4py import MPI
        myid     = MPI.COMM_WORLD.rank        
        if (myid == 0):
               dprint1('Number of finite element unknowns: '+  str(fe_sizes))
        return fe_sizes
     
    def save_mesh(self):
        from mpi4py import MPI                               
        num_proc = MPI.COMM_WORLD.size
        myid     = MPI.COMM_WORLD.rank
        smyid = '{:0>6d}'.format(myid)

        mesh_names = []
        for k, mesh in enumerate(self.emeshes):
            if mesh is None: continue
            mesh_name  =  "solmesh_"+str(k)+"."+smyid
            mesh.PrintToFile(mesh_name, 8)
            mesh_names.append(mesh_name)
        return mesh_names
     
    def save_parmesh(self):
        from mpi4py import MPI                               
        num_proc = MPI.COMM_WORLD.size
        myid     = MPI.COMM_WORLD.rank
        smyid = '{:0>6d}'.format(myid)

        mesh_names = []
        for k, mesh in enumerate(self.meshes):
            if mesh is None: continue
            mesh_name  =  "solparmesh_"+str(k)+"."+smyid
            mesh.ParPrintToFile(mesh_name, 8)
        return
     
    def solfile_name(self, name, mesh_idx,
                     namer = 'solr', namei = 'soli' ):
        from mpi4py import MPI                               
        num_proc = MPI.COMM_WORLD.size
        myid     = MPI.COMM_WORLD.rank
        smyid = '{:0>6d}'.format(myid)
       
        fnamer = '_'.join((namer, name, str(mesh_idx)))+"."+smyid
        fnamei = '_'.join((namei, name, str(mesh_idx)))+"."+smyid
        mesh_name  =  "solmesh_"+str(mesh_idx)+"."+smyid        
        return fnamer, fnamei, mesh_name

    def extrafile_name(self):
        from mpi4py import MPI                               
        num_proc = MPI.COMM_WORLD.size
        myid     = MPI.COMM_WORLD.rank
        smyid = '{:0>6d}'.format(myid)
       
        return 'sol_extended.data.'+smyid

    def fill_block_matrix_fespace(self, blocks, mv,
                                        gl_ess_tdof, interp,
                                        offset, convert_real = False):
                                      
        '''
        fill block matrix for the left hand side
        '''
        from mpi4py import MPI
        myid     = MPI.COMM_WORLD.rank
                                      
        if len(mv) == 6:
            r_X, r_B, r_A, i_X, i_B, i_A = mv 
            is_complex = True
        else:
            r_X, r_B, r_A = mv ; i_A = None
            is_complex = False
            
        M, B, Me = blocks

        A1 = chypre.MfemMat2PyMat(r_A, i_A)
        M[offset, offset] = A1;  A1 = M[offset, offset]
        # use the same as in the serial 
        #M.set_element(r_A, i_A, offset, offset)
        #A1 = M[offset, offset]

        A1.setDiag(gl_ess_tdof, 1.0) # fix diagonal since they are set 1+1j
        P, nonzeros, zeros = interp
        
        if P is not None:
           dprint1("P is not None")
           A1 = A1.rap(P.transpose())
           A1.setDiag(zeros, 1.0) # comment this when making final matrix smaller

        M[offset, offset] = A1
        all_extras = [(key, self.extras[phys][key])  for phys in self.extras
                       for key in self.extras[phys]]
                      
        for key, v in all_extras:
            dep_var, extra_name = key
            idx0 = self.dep_var_offset(dep_var)
            idx1 = self.dep_var_offset(extra_name)                      
            t1, t2, t3, t4, t5 = v[0]
            mm = v[1]
            kk = mm_list.index(mm.fullname())
            if (isinstance(t1, chypre.CHypreMat) or
                isinstance(t2, chypre.CHypreMat)):
                if t1 is not None: dprint1("t1, shape", t1.shape)
                if t2 is not None: dprint1("t2, shape", t2.shape)
                if P is not  None:
                    if t1 is not None: t1 = P.conj().dot(t1); P.conj()
                    if t2 is not None: t2 = P.dot(t2)
            elif isinstance(t1, chypre.CHypreVec): # 1D array
                if P is not  None:
                    if t1 is not None: t1 = P.conj().dot(t1); P.conj()
                    if t2 is not None: t2 = P.dot(t2)
                # this should be taken care in finalization                      
                #for x in ess_tdof_list:
                #    t1.set_element(x, 0.0)
                #from petram.helper.chypre_to_pymatrix import Vec2MatH, Vec2MatV
                #t1 = Vec2MatV(t1, is_complex)
                #t2 = Vec2MatH(t2, is_complex)                
            else:
                pass
            #nicePrint('t2', t2[0].GetRowPartArray(), t2[0].GetColPartArray())
            
            if t1 is not None: M[idx0,   idx1] = t1
            if t2 is not None: M[idx1,   idx0] = t2.transpose()
            if t3 is not None: M[idx1,   idx1] = t3
    '''
    def fill_block_rhs_fespace(self, blocks, mv, interp, offset):
        from mpi4py import MPI
        myid     = MPI.COMM_WORLD.rank

        M, B,  Me = blocks
        if len(mv) == 6:
            r_X, r_B, r_A, i_X, i_B, i_A = mv
            is_complex = True
        else:
            r_X, r_B, r_A = mv; i_B = None
            is_complex = False

        b1 = chypre.MfemVec2PyVec(r_B, i_B)

        P, nonzeros, zeros = interp
        if P is not None:
           b1 = P.conj().dot(b1)
           P.conj() # to set P back
        B[offset] = b1

        all_extras = [(key, self.extras[phys][key])  for phys in self.extras
                       for key in self.extras[phys]]
                      
        for key, v in all_extras:
            dep_var, extra_name = key
            idx0 = self.dep_var_offset(dep_var)
            idx1 = self.dep_var_offset(extra_name)                      
            t1, t2, t3, t4, t5 = v[0]
            mm = v[1]
            kk = mm_list.index(mm.fullname())
            if t4 is None: continue
            try:
               void = len(t4)
               t4 = t4
            except:
               raise ValueError("This is not supported")
               t4 = np.zeros(t2.M())+t4
            B[idx1] = t4

    def fill_block_from_mixed(self, loc, m, interp1, interp2):
        if loc[2]  == -1:
           m = m.transpose()
        if loc[3]  == -1:
           m = m.conj()
       
        # should fix here
        P1, nonzeros, zeros = interp1[1]
        P2, nonzeros, zeros = interp2[1]

        if P1 is not None:
           m = P1.dot(m)
        if P2 is not None:
           m = m.dot(P2.conj().transpose())        
           P2.conj() # set P2 back...
        return m
    '''
    ''' 
    def finalize_coo_matrix(self, M_block, is_complex, convert_real = False):     
        if not convert_real:
            if is_complex:
                M = M_block.get_global_coo(dtype='complex')           
            else:
                M = M_block.get_global_coo(dtype='float')                          
        else:
            M = M_block.get_global_coo(dtype='complex')                      
            M = scipy.sparse.bmat([[M.real, -M.imag], [-M.imag, -M.real]], format='coo')
        return M
    '''
    def split_sol_array_fespace(self, sol, P):
        sol0 = sol[0, 0]
        if P is not None:
           sol0 = (P.transpose()).dot(sol0)
        return sol0

    def collect_all_ess_tdof(self, M = None):
        from mpi4py import MPI

        #gl_ess_tdofs = []
        #for name in phys.dep_vars:
        #    fes = self.fespaces[name]
            
        for name in self.ess_tdofs:
            tdof = self.ess_tdofs[name]
            fes =  self.fespaces[name]
            data = (np.array(tdof) +
                    fes.GetMyTDofOffset()).astype(np.int32)
   
            gl_ess_tdof = allgather_vector(data, MPI.INT)
            MPI.COMM_WORLD.Barrier()
            #gl_ess_tdofs.append((name, gl_ess_tdof))
            ## TO-DO intArray must accept np.int32
            tmp = [int(x) for x in gl_ess_tdof]
            self.gl_ess_tdofs[name] = tmp
     
    def mkdir(self, path):
        myid     = MPI.COMM_WORLD.rank                
        if myid == 0:
           if not os.path.exists(path): os.mkdir(path)           
        else:
           pass
        MPI.COMM_WORLD.Barrier()                

    def cleancwd(self):
        myid     = MPI.COMM_WORLD.rank                
        if myid == 0:
            for f in os.listdir("."): os.remove(f)            
        else:
            pass
        MPI.COMM_WORLD.Barrier()                

    def remove_solfiles(self):       
        dprint1("clear sol: ", os.getcwd())                  
        myid     = MPI.COMM_WORLD.rank                
        if myid == 0:
            d = os.getcwd()
            files = os.listdir(d)
            for file in files:
                if file.startswith('solmesh'): os.remove(os.path.join(d, file))
                if file.startswith('solr'): os.remove(os.path.join(d, file))
                if file.startswith('soli'): os.remove(os.path.join(d, file))
        else:
            pass
        MPI.COMM_WORLD.Barrier()

    def a2A(self, a):   # BilinearSystem to matrix
        # we dont eliminate essentiaal at this level...                 
        inta = mfem.intArray()
        m = self.new_matrix()
        a.FormSystemMatrix(inta, m)
        return m

    def a2Am(self, a):  # MixedBilinearSystem to matrix
        a.Finalize()
        return a.ParallelAssemble()
    
    def b2B(self, b):
        fes = b.ParFESpace()
        B = mfem.HypreParVector(fes)
        P = fes.GetProlongationMatrix()       
        B.SetSize(fes.TrueVSize())
        P.MultTranspose(b, B)

        return B
     
    def x2X(self, x):
        fes = x.ParFESpace()
        X = mfem.HypreParVector(fes)        
        R = fes.GetRestrictionMatrix()       
        X.SetSize(fes.TrueVSize())
        R.Mult(x, X)            
        return X
     
    def X2x(self, X, x): # RecoverFEMSolution
        fes = x.ParFESpace()
        P = fes.GetProlongationMatrix()
        x.SetSize(P.Height())
        P.Mult(X, x)
     
        
        
  
