#!/bin/env python
import sys
import os
import numpy as np
import scipy.sparse
from warnings import warn
import weakref
from weakref import WeakKeyDictionary

from petram.mfem_config import use_parallel
if use_parallel:
   from petram.helper.mpi_recipes import *
   import mfem.par as mfem   
else:
   import mfem.ser as mfem
import mfem.common.chypre as chypre

#this is only for debuging
from mfem.common.parcsr_extra import ToScipyCoo

from petram.model import Domain, Bdry
import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('Engine')
from petram.helper.matrix_file import write_coo_matrix, write_vector

#groups = ['Domain', 'Boundary', 'Edge', 'Point', 'Pair']
groups = ['Domain', 'Boundary', 'Pair']

class ModelDict(WeakKeyDictionary):
    def __init__(self, root):
        WeakKeyDictionary.__init__(self)
        self.root = weakref.ref(root)
        
    def __setitem__(self, key, value):
        hook = key.get_hook()
        return WeakKeyDictionary.__setitem__(self, hook, value)

    def __getitem__(self, key):
        hook = key.get_hook()
        return WeakKeyDictionary.__getitem__(self, hook)
        
    def __iter__(self):
        return [reduce(lambda x, y: x[y], [self.root()] + hook().names)
                for hook in self.keys()]

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
        
        # place holder : key is base physics modules, such as EM3D1...
        #
        # for example : self.r_b['EM3D1'] = [LF for E, LF for psi]
        #               physics moduel provides a map form variable name to index.

        self.case_base = 0
        
    def set_model(self, model):
        self.model = model
        self.is_assembled = False
        self.is_initialized = False        
        self.meshes = []
        
        if model is not None:
            self.fespaces = ModelDict(model)
            self.fec = ModelDict(model)            
            self.r_b = ModelDict(model)
            self.r_x = ModelDict(model)
            self.r_a = ModelDict(model)
            self.i_b = ModelDict(model)
            self.i_x = ModelDict(model)
            self.i_a = ModelDict(model)
            self.mixed_bf = ModelDict(model)
            self.interps = ModelDict(model)
            self.extras = ModelDict(model)
            self.gl_ess_tdofs = ModelDict(model)
            self.alloc_flag  = ModelDict(model)
        else:
            self.fespaces = None
            self.fec = None            
            self.r_b = None
            self.r_x = None
            self.r_a = None
            self.i_b = None
            self.i_x = None
            self.i_a = None
            self.mixed_bf = None
            self.interps = None
            self.extras = None
            self.gl_ess_tdofs = None
            self.alloc_flag = ModelDict(model)
            

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
        self.run_mesh()
        self.assign_sel_index()

        if dir is None:
            dir = os.path.dirname(os.path.realpath(mainfile))           
        for node in model.walk():
            if node.has_ns() and node.ns_name is not None:
                node.read_ns_script_data(dir = dir)
        self.build_ns()
        self.run_preprocess()
        
        solver = model["Solver"].get_active_solver()
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
            #if phys.check_new_fespace(self.fespaces, self.meshes):
            self.allocate_fespace(phys)
 
            self.assign_sel_index(phys)
            for node in phys.walk():
                if not node.enabled: continue
                node.preprocess_params(self)
        for k in self.model['InitialValue'].keys():
            init = self.model['InitialValue'][k]
            init.preprocess_params(self)            

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
              if self.alloc_flag[phys]: alloced_phys.append[phys]
           except:
              pass
        phys_target = [phys for phys in phys_target
                       if not phys in allocated_phys]
        dprint1("allocating fespace/sol vector for " + str(phys_target))
        
        for phys in phys_target:
            self.run_update_param(phys)
        for phys in phys_target:
            self.initialize_phys(phys)

        from petram.helper.variables import Variables
        variables = Variables()
        
        for phys in phys_target:            
            for kfes, rgf, igf in enum_fes(phys, self.r_x, self.i_x):
                name = phys.dep_vars[kfes]
                phys.add_variables(variables, name, rgf, igf)

        keys = self.model._variables.keys()
        self.model._variables.clear()
        dprint1("===  List of variables ===")
        dprint1(variables)
        for k in variables.keys():
           self.model._variables[k] = variables[k]
        self.is_initialized = True
        for phys in phys_target:        
            self.alloc_flag[phys] = True

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
        for phys in phys_target:
           if mode == 0:
               for kfes, rgf, igf in enum_fes(phys, self.r_x, self.i_x):
                   rgf.Assign(0.0)
                   if igf is not None: igf.Assign(0.0)
           elif mode == 1: 
               for kfes, rgf, igf in enum_fes(phys, self.r_x, self.i_x):
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
            self.apply_essential(phys)
     
    def run_assemble(self, phys_target = None):
        matvecs = ModelDict(self.model)
        matvecs_c = ModelDict(self.model)
        #for phys in phys_target:
        #    self.run_update_param(phys)
        for phys in phys_target:
            matvec = self.assemble_phys(phys)
            matvecs[phys] = matvec
            if "Coupling" in self.model:
               matvec = assemble_coupling(self.model["Coupling"],
                                          oneway_only = False)
               matvecs_c[phys] = matvec
            else:
               matvecs_c[phys] = None
        return matvecs, matvecs_c

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
        vecs = ModelDict(self.model)
        vecs_c = ModelDict(self.model)
        for phys in phys_target:
            self.run_update_param(phys)

        for phys in phys_target:
            vec = self.assemble_rhs(phys)
            vecs[phys] = vec
            if "Coupling" in self.model:
               vec = assemble_coupling(self.model["Coupling"], oneway_only = True)
               vecs_c[phys] = vec
            else:
               vecs_c[phys] = None
        return vecs, vecs_c
    #
    #  update mode param
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
        ess_tdofs = self.get_essential_bdr_tofs(phys, flags)

        self.allocate_gf(phys)
        
    def assemble_phys(self, phys):
        '''
        assemble matrix made from block matrices
        each block can be either PyMatrix or HypreMatrix
        '''
        is_complex = phys.is_complex()        
        #self.assign_sel_index(phys)
        #self.allocate_fespace(phys)
        #true_v_sizes = self.get_true_v_sizes(phys)
        flags = self.get_essential_bdr_flag(phys)
        ess_tdofs = self.get_essential_bdr_tofs(phys, flags)

        self.allocate_lf(phys)
        self.allocate_bf(phys)

        matvec = self.allocate_matvec(phys)
                
        self.apply_essential(phys)
        self.assemble_lf(phys)
        self.assemble_bf(phys)
        self.assemble_mixed(phys)        

        self.call_FormLinearSystem(phys, ess_tdofs, matvec)

        # collect data for essential elimination
        self.collect_all_ess_tdof(phys, ess_tdofs)
        self.assemble_extra(phys)
        self.assemble_interp(phys)

        return matvec #r_X, r_B, r_A, i_X, i_B, i_A 
        
    def assemble_coupling(self, coupling, oneway_only = False):
        raise NotImplementedError(
             "between Module coupling")
        '''
        it should return array of mixed-bf and lf
     
            for kk in self.model['Coupling'].keys():
               coupling = model['Coupling'][kk]
               if coupling.has_oneway(phys): # this is linear form
                  vec = self.assemble_oneway(coupling, phys)
               else:
                  vec = None
               if coupling.has_twoway(phys): # this is linear form
                  mat = self.assemble_twoway(coupling, phys)
               else:
                  mat = None
        '''  

    def assemble_rhs(self, phys):
        is_complex = phys.is_complex()

        flags = self.get_essential_bdr_flag(phys)
        ess_tdofs = self.get_essential_bdr_tofs(phys, flags)
        self.collect_all_ess_tdof(phys, ess_tdofs)
        
        self.allocate_lf(phys)
        
        self.apply_essential(phys)
        self.assemble_lf(phys)        

        matvec = self.allocate_matvec(phys)
        self.call_FormLinearSystem(phys, ess_tdofs, matvec)
        
        self.assemble_extra(phys)
        
        return matvec
        
    def apply_essential(self, phys):
        for kfes, rgf, igf in enum_fes(phys, self.r_x, self.i_x):
            #rgf.Assign(0.0)
            #if igf is not None:
            #   igf.Assign(0.0)
            for mm in phys.walk():
                if not mm.enabled: continue
                if not mm.has_essential: continue
                if len(mm.get_essential_idx(kfes)) == 0: continue
                mm.apply_essential(self, rgf, real = True, kfes = kfes)
                if igf is not None:
                    mm.apply_essential(self, igf, real = False, kfes = kfes)

    def apply_init_from_init_panel(self, phys):
        for kfes, rgf, igf in enum_fes(phys, self.r_x, self.i_x):
            tmp = self.new_gf(None, gf = rgf)
            for mm in phys.walk():
                if not mm.enabled: continue
                c = mm.get_init_coeff(self, real = True, kfes = kfes)
                if c is None: continue
                tmp.ProjectCoefficient(c)                
                rgf += tmp
            if igf is None: continue
            tmp *= 0.0
            for mm in phys.walk():
                if not mm.enabled: continue
                c = mm.get_init_coeff(self, real = False, kfes = kfes)
                if c is None: continue
                tmp.ProjectCoefficient(c)
                igf += tmp

    def apply_init_from_file(self, phys, init_path):
        '''
        read initial gridfunction from solution
        if init_path is "", then file is read from cwd.
        if file is not found, then it zeroes the gf
        '''
        mesh_idx = phys.mesh_idx
        names = phys.dep_vars
        
        for kfes, rgf, igf in enum_fes(phys, self.r_x, self.i_x):
            fr, fi, meshname = self.solfile_name(names[kfes],
                                                         mesh_idx)
            path = os.path.expanduser(init_path)
            if path == '': path = os.getcwd()
            fr = os.path.join(path, fr)
            fi = os.path.join(path, fi)
            meshname = os.path.join(path, meshname)

            print meshname, fr
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
       
    def assemble_bf(self, phys):
        for kfes, ra, ia in enum_fes(phys, self.r_a, self.i_a):
            for mm in phys.walk():
                if not mm.enabled: continue
                if not mm.has_bf_contribution(kfes):continue
                if len(mm._sel_index) == 0: continue 
                mm.add_bf_contribution(self, ra, real = True, kfes = kfes)
                if ia is not None:
                    mm.add_bf_contribution(self, ia, real = False, kfes = kfes)
            ra.Assemble()
            if ia is not None: ia.Assemble()
            
    def assemble_lf(self, phys):
        for kfes, rb, ib in enum_fes(phys, self.r_b, self.i_b):       
            rb.Assign(0.0)
            if ib is not None: ib.Assign(0.0)
            for mm in phys.walk():
               if not mm.enabled: continue
               if not mm.has_lf_contribution(kfes): continue
               if len(mm._sel_index) == 0: continue                          
               mm.add_lf_contribution(self, rb, real=True, kfes=kfes)
               if ib is not None:
                  mm.add_lf_contribution(self, ib, real=False, kfes=kfes)
            rb.Assemble()
            if ib is not None: ib.Assemble()
            
    def assemble_mixed(self, phys):
        is_complex = phys.is_complex()
        mixed_bf = {}
        tasks = {}
        fespaces = self.fespaces[phys]
        for mm in phys.walk():
            if not mm.enabled: continue
            if not mm.has_mixed_contribution():continue
            if len(mm._sel_index) == 0: continue
                          
            loc_list = mm.get_mixedbf_loc()

            for loc in loc_list:
                r,c, is_trans, is_conj= loc
                if not loc in mixed_bf:
                    if loc[2] > 0:
                        fes1 = fespaces[c][1]
                        fes2 = fespaces[r][1]
                    else: # fill transpose
                        fes1 = fespaces[r][1]
                        fes2 = fespaces[c][1]
                    if is_complex:
                        mixed_bf[loc] = [self.new_mixed_bf(fes1, fes2),
                                         self.new_mixed_bf(fes1, fes2)]
                    else:
                        mixed_bf[loc] = [self.new_mixed_bf(fes1, fes2)]
                    tasks[loc] = [mm]
                else:
                    tasks[loc].append(mm)
        for loc in tasks:
            r,c, is_trans, is_conj= loc           
            for mm in tasks[loc]:
                if not mm.enabled: continue               
                mm.add_mix_contribution(self, mixed_bf[loc][0], r, c, real = True)
                if is_complex:
                     mm.add_mix_contribution(self, mixed_bf[loc][1], r, c, real = False)
        for loc in mixed_bf:
            for mbf in mixed_bf[loc]: mbf.Assemble()
        self.mixed_bf[phys]  = mixed_bf

    def assemble_extra(self, phys):
        names = phys.dep_vars      
        extras = []
        for kfes, gl_ess_tdof in enum_fes(phys, self.gl_ess_tdofs):
            extra = []
            for mm in phys.walk():
                if not mm.enabled: continue           
                if mm.has_extra_DoF(kfes):
                    tmp  = mm.add_extra_contribution(self,
                                                     ess_tdof=gl_ess_tdof, 
                                                     kfes = kfes)
                    if tmp is None: continue
                    extra.append((tmp, mm))
            extras.append((names[kfes], extra))
        self.extras[phys] = extras


    def assemble_interp(self, phys):
        names = phys.dep_vars
        interps = []
        for kfes, gl_ess_tdof in enum_fes(phys, self.gl_ess_tdofs):        
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

            interps.append((names[kfes], (P, nonzeros, zeros)))
        self.interps[phys] = interps

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
                ib = i_B.GetDataArray().astype(float) # astype will allocate
                                                      # new memory
                i_B *= 0.0; i_b.Assign(0.0)
                i_a.FormLinearSystem(ess_tdofs[kfes][1], r_x, i_b, i_A, r_X, i_B)
                i_B += mfem.Vector(ib)
                for k in ess: i_B[k] = ib[k]

        import petram.debug as debug
        if debug.debug_essential_bc:
            name = self.fespaces[phys][kfes][0]
            mesh_idx = phys.mesh_idx 
            self.save_solfile_fespace(name, mesh_idx, r_x, i_x,
                                          namer = 'x_r',
                                          namei = 'x_i')

    #
    #  build linear system construction
    #
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
        blocks = self.prepare_blocks(phys_target)
        self.fill_block_rhs(phys_target, blocks, vecs, vecs_c)
        return blocks

    def fill_block_matrix(self, phys_target, blocks, matvecs, matvecs_c):
        '''
        assemble block matrix into one coo matrix

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
              x = P^t y
        
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
        M_block, B_block,  Me_block = blocks
        # (step1) each FESpace is handled 
        #         apply projection

        for phys, offsets in iter_phys(phys_target, self.phys_offsets):
            for kfes, extra, interp, gl_ess_tdof in enum_fes(phys,
                         self.extras, self.interps, self.gl_ess_tdofs):
                offset = offsets[kfes]
                mvs = matvecs[phys]
                mv = [mm[kfes] for mm in mvs]                
                #matvec[k]  =  r_X, r_B, r_A, i_X, i_B, i_A or r only

                self.fill_block_matrix_fespace(blocks, mv,
                                               gl_ess_tdof, extra, 
                                               interp, 
                                               offset)
        # (step2) in-physics coupling (MixedForm)
        for phys, offsets, mixed_bf, interps in iter_phys(
                                          phys_target, self.phys_offsets,
                                          self.mixed_bf, self.interps):        
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

    def fill_block_rhs(self, phys_target, blocks, matvecs, matvecs_c):
        # (step1) each FESpace is handled 
        #         apply projection
        dprint1("Entering Filling Block RHS")
        dprint2("\n", blocks[1])
        
        for phys, offsets, mvs in iter_phys(phys_target, self.phys_offsets,
                                                  matvecs):
            for kfes, extra, interp, gl_ess_tdof in enum_fes(phys,
                         self.extras, self.interps, self.gl_ess_tdofs):
                offset = offsets[kfes]
                mv = [mm[kfes] for mm in mvs]                
                self.fill_block_rhs_fespace(blocks, mv, extra, interp,
                                            offset)

        # (step3) mulit-physics coupling (???)
        #         apply projection + place in a block format
        dprint2("\n", blocks[1])
        dprint1("Exiting Filling Block RHS")        
        return
       
    def fill_elimination_block(self, phys_target, blocks):
        '''
        make block eliminaition matrix for off-diagonal blocks
        for essential DoF
        '''
        dprint1("Entering Filling Elimination Block")

        M = blocks[0]
        Me = blocks[2]
        size = Me.shape[0]
        dprint2("\n", M)

        # eliminate horizontal.
        dprint2("Filling Elimination Block (step1)")    
        for phys, gl_ess_tdofs, offsets in iter_phys(phys_target,
                                                     self.gl_ess_tdofs,
                                                     self.phys_offsets):
            for offset, gl_ess_tdof in zip(offsets, gl_ess_tdofs):
                for ib in range(size):
                    if ib == offset: continue
                    if M[offset, ib] is None: continue                    
                    M[offset, ib].resetRow(gl_ess_tdof[1])

        # store vertical to Me (choose only essential col)
        dprint2("Filling Elimination Block (step2)")        
        for phys, gl_ess_tdofs, offsets in iter_phys(phys_target,
                                                     self.gl_ess_tdofs,
                                                     self.phys_offsets):
            for offset, gl_ess_tdof in zip(offsets, gl_ess_tdofs):
                for ib in range(size):
                    if ib == offset: continue
                    if M[ib, offset] is None: continue
                    SM = M.get_squaremat_from_right(ib, offset)
                    SM.setDiag(gl_ess_tdof[1])
                    dprint3("calling dot", ib, offset, M[ib, offset], SM)
                    Me[ib, offset] = M[ib, offset].dot(SM)

        dprint1("Exiting fill_elimination_block\n", M)
        
    def prepare_blocks(self, phys_target):
        phys_offsets = ModelDict(self.model)
        base_offset = 0
        for phys, extras in iter_phys(phys_target, self.extras):
            num_blocks = [1 + len(extra) for name, extra in extras]
            offsets = np.hstack((0, np.cumsum(num_blocks)))+base_offset
            phys_offsets[phys] = offsets
            base_offset = offsets[-1]
            dprint3("offset ", offsets)
        
        size = base_offset
        M_block = self.new_blockmatrix((size, size))
        B_block = self.new_blockmatrix((size, 1))
        Me_block = self.new_blockmatrix((size, size))
        
        self.phys_offsets = phys_offsets
        return (M_block, B_block, Me_block)

    #
    #  finalize linear system construction
    #
    def eliminate_and_shrink(self,  M_block, B_blocks, Me):
        # eliminate dof
        dprint1("Eliminate and Shrink")
        dprint1("Me\n", Me)

        # essentailBC is stored in b
        B_blocks = [b - Me.dot(b) for b in B_blocks]
        
        # shrink size
        dprint2("M (before shrink)\n", M_block)        
        M_block2, P2 = M_block.eliminate_empty_rowcol()
        dprint1("P2\n", P2)
        
        B_blocks = [P2.dot(b) for b in B_blocks]

        dprint2("M (after shrink)\n", M_block2)                
        dprint2("B (after shrink)\n", B_blocks[0])

        return M_block2, B_blocks, P2
     
    def finalize_linearsystem(self, M_block, B_blocks, is_complex,
                              format = 'coo'):
        
        if format == 'coo': # coo either real or complex
            M = self.finalize_coo_matrix(M_block, is_complex)
            B = [self.finalize_coo_rhs(b, is_complex) for b in B_blocks]
            B = np.hstack(B)

        elif format == 'coo_real': # real coo converted from complex
            M = self.finalize_coo_matrix(M_block, is_complex,
                                            convert_real = True)

            B = [self.finalize_coo_rhs(b, is_complex,
                                   convert_real = True)
                    for b in B_blocks]
            B = np.hstack(B)


        #S = self.finalize_flag(S_block)
        self.is_assembled = True
        return M, B

    def finalize_coo_rhs(self, b, is_complex,
                     convert_real = False):
        dprint1("b \n",  b)
        B = b.gather_densevec()
        if convert_real:
            B = np.hstack((B.real, -B.imag))
        return B
    #
    #  processing solution
    #
    def split_sol_array(self, phys_target, sol):

        s = []
        e = []

        for phys in phys_target:
            offsets = self.phys_offsets[phys]
            for kfes, interps, in enum_fes(phys, self.interps):
               P, nonzeros, zeros = interps
               sol_section = sol[offsets[kfes]:offsets[kfes+1], 0]
               s1, s2 = self.split_sol_array_fespace(sol_section, P)
               s.append(s1)  ## I need to add toarray to hypre?
               e.append(s2)

        return s, e
     
    def recover_sol(self, phys_target, matvecs, sol):
        k = 0

        for phys in phys_target:
            for kfes, r_a, r_b, r_x, i_a, i_b, i_x  in enum_fes(phys,
                                     self.r_a, self.r_b, self.r_x,
                                     self.i_a, self.i_b, self.i_x):
                mvs = matvecs[phys]
                mv = [mm[kfes] for mm in mvs]                
                r_X = mv[0]
                s = sol[k].toarray()
                r_X.SetVector(mfem.Vector(s.real), 0)           
                r_a.RecoverFEMSolution(r_X, r_b, r_x)
                if len(mvs)== 6:  #r_X, r_B, r_A, i_X, i_B, i_A
                   i_X = mv[3]
                   i_X.SetVector(mfem.Vector(s.imag), 0)
                   i_a.RecoverFEMSolution(i_X, i_b, i_x)
                k = k + 1

    def process_extra(self, phys_target, sol_extra):
        ret = {}
        k = 0
        for phys in phys_target:
            for name, extra in self.extras[phys]:
               dataset = sol_extra[k]
               key = phys.name()+ "." + name
               ret[key] = {}
               for kextra, v in enumerate(extra):
                   data = dataset[kextra, 0]
                   t1, t2, t3, t4, t5 = v[0]
                   if not t5: continue            
                   mm = v[1]
                   t5 = [t5]*(t2.shape[0])
                   if data is None:
                       # extra can be none in MPI child nodes
                       # this is called so that we can use MPI
                       # in postprocess_extra in future
                       mm.postprocess_extra(None, t5, ret[key])
                   else:
                       mm.postprocess_extra(data, t5, ret[key])
               k = k + 1

        return ret

    #
    #  save to file
    #
    
    def save_sol_to_file(self, phys_target, skip_mesh = False,
                               mesh_only = False):
        if not skip_mesh:
            mesh_filenames =self.save_mesh()
        if mesh_only: return mesh_filenames
       
        for phys in phys_target:
            mesh_idx = phys.mesh_idx 
            for kfes, r_x, i_x in enum_fes(phys, self.r_x, self.i_x):
                name = self.fespaces[phys][kfes][0]
                self.save_solfile_fespace(name, mesh_idx, r_x, i_x)
     
    def save_extra_to_file(self, sol_extra):
        if sol_extra is None: return
        fid = open('sol_extended.data', 'w')
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
            self.do_assign_sel_index(p, mesh.attributes.Max(), Domain)
            self.do_assign_sel_index(p, mesh.bdr_attributes.Max(), Bdry)
     
    def do_assign_sel_index(self, m, size, cls):
        dprint1("## setting _sel_index (1-based number): "+m.fullname())
        #_sel_index is 0-base array
        def _walk_physics(node):
            yield node
            for k in node.keys():
                yield node[k]
        rem = None
        checklist = [True]*size
        for node in m.walk():
           if not isinstance(node, cls): continue
           if not node.enabled: continue
           ret = node.process_sel_index()
           if ret is None:
              if rem is not None: rem._sel_index = []
              rem = node
           else:
              dprint1(node.fullname(), ret)
              for k in ret:
                 if k-1 >= len(checklist): continue
                 if node.is_secondary_condition: continue
                 checklist[k-1] = False
        if rem is not None:
           rem._sel_index = list(np.where(checklist)[0]+1)
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

    def get_essential_bdr_flag(self, phys):
        flag = []
        for k, pair in enumerate(self.fespaces[phys]):
            index = []
            name, fes = pair
            for node in phys.walk():
                #if not isinstance(node, Bdry): continue           
                if not node.enabled: continue
                if node.has_essential:
                    index = index + node.get_essential_idx(k)

            ess_bdr = [0]*self.meshes[phys.mesh_idx].bdr_attributes.Max()
            for k in index: ess_bdr[k-1] = 1
            flag.append((name, ess_bdr))
        dprint1("esse flag", flag)
        return flag

    def get_essential_bdr_tofs(self, phys, flags):
        ess_tdofs = []
        for flag_pair, fespace_pair in zip(flags, self.fespaces[phys]):
            name = flag_pair[0]
            ess_tdof_list = mfem.intArray()
            ess_bdr = mfem.intArray(flag_pair[1])
            fespace_pair[1].GetEssentialTrueDofs(ess_bdr, ess_tdof_list)
            ess_tdofs.append((name, ess_tdof_list))
        return ess_tdofs

    def allocate_fespace(self, phys):
        #
        self.fespaces[phys] = []
        self.fec[phys] = []
        
        for name, elem in phys.get_fec():
            dprint1("allocate_fespace" + name)
            mesh = self.meshes[phys.mesh_idx]
            fec = getattr(mfem, elem)
            if fec is mfem.ND_FECollection:
                mesh.ReorientTetMesh()
            dim = mesh.Dimension()
            sdim= mesh.SpaceDimension()
            f = fec(phys.order, sdim)
            self.fec[phys].append(f)
            fes = self.new_fespace(mesh, f)
            self.fespaces[phys].append((name, fes))
            
    def get_fes(self, phys, kfes):
        return self.fespaces[phys][kfes][1]

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

    def build_ns(self):
        for node in self.model.walk():
           if node.has_ns():
              try:
                  node.eval_ns()
              except:
                  import traceback
                  traceback.print_exc()
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
     
class SerialEngine(Engine):
    def __init__(self, modelfile='', model=None):
        super(SerialEngine, self).__init__(modelfile = modelfile, model=model)

    def run_mesh(self, meshmodel = None):
        from petram.mesh.mesh_model import MeshFile
        
        self.meshes = []
        if meshmodel is None:
            for idx, g in enumerate(self.model['Mesh'].keys()):
                self.meshes.append(None)
                if not self.model['Mesh'][g].enabled: continue
                target = None
                for k in self.model['Mesh'][g].keys():
                    o = self.model['Mesh'][g][k]
                    if not o.enabled: continue
                    if isinstance(o, MeshFile):
                        self.meshes[idx] = o.run()
                        target = self.meshes[idx]
                    else:
                        if hasattr(o, 'run') and target is not None:
                            self.meshes[idx] = o.run(target)
        
    def run_assemble(self, phys):
        self.is_matrix_distributed = False       
        return super(SerialEngine, self).run_assemble(phys)

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

    def new_fespace(self, mesh, fec):
        return  mfem.FiniteElementSpace(mesh, fec)
     

    def fill_block_matrix_fespace(self, blocks, mv,
                                        gl_ess_tdof, extra, interp,
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

        for k, v in enumerate(extra):
            t1, t2, t3, t4, t5 = v[0]
            dprint2("extra matrix nnz before elimination ",
                    len(t1.nonzero()[0]), " ",  len(t2.nonzero()[0]))

            if isinstance(t2, np.ndarray):
                if P is not None:
                    t2 = P.dot(t2.transpose()).transpose()
                    t1 = PP.dot(t1)
                M[k+1+offset, offset] = t2
                M[offset, k+1+offset] = t1
                M[k+1+offset, k+1+offset] = t3                

            else:
                t1 = t1.tolil(); t2 = t2.tolil()
                if P is not None:
                    for i in  zeros:
                        t1[:, i] = 0.0
                    for i in  zeros:
                        t2[i, :] = 0.0
                    t2 = P.dot(t2.transpose()).transpose().tolil()
                    t1 = PP.dot(t1).tolil()
                M[k+1+offset, offset] = t2
                M[offset, k+1+offset] = t1
                M[k+1+offset, k+1+offset] = t3
                #t4 = np.zeros(t2.shape[0])+t4 (t4 should be vector...)
                t5 = [t5]*(t2.shape[0])

        return 

    def fill_block_rhs_fespace(self, blocks, mv, extra, interp, offset):

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
        size = 1 + len(extra)
        for k, v in enumerate(extra):
            t1, t2, t3, t4, t5 = v[0]
            try:
                void = len(t4)
                t4 = t4
            except:
                print "here", t4, type(t4), t4.shape
                raise ValueError("This is not supported")                
                t4 = np.zeros(t2.shape[0])+t4
            B[k+1+offset] = t4

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


    def collect_all_ess_tdof(self, phys, ess_tdofs):
        self.gl_ess_tdofs[phys] = [(name, ess_tdof.ToList())
                                   for name, ess_tdof in ess_tdofs]

    def save_mesh(self):
        mesh_names = []
        for k, mesh in enumerate(self.meshes):
            if mesh is None: continue
            name = 'solmesh_' + str(k)           
            mesh.PrintToFile(name, 8)
            mesh_names.append(name)
        return mesh_names

    def solfile_name(self, name, mesh_idx,
                     namer = 'solr', namei = 'soli' ):
        fnamer = '_'.join((namer, name, str(mesh_idx)))
        fnamei = '_'.join((namei, name, str(mesh_idx)))
        mesh_name  =  "solmesh_"+str(mesh_idx)              
        return fnamer, fnamei, mesh_name

    def get_true_v_sizes(self, phys):
        fe_sizes = [fes[1].GetTrueVSize() for fes in self.fespaces[phys]]
        dprint1('Number of finite element unknowns: '+  str(fe_sizes))
        return fe_sizes

    def split_sol_array_fespace(self, sol, P):
        sol0 = sol[0, 0]
        if P is not None:
           sol0 = P.transpose().dot(sol0)
        return sol0, sol[1:, 0]

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

class ParallelEngine(Engine):
    def __init__(self, modelfile='', model=None):
        super(ParallelEngine, self).__init__(modelfile = modelfile, model=model)


    def run_mesh(self, meshmodel = None):
        from mpi4py import MPI
        from petram.mesh.mesh_model import MeshFile        
        self.meshes = []
        if meshmodel is None:
            for idx, g in enumerate(self.model['Mesh'].keys()):
                self.meshes.append(None)
                if not self.model['Mesh'][g].enabled: continue
                for k in self.model['Mesh'][g].keys():
                    o = self.model['Mesh'][g][k]
                    if not o.enabled: continue   
                    if isinstance(o, MeshFile):
                        smesh = o.run()
                        self.meshes[idx] = mfem.ParMesh(MPI.COMM_WORLD, smesh)
                        target = self.meshes[idx]
                    else:
                        if hasattr(o, 'run'): self.meshes[idx] = o.run(target)

    def run_assemble(self, phys):
        self.is_matrix_distributed = True       
        return super(ParallelEngine, self).run_assemble(phys)
     
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
               
    def new_fespace(self,mesh, fec):
        return  mfem.ParFiniteElementSpace(mesh, fec)
 
    def new_matrix(self, init = True):
        return  mfem.HypreParMatrix()

    def new_blockmatrix(self, shape):
        from petram.helper.block_matrix import BlockMatrix
        return BlockMatrix(shape, kind = 'hypre')

    def get_true_v_sizes(self, phys):
        fe_sizes = [fes[1].GlobalTrueVSize() for fes in self.fespaces[phys]]
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
        for k, mesh in enumerate(self.meshes):
            if mesh is None: continue
            mesh_name  =  "solmesh_"+str(k)+"."+smyid
            mesh.PrintToFile(mesh_name, 8)
            mesh_names.append(mesh_name)
        return mesh_names
     
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

    '''
    def save_solfile_fespace(self, name, mesh_idx, r_x, i_x, 
                             namer = 'solr', namei = 'soli' ):
        from mpi4py import MPI                               
        num_proc = MPI.COMM_WORLD.size
        myid     = MPI.COMM_WORLD.rank
        smyid = '{:0>6d}'.format(myid)

        fname = '_'.join((namer, name, str(mesh_idx)))+"."+smyid
        r_x.SaveToFile(fname, 8)
        if i_x is not None:
            fname = '_'.join((namei, name, str(mesh_idx)))+"."+smyid
            i_x.SaveToFile(fname, 8)
    '''

    def fill_block_matrix_fespace(self, blocks, mv,
                                        gl_ess_tdof, extra, interp,
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
           A1 = A1.rap(P.transpose())
           A1.setDiag(zeros, 1.0) # comment this when making final matrix smaller

        M[offset, offset] = A1


                                      
        for k, v in enumerate(extra):
            t1, t2, t3, t4, t5 = v[0]
            if isinstance(t1, chypre.CHypreMat):
                dprint1("t1, t2, shape", t1.shape, t2.shape)
                if P is not  None:
                    t1 = P.conj().dot(t1); P.conj()
                    t2 = t2.dot(P.transpose())
                #                      
                #t1.resetRow(ess_tdof_list)
                #t2.resetCol(ess_tdof_list)
                #t1.resetRow(zeros)
                #t2.resetCol(zeros)
                '''
                elim = t3 is None
                t1 = Mat2CooV(t1, is_complex, elimination = elim)
                t2 = Mat2CooH(t2, is_complex, elimination = elim)               
                '''
                #dprint1("t1, t2, shape, shrunk", (t1.M(), t1.N()),(t2.M(), t2.N()))
                '''
                for x in ess_tdof_list:
                   t1.set_row(0.0, 0.0, int(x))
                   t2.set_col(0.0, 0.0, int(x))
                for x in zeros:
                   t1.set_row(0.0, 0.0, int(x))
                   t2.set_col(0.0, 0.0, int(x))
                '''
                   
            elif isinstance(t1, chypre.CHypreVec): # 1D array
                if P is not  None:
                    t1 = P.conj().dot(t1); P.conj()
                    t2 = P.dot(t2)  # = t2*P^t should check if t2 is not hirizontal
                # this should be taken care in finalization                      
                #for x in ess_tdof_list:
                #    t1.set_element(x, 0.0)
                #from petram.helper.chypre_to_pymatrix import Vec2MatH, Vec2MatV
                #t1 = Vec2MatV(t1, is_complex)
                #t2 = Vec2MatH(t2, is_complex)                
            else:
                pass

            M[k+1+offset,offset] = t2
            M[offset, k+1+offset] = t1
            M[k+1+offset, k+1+offset] = t3

    def fill_block_rhs_fespace(self, blocks, mv, extra, interp, offset):
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

        for k, v in enumerate(extra):
            t1, t2, t3, t4, t5 = v[0]
            try:
               void = len(t4)
               t4 = t4
            except:
               raise ValueError("This is not supported")
               t4 = np.zeros(t2.M())+t4
            B[k+1+offset] = t4

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


    def split_sol_array_fespace(self, sol, P):
        sol0 = sol[0, 0]
        if P is not None:
           sol0 = (P.transpose()).dot(sol0)
        return sol0, sol[1:, 0]

    def collect_all_ess_tdof(self, phys, ess_tdofs, M = None):
        from mpi4py import MPI

        gl_ess_tdofs = []
        for tdof_pair, fes_pair in zip(ess_tdofs, self.fespaces[phys]):
            tdof = tdof_pair[1]
            name = tdof_pair[0]
            fes  = fes_pair[1]
            data = (np.array(tdof.ToList()) +
                    fes.GetMyTDofOffset()).astype(np.int32)
   
            gl_ess_tdof = allgather_vector(data, MPI.INT)
            MPI.COMM_WORLD.Barrier()
            gl_ess_tdofs.append((name, gl_ess_tdof))
        self.gl_ess_tdofs[phys] = gl_ess_tdofs
     
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
        
  
