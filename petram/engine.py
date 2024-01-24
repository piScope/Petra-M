from __future__ import print_function
from petram.helper.matrix_file import write_coo_matrix, write_vector

import sys
import os
import six
import shutil
import traceback
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

# these are only for debuging
from mfem.common.parcsr_extra import ToScipyCoo
from mfem.common.mpi_debug import nicePrint

from petram.model import Domain, Bdry, Point, Pair

import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('Engine')

# if you need to turn a specific warning to exception
#import scipy.sparse
#import warnings
#warnings.filterwarnings('error', category=scipy.sparse.SparseEfficiencyWarning)


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

# Number of matrices to handle.
#   must be even number.
#    First half is for time-dependent
#    Second half is for gradient


max_matrix_num = 10


class Engine(object):
    max_levels = 10

    def __init__(self, modelfile='', model=None):
        if modelfile != '':
            import petram.helper.pickle_wrapper as pickle
            model = pickle.load(open(modelfile, 'rb'))

        from collections import deque

        self.data_stack = deque()
        self.set_model(model)

        # mulitlple matrices in equations.
        self._num_matrix = 1
        self._access_idx = -1

        # multi-level assembly
        self._num_levels = 1
        self._level_idx = 0

        self._dep_vars = []
        self._isFESvar = []
        self._rdep_vars = []
        self._rdep_var_grouped = []
        self._risFESvar = []
        self._risFESvar_grouped = []
        self._rfes_vars = []
        self._aux_essential = []

        self.max_bdrattr = -1
        self.max_attr = -1
        self.sol_extra = None
        self.sol = None

        self._r_x_old = {}
        self._i_x_old = {}
        self._assembled_blocks = [[None]*7, ]

        # place holder : key is base physics modules, such as EM3D1...
        #
        # for example : self.r_b['EM3D1'] = [LF for E, LF for psi]
        #               physics moduel provides a map form variable name to index.

        self.case_base = 0
        self._init_done = []

        self._ppname_postfix = ''

    def initialize_datastorage(self):
        self.is_assembled = False
        self.is_initialized = False
        self.meshes = []
        self.emeshes = []
        self.emesh_data = None
        # place holder max level 10
        #self._fec_storage = {}
        #self._fes_storage = {}
        self.pp_extra = {}
        self.alloc_flag = {}
        self.initialize_fespaces()

    def initialize_fespaces(self):
        from petram.helper.hierarchical_finite_element_spaces import HierarchicalFiniteElementSpaces
        self._fespace_hierarchy = HierarchicalFiniteElementSpaces(owner=self)

    stored_data_names = ("is_assembled",
                         "self.is_initialized",
                         "meshes",
                         "emeshes",
                         "emesh_data",
                         "_fespaces",
                         "_fes_storage",
                         "_fec_storage",
                         "pp_extra",
                         "alloc_flag")

    def push_datastorage(self):
        v = self.model.root()._variables
        p = self.model.root()._parameters
        data = [getattr(self, n) for n in stored_data_names]
        dataset = (v, p, data)

        self.data_stack.append(dataset)

    def access_datastorage(self, idx):
        data = self.data_stack.index(idx)
        return data

    def pop_datastorage(self):
        data = self.data_stack.pop()
        self.model.root()._variables = data[0]
        self.model.root()._parameters = data[1]
        for k, n in enumerate(stored_data_names):
            setattr(self, n, data[2][k])

    #
    #  active_matrix
    #
    def is_matrix_active(self, k_matrix):
        return self._active_matrix[k_matrix]

    def set_active_matrix(self, active_matrix):
        self._active_matrix = active_matrix

    def activate_matrix(self, k):
        self._active_matrix[k] = True

    def deactivate_matrix(self, k):
        self._active_matrix[k] = False

    def iter_active_matrix(self):
        d = self._active_matrix
        return [i*d[i] for i in range(len(d)) if d[i]]
    #
    #  formblocks
    #

    def set_formblocks(self, phys_target, phys_range, n_matrix):
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

        self.collect_dependent_vars(phys_target, phys_range)
        self.collect_dependent_vars(phys_range,  phys_range, range_space=True)

        n_fes = len(self.fes_vars)
        n_rfes = len(self.r_fes_vars)
        diag = [self.r_fes_vars.index(n) for n in self.fes_vars]
        n_mat = self.n_matrix

        # assembled block [A, X, RHS, Ae,  B,  M, self.dep_vars[:]]
        if self.level_idx == 0:
            self._initialize_variable_lists()
            self.n_levels = 0
            self._assembled_blocks = [[None]*7, ]
        else:
            self.n_levels = self.n_levels + 1
            self._assembled_blocks.append([None]*7)

        self._r_a.append([FormBlock((n_fes, n_rfes), diag=diag,
                                    new=self.alloc_bf, mixed_new=self.alloc_mbf)
                          for k in range(n_mat)])
        self._i_a.append([FormBlock((n_fes, n_rfes), diag=diag,
                                    new=self.alloc_bf, mixed_new=self.alloc_mbf)
                          for k in range(n_mat)])
        self._r_at.append([FormBlock((n_fes, n_rfes), diag=diag,
                                     new=self.alloc_bf, mixed_new=self.alloc_mbf)
                           for k in range(n_mat)])
        self._i_at.append([FormBlock((n_fes, n_rfes), diag=diag,
                                     new=self.alloc_bf, mixed_new=self.alloc_mbf)
                           for k in range(n_mat)])
        self._r_x.append([FormBlock(n_rfes, new=self.alloc_gf)
                          for k in range(n_mat)])
        self._i_x.append([FormBlock(n_rfes, new=self.alloc_gf)
                          for k in range(n_mat)])
        self._r_b.append(FormBlock(n_fes, new=self.alloc_lf))
        self._i_b.append(FormBlock(n_fes, new=self.alloc_lf))

        self._extras.append([None for i in range(n_mat)])
        self._aux_ops.append([None for i in range(n_mat)])

        self._interps.append({})
        self._projections.append({})
        self._projections_hash.append({})
        self._gl_ess_tdofs.append({n: [] for n in self.fes_vars})
        self._ess_tdofs.append({n: [] for n in self.fes_vars})

    def _initialize_variable_lists(self):
        self._r_a = []
        self._i_a = []
        self._r_at = []
        self._i_at = []
        self._r_b = []
        self._i_b = []
        self._r_x = []
        self._i_x = []
        self._extras = []
        self._aux_ops = []
        self._interps = []
        self._projections = []
        self._projections_hash = []
        self._gl_ess_tdofs = []
        self._ess_tdofs = []

    @property
    def n_matrix(self):
        return self._num_matrix

    @n_matrix.setter
    def n_matrix(self, i):
        self._num_matrix = i

    @property
    def n_levels(self):
        return self._num_levels

    @n_levels.setter
    def n_levels(self, i):
        self._num_levels = i

    @property
    def access_idx(self):
        return self._access_idx

    @access_idx.setter
    def access_idx(self, i):
        self._access_idx = i

    @property
    def level_idx(self):
        return self._level_idx

    @level_idx.setter
    def level_idx(self, i):
        self._level_idx = i

    # bilinearforms (real)
    @property
    def r_a(self):
        return self._r_a[self._level_idx][self._access_idx]

    @r_a.setter
    def r_a(self, v):
        self._r_a[self._level_idx][self._access_idx] = v
    # bilinearforms (imag)

    @property
    def i_a(self):
        return self._i_a[self._level_idx][self._access_idx]

    @i_a.setter
    def i_a(self, v):
        self._i_a[self._level_idx][self._access_idx] = v

    # bilinearforms transposed contribution (real)
    @property
    def r_at(self):
        return self._r_at[self._level_idx][self._access_idx]

    @r_at.setter
    def r_at(self, v):
        self._r_at[self._level_idx][self._access_idx] = v
    # bilinearforms transposed contribution (imag)

    @property
    def i_at(self):
        return self._i_at[self._level_idx][self._access_idx]

    @i_at.setter
    def i_at(self, v):
        self._i_at[self._level_idx][self._access_idx] = v

    # grid functions (real)
    @property
    def r_x(self):
        return self._r_x[self._level_idx][self._access_idx]

    @r_x.setter
    def r_x(self, v):
        self._r_x[self._level_idx][self._access_idx] = v

    # grid functions (imag)
    @property
    def i_x(self):
        return self._i_x[self._level_idx][self._access_idx]

    @i_x.setter
    def i_x(self, v):
        self._i_x[self._level_idx][self._access_idx] = v

    # rhs functions (real)
    @property
    def r_b(self):
        return self._r_b[self._level_idx]

    @r_b.setter
    def r_b(self, v):
        self._r_b[self._level_idx] = v

    # rhs functions (imag)
    @property
    def i_b(self):
        return self._i_b[self._level_idx]

    @i_b.setter
    def i_b(self, v):
        self._i_b[self._level_idx] = v

    @property
    def extras(self):
        return self._extras[self._level_idx][self._access_idx]

    @extras.setter
    def extras(self, v):
        self._extras[self._level_idx][self._access_idx] = v

    @property
    def aux_ops(self):
        return self._aux_ops[self._level_idx][self._access_idx]

    @aux_ops.setter
    def aux_ops(self, v):
        self._aux_ops[self._level_idx][self._access_idx] = v

    @property
    def interps(self):
        return self._interps[self._level_idx]

    @interps.setter
    def interps(self, v):
        self._interps[self._level_idx] = v

    @property
    def projections(self):
        return self._projections[self._level_idx]

    @projections.setter
    def projections(self, v):
        self._projections[self._level_idx] = v

    @property
    def projections_hash(self):
        return self._projections_hash[self._level_idx]

    @projections_hash.setter
    def projections_hash(self, v):
        self._projections_hash[self._level_idx] = v

    @property
    def gl_ess_tdofs(self):
        return self._gl_ess_tdofs[self._level_idx]

    @gl_ess_tdofs.setter
    def gl_ess_tdofs(self, v):
        self._gl_ess_tdofs[self._level_idx] = v

    @property
    def ess_tdofs(self):
        return self._ess_tdofs[self._level_idx]

    @ess_tdofs.setter
    def ess_tdofs(self, v):
        self._ess_tdofs[self._level_idx] = v

    @property
    def fespaces(self):
        return self._fespace_hierarchy

    @property
    def assembled_blocks(self):
        return self._assembled_blocks[self._level_idx]

    @assembled_blocks.setter
    def assembled_blocks(self, v):
        self._assembled_blocks[self._level_idx] = v

    '''
    @property
    def matvecs(self):
        return self._matrix_block[self._access_idx]

    @matvecs.setter
    def matvecs(self, v):
        self._matrix_block[self._access_idx] = v
    '''

    def store_x(self):
        for k, name in enumerate(self.r_fes_vars):
            self._r_x_old[name] = self._r_x[0][0][k]
            self._i_x_old[name] = self._i_x[0][0][k]

    def set_model(self, model):
        self.model = model
        self.initialize_datastorage()
        if model is None:
            return

        self.alloc_flag = {}
        # below is to support old version
        from petram.mesh.mesh_model import MeshGroup
        g = None
        items = []

        for k in model['Mesh'].keys():
            if not hasattr(model['Mesh'][k], 'isMeshGroup'):
                if g is None:
                    name = model['Mesh'].add_item('MFEMMesh', MeshGroup)
                    g = model['Mesh'][name]
                items.append((k, model['Mesh'][k]))

        for name, obj in items:
            del model['Mesh'][name]
            model['Mesh']['MFEMMesh1'][name] = obj

        # convert old model which does not use SolveStep...
        from petram.solver.solver_model import SolveStep
        from petram.solver.solver_controls import SolveControl

        solk = list(model['Solver'])
        flag = any([not isinstance(model['Solver'][k], SolveStep)
                    for k in solk])
        if flag:
            box = None
            for k in solk:
                if not (isinstance(model['Solver'][k], SolveStep) or
                        isinstance(model['Solver'][k], SolveControl)):
                    if box is None:
                        name = model['Solver'].add_item('SolveStep', SolveStep)
                        box = model['Solver'][name]
                    obj = model['Solver'][k]
                    del model['Solver'][k]
                    box.add_itemobj(k, obj, nosuffix=True)
                else:
                    box = None

        if not 'InitialValue' in model:
            idx = list(model).index('Phys')+1
            from petram.mfem_model import MFEM_InitRoot
            model.insert_item(idx, 'InitialValue', MFEM_InitRoot())
        if not 'PostProcess' in model:
            idx = list(model).index('InitialValue')+1
            from petram.mfem_model import MFEM_PostProcessRoot
            model.insert_item(idx, 'PostProcess', MFEM_PostProcessRoot())

        from petram.mfem_model import has_geom
        if not 'Geom' in model and has_geom:
            try:
                from petram.geom.geom_model import MFEM_GeomRoot
            except:
                from petram.mfem_model import MFEM_GeomRoot
            model.insert_item(1, 'Geometry', MFEM_GeomRoot())

        #
        #  set debug parameters to mfem_config
        #
        import petram.mfem_config
        if model["General"].debug_numba_jit == 'on':
            petram.mfem_config.numba_debug = True
        else:
            petram.mfem_config.numba_debug = False

        dprint1("petram.mfem_config.numba_debug",
                petram.mfem_config.numba_debug)
        petram.mfem_config.allow_python_function_coefficient = model[
            "General"].allow_fallback_nonjit

    def get_mesh(self, idx=0, mm=None):
        if len(self.meshes) == 0:
            return None
        if mm is not None:
            idx = mm.get_root_phys().mesh_idx
        return self.meshes[idx]

    def get_smesh(self, idx=0, mm=None):
        if len(self.smeshes) == 0:
            return None
        if mm is not None:
            idx = mm.get_root_phys().mesh_idx
        return self.smeshes[idx]

    def get_emesh(self, idx=0, mm=None):
        if len(self.emeshes) == 0:
            return None
        if mm is not None:
            idx = mm.get_root_phys().emesh_idx
        return self.emeshes[idx]

    def get_emesh_idx(self, mm=None, name=None):
        if len(self.emeshes) == 0:
            return -1
        if mm is not None:
            return mm.get_root_phys().emesh_idx

        if name is None:
            for item in self.model['Phys']:
                mm = self.model['Phys'][item]
                if not mm.enabled:
                    continue
                if name in mm.dep_vars():
                    return mm.emesh_idx
        return -1

    def preprocess_modeldata(self, dir=None):
        '''
        do everything it takes to run a newly built
        model data strucutre.
        name space is already build
        used from text script execution
        '''
        import os

        model = self.model
        self.initialize_datastorage()

        model['General'].run()

        from petram.mfem_config import use_parallel

        if use_parallel:
            self.run_mesh_serial(skip_refine=True)
        else:
            self.run_mesh_serial()

        self.run_preprocess()  # this must run when mesh is serial

        if use_parallel:
            self.initialize_fespaces()
            self.run_mesh()
            self.emeshes = []
            for k in self.model['Phys'].keys():
                phys = self.model['Phys'][k]
                if not phys.enabled:
                    continue
                self.run_mesh_extension(phys)
                self.allocate_fespace(phys)

        self.save_processed_model()

        solver = model["Solver"].get_active_solvers()
        return solver

    def run_build_ns(self, dir=None):
        model = self.model
        model['General'].run()

        if dir is None:
            from __main__ import __file__ as mainfile
            dir = os.path.dirname(os.path.realpath(mainfile))

        for node in model.walk():
            if node.has_ns() and node.ns_name is not None:
                node.read_ns_script_data(dir=dir)
        self.build_ns()
        solver = model["Solver"].get_active_solvers()

        dprint1("solver", solver)
        return solver

    def run_config(self):
        self.model['General'].run()
        try:
            self.build_ns()
        except:
            exception = traceback.format_exc()
            return -2, exception
        return 0, None
        '''
        import traceback
        try:        
            self.model['General'].run()
            self.run_mesh_serial(skip_refine=skip_refine)
        except:
            exception = traceback.format_exc()
            return -1, exception
        try:
            self.build_ns()
            self.assign_phys_pp_sel_index()                    
            self.run_mesh_extension_prep()        
            self.assign_sel_index()
        except:
            exception = traceback.format_exc()
            return -2, exception
        return 0, None
        '''

    def run_preprocess(self, ns_folder=None, data_folder=None):
        dprint1("!!!!! run preprocess !!!!!")
        if ns_folder is not None:
            self.preprocess_ns(ns_folder, data_folder)

        from petram.model import Domain, Bdry

        self.assign_sel_index()
        self.assign_phys_pp_sel_index()
        self.run_mesh_extension_prep()

        for k in self.model['Phys'].keys():
            phys = self.model['Phys'][k]
            if not phys.enabled:
                continue

            self.run_mesh_extension(phys)
            self.allocate_fespace(phys)
            # this is called already from preprocess_modeldata
            #
            for node in phys.walk():
                if not node.enabled:
                    continue
                node.preprocess_params(self)

        for k in self.model['InitialValue'].keys():
            init = self.model['InitialValue'][k]
            init.preprocess_params(self)

    def run_verify_setting(self, phys_target=None, solver=None):
        if phys_target is None and solver is None:
            top = self.model
            for mm in top.walk_enabled():
                if hasattr(mm, 'verify_setting'):
                    error, txt, long_txt = mm.verify_setting()
                assert error, mm.fullname() + ":" + long_txt
            return

        for phys in phys_target:
            for mm in phys.walk():
                if not mm.enabled:
                    continue
                error, txt, long_txt = mm.verify_setting()
                assert error, mm.fullname() + ":" + long_txt

        for mm in solver.walk():
            if not mm.enabled:
                continue
            error, txt, long_txt = mm.verify_setting()
            assert error, mm.fullname() + ":" + long_txt

    #  mesh manipulation
    #

    def run_mesh_extension_prep(self, reset=False):
        if reset:
            self.reset_emesh_data()

        for k in self.model['Phys'].keys():
            phys = self.model['Phys'][k]
            # (we do this even disabled physics, until different problem develops)
            # if not phys.enabled:
            #     continue
            self.do_run_mesh_extension_prep(phys)

    def do_run_mesh_extension_prep(self, phys):
        from petram.mesh.mesh_model import MFEMMesh

        if len(self.emeshes) == 0:
            self.emeshes = self.meshes[:]
            for j in range(len(self.emeshes)):
                attrs = np.unique(self.emeshes[j].GetAttributeArray())
                self.emesh_data.add_default_info(j,
                                                 self.emeshes[j].Dimension(),
                                                 attrs)
        info = phys.get_mesh_ext_info(self.meshes[phys.mesh_idx])
        if info is not None:
            idx = self.emesh_data.add_info(info)
            phys.emesh_idx = idx
        elif phys.enabled:
            assert False, "failed to run mesh extension. check selection of " + \
                str(phys)
        else:
            phys.emesh_idx = 0   # this is when
        dprint1(phys.name() + ":  emesh index =", phys.emesh_idx)

    def run_mesh_extension(self, phys):

        import petram.mesh.partial_mesh

        p_method = self.get_submesh_partitiong_method()
        if p_method == "auto":
            petram.mesh.partial_mesh.partition_method = "default"
        else:
            petram.mesh.partial_mesh.partition_method = "0"

        from petram.mesh.mesh_extension import MeshExt, generate_emesh
        from petram.mesh.mesh_model import MFEMMesh

        if len(self.emeshes) == 0:
            self.emeshes = self.meshes[:]

        idx = phys.emesh_idx
        info = self.emesh_data.get_info(idx)

        if len(self.emeshes) <= idx or self.emeshes[idx] is None:
            m = generate_emesh(self.emeshes, info)
            # 2021 Nov.
            # m.ReorientTetMesh()
            if len(self.emeshes) <= idx:
                self.emeshes.extend([None]*(1+idx-len(self.emeshes)))
            self.emeshes[idx] = m
        dprint1(phys.name() + ":  emesh index =", idx)

    #
    #  assembly
    #
    def run_alloc_sol(self, phys_target=None):
        '''
        allocate fespace and gridfunction (unknowns)
        apply essentials
        define model variables

        alloc_flag is used to avoid repeated allocation.
        '''

        allocated_phys = []
        for phys in phys_target:
            try:
                if self.alloc_flag[phys.name()]:
                    alloced_phys.append[phys.name()]
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

        self.is_initialized = True

        for phys in phys_target:
            self.alloc_flag[phys.name()] = True

    @property
    def isInitialized(self):
        return self.is_initialized

    def run_apply_init0(self, phys_range, mode,
                        init_value=0.0, init_path='', init_dwc=("", ""), init_var=""):
        # mode
        #  0: zero
        #  1: init to constant
        #  2: use init panel values
        #  3: load file
        #  4: do nothing
        dprint1("run_apply_init0", phys_range, mode)
        for j in range(self.n_matrix):
            self.access_idx = j
            if not self.is_matrix_active(j):
                continue

            if mode in [0, 1, 2, 3, 4]:
                for phys in phys_range:
                    names = phys.dep_vars
                    if mode == 0:
                        for name in names:
                            if init_var != "" and name != init_var:
                                continue

                            r_ifes = self.r_ifes(name)
                            rgf = self.r_x[r_ifes]
                            igf = self.i_x[r_ifes]
                            rgf.Assign(0.0)
                            if igf is not None:
                                igf.Assign(0.0)
                            if not name in self._init_done:
                                self._init_done.append(name)
                    elif mode == 1:
                        from petram.helper.variables import project_variable_to_gf

                        global_ns = phys._global_ns.copy()
                        for key in self.model.root()._variables:
                            global_ns[key] = self.model.root()._variables[key]
                        for name in names:
                            if init_var != "" and name != init_var:
                                continue
                            r_ifes = self.r_ifes(name)
                            rgf = self.r_x[r_ifes]
                            igf = self.i_x[r_ifes]
                            ind_vars = phys.ind_vars
                            dprint1(
                                "applying init value to entire discrete space:" + name, init_value)
                            project_variable_to_gf(init_value,
                                                   ind_vars,
                                                   rgf, igf,
                                                   global_ns=global_ns)

                            # rgf.ProjectCoefficient(rc)
                            # rgf.Assign(rinit)
                            # if igf is not None:
                            #   igf.ProjectCoefficient(ic)
                            if not name in self._init_done:
                                self._init_done.append(name)
                    elif mode == 2:  # apply Einit
                        self.apply_init_from_init_panel(phys)
                    elif mode == 3:
                        self.apply_init_from_file(phys, init_path)
                    elif mode == 4:
                        self.apply_init_from_previous(names)
            elif mode == 5:
                names = []
                for phys in phys_range:
                    names.extend(phys.dep_vars)
                self.apply_init_by_dwc(names, init_dwc)
            else:
                raise NotImplementedError("unknown init mode")

        self.add_FESvariable_to_NS(phys_range, verbose=True)

    def run_apply_init_autozero(self, phys_range):

        # mode
        #  0: zero
        #  1: init to constant
        #  2: use init panel values
        #  3: load file
        #  4: do nothing
        dprint1("run_apply_init_autozero", phys_range)
        for j in range(self.n_matrix):
            self.access_idx = j
            if not self.is_matrix_active(j):
                continue

            for phys in phys_range:
                names = phys.dep_vars
                for name in names:
                    if name in self._init_done:
                        continue
                    self._init_done.append(name)
                    r_ifes = self.r_ifes(name)
                    rgf = self.r_x[r_ifes]
                    igf = self.i_x[r_ifes]
                    rgf.Assign(0.0)
                    if igf is not None:
                        igf.Assign(0.0)

    def run_apply_init(self, phys_range, inits=None):
        if len(inits) == 0:
            self.run_apply_init_autozero(phys_range)
        else:
            for init in inits:
                tmp = init.run(self)
                phys_range = [phys for phys in phys_range if not phys in tmp]
            if len(phys_range) > 0:
                dprint1(
                    "!!!!! These phys are not initiazliaed (FES variable is not available)!!!!!", phys_range)

    def run_apply_essential(self, phys_target, phys_range, update=False):
        L = len(self.r_dep_vars)
        self.mask_X = np.array([not update]*L*self.n_matrix,
                               dtype=bool).reshape(-1, L)

        self.gl_ess_tdofs = {n: ([], []) for n in self.fes_vars}
        self.ess_tdofs = {n: ([], []) for n in self.fes_vars}

        for phys in phys_range:
            self.gather_essential_tdof(phys)
        self.collect_all_ess_tdof()

        for j in range(self.n_matrix):
            self.access_idx = j
            if not self.is_matrix_active(j):
                continue

            for phys in phys_target:
                self.apply_essential(phys, update=update)

    def run_assemble_mat(self, phys_target, phys_range, update=False):
        # for phys in phys_target:
        #    self.gather_essential_tdof(phys)

        R = len(self.dep_vars)
        C = len(self.r_dep_vars)
        self.mask_M = np.array([not update]*R*C*self.n_matrix,
                               dtype=bool).reshape(-1, R, C)

        for phys in phys_target:
            self.assemble_interp(phys)  # global interpolation (periodic BC)
            # global interpolation (mesh coupling)
            self.assemble_projection(phys)

        self.extras_mm = {}

        for j in range(self.n_matrix):
            self.access_idx = j
            for phys in phys_target:
                for mm in phys.walk_enabled():
                    mm.compile_coeffs()

        for j in range(self.n_matrix):
            self.access_idx = j
            if not self.is_matrix_active(j):
                continue

            for phys in phys_target:
                self.fill_bf(phys, update)
                self.fill_mixed(phys, update)

            self.r_a.set_no_allocator()
            self.i_a.set_no_allocator()
            self.r_at.set_no_allocator()
            self.i_at.set_no_allocator()

            rcforms = ([(r, c, form) for r, c, form in self.r_a] +
                       [(r, c, form) for r, c, form in self.i_a] +
                       [(r, c, form) for r, c, form in self.r_at] +
                       [(r, c, form) for r, c, form in self.i_at])

            for r, c, form in rcforms:
                r1 = self.dep_var_offset(self.fes_vars[r])
                c1 = self.r_dep_var_offset(self.r_fes_vars[c])
                if self.mask_M[j, r1, c1]:
                    try:
                        form.Assemble(0)
                    except BaseException:
                        print("failed to assemble (r, c) = ", r1, c1)
                        raise

            self.extras = {}
            updated_extra = []
            for phys in phys_target:
                keys_to_update = self.extra_update_check_M(phys, phys_range)

                if update:
                    self.assemble_extra(phys, phys_range, keys_to_update)
                else:
                    self.assemble_extra(phys, phys_range, None)

                updated_extra.extend(keys_to_update)

            for extra_name, dep_name, kfes in updated_extra:
                r = self.dep_var_offset(extra_name)
                c = self.r_dep_var_offset(dep_name)
                self.mask_M[j, r, c] = True

            self.aux_ops = {}
            updated_extra = []
            # for phys in phys_target:
            updated_aux_ops = self.assemble_aux_ops(
                phys_target, phys_range, update)
            for key in updated_aux_ops:
                testname, trialname, mm_fullpath = key
                r = self.dep_var_offset(testname)
                c = self.r_dep_var_offset(trialname)
                self.mask_M[j, r, c] = True

        return np.any(self.mask_M) or len(updated_extra) > 0

    def run_assemble_extra_rhs(self, phys_target, phys_range, update=False):
        # assemble extra only
        #    this is used when filling RHS only
        #    (TODO) split assemble_extra to assemble_extra_matrix and
        #           assemble_extra_rhs

        R = len(self.dep_vars)
        C = len(self.r_dep_vars)
        self.mask_M = np.array([not update]*R*C*self.n_matrix,
                               dtype=bool).reshape(-1, R, C)

        for phys in phys_target:
            self.assemble_interp(phys)  # global interpolation (periodic BC)
            # global interpolation (mesh coupling)
            self.assemble_projection(phys)

        #self.extras_mm = {}

        for j in range(self.n_matrix):
            self.access_idx = j
            if not self.is_matrix_active(j):
                continue

            # self.extras = {} (we keep old extra and update only where it is needed)
            updated_extra = []
            for phys in phys_target:
                # dprint1("checking extra_updat for B",
                #        self.extra_update_check_B(phys, phys_target))
                keys_to_update = self.extra_update_check_B(phys, phys_target)

                self.assemble_extra(phys, phys_range, keys_to_update)
                updated_extra.extend(
                    self.extra_update_check_M(phys, phys_range))

            for extra_name, dep_name, kfes in updated_extra:
                r = self.dep_var_offset(extra_name)
                c = self.r_dep_var_offset(dep_name)
                self.mask_M[j, r, c] = True

        return np.any(self.mask_M) or len(updated_extra) > 0

    def run_assemble_b(self, phys_target=None, update=False):
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
        L = len(self.dep_vars)
        self.mask_B = np.array([not update]*L)

        for phys in phys_target:
            self.run_update_param(phys)

        # for phys in phys_target:
        #    self.gather_essential_tdof(phys)
        # self.collect_all_ess_tdof()

        self.access_idx = 0
        for phys in phys_target:
            self.fill_lf(phys, update)

        self.r_b.set_no_allocator()
        self.i_b.set_no_allocator()

        for r, c, form in self.r_b:
            name = self.fes_vars[r]
            offset = self.dep_var_offset(name)
            if self.mask_B[offset]:
                form.Assemble()

        for r, c, form in self.i_b:
            name = self.fes_vars[r]
            offset = self.dep_var_offset(name)
            if self.mask_B[offset]:
                form.Assemble()

        updated_extra = []
        for phys in phys_target:
            updated_extra.extend(self.extra_update_check_B(phys, phys_target))

        #  (extra_name, dep_var, kfes)
        for n, dep_var, kfes in updated_extra:
            r = self.dep_var_offset(n)
            self.mask_B[r] = True

        return np.any(self.mask_B)

    def run_fill_X_block(self, update=False):
        if update:
            X = self.assembled_blocks[1]
        else:
            X = self.prepare_X_block()

        X = self.fill_X_block(X)
        self.assembled_blocks[1] = X

    def run_assemble_blocks(self, compute_A, compute_rhs,
                            inplace=True, update=False):
        '''
        assemble M, B, X blockmatrices.

        in parallel, inplace = False makes sure that blocks in A and RHS  
        are not shared by M, B, X

        daigpolicy = 0  # DiagOne
        daigpolicy = 1  # DiagKeep
        '''
        if update:
            M = self.assembled_blocks[5]
            B = self.assembled_blocks[4]
            X = self.assembled_blocks[1]
            Ae = self.assembled_blocks[3]
            A = self.assembled_blocks[0]
        else:
            M, B = self.prepare_M_B_blocks()
            X = self.assembled_blocks[1]
            Ae = None

        M, B, M_changed = self.fill_M_B_blocks(M, B, update=update)

        # B.save_to_file("B")
        # M[0].save_to_file("M0")
        # M[1].save_to_file("M1")
        # X[0].save_to_file("X0")
        A2, isAnew = compute_A(M, B, X,
                               self.mask_M,
                               self.mask_B)  # solver determins A

        if isAnew:
            # generate Ae and eliminated A
            A, Ae = self.fill_BCeliminate_matrix(A2, B,
                                                 inplace=inplace,
                                                 update=update)

        RHS = compute_rhs(M, B, X)          # solver determins RHS
        RHS = self.eliminateBC(Ae, X[0], RHS)  # modify RHS and

        # A and RHS is modifedy by global DoF coupling P
        A, RHS = self.apply_interp(A, RHS)

        # RHS.save_to_file("RHS")
        # M[0].save_to_file("M0there2")
        # M[1].save_to_file("M1")
        # X[0].save_to_file("X0")
        # RHS.save_to_file("RHS")

        # = [A, X, RHS, Ae,  B,  M, self.dep_vars[:]]
        self.assembled_blocks = [A, X, RHS, Ae,  B,  M, self.dep_vars[:]]

        return self.assembled_blocks, M_changed

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
        dprint1("run update_param : ", phys)
        for mm in phys.walk():
            if not mm.enabled:
                continue
            mm.update_param()

    def initialize_phys(self, phys):
        is_complex = phys.is_complex()

        # this is called from preprocess_modeldata
        # self.assign_sel_index(phys)

        self.allocate_fespace(phys)
        true_v_sizes = self.get_true_v_sizes(phys)

        flags = self.get_essential_bdr_pnt_flag(phys)
        self.get_essential_bdr_pnt_tdofs(phys, flags)

        # this loop alloates GridFunctions
        for j in range(self.n_matrix):
            self.access_idx = j
            if not self.is_matrix_active(j):
                continue

            is_complex = phys.is_complex()
            for n in phys.dep_vars:
                r_ifes = self.r_ifes(n)
                void = self.r_x[r_ifes]
                if is_complex:
                    void = self.i_x[r_ifes]

    #
    #  Step 1  set essential and initial values to the solution vector.
    #
    def apply_essential(self, phys, update=False):
        is_complex = phys.is_complex()

        for kfes, name in enumerate(phys.dep_vars):
            r_ifes = self.r_ifes(name)
            rgf = self.r_x[r_ifes]
            igf = None if not is_complex else self.i_x[r_ifes]
            for mm in phys.walk():
                if not mm.enabled:
                    continue
                if not mm.has_essential:
                    continue
                if len(mm.get_essential_idx(kfes)) == 0:
                    continue
                if update and not mm.update_flag:
                    continue
                self.mask_X[self.access_idx,
                            self.r_dep_var_offset(name)] = True
                mm.apply_essential(self, rgf, real=True, kfes=kfes)
                if igf is not None:
                    mm.apply_essential(self, igf, real=False, kfes=kfes)

    def apply_init_from_init_panel(self, phys):
        from petram.model import Domain, Bdry
        from petram.phys.coefficient import sum_coefficient

        is_complex = phys.is_complex()

        def loop_over_phys_mm(gf, phys, kfes):
            bdrs = []
            c1_arr = []
            c2_arr = []

            for mm in phys.walk():
                if not mm.enabled:
                    continue
                if len(mm._sel_index) == 0:
                    continue

                if isinstance(mm, Domain):
                    c = mm.get_init_coeff(self, real=True, kfes=kfes)
                    if c is None:
                        continue
                    c1_arr.append(c)
                if isinstance(mm, Bdry):
                    c = mm.get_init_coeff(self, real=True, kfes=kfes)
                    if c is None:
                        continue
                    bdrs.extend(mm._sel_index)
                    c2_arr.append(c)

            if len(c1_arr) > 0:
                cc = sum_coefficient(c1_arr)
                gf.ProjectCoefficient(cc)

            if len(c2_arr) > 0:
                attrs = mfem.intArray(bdrs)
                cc = sum_coefficient(c2_arr)
                name = gf.FESpace().FEColl().Name()
                if name.startswith('ND'):
                    gf.ProjectBdrCoefficientTangent(cc, attrs)
                elif name.startswith('RT'):
                    gf.ProjectBdrCoefficientNormal(cc, attrs)
                else:
                    gf.ProjectBdrCoefficient(cc, attr)

        for kfes, name in enumerate(phys.dep_vars):
            if not name in self._init_done:
                self._init_done.append(name)
            r_ifes = self.r_ifes(name)
            rgf = self.r_x[r_ifes]

            loop_over_phys_mm(rgf, phys, kfes)

            if not is_complex:
                continue
            igf = self.i_x[r_ifes]
            loop_over_phys_mm(igf, phys, kfes)
            '''
            for mm in phys.walk():
                if not mm.enabled: continue
                c = mm.get_init_coeff(self, real = False, kfes = kfes)
                if c is None: continue
                ifg.ProjectCoefficient(c)
                #igf += tmp
            '''

    def apply_init_from_previous(self, names):

        for name in names:
            if not name in self._init_done:
                self._init_done.append(name)
            assert name in self._r_x_old, name + \
                " is not available from previous (real)"
            assert name in self._i_x_old, name + \
                " is not available from previous (imag)"
            rgf_old = self._r_x_old[name]
            igf_old = self._i_x_old[name]

            for j in range(self.n_matrix):
                self.access_idx = j
                if not self.is_matrix_active(j):
                    continue

                # if it is not using name defined in init, skip it...
                if not self.has_rfes(name):
                    continue

                r_ifes = self.r_ifes(name)
                rgf = self.r_x[r_ifes]
                igf = self.i_x[r_ifes]

                rgf.Assign(rgf_old)
                if igf is not None and igf_old is not None:
                    igf.Assign(igf_old)
                elif igf_old is None:
                    if igf is not None:
                        igf.Assign(0.0)
                        dprint1(
                            "New FESvar is complex while the previous one is real")
                else:
                    pass

    def apply_init_from_file(self, phys, init_path):
        '''
        read initial gridfunction from solution
        if init_path is "", then file is read from cwd.
        if file is not found, then it zeroes the gf
        '''
        dprint1("apply_init_from_file", phys, init_path)
        emesh_idx = phys.emesh_idx
        names = phys.dep_vars
        suffix = self.solfile_suffix()

        for kfes, name in enumerate(phys.dep_vars):
            if not name in self._init_done:
                self._init_done.append(name)
            r_ifes = self.r_ifes(name)
            rgf = self.r_x[r_ifes]
            if phys.is_complex():
                igf = self.i_x[r_ifes]
            else:
                igf = None
            fr, fi = self.solfile_name(names[kfes], emesh_idx)
            meshname = 'solmesh_' + str(emesh_idx) + suffix
            fr = fr + suffix
            fi = fi + suffix

            path = os.path.expanduser(init_path)
            if path == '':
                path = os.getcwd()
            fr = os.path.join(path, fr)
            fi = os.path.join(path, fi)
            meshname = os.path.join(path, meshname)

            rgf.Assign(0.0)
            if igf is not None:
                igf.Assign(0.0)
            if not os.path.exists(meshname):
                assert False, "Meshfile for sol does not exist:"+meshname
            if not os.path.exists(fr):
                assert False, "Solution (real) does not exist:"+fr
            if igf is not None and not os.path.exists(fi):
                assert False, "Solution (imag) does not exist:"+fi

            m = mfem.Mesh(str(meshname), 1, 1)
            # 2021. Nov
            # m.ReorientTetMesh()
            solr = mfem.GridFunction(m, str(fr))
            if solr.Size() != rgf.Size():
                assert False, "Solution file (real) has different length!!!"
            rgf += solr
            if igf is not None:
                soli = mfem.GridFunction(m, str(fi))
                if soli.Size() != igf.Size():
                    assert False, "Solution file (imag) has different length!!!"
                igf += soli

        check, val = self.load_extra_from_file(init_path)
        if check:
            self.sol_extra = val
        else:
            dprint1("(warining) extra is not loaded ...")
        # print self.sol_extra

    def apply_init_by_dwc(self, names, init_dwc):
        for n in names:
            if n not in self._init_done:
                self._init_done.append(n)

        self.call_dwc(None, method='init', callername=init_dwc[0],
                      dwcname=init_dwc[1],
                      args=init_dwc[2],
                      fesnames=names)

    #
    #  Step 2  fill matrix/rhs elements
    #

    def fill_bf(self, phys, update):
        renewargs = []

        if update:
            # make mask to renew r_a/i_a
            mask = [False]*len(phys.dep_vars)

            for kfes, name in enumerate(phys.dep_vars):
                ifes = self.ifes(name)
                rifes = self.r_ifes(name)
                for mm in phys.walk():
                    if not mm.enabled:
                        continue
                    if not mm.has_bf_contribution2(kfes, self.access_idx):
                        continue
                    if len(mm._sel_index) == 0:
                        continue
                    if not mm.update_flag:
                        continue
                    proj = mm.get_projection()
                    mask[kfes] = True
                    renewargs.append((ifes, rifes, proj))
                    self.mask_M[self.access_idx, self.dep_var_offset(name),
                                self.r_dep_var_offset(name)] = True

        else:
            mask = [True]*len(phys.dep_vars)
        is_complex = phys.is_complex()

        for args in renewargs:
            self.r_a.renew(args)

        for kfes, name in enumerate(phys.dep_vars):
            if not mask[kfes]:
                continue
            ifes = self.ifes(name)
            rifes = self.r_ifes(name)

            for mm in phys.walk():
                if not mm.enabled:
                    continue
                if not mm.has_bf_contribution2(kfes, self.access_idx):
                    continue
                if len(mm._sel_index) == 0:
                    continue
                proj = mm.get_projection()
                ra = self.r_a[ifes, rifes, proj]

                mm.set_integrator_realimag_mode(True)
                mm.add_bf_contribution(self, ra, real=True, kfes=kfes)

        if not is_complex:
            return

        for args in renewargs:
            self.i_a.renew(args)

        for kfes, name in enumerate(phys.dep_vars):
            if not mask[kfes]:
                continue

            ifes = self.ifes(name)
            rifes = self.r_ifes(name)
            for mm in phys.walk():
                if not mm.enabled:
                    continue
                if not mm.has_bf_contribution2(kfes, self.access_idx):
                    continue
                if len(mm._sel_index) == 0:
                    continue
                proj = mm.get_projection()
                ia = self.i_a[ifes, rifes, proj]

                mm.set_integrator_realimag_mode(False)
                mm.add_bf_contribution(self, ia, real=False, kfes=kfes)

    def fill_lf(self, phys, update):
        renewargs = []
        if update:
            # make mask to renew r_a/i_a
            mask = [False]*len(phys.dep_vars)

            for kfes, name in enumerate(phys.dep_vars):
                ifes = self.ifes(name)
                for mm in phys.walk():
                    if not mm.enabled:
                        continue
                    if not mm.has_lf_contribution2(kfes, self.access_idx):
                        continue
                    if len(mm._sel_index) == 0:
                        continue
                    if not mm.update_flag:
                        continue
                    mask[kfes] = True
                    renewargs.append(ifes)
                    self.mask_B[self.dep_var_offset(name)] = True
        else:
            mask = [True]*len(phys.dep_vars)

        is_complex = phys.is_complex()
        for args in renewargs:
            self.r_b.renew(args)

        for kfes, name in enumerate(phys.dep_vars):
            ifes = self.ifes(name)
            rb = self.r_b[ifes]
            rb.Assign(0.0)
            for mm in phys.walk():
                if not mm.enabled:
                    continue
                if not mm.has_lf_contribution2(kfes, self.access_idx):
                    continue
                if len(mm._sel_index) == 0:
                    continue

                mm.set_integrator_realimag_mode(True)
                mm.add_lf_contribution(self, rb, real=True, kfes=kfes)

        if not is_complex:
            return

        for args in renewargs:
            self.i_b.renew(args)

        for kfes, name in enumerate(phys.dep_vars):
            ifes = self.ifes(name)
            ib = self.i_b[ifes]
            ib.Assign(0.0)
            for mm in phys.walk():
                if not mm.enabled:
                    continue
                if not mm.has_lf_contribution2(kfes, self.access_idx):
                    continue
                if len(mm._sel_index) == 0:
                    continue

                mm.set_integrator_realimag_mode(False)
                mm.add_lf_contribution(self, ib, real=False, kfes=kfes)

    def fill_mixed(self, phys, update):

        renewflag1 = {}
        renewflag2 = {}
        fillflag = {}
        phys_offset = self.phys_offsets(phys)[0]
        rphys_offset = self.r_phys_offsets(phys)[0]

        for mm in phys.walk():
            if not mm.enabled:
                continue
            if not mm.has_mixed_contribution2(self.access_idx):
                continue
            if len(mm._sel_index) == 0:
                continue

            loc_list = mm.get_mixedbf_loc()

            for loc in loc_list:
                r, c, is_trans, is_conj = loc
                if isinstance(r, int):
                    #idx1 = phys_offset + r
                    #idx2 = rphys_offset + c
                    idx1 = self.ifes(phys.dep_vars[r])
                    idx2 = self.r_ifes(phys.dep_vars[c])
                else:
                    idx1 = self.ifes(r)
                    idx2 = self.r_ifes(c)
                if loc[2] < 0:
                    idx1, idx2 = idx2, idx1

                if not update:
                    fillflag[(idx1, idx2)] = True
                elif mm.update_flag:
                    if is_trans < 0:
                        renewflag2[(idx2, idx1)] = True
                        fillflag[(idx1, idx2)] = True
                    else:
                        renewflag1[(idx1, idx2)] = True
                        fillflag[(idx1, idx2)] = True
                else:
                    if not (idx1, idx2) in fillflag:
                        fillflag[(idx1, idx2)] = False

        is_complex = phys.is_complex()
        mixed_bf = {}
        tasks = {}

        for idx in renewflag1:
            self.r_a.renew(idx)
            if is_complex:
                self.i_a.renew(idx)

        for idx in renewflag2:
            self.r_at.renew(idx)
            if is_complex:
                self.i_at.renew(idx)

        for mm in phys.walk():
            if not mm.enabled:
                continue
            if not mm.has_mixed_contribution2(self.access_idx):
                continue
            if len(mm._sel_index) == 0:
                continue

            loc_list = mm.get_mixedbf_loc()

            for loc in loc_list:
                r, c, is_trans, is_conj = loc

                is_trans = (is_trans < 0)
                is_conj = (is_conj == -1)

                if isinstance(r, int):
                    #idx1 = phys_offset + r
                    #idx2 = rphys_offset + c
                    idx1 = self.ifes(phys.dep_vars[r])
                    idx2 = self.r_ifes(phys.dep_vars[c])
                else:
                    idx1 = self.ifes(r)
                    idx2 = self.r_ifes(c)

                if is_trans:
                    idx1_rec, idx2_rec = idx2, idx1
                else:
                    idx1_rec, idx2_rec = idx1, idx2

                if not fillflag[(idx1_rec, idx2_rec)]:
                    continue
                ridx1 = self.dep_var_offset(self.fes_vars[idx1_rec])
                ridx2 = self.r_dep_var_offset(self.r_fes_vars[idx2_rec])
                self.mask_M[self.access_idx, ridx1, ridx2] = True

                # real part
                if is_trans:
                    bfr = self.r_at[idx1, idx2]
                else:
                    bfr = self.r_a[idx1, idx2]

                mm.set_integrator_realimag_mode(True)
                mm.add_mix_contribution2(
                    self, bfr, r, c, False, is_conj, real=True)

                # imag part
                if is_complex:
                    if is_trans:
                        bfi = self.i_at[idx1, idx2]
                    else:
                        bfi = self.i_a[idx1, idx2]
                    mm.set_integrator_realimag_mode(False)
                    mm.add_mix_contribution2(
                        self, bfi, r, c, False, is_conj, real=False)

    def update_bf(self):
        fes_vars = self.fes_vars
        for j in range(self.n_matrix):
            self.access_idx = j
            if not self.is_matrix_active(j):
                continue

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

                projs = self.r_at.get_projections(ifes, ifes)
                fes = self.fespaces[name]
                for p in projs:
                    rat = self.r_at[ifes, ifes, p]
                    rat.Update(fes)
                projs = self.i_at.get_projections(ifes, ifes)
                for p in projs:
                    iat = self.i_at[ifes, ifes, p]
                    iat.Update(fes)

    def fill_coupling(self, coupling, phys_target):
        raise NotImplementedError("Coupling is not supported")

    def assemble_extra(self, phys, phys_range, keys_to_update):
        for mm in phys.walk():
            if not mm.enabled:
                continue
            for phys2 in phys_range:
                names = phys2.dep_vars
                for kfes, name in enumerate(names):
                    if not mm.has_extra_DoF2(kfes, phys2, self.access_idx):
                        continue

                    dep_var = names[kfes]
                    extra_name = mm.extra_DoF_name2(kfes)
                    key = (extra_name, dep_var, kfes)

                    if keys_to_update is not None and key not in keys_to_update:
                        continue

                    gl_ess_tdof1, gl_ess_tdof2 = self.gl_ess_tdofs[name]
                    gl_ess_tdof = gl_ess_tdof1 + gl_ess_tdof2
                    tmp = mm.add_extra_contribution(self,
                                                    ess_tdof=gl_ess_tdof,
                                                    kfes=kfes,
                                                    phys=phys2)
                    if tmp is None:
                        continue

                    if key in self.extras and keys_to_update is None:
                        assert False, "extra with key= " + \
                            str(key) + " already exists."
                    self.extras[key] = tmp
                    self.extras_mm[key] = mm.fullpath()

    def _extra_update_check(self, phys, phys_range, mode='B'):
        updated_name = []
        for mm in phys.walk():
            if not mm.enabled:
                continue
            for phys2 in phys_range:
                names = phys2.dep_vars
                for kfes, name in enumerate(names):
                    if not mm.has_extra_DoF2(kfes, phys2, self.access_idx):
                        continue

                    dep_var = names[kfes]
                    extra_name = mm.extra_DoF_name2(kfes)
                    key = (extra_name, dep_var, kfes)

                    if mm.update_flag:
                        if mode == 'B' and mm.check_extra_update('B'):
                            updated_name.append(key)
                        if mode == 'M' and mm.check_extra_update('M'):
                            updated_name.append(key)
        return updated_name

    def extra_update_check_M(self, phys, phys_range):
        return self._extra_update_check(phys, phys_range, mode='M')

    def extra_update_check_B(self, phys, phys_range):
        return self._extra_update_check(phys, phys_range, mode='B')

    def assemble_aux_ops(self, phys_target, phys_range, update):
        updated_name = []
        allmm = [mm for phys in phys_target for mm in phys.walk()
                 if mm.is_enabled()]

        self._aux_essential = []

        for phys1 in phys_target:
            names = phys1.dep_vars
            for kfes1, name1 in enumerate(names):
                for phys2 in phys_range:
                    names2 = phys2.dep_vars
                    for kfes2, name2 in enumerate(names2):
                        for mm in allmm:
                            if update and not mm.update_flag:
                                continue

                            if not mm.has_aux_op2(phys1, kfes1,
                                                  phys2, kfes2, self.access_idx):
                                continue
                            gl_ess_tdof1 = self.gl_ess_tdofs[name1]
                            gl_ess_tdof2 = self.gl_ess_tdofs[name2]
                            op = mm.get_aux_op(self, phys1, kfes1, phys2, kfes2,
                                               test_ess_tdof=gl_ess_tdof1,
                                               trial_ess_tdof=gl_ess_tdof2)
                            key = (name1, name2, mm.fullpath())
                            self.aux_ops[key] = op
                            if mm.no_elimination:
                                idx1 = self.dep_var_offset(name1)
                                idx2 = self.r_dep_var_offset(name2)
                                self._aux_essential.append((idx1, idx2))
                            updated_name.append(key)
        return updated_name

    def assemble_interp(self, phys):
        names = phys.dep_vars
        for name in names:
            gl_ess_tdof1, gl_ess_tdof2 = self.gl_ess_tdofs[name]
            gl_ess_tdof = gl_ess_tdof1 + gl_ess_tdof2
            kfes = names.index(name)
            interp = []
            for mm in phys.walk():
                if not mm.enabled:
                    continue
                if not mm.has_interpolation_contribution(kfes):
                    continue
                interp.append(mm.add_interpolation_contribution(self,
                                                                ess_tdof=gl_ess_tdof,
                                                                kfes=kfes))
            # merge all interpolation constraints
            P = None
            nonzeros = []
            zeros = []
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
    def prepare_M_B_blocks(self):
        size1 = len(self.dep_vars)
        size2 = len(self.r_dep_vars)
        M_block = [self.new_blockmatrix((size1, size2))
                   for i in range(self.n_matrix)]
        B_block = self.new_blockmatrix((size1, 1))
        return (M_block, B_block)

    def prepare_X_block(self):
        size = len(self.r_dep_vars)
        X_block = [self.new_blockmatrix((size, 1))
                   for i in range(self.n_matrix)]
        return X_block

    def prepare_B_blocks(self):
        size = len(self.dep_vars)
        B_block = self.new_blockmatrix((size, 1))
        return B_block

    def fill_M_B_blocks(self, M, B, update=False):
        from petram.helper.formholder import convertElement
        from mfem.common.chypre import BF2PyMat, LF2PyVec, Array2PyVec
        from mfem.common.chypre import MfemVec2PyVec, MfemMat2PyMat
        from itertools import product

        if update:
            M_changed = False
            R = len(self.dep_vars)
            C = len(self.r_dep_vars)
            for k in range(self.n_matrix):
                if not self.is_matrix_active(k):
                    continue

                for i, j in product(range(R), range(C)):
                    if self.mask_M[k, i, j]:
                        M[k][i, j] = None
                        M_changed = True
            for i in range(R):
                if self.mask_B[i]:
                    B[i] = None
        else:
            M_changed = True

        nfes = len(self.fes_vars)
        nrfes = len(self.r_fes_vars)

        for k in range(self.n_matrix):
            self.access_idx = k
            if not self.is_matrix_active(k):
                continue

            self.r_a.generateMatVec(self.a2A, self.a2Am)
            self.i_a.generateMatVec(self.a2A, self.a2Am)
            self.r_at.generateMatVec(self.a2A, self.a2Am)
            self.i_at.generateMatVec(self.a2A, self.a2Am)

            for i, j in product(range(nfes), range(nrfes)):
                r = self.dep_var_offset(self.fes_vars[i])
                c = self.r_dep_var_offset(self.r_fes_vars[j])

                if update and not self.mask_M[k, r, c]:
                    continue

                m1 = convertElement(self.r_a, self.i_a,
                                    i, j, MfemMat2PyMat,
                                    projections=(self.projections, self.projections_hash))
                if m1 is not None:
                    M[k][r, c] = m1 if M[k][r, c] is None else M[k][r, c] + m1

                m2 = convertElement(self.r_at, self.i_at,
                                    i, j, MfemMat2PyMat,
                                    projections=(self.projections, self.projections_hash))

                if m2 is not None:
                    m2t = m2.transpose()
                    M[k][c, r] = m2t if M[k][c, r] is None else M[k][c, r] + m2t

            for extra_name, dep_name, kfes in self.extras.keys():
                r = self.dep_var_offset(extra_name)
                c = self.r_dep_var_offset(dep_name)
                c1 = self.dep_var_offset(extra_name)
                if dep_name in self._dep_vars:
                    r1 = self.dep_var_offset(dep_name)
                else:
                    r1 = -1

                if update and not self.mask_M[k, r, c]:
                    continue

                # t1, t2, t3, t4 = (vertical, horizontal, diag, rhs).
                t1, t2, t3, t4, t5 = self.extras[(extra_name, dep_name, kfes)]

                ifes = self.r_ifes(dep_name)
                x = self.r_x[ifes]

                T1 = self.t1_2_T1(t1, x)
                T2 = self.t2_2_T2(t2, x)

                if r1 != -1:
                    M[k][r1, c1] = T1 if M[k][r1, c1] is None else M[k][r1, c1]+T1
                M[k][r, c] = T2 if M[k][r, c] is None else M[k][r, c]+T2
                #M[k][r,r] = t3 if M[k][r,r] is None else M[k][r,r]+t3
                M[k][r, c1] = t3 if M[k][r, c1] is None else M[k][r, c1]+t3

            #print("aux", k, self.aux_ops.keys())
            for key in self.aux_ops.keys():
                testname, trialname, mm_fullpath = key
                r = self.dep_var_offset(testname)
                c = self.r_dep_var_offset(trialname)
                if update and not self.mask_M[k, r, c]:
                    continue

                m = self.aux_ops[key]
                M[k][r, c] = m if M[k][r, c] is None else M[k][r, c]+m

        self.fill_B_blocks(B, update=update)

        return M, B, M_changed

    def fill_B_blocks(self, B, update=False):
        from petram.helper.formholder import convertElement
        from mfem.common.chypre import MfemVec2PyVec

        dprint1("fill_B_blocks mask_B", self.mask_B)
        nfes = len(self.fes_vars)
        self.access_idx = 0
        self.r_b.generateMatVec(self.b2B)
        self.i_b.generateMatVec(self.b2B)
        for i in range(nfes):
            r = self.dep_var_offset(self.fes_vars[i])
            if update and not self.mask_B[r]:
                continue

            v = convertElement(self.r_b,
                               self.i_b,
                               i, 0, MfemVec2PyVec)
            B[r] = v

        self.access_idx = 0
        for extra_name, dep_name, kfes in self.extras.keys():
            r = self.dep_var_offset(extra_name)
            if update and not self.mask_B[r]:
                continue

            t1, t2, t3, t4, t5 = self.extras[(extra_name, dep_name, kfes)]
            B[r] = t4

    def fill_X_block(self, X):
        from petram.helper.formholder import convertElement
        from mfem.common.chypre import BF2PyMat, LF2PyVec, Array2PyVec
        from mfem.common.chypre import MfemVec2PyVec, MfemMat2PyMat
        from itertools import product

        for k in range(self.n_matrix):
            self.access_idx = k
            if not self.is_matrix_active(k):
                continue

            self.r_x.generateMatVec(self.x2X)
            self.i_x.generateMatVec(self.x2X)
            for dep_var in self.r_dep_vars:
                r = self.r_dep_var_offset(dep_var)
                if not self.mask_X[k, r]:
                    continue

                if self.r_isFESvar(dep_var):
                    i = self.r_ifes(dep_var)
                    v = convertElement(self.r_x, self.i_x,
                                       i, 0, MfemVec2PyVec)
                    X[k][r] = v
                else:
                    if self.sol_extra is not None:
                        for key in self.sol_extra:
                            if dep_var in self.sol_extra[key]:
                                value = self.sol_extra[key][dep_var]
                                X[k][r] = Array2PyVec(value)
                    else:
                        pass
                        # For now, it leaves as None for Lagrange Multipler?
                        # May need to allocate zeros...
        return X

    def fill_BCeliminate_matrix(self, A, B, inplace=True, update=False):
        diagpolicy = self.get_diagpolicy()

        nblock1 = A.shape[0]
        nblock2 = A.shape[1]

        Ae = self.new_blockmatrix(A.shape)

        for name in self.gl_ess_tdofs:
            # we do elimination only for the varialbes to be solved
            if not name in self._dep_vars:
                continue

            gl_ess_tdof1, gl_ess_tdof2 = self.gl_ess_tdofs[name]
            ess_tdof1, ess_tdof2 = self.ess_tdofs[name]
            idx1 = self.dep_var_offset(name)
            idx2 = self.r_dep_var_offset(name)

            if A[idx1, idx2] is None:
                A.add_empty_square_block(idx1, idx2)

            if A[idx1, idx2] is not None:
                # note: this check is necessary, since in parallel environment,
                # add_empty_square_block could not create any block because
                # locally number or rows is zero.
                if self.get_autofill_diag():
                    self.fill_empty_diag(A[idx1, idx2])

                Aee, A[idx1, idx2], Bnew = A[idx1, idx2].eliminate_RowsCols(B[idx1], ess_tdof1,
                                                                            inplace=inplace,
                                                                            diagpolicy=diagpolicy)
                A[idx1, idx2] = A[idx1, idx2].resetRow(
                    gl_ess_tdof2, inplace=inplace)
                A[idx1, idx2].setDiag(gl_ess_tdof2)

                Ae[idx1, idx2] = Aee

                B[idx1] = Bnew

            '''
            note: minor differece between serial/parallel
 
            Aee in serial ana parallel are not equal. The definition of Aee in MFEM is
            A_original = Aee + A, where A_diag is set to one for Esseential DoF
            In the serial mode, Aee_diag is not properly set. But this element
            does not impact the final RHS.
            '''
            for j in range(nblock2):
                if j == idx2:
                    continue
                if A[idx1, j] is None:
                    continue

                A[idx1, j] = A[idx1, j].resetRow(gl_ess_tdof1, inplace=inplace)
                if not (idx1, j) in self._aux_essential and len(gl_ess_tdof2) > 0:
                    A[idx1, j] = A[idx1, j].resetRow(
                        gl_ess_tdof2, inplace=inplace)

            for j in range(nblock1):
                if j == idx1:
                    continue
                if A[j, idx2] is None:
                    continue

                SM = A.get_squaremat_from_right(j, idx2)
                SM.setDiag(gl_ess_tdof1)

                Ae[j, idx2] = A[j, idx2].dot(SM)
                A[j, idx2] = A[j, idx2].resetCol(gl_ess_tdof1, inplace=inplace)

        return A, Ae

    def eliminateJac(self, Jac):
        '''
        eliminate both col/rows from matrix

        '''
        for name in self.gl_ess_tdofs:
            if not name in self._dep_vars:
                continue

            idx1 = self.dep_var_offset(name)
            idx2 = self.r_dep_var_offset(name)

            gl_ess_tdof1, gl_ess_tdof2 = self.gl_ess_tdofs[name]
            if Jac[idx1, idx2] is not None:
                Jac[idx1, idx2].resetRow(gl_ess_tdof1)
                Jac[idx1, idx2].resetCol(gl_ess_tdof1)

    def eliminateBC(self, Ae, X, RHS):
        try:
            AeX = Ae.dot(X)
            for name in self.gl_ess_tdofs:
                if not name in self._dep_vars:
                    continue

                idx = self.dep_var_offset(name)
                gl_ess_tdof1, gl_ess_tdof2 = self.gl_ess_tdofs[name]
                if AeX[idx, 0] is not None:
                    AeX[idx, 0].resetRow(gl_ess_tdof1)

            RHS = RHS - AeX
        except:
            print("Ae", Ae)
            print("X", X)
            raise

        for name in self.gl_ess_tdofs:
            if not name in self._dep_vars:
                continue

            idx = self.dep_var_offset(name)
            ridx = self.r_dep_var_offset(name)
            gl_ess_tdof1, gl_ess_tdof2 = self.gl_ess_tdofs[name]
            ess_tdof1, ess_tdof2 = self.ess_tdofs[name]

            x1 = X[ridx].get_elements(gl_ess_tdof1)
            x2 = RHS[idx].get_elements(gl_ess_tdof1)
            RHS[idx].set_elements(gl_ess_tdof1, x1*x2)

        return RHS

    def collect_local_ess_TDofs(self, opr, format, is_complex):
        '''
        Find essential TDoFs index in solution block vector

        this method assumes format is either (blk_interleave, blk_merged, blk_merged_s)
        '''
        nblock1 = opr.NumColBlocks()
        nblock2 = opr.NumRowBlocks()

        offsets1 = opr.ColOffsets().ToList()
        offsets2 = opr.RowOffsets().ToList()

        ret = []

        for name in self.gl_ess_tdofs:
            # we do elimination only for the varialbes to be solved
            if not name in self._dep_vars:
                continue

            # collecto only essentials which are eliminated
            ess_tdof = np.array(self.ess_tdofs[name][0], dtype=int)
            idx = self.dep_var_offset(name)

            if is_complex and format == 'blk_interleave':
                o1 = offsets1[2*idx]
                ret.append(ess_tdof + o1)
                o2 = offsets1[2*idx+1]
                ret.append(ess_tdof + o2)
            elif is_complex and format.startswith('blk_merged'):
                o1 = offsets1[idx]
                size = offsets1[idx+1]-offsets1[idx]
                ret.append(ess_tdof + o1)
                ret.append(ess_tdof + o1 + size//2)
            else:
                o1 = offsets1[idx]
                ret.append(ess_tdof + o1)

        return np.hstack(ret)

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
        #import traceback
        # traceback.print_stack()
        for name in self.interps:
            idx1 = self.dep_var_offset(name)
            idx2 = self.r_dep_var_offset(name)
            P, nonzeros, zeros = self.interps[name]
            if P is None:
                continue

            if A is not None:
                shape = A.shape
                A1 = A[idx1, idx2]
                A1 = A1.rap(P.transpose())
                A1.setDiag(zeros, 1.0)
                A[idx1, idx2] = A1

                PP = P.conj(inplace=True)
                for i in range(shape[1]):
                    if idx1 == i:
                        continue
                    if A[idx1, i] is not None:
                        A[idx1, i] = PP.dot(A[idx1, i])
                P = PP.conj(inplace=True)
                PP = P.transpose()
                for i in range(shape[0]):
                    if idx2 == i:
                        continue
                    if A[i, idx2] is not None:
                        A[i, idx2] = A[i, idx2].dot(PP)
            if RHS is not None:
                RHS[idx1] = P.conj(inplace=True).dot(RHS[idx1])
                P.conj(inplace=True)

        if A is not None and RHS is not None:
            return A, RHS
        if A is not None:
            return A,
        if RHS is not None:
            return RHS

    #
    #  step4 : matrix finalization (to form a data being passed to a linear solver)
    #
    def finalize_matrix(self, M_block, mask, is_complex, format='coo',
                        verbose=True):
        if verbose:
            dprint1("A (in finalizie_matrix) \n", format, mask)
            dprint1(M_block, notrim=True)
        M_block = M_block.get_subblock(mask[0], mask[1])

        if format == 'coo':  # coo either real or complex
            M = self.finalize_coo_matrix(M_block, is_complex, verbose=verbose)

        elif format == 'coo_real':  # real coo converted from complex
            M = self.finalize_coo_matrix(M_block, is_complex,
                                         convert_real=True, verbose=verbose)

        elif format == 'blk_interleave':  # real coo converted from complex
            M = M_block.get_global_blkmat_interleave()

        elif format == 'blk_merged':  # real coo converted from complex
            M = M_block.get_global_blkmat_merged()

        elif format == 'blk_merged_s':  # real coo converted from complex
            M = M_block.get_global_blkmat_merged(symmetric=True)

        dprint2('exiting finalize_matrix')
        self.is_assembled = True
        return M

    def finalize_rhs(self,  B_blocks, M_block, X_block,
                     mask, is_complex, format='coo', verbose=True,
                     use_residual=False):
        #
        #  RHS = B - A[not solved]*X[not solved]
        #
        inv_mask = [not x for x in mask[1]]
        MM = M_block.get_subblock(mask[0], inv_mask)
        XX = X_block.get_subblock(inv_mask, [True])
        xx = MM.dot(XX)

        B_blocks = [b.get_subblock(mask[0], [True]) - xx for b in B_blocks]

        if use_residual:
            M_block_use = M_block.get_subblock(mask[0], mask[1])
            X_block_use = X_block.get_subblock(mask[1], [True])
            B_blocks = [b - M_block_use.dot(X_block_use) for b in B_blocks]

        if format == 'coo':  # coo either real or complex
            BB = [self.finalize_coo_rhs(
                b, is_complex, verbose=verbose) for b in B_blocks]
            BB = np.hstack(BB)

        elif format == 'coo_real':  # real coo converted from complex
            BB = [self.finalize_coo_rhs(b, is_complex,
                                        convert_real=True, verbose=verbose)
                  for b in B_blocks]
            BB = np.hstack(BB)

        elif format == 'blk_interleave':  # real coo converted from complex
            BB = [b.gather_blkvec_interleave() for b in B_blocks]

        elif format == 'blk_merged':
            BB = [b.gather_blkvec_merged() for b in B_blocks]

        elif format == 'blk_merged_s':
            BB = [b.gather_blkvec_merged(symmetric=True) for b in B_blocks]

        else:
            assert False, "unsupported format for B"

        return BB

    def finalize_x(self,  X_block, RHS, mask, is_complex,
                   format='coo', verbose=True):
        X_block = X_block.get_subblock(mask[1], [True])
        RHS = RHS.get_subblock(mask[0], [True])
        if format == 'blk_interleave':  # real coo converted from complex
            X = X_block.gather_blkvec_interleave(size_hint=RHS)

        elif format == 'blk_merged' or format == 'blk_merged_s':
            X = X_block.gather_blkvec_merged(size_hint=RHS)

        else:
            assert False, "unsupported format for X"

        return X

    def finalize_coo_matrix(self, M_block, is_complex, convert_real=False,
                            verbose=False):
        # if verbose:
        #    dprint1("A (in finalizie_coo_matrix) \n",  M_block)

        if not convert_real:
            if is_complex:
                M = M_block.get_global_coo(dtype='complex')
            else:
                M = M_block.get_global_coo(dtype='float')
        else:
            M = M_block.get_global_coo(dtype='complex')
            M = scipy.sparse.bmat(
                [[M.real, -M.imag], [M.imag, M.real]], format='coo')
            # (this one make matrix symmetric, for now it is off to do the samething
            #  as GMRES case)
            # M = scipy.sparse.bmat([[M.real, -M.imag], [-M.imag, -M.real]], format='coo')
        return M

    def finalize_coo_rhs(self, b, is_complex,
                         convert_real=False,
                         verbose=True):
        if verbose:
            dprint1("b (in finalizie_coo_rhs) \n",  b, notrim=True)
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
        s = [None]*len(self.r_fes_vars)
        #nicePrint("sol", sol, self._rdep_vars)
        for name in self.fes_vars:
            # print name
            j = self.r_dep_var_offset(name)
            sol_section = sol[j, 0]

            if name in self.interps:
                P, nonzeros, zeros = self.interps[name]
                if P is not None:
                    sol_section = (P.transpose()).dot(sol_section)

            ifes = self.r_ifes(name)
            s[ifes] = sol_section
            sol[j, 0] = sol_section

        e = []
        for name in self.dep_vars:
            if not self.isFESvar(name):
                e.append(sol[self.dep_var_offset(name)])
        #nicePrint(s, e)
        return s, e

    def recover_sol(self, sol, access_idx=0):
        self.access_idx = access_idx

        for k, s in enumerate(sol):
            if s is None:
                continue  # None=linear solver didnot solve this value, so no update
            name = self.r_fes_vars[k]
            r_ifes = self.r_ifes(name)
            ridx = self.r_dep_var_offset(name)
            s = s.toarray()
            X = self.r_x.get_matvec(r_ifes)

            X.Assign(s.flatten().real)

            self.X2x(X, self.r_x[r_ifes])
            if self.i_x[r_ifes] is not None:
                X = self.i_x.get_matvec(r_ifes)
                X.Assign(s.flatten().imag)
                self.X2x(X, self.i_x[r_ifes])
            else:
                dprint2("real value problem skipping i_x")

    def process_extra(self, sol_extra):
        ret = {}
        k = 0
        extra_names = [name for name in self.dep_vars
                       if not self.isFESvar(name)]

        if self.extras is None:
            # when init_only with fixed initial is chosen
            return ret

        print_flag = []
        for extra_name, dep_name, kfes in self.extras.keys():
            data = sol_extra[extra_names.index(extra_name)]
            t1, t2, t3, t4, t5 = self.extras[(extra_name, dep_name, kfes)]
            mm_path = self.extras_mm[(extra_name, dep_name, kfes)]
            mm = self.model[mm_path]
            ret[extra_name] = {}
            if mm.extra_diagnostic_print:
                print_flag.append(extra_name)

            if not t5:
                continue
            if data is not None:
                ret[extra_name][mm.extra_DoF_name2(kfes)] = data.toarray()
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
        for k in ret:
            tmp = {x: ret[k][x].flatten() for x in ret[k]}
            tmp2 = {x: ret[k][x].flatten() for x in ret[k] if x in print_flag}
            dprint1("extra (diagnostic)", tmp2)
        return ret

    #
    #  save to file
    #

    def save_sol_to_file(self, phys_target, skip_mesh=False,
                         mesh_only=False,
                         save_parmesh=False,
                         save_mesh_linkdir=None):
        if not skip_mesh:
            m1 = [self.save_mesh0(save_mesh_linkdir), ]
            mesh_filenames = self.save_mesh(phys_target, save_mesh_linkdir)
            mesh_filenames = m1 + mesh_filenames

        if save_parmesh:
            self.save_parmesh(phys_target)
        if mesh_only:
            return mesh_filenames

        self.access_idx = 0
        for phys in phys_target:
            emesh_idx = phys.emesh_idx
            for name in phys.dep_vars:
                ifes = self.r_ifes(name)
                r_x = self.r_x[ifes]
                i_x = self.i_x[ifes]
                self.save_solfile_fespace(name, emesh_idx, r_x, i_x)

    def extrafile_name(self):
        return 'sol_extended.data'

    def save_extra_to_file(self, sol_extra, extrafile_name=''):
        if sol_extra is None:
            return

        if extrafile_name == '':
            extrafile_name = self.extrafile_name()
        extrafile_name += self.solfile_suffix()

        self.sol_extra = sol_extra  # keep it for future reuse

        # count extradata length
        ll = 0
        for name in sol_extra.keys():
            for k in sol_extra[name].keys():
                data = sol_extra[name][k]
                if data.ndim == 0:
                    ll = ll + data.size
                else:
                    data = data.flatten()
                    sol_extra[name][k] = data
                    ll = ll + data.size
        if ll == 0:
            return

        fid = open(extrafile_name, 'w')
        for name in sol_extra.keys():
            for k in sol_extra[name].keys():
                data = sol_extra[name][k]
                #  data must be NdArray
                #  dataname : "E1.E_out"
                fid.write('name : ' + name + '.' + str(k) + '\n')
                if data.ndim == 0:
                    fid.write('size : ' + str(data.size) + '\n')
                    fid.write('dim : ' + str(data.ndim) + '\n')
                    fid.write('dtype: ' + str(data.dtype) + '\n')
                    fid.write(str(0) + ' ' + str(data) + '\n')
                else:
                    data = data.flatten()
                    sol_extra[name][k] = data
                    fid.write('size : ' + str(data.size) + '\n')
                    fid.write('dim : ' + str(data.ndim) + '\n')
                    fid.write('dtype: ' + str(data.dtype) + '\n')
                    for kk, d in enumerate(data):
                        fid.write(str(kk) + ' ' + str(d) + '\n')
        fid.close()

    def load_extra_from_file(self, init_path):
        sol_extra = {}
        extrafile_name = self.extrafile_name()+self.solfile_suffix()

        path = os.path.join(init_path, extrafile_name)

        if not os.path.exists(path):
            return False, None

        fid = open(path, 'r')
        line = fid.readline()
        while line:
            if line.startswith('name'):
                name, name2 = line.split(':')[1].strip().split('.')
                if not name in sol_extra:
                    sol_extra[name] = {}
            size = long(fid.readline().split(':')[1].strip())
            dim = long(fid.readline().split(':')[1].strip())
            dtype = fid.readline().split(':')[1].strip()
            if dtype.startswith('complex'):
                data = [complex(fid.readline().split(' ')[1])
                        for k in range(size)]
                data = np.array(data, dtype=dtype)
            else:
                data = [float(fid.readline().split(' ')[1])
                        for k in range(size)]
                data = np.array(data, dtype=dtype)
            sol_extra[name][name2] = data
            line = fid.readline()
        fid.close()
        return True, sol_extra
    #
    #  postprocess
    #

    @property
    def ppname_postfix(self):
        return self._ppname_postfix

    @ppname_postfix.setter
    def ppname_postfix(self, value):
        self._ppname_postfix = value

    def store_pp_extra(self, name, data, save_once=False):
        name = name + self._ppname_postfix
        self.model._parameters[name] = data
        self._pp_extra_update.append(name)

    def run_postprocess(self, postprocess, name=''):
        self._pp_extra_update = []

        for pp in postprocess:
            if not pp.enabled:
                continue
            pp.run(self)

        extra = {n: self.model._parameters[n] for n in self._pp_extra_update}
        extra = {name: extra}

        self.save_extra_to_file(extra, extrafile_name=name + '.data')

    #
    #  helper methods
    #
    def assign_phys_pp_sel_index(self):
        if len(self.meshes) == 0:
            #dprint1('!!!! mesh is None !!!!')
            return
        all_phys = [self.model['Phys'][k] for
                    k in self.model['Phys'].keys()]
        all_pp = []
        for k in self.model['PostProcess']:
            for kk in self.model['PostProcess'][k]:
                all_pp.append(self.model['PostProcess'][k][kk])

        for p in all_phys + all_pp:
            #
            if hasattr(p, "mesh_idx") and p.mesh_idx != 0:
                assert False, "We don't support mesh_idx != 0, Contact developer if this is needed"
            base_mesh = self.meshes[0]

            if base_mesh is None:
                assert False, "base_mesh not selected"

            ec = base_mesh.extended_connectivity
            allv = list(ec['vol2surf']) if ec['vol2surf'] is not None else []
            alls = list(ec['surf2line']) if ec['surf2line'] is not None else []
            alle = list(ec['line2vert']) if ec['line2vert'] is not None else []

            p.update_dom_selection(all_sel=(allv, alls, alle))

    def assign_sel_index(self, phys=None):
        if len(self.meshes) == 0:
            #dprint1('!!!! mesh is None !!!!')
            return
        if phys is None:
            all_phys = [self.model['Phys'][k] for
                        k in self.model['Phys'].keys()]
        else:
            all_phys = [phys]

        for p in all_phys:
            if p.mesh_idx < 0:
                continue
            mesh = self.meshes[p.mesh_idx]
            if mesh is None:
                continue

            if len(p.sel_index) == 0:
                continue

            dom_choice, bdr_choice, pnt_choice, internal_bdr = p.get_dom_bdr_pnt_choice(
                self.meshes[p.mesh_idx])

            dprint1("## internal bdr index " + str(internal_bdr))

            p._phys_sel_index = dom_choice
            self.do_assign_sel_index(p, dom_choice, Domain)
            self.do_assign_sel_index(
                p, bdr_choice, Bdry, internal_bdr=internal_bdr)
            self.do_assign_sel_index(p, pnt_choice, Point)

    def do_assign_sel_index(self, m, choice, cls, internal_bdr=None):
        dprint1("## setting _sel_index (1-based number): " + cls.__name__ +
                ":" + m.fullname() + ":" + str(choice))
        # _sel_index is 0-base array

        def _walk_physics(node):
            yield node
            for k in node.keys():
                yield node[k]
        rem = None
        checklist = np.array([True]*len(choice), dtype=bool)

        for node in m.walk():
            if not isinstance(node, cls):
                continue
            if not node.is_enabled():
                continue
            ret = node.process_sel_index(choice, internal_bdr=internal_bdr)

            if ret is None:
                if rem is not None:
                    rem._sel_index = []
                rem = node
            elif ret == -1:
                node._sel_index = choice
                if not node.is_secondary_condition:
                    checklist[np.in1d(choice, node._sel_index)] = False

                dprint1(node.fullname(), str(node._sel_index))
            else:
                dprint1(node.fullname(), str(ret))
                # for k in ret:
                #   idx = list(choice).index(k)
                #   if node.is_secondary_condition: continue
                #   checklist[idx] = False
                if not node.is_secondary_condition:
                    checklist[np.in1d(choice, ret)] = False
        if rem is not None:
            rem._sel_index = list(np.array(choice)[checklist])
            dprint1(rem.fullname() + ':' + str(rem._sel_index))

    def find_domain_by_index(self, phys, idx,  check_enabled=False):
        return self._do_find_by_index(phys, idx, Domain,
                                      check_enabled=check_enabled)

    def find_bdry_by_index(self, phys, idx, check_enabled=False):
        return self._do_find_by_index(phys, idx, Bdry,
                                      check_enabled=check_enabled)

    def _do_find_by_index(self, phys, idx, cls, ignore_secondary=True,
                          check_enabled=False):
        for node in phys.walk():
            if (check_enabled and (not node.enabled)):
                continue
            if not isinstance(node, cls):
                continue
            if idx in node._sel_index:
                if ((ignore_secondary and not node.is_secondary_condition)
                        or not ignore_secondary):
                    return node

    def gather_essential_tdof(self, phys):
        flags = self.get_essential_bdr_pnt_flag(phys)
        self.get_essential_bdr_pnt_tdofs(phys, flags)

    def get_essential_bdr_pnt_flag(self, phys):
        flag = []
        for k,  name in enumerate(phys.dep_vars):
            fes = self.fespaces[name]
            index1 = []    # with elimination
            index2 = []    # w/o elimination
            ptx1 = []    # with elimination (point)
            ptx2 = []    # w/o elimination (point)

            for node in phys.walk():
                # if not isinstance(node, Bdry): continue
                if not node.enabled:
                    continue
                if node.has_essential and isinstance(node, Bdry):
                    if node.use_essential_elimination():
                        index1 = index1 + node.get_essential_idx(k)
                    else:
                        index2 = index2 + node.get_essential_idx(k)
                if node.has_essential and isinstance(node, Point):
                    if node.use_essential_elimination():
                        ptx1 = ptx1 + node.get_ess_point_array(k)
                    else:
                        ptx2 = ptx2 + node.get_ess_point_array(k)

            if len(self.emeshes[phys.emesh_idx].bdr_attributes.ToList()) > 0:
                ess_bdr1 = [0] * \
                    self.emeshes[phys.emesh_idx].bdr_attributes.Max()
                ess_bdr2 = [0] * \
                    self.emeshes[phys.emesh_idx].bdr_attributes.Max()
            else:
                ess_bdr1 = []
                ess_bdr2 = []
            for kk in index1:
                ess_bdr1[kk-1] = 1
            for kk in index2:
                ess_bdr2[kk-1] = 1
            flag.append((name, ess_bdr1, ess_bdr2, ptx1, ptx2))

        return flag

    def get_point_essential_tdofs(self, fespace, ess_point_array):
        raise NotImplementedError(
            "you must specify this method in subclass")

    def get_essential_bdr_pnt_tdofs(self, phys, flags):
        for name, ess_bdr1, ess_bdr2, ptx1, ptx2 in flags:
            fespace = self.fespaces[name]

            ess_tdof_list = mfem.intArray()
            ess_bdr1 = mfem.intArray(ess_bdr1)
            fespace.GetEssentialTrueDofs(ess_bdr1, ess_tdof_list)
            ess_tdofs1 = ess_tdof_list.ToList()

            ess_tdof_list = mfem.intArray()
            ess_bdr2 = mfem.intArray(ess_bdr2)
            fespace.GetEssentialTrueDofs(ess_bdr2, ess_tdof_list)
            ess_tdofs2 = ess_tdof_list.ToList()
            self.ess_tdofs[name] = (ess_tdofs1, ess_tdofs2)

            if len(ptx1) > 0:
                tmp = self.get_point_essential_tdofs(fespace, ptx1)
                self.ess_tdofs[name][0].extend(tmp)
            if len(ptx2) > 0:
                tmp = self.get_point_essential_tdofs(fespace, ptx2)
                self.ess_tdofs[name][1].extend(tmp)

            #print(name, len(self.ess_tdofs[name]))
        return

    def allocate_fespace(self, phys):
        num_fec = len(phys.get_fec())

        count = 0
        for name, elem in phys.get_fec():
            vdim = phys.vdim
            if hasattr(vdim, '__iter__'):
                vdim = vdim[count]
            else:
                pass
            emesh_idx = phys.emesh_idx
            order = phys.fes_order(count)

            if elem.startswith('RT'):
                vdim = 1
            if elem.startswith('ND'):
                vdim = 1

            dprint1("allocate_fespace: " + name)
            is_new, fes = self.get_or_allocate_fecfes(name, emesh_idx, elem,
                                                      order, vdim)
            count = count+1

    def get_or_allocate_fecfes(self, name, emesh_idx, elem, order, vdim, make_new=True):
        mesh = self.emeshes[emesh_idx]
        isParMesh = hasattr(mesh, 'ParPrint')
        sdim = mesh.SpaceDimension()
        dim = mesh.Dimension()

        is_new = False
        key = (emesh_idx, elem, order, dim, sdim, vdim, isParMesh)
        dkey = ("emesh_idx", "elem", "order",
                "dim", "sdim", "vdim", "isParMesh")
        #dprint1("Allocate/Reuse fec/fes:", {d: v for d, v in zip(dkey, key)})

        dprint1("Looking for already allocated fespaces ", name)
        if name in self.fespaces:
            fes1 = self.fespaces[name]
            isFESparallel = hasattr(fes1, 'GroupComm')
            if isFESparallel == isParMesh:
                return False, fes1
        # elif not make_new:
        #    return False, None
        dprint1("Making a new fec/fes", {d: v for d, v in zip(dkey, key)})
        is_new = True
        element = elem.split('(')[0].strip()

        if "(" in elem:
            fecdim = dim
        else:
            fecdim = dim
            if elem.startswith('RT'):
                fecdim = sdim
            if elem.startswith('ND'):
                fecdim = sdim

        self.fespaces.new_hierarchy(name,
                                    parameters=(emesh_idx, element, order, fecdim, vdim))
        fes1 = self.fespaces[name]

        return True, fes1

    # mesh.GetEdgeVertexTable()
    #        self._fes_storage[key] = fes
        #self.add_fec_fes(name, fec, fes)

    # def add_fec_fes(self, name, fec, fes):
    #    self.fec[name] = fec
    #    self.fespaces[name] = fes
    def prepare_refined_level(self, phys, mode, inc=1, refine_dom=None):
        '''
        mode = 'H' or 'P'
        inc = increment of order
        '''

        names = [n for n in phys.dep_vars]
        for name in names:
            if inc == 0:
                nlevels = self.fespaces.add_same_level(name, self)

            elif mode == 'H':
                nlevels = self.fespaces.add_mesh_refined_level(
                    name, self, inc, refine_dom)

            elif mode == 'P':
                nlevels = self.fespaces.add_order_refined_level(
                    name, self, inc)

            else:
                assert False, "Unknown refinement mode"

    def get_fes(self, phys, kfes=0, name=None):
        if name is None:
            name = phys.dep_vars[kfes]
            return self.fespaces[name]
        else:
            return self.fespaces[name]

    def alloc_gf(self, idx, idx2=0):
        fes = self.fespaces[self.r_fes_vars[idx]]
        return self.new_gf(fes)

    def alloc_lf(self, idx, idx2=0):
        name = self.fes_vars[idx]
        dprint2("")
        dprint2("< *** > Generating a new LF between " + name)
        fes = self.fespaces[name]
        return self.new_lf(fes)

    def alloc_bf(self, idx, idx2=None):
        name = self.fes_vars[idx]
        dprint2("")
        dprint2("< *** > Generating a new BF between " + name)
        fes = self.fespaces[name]
        return self.new_bf(fes)

    def alloc_mbf(self, idx1, idx2):  # row col

        name1 = self.fes_vars[idx1]
        name2 = self.r_fes_vars[idx2]
        dprint2("")
        dprint2("< *** > Generating a new mixed-BF between " +
                name1 + " and " + name2)
        fes1 = self.fespaces[name1]
        fes2 = self.fespaces[name2]

        info1 = self.get_fes_info(fes1)
        info2 = self.get_fes_info(fes2)
        if info1["emesh_idx"] != info2["emesh_idx"]:
            info1 = self.get_fes_info(fes1)
            info2 = self.get_fes_info(fes2)
            dprint1(
                "fes1 and fes2 are on different mesh. Constructing a DoF map.", name1, name2)
            dprint1("info1", info1)
            dprint1("info2", info2)

            name = name2 + '_to_' + name1
            namehash = str(hash(tuple(info1.items()))) + \
                "___" + str(hash(tuple(info1.items())))
            self.projections_hash[name] = namehash

            if abs(info1["dim"]-info2["dim"]) > 1:
                assert False, "does not support direct mapping from volume to edge (or face to vertex)"

            from petram.helper.projection import simple_projection, fes_mapping

            '''
            transpose = False
            if info2["dim"] >= info1["dim"]:
                transpose = False
            else:
                _info1 = info1
                info1 = info2
                info2 = _info1
                _fes2 = fes1
                fes1 = fes2
                fes2 = _fes2
                transpose = True
            '''
            el, order, mbfdim = fes_mapping(info2["element"], info2["order"], info2["dim"],
                                            info1["dim"])

            #emesh_idx = info2["emesh_idx"] if mbfdim == 1 else info1["emesh_idx"]
            emesh_idx = info1["emesh_idx"]
            is_new, fes = self.get_or_allocate_fecfes(name,
                                                      emesh_idx,
                                                      el,
                                                      order,
                                                      info2["vdim"])

            if namehash not in self.projections:
                if info2["dim"]-info1["dim"] == 1:  # (ex)volume to surface
                    mode = "boundary"
                elif info2["dim"]-info1["dim"] == 0:
                    mode = "domain"
                else:
                    mode = "domain"
                #assert False, "should not come here."

                # fes2  -> fes (intermediate space using fes1's mesh)
                p = simple_projection(fes2, fes, mode)

                self.projections[namehash] = p
                '''
                if transpose:
                    # first -1 is flag to apply projecton from right
                    proj = (-1, name)
                    fes2 = fes
                    return self.new_mixed_bf(fes1, fes2), proj
                else:
                '''
            else:
                p = self.projections[namehash]

            # first 1 is flag to apply projecton from left. A_ij = A_ij*Map
            proj = (1, name)
            fes2 = fes
            return self.new_mixed_bf(fes2, fes1), proj
        else:
            proj = 1
            return self.new_mixed_bf(fes2, fes1), proj

    def build_ns(self):
        errors = []
        for node in self.model.walk():
            if node.has_ns():
                try:
                    node.eval_ns()

                except Exception as e:
                    node._global_ns = {}
                    m = traceback.format_exc()
                    errors.append("failed to build ns for " + node.fullname() +
                                  "\n" + m)
            else:
                #node._global_ns = None
                node._local_ns = self.model.root()._variables

        if len(errors) > 0:
            dprint1("\n".join(errors), notrim=True)
            assert False, "\n".join(errors)

    def preprocess_ns(self, ns_folder, data_folder):
        '''
        folders are tree object
        '''
        for od in self.model.walk():
            if od.has_ns():
                od.preprocess_ns(ns_folder, data_folder)
            if od.has_nsref():
                od.reset_ns()

    def form_linear_system(self, ess_tdof_list, extra, interp, r_A, r_B, i_A, i_B):
        raise NotImplementedError(
            "you must specify this method in subclass")

    def reset_emesh_data(self):
        from petram.mesh.mesh_extension import MeshExt
        self.emesh_data = MeshExt()

    def prep_emesh_data_ifneeded(self):
        if self.emesh_data is None:
            self.reset_emesh_data()

    def run_mesh_serial(self, meshmodel=None,
                        skip_refine=False):

        from petram.mesh.mesh_model import MeshFile, MFEMMesh
        #from petram.mesh.mesh_extension import MeshExt
        from petram.mesh.mesh_utils import get_extended_connectivity

        self.meshes = []
        self.prep_emesh_data_ifneeded()
        # if self.emesh_data is None:
        #    self.emesh_data = MeshExt()
        self.emeshes = []

        if meshmodel is None:
            parent = self.model['Mesh']
            children = [parent[g] for g in parent.keys()
                        if isinstance(parent[g], MFEMMesh) and parent[g].enabled]
            for idx, child in enumerate(children):
                self.meshes.append(None)
                # if not child.enabled: continue
                target = None
                for k in child.keys():
                    o = child[k]
                    if not o.enabled:
                        continue

                    if o.isMeshGenerator:
                        dprint1("Loading mesh (serial)")
                        self.meshes[idx] = o.run_serial()
                        target = self.meshes[idx]
                    else:
                        if (o.isRefinement and
                                skip_refine):
                            continue
                        if hasattr(o, 'run') and target is not None:
                            self.meshes[idx] = o.run(target)
        self.max_bdrattr = -1
        self.max_attr = -1

        for m in self.meshes:
            if len(m.GetBdrAttributeArray()) > 0:
                self.max_bdrattr = np.max(
                    [self.max_bdrattr, max(m.GetBdrAttributeArray())])
            if len(m.GetAttributeArray()) > 0:
                self.max_attr = np.max(
                    [self.max_attr, max(m.GetAttributeArray())])

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

    def new_fespace(self, mesh, fec, vdim):
        raise NotImplementedError(
            "you must specify this method in subclass")

    def new_fespace_hierarchy(self, mesh, fes, ownM, ownF):
        parallel = hasattr(mesh, 'GetComm')
        if parallel:
            return mfem.ParFiniteElementSpaceHierarchy(mesh, fes, ownM, ownF)
        else:
            return mfem.FiniteElementSpaceHierarchy(mesh, fes, ownM, ownF)

    def new_mesh_from_mesh(self, mesh):
        parallel = hasattr(mesh, 'GetComm')
        if parallel:
            return mfem.ParMesh(mesh)
        else:
            return mfem.Mesh(mesh)

    def eliminate_ess_dof(self, ess_tdof_list, M, B):
        raise NotImplementedError(
            "you must specify this method in subclass")

    def solfile_suffix(self):
        raise NotImplementedError(
            "you must specify this method in subclass")

    def solfile_name(self, name, mesh_idx,
                     namer='solr', namei='soli'):
        fnamer = '_'.join((namer, name, str(mesh_idx)))
        fnamei = '_'.join((namei, name, str(mesh_idx)))

        return fnamer, fnamei

    def remove_solfiles(self):
        dprint1("clear sol: ", os.getcwd())
        d = os.getcwd()
        files = os.listdir(d)
        for file in files:
            if file.startswith('solmesh'):
                os.remove(os.path.join(d, file))
            if file.startswith('solr'):
                os.remove(os.path.join(d, file))
            if file.startswith('soli'):
                os.remove(os.path.join(d, file))
            if file.startswith('checkpoint.'):
                os.remove(os.path.join(d, file))
            if file.startswith('sol_extended'):
                os.remove(os.path.join(d, file))
            if file.startswith('probe'):
                os.remove(os.path.join(d, file))
            if file.startswith('matrix'):
                os.remove(os.path.join(d, file))
            if file.startswith('rhs'):
                os.remove(os.path.join(d, file))
            if file.startswith('SolveStep'):
                os.remove(os.path.join(d, file))
            if file.startswith('cProfile_'):
                os.remove(os.path.join(d, file))
            if file.startswith('checkpoint_') and os.path.isdir(file):
                dprint1("removing checkpoint_", file)
                shutil.rmtree(os.path.join(d, file))

    def remove_case_dirs(self):
        dprint1("clear case directories: ", os.getcwd())
        d = os.getcwd()
        files = os.listdir(d)
        for file in files:
            if file.startswith('case') and os.path.isdir(file):
                dprint1("removing case directory", file)
                shutil.rmtree(os.path.join(d, file))

    def clear_solmesh_files(self, header):
        try:
            from mpi4py import MPI
        except:
            from petram.helper.dummy_mpi import MPI
        myid = MPI.COMM_WORLD.rank
        nproc = MPI.COMM_WORLD.size

        MPI.COMM_WORLD.Barrier()
        if myid == 0:
            for f in os.listdir("."):
                if f.startswith(header):
                    os.remove(f)
        MPI.COMM_WORLD.Barrier()

    def save_solfile_fespace(self, name, mesh_idx, r_x, i_x):
        fnamer, fnamei = self.solfile_name(name, mesh_idx)
        suffix = self.solfile_suffix()

        self.clear_solmesh_files(fnamer)
        self.clear_solmesh_files(fnamei)

        fnamer = fnamer+suffix
        fnamei = fnamei+suffix

        if self.get_savegz():
            r_x.SaveGZ(fnamer, 8)
            if i_x is not None:
                i_x.SaveGZ(fnamei, 8)
        else:
            r_x.Save(fnamer, 8)
            if i_x is not None:
                i_x.Save(fnamei, 8)

    def save_mesh0(self, save_mesh_linkdir=None):
        mesh_names = []
        suffix = self.solfile_suffix()
        mesh = self.emeshes[0]
        header = 'solmesh_0'
        self.clear_solmesh_files(header)
        name = header+suffix

        if save_mesh_linkdir is None:
            if self.get_savegz():
                mesh.PrintGZ(name, 16)
            else:
                mesh.Print(name, 16)
        else:
            src = os.path.join(save_mesh_linkdir, name)
            dst = os.path.join(os.getcwd(), name)
            os.symlink(src, dst)
        return name

    def save_mesh(self, phys_target, save_mesh_linkdir=None):
        mesh_names = []
        suffix = self.solfile_suffix()

        done = []

        for phys in phys_target:
            k = phys.emesh_idx
            if k in done:
                continue
            done.append(k)

            name = phys.dep_vars[0]

            mesh = self.fespaces.get_mesh(name)

            header = 'solmesh_' + str(k)
            self.clear_solmesh_files(header)

            name = header+suffix

            if save_mesh_linkdir is None:
                if self.get_savegz():
                    mesh.PrintGZ(name, 16)
                else:
                    mesh.Print(name, 16)
            else:
                src = os.path.join(save_mesh_linkdir, name)
                dst = os.path.join(os.getcwd(), name)
                os.symlink(src, dst)

            mesh_names.append(name)

        return mesh_names

    @property  # ALL dependent variables including Lagrange multipliers
    def dep_vars(self):
        return self._dep_vars

    @property  # ALL finite element space variables
    def fes_vars(self):
        return self._fes_vars

    @property  # ALL dependent variables including Lagrange multipliers
    def r_dep_vars(self):
        return self._rdep_vars

    @property  # ALL finite element space variables
    def r_fes_vars(self):
        return self._rfes_vars

    def ifes(self, name):
        return self._fes_vars.index(name)

    def r_ifes(self, name):
        if name in self._rfes_vars:
            return self._rfes_vars.index(name)
        return -1

    def has_rfes(self, name):
        return name in self._rfes_vars

    def has_fes(self, name):
        return name in self._fes_vars

    def phys_offsets(self, phys):
        name = phys.dep_vars[0]
        idx0 = self._dep_vars.index(name)
        for names in self._dep_var_grouped:
            if name in names:
                l = len(names)
        return range(idx0, idx0+l)

    def dep_var_offset(self, name):
        return self._dep_vars.index(name)

    def masked_dep_var_offset(self, name):
        return [x for i, x in enumerate(self._dep_vars)
                if self._matrix_blk_mask[0][i]].index(name)

    def masked_dep_var_names(self):
        return [x for i, x in enumerate(self._dep_vars)
                if self._matrix_blk_mask[0][i]]

    def isFESvar(self, name):
        if not name in self._dep_vars:
            assert False, "Variable " + name + " not used in the model"
        idx = self._dep_vars.index(name)
        return self._isFESvar[idx]

    def r_phys_offsets(self, phys):
        name = phys.dep_vars[0]
        idx0 = self._rdep_vars.index(name)
        for names in self._rdep_var_grouped:
            if name in names:
                l = len(names)
        return range(idx0, idx0+l)

    def r_dep_var_offset(self, name):
        return self._rdep_vars.index(name)

    def masked_r_dep_var_offset(self, name):
        return [x for i, x in enumerate(self._rdep_vars)
                if self._matrix_blk_mask[1][i]].index(name)

    def r_isFESvar(self, name):
        if not name in self._rdep_vars:
            assert False, "Variable " + name + " not used in the model"
        idx = self._rdep_vars.index(name)
        return self._risFESvar[idx]

    def get_block_mask(self, all_phys, phys_target, use_range=False):

        # all_phys = [self.model['Phys'][k] for k in self.model['Phys']
        #            if self.model['Phys'][k].enabled]
        # if phys_target is None:
        #   phys_target = all_phys
        # self.collect_dependent_vars()

        if use_range:
            mask = [False]*len(self._rdep_vars)
        else:
            mask = [False]*len(self._dep_vars)

        for phys in phys_target:
            idx = all_phys.index(phys)
            dep_vars0 = phys.dep_vars0
            dep_vars = phys.dep_vars
            if use_range:
                extra_vars = [x for x in self._rdep_var_grouped[idx]
                              if not x in dep_vars]
            else:
                extra_vars = [x for x in self._dep_var_grouped[idx]
                              if not x in dep_vars]

            for name in dep_vars0+extra_vars:
                if use_range:
                    offset = self.r_dep_var_offset(name)
                else:
                    offset = self.dep_var_offset(name)
                mask[offset] = True
        return mask

    def copy_block_mask(self, mask):
        self._matrix_blk_mask = mask

    def check_block_matrix_changed(self, mask):
        from itertools import product
        R = len(self.dep_vars)
        C = len(self.r_dep_vars)
        for k in range(self.n_matrix):
            if not self.is_matrix_active(k):
                continue

            for i, j in product(range(R), range(C)):
                if self.mask_M[k, i, j] and mask[0][j] and mask[1][i]:
                    return True
        return False

    def collect_dependent_vars(self, phys_target=None, phys_range=None, range_space=False):
        if phys_target is None:
            phys_target = [self.model['Phys'][k] for k in self.model['Phys']
                           if self.model['Phys'].enabled]

        dep_vars_g = []
        isFesvars_g = []

        for phys in phys_target:
            dep_vars = []
            isFesvars = []
            if not phys.enabled:
                continue

            dv = phys.dep_vars
            dep_vars.extend(dv)
            isFesvars.extend([True]*len(dv))

            extra_vars = []
            for mm in phys.walk():
                if not mm.enabled:
                    continue

                for j in range(self.n_matrix):
                    for k in range(len(dv)):
                        # for phys2 in phys_target:
                        for phys2 in phys_range:
                            if not phys2.enabled:
                                continue
                            if not mm.has_extra_DoF2(k, phys2, j):
                                continue

                            name = mm.extra_DoF_name2(k)
                            if not name in extra_vars:
                                extra_vars.append(name)
            dep_vars.extend(extra_vars)
            isFesvars.extend([False]*len(extra_vars))

            dep_vars_g.append(dep_vars)
            isFesvars_g.append(isFesvars)

        if range_space:
            self._rdep_vars = sum(dep_vars_g, [])
            self._rdep_var_grouped = dep_vars_g
            self._risFESvar = sum(isFesvars_g, [])
            self._risFESvar_grouped = isFesvars_g
            self._rfes_vars = [x for x, flag in zip(
                self._rdep_vars, self._risFESvar) if flag]
            dprint1("dependent variables(range)", self._rdep_vars)
            dprint1("is FEspace variable?(range)", self._risFESvar)

        else:
            self._dep_vars = sum(dep_vars_g, [])
            self._dep_var_grouped = dep_vars_g
            self._isFESvar = sum(isFesvars_g, [])
            self._isFESvar_grouped = isFesvars_g
            self._fes_vars = [x for x, flag in zip(
                self._dep_vars, self._isFESvar) if flag]
            dprint1("dependent variables", self._dep_vars)
            dprint1("is FEspace variable?", self._isFESvar)

    def add_PP_to_NS(self, variables):
        for k in variables.keys():
            self.model._variables[k] = variables[k]

    def add_FESvariable_to_NS(self, phys_range, verbose=False):
        '''
        bring FESvariable to NS so that it is available in matrix assembly.
        note:
            if SolveStep is used. This overwrite the FESvaialbe from the
            previous SolveStep. In order to use values from the previous step,
            a user needs to load in using InitSetting
        '''
        from petram.helper.variables import Variables
        variables = Variables()

        self.access_idx = 0
        for phys in phys_range:
            for name in phys.dep_vars:
                if not self.has_rfes(name):
                    continue
                rifes = self.r_ifes(name)
                rgf = self.r_x[rifes]
                igf = self.i_x[rifes]
                phys.add_variables(variables, name, rgf, igf)

        keys = list(self.model._variables)
        # self.model._variables.clear()
        if verbose:
            dprint1("===  List of variables ===")
            dprint1(variables)
        for k in variables.keys():
            # if k in self.model._variables:
            #   dprint1("Note : FES variable from previous step exists, but overwritten. \n" +
            #           "Use InitSetting to load value from previous SolveStep: ", k)
            self.model._variables[k] = variables[k]

    def set_update_flag(self, mode):
        for k in self.model['Phys'].keys():
            phys = self.model['Phys'][k]
            for mm in self.model['Phys'][k].walk():
                mm._update_flag = False
                if mode == 'TimeDependent':
                    if mm.isTimeDependent:
                        mm._update_flag = True
                    if mm.isTimeDependent_RHS:
                        mm._update_flag = True
                elif mode == 'UpdateAll':
                    mm._update_flag = True
                elif mode == 'ParametricRHS':
                    if self.n_matrix > 1:
                        assert False,  "RHS-only parametric is not allowed for n__matrix > 1"
                    for kfes, name in enumerate(phys.dep_vars):
                        if mm.has_lf_contribution2(kfes, 0):
                            mm._update_flag = True
                    if mm.has_essential:
                        mm._update_flag = True
                    # if mm.is_extra_RHSonly():
                    if mm.isTimeDependent_RHS:
                        for kfes, name in enumerate(phys.dep_vars):
                            if mm.has_extra_DoF2(kfes, phys, 0):
                                mm._update_flag = True
                    if mm.isTimeDependent:
                        for kfes, name in enumerate(phys.dep_vars):
                            if mm.has_extra_DoF2(kfes, phys, 0):
                                assert False, "RHS only parametric is invalid for general extra DoF"
                    if mm._update_flag:
                        for kfes, name in enumerate(phys.dep_vars):
                            if mm.has_bf_contribution2(kfes, 0):
                                assert False, "RHS only parametric is not possible for BF :"+mm.name()

                else:
                    assert False, "update mode not supported: mode = "+mode

    def call_dwc(self, phys_range, method='', callername='',
                 dwcname='', args='', fesnames=None, **kwargs):

        names = []
        if phys_range is not None:
            for phys in phys_range:
                for name in phys.dep_vars:
                    names.append(name)
        if fesnames is not None:
            names.extend(fesnames)

        for name in names:
            rifes = self.r_ifes(name)
            if rifes == -1:
                continue
            rgf = self.r_x[rifes]
            igf = self.i_x[rifes]
            if igf is None:
                kwargs[name] = rgf
            else:
                kwargs[name] = (rgf, igf)

        g = self.model['General']._global_ns
        if dwcname == '':
            dwc = g[self.model['General'].dwc_object_name]
        else:
            dwc = g[dwcname]
        args0, kwargs = dwc.make_args(method, kwargs)

        m = getattr(dwc, method)

        def f(*args, **kargs):
            return args, kargs
        try:
            args, kwargs2 = eval('f('+args+')',  g, {'f': f})
        except:
            traceback
            traceback.print_exc()
            assert False, "Failed to convert text to argments"

        for k in kwargs2:
            kwargs[k] = kwargs2[k]
        args = tuple(list(args0) + list(args))

        try:
            return m(callername, *args, **kwargs)

        except:
            import traceback
            traceback.print_exc()
            assert False, "Direct Wrapper Call Failed"

    def get_fes_info(self, fes):
        return self.fespaces.get_fes_info(fes)

        '''
        # k = emesh_idx, elem, order, sdim, vdim, isParMesh
        for k in self.fecfes_storage:
            if self.fecfes_storage[k][1] == fes:
                return {'emesh_idx': k[0],
                        'element': k[1],
                        'order': k[2],
                        'dim': k[3],
                        'sdim': k[4],
                        'vdim': k[5], }
        return None
        '''

    def variable2vector(self, v, horizontal=False):
        '''
        GFFunctionVariable to PyVec
        '''
        from mfem.common.chypre import MfemVec2PyVec
        rlf = self.x2X(v.deriv_args[0])
        if v.deriv_args[1] is not None:
            ilf = self.x2X(v.deriv_args[1])
        else:
            ilf = None
        return MfemVec2PyVec(rlf, ilf, horizontal=horizontal)

    def run_geom_gen(self, gen):
        raise NotImplementedError(
            "you must specify this method in subclass")

    def run_mesh_gen(self, gen):
        raise NotImplementedError(
            "you must specify this method in subclass")

    def get_diagpolicy(self):
        policy = self.model.root()['General'].diagpolicy
        if policy == 'one':
            return 0
        elif policy == 'keep':
            return 1
        else:
            assert False, "unknow diag polcy: " + policy

    def get_savegz(self):
        val = self.model.root()['General'].savegz
        if val == 'on':
            return True
        return False

    def get_partitiong_method(self):
        return self.model.root()['General'].partitioning

    def get_submesh_partitiong_method(self):
        return self.model.root()['General'].submeshpartitioning

    def get_autofill_diag(self):
        return self.model.root()['General'].autofilldiag == 'on'


class SerialEngine(Engine):
    def __init__(self, modelfile='', model=None):
        super(SerialEngine, self).__init__(modelfile=modelfile, model=model)
        self.isParallel = False

    def run_mesh(self, meshmodel=None, skip_refine=False):
        '''
        skip_refine is for mfem_viewer
        '''
        return self.run_mesh_serial(meshmodel=meshmodel,
                                    skip_refine=skip_refine,)

    def run_assemble_mat(self, phys_target, phys_range, update=False):
        self.is_matrix_distributed = False
        return super(SerialEngine, self).run_assemble_mat(phys_target, phys_range,
                                                          update=update)

    def new_lf(self, fes):
        return mfem.LinearForm(fes)

    def new_bf(self, fes, fes2=None):
        return mfem.BilinearForm(fes)

    def new_mixed_bf(self, fes1, fes2):
        bf = mfem.MixedBilinearForm(fes1, fes2)
        bf._finalized = False
        return bf

    def new_gf(self, fes, init=True, gf=None):
        if gf is None:
            gf = mfem.GridFunction(fes)
        else:
            assert False, "I don't think this is used..."
            gf = mfem.GridFunction(gf.FESpace())

        if init:
            gf.Assign(0.0)

        idx = self.fespaces.get_fes_emesh_idx(fes)
        if idx is not None:
            gf._emesh_idx = idx
        else:
            assert False, "new gf is called with unknonw fes"

        return gf
        '''
        for k in self.fecfes_storage:
            if self.fecfes_storage[k][1] == fes:
                gf._emesh_idx = k[0]
                break
        else:
            assert False, "new gf is called with unknonw fes"
        return gf
        '''

    def new_matrix(self, init=True):
        return mfem.SparseMatrix()

    def new_blockmatrix(self, shape):
        from petram.helper.block_matrix import BlockMatrix
        return BlockMatrix(shape, kind='scipy')

    def new_fespace(self, mesh, fec, vdim):
        return mfem.FiniteElementSpace(mesh, fec, vdim)

    def collect_all_ess_tdof(self):
        self.gl_ess_tdofs = self.ess_tdofs

    def get_point_essential_tdofs(self, fes, ess_point_array):
        fec_name = fes.FEColl().Name()
        if not fec_name.startswith("H1"):
            assert False, "Pointwise Essential supports only H1 element"

        mesh = fes.GetMesh()

        dofs = []
        for iv in range(mesh.GetNV()):
            if tuple(mesh.GetVertexArray(iv)) in ess_point_array:
                tmp = fes.GetVertexDofs(iv)
                dofs.extend(tmp)

        return dofs

    def save_parmesh(self, phys_target):
        # serial engine does not do anything
        return

    def solfile_suffix(self):
        return ""

    def get_true_v_sizes(self, phys):
        fe_sizes = [self.fespaces[name].GetTrueVSize()
                    for name in phys.dep_vars]
        dprint1('Number of finite element unknowns: ' + str(fe_sizes))
        return fe_sizes

    def split_sol_array_fespace(self, sol, P):
        sol0 = sol[0, 0]
        if P is not None:
            sol0 = P.transpose().dot(sol0)
        return sol0

    def mkdir(self, path):
        if not os.path.exists(path):
            os.mkdir(path)

    def symlink(self, target, link):
        os.symlink(target, link)

    def open_file(self, *args, **kwargs):
        return open(*args, **kwargs)

    def cleancwd(self):
        for f in os.listdir("."):
            os.remove(f)

    '''
    def remove_solfiles(self):       
        dprint1("clear sol: ", os.getcwd())                  
        d = os.getcwd()
        files = os.listdir(d)
        for file in files:
            if file.startswith('solmesh'): os.remove(os.path.join(d, file))
            if file.startswith('solr'): os.remove(os.path.join(d, file))
            if file.startswith('soli'): os.remove(os.path.join(d, file))
            if file.startswith('checkpoint.'): os.remove(os.path.join(d, file))
            if file.startswith('sol_extended'): os.remove(os.path.join(d, file))
            if file.startswith('proble'): os.remove(os.path.join(d, file))
            if file.startswith('matrix'): os.remove(os.path.join(d, file))
            if file.startswith('rhs'): os.remove(os.path.join(d, file))
            if file.startswith('checkpoint_'): shutil.removetree(os.path.joij(d, file))   
    '''

    def a2A(self, a):  # BilinearSystem to matrix
        # we dont eliminate essentiaal at this level...
        inta = mfem.intArray()
        m = self.new_matrix()
        a.FormSystemMatrix(inta, m)
        return m

    def a2Am(self, a):  # MixedBilinearSystem to matrix
        if not a._finalized:
            a.ConformingAssemble()
            a._finalized = True
        return a.SpMat()

    def b2B(self, b):  # FormLinearSystem w/o elimination
        fes = b.FESpace()
        B = mfem.Vector()
        if not fes.Conforming():
            P = fes.GetConformingProlongation()
            #R = fes.GetConformingRestriction()
            # if R is not None:
            if P is not None:
                B.SetSize(P.Width())
                P.MultTranspose(b, B)
            else:
                B.NewDataAndSize(b.GetData(), b.Size())
        else:
            B.NewDataAndSize(b.GetData(), b.Size())
        return B

    def x2X(self, x):  # gridfunction to vector
        fes = x.FESpace()
        X = mfem.Vector()
        if not fes.Conforming():
            R = fes.GetConformingRestriction()
            if R is not None:
                X.SetSize(R.Height())
                R.Mult(x, X)
            else:
                X.NewDataAndSize(x.GetData(), x.Size())
        else:
            X.NewDataAndSize(x.GetData(), x.Size())
        return X

    def X2x(self, X, x):  # RecoverFEMSolution
        fes = x.FESpace()
        if fes.Conforming():
            pass
        else:
            P = fes.GetConformingProlongation()
            if P is not None:
                x.SetSize(P.Height())
                P.Mult(X, x)

    def t1_2_T1(self, t1, x):
        fes = x.FESpace()
        if not fes.Conforming():
            R = fes.GetConformingRestriction()
            if R is not None:
                from mfem.common.sparse_utils import sparsemat_to_scipycsr
                R2 = sparsemat_to_scipycsr(R, float)
                T1 = R2.dot(t1)
                return T1
            else:
                return t1
        else:
            return t1

    def t2_2_T2(self, t2, x):
        fes = x.FESpace()
        if not fes.Conforming():
            R = fes.GetConformingRestriction()
            if R is not None:
                from mfem.common.sparse_utils import sparsemat_to_scipycsr
                R2 = sparsemat_to_scipycsr(R, float)
                T2 = t2.dot(R2.transpose())
                return T2
            else:
                return t2
        else:
            return t2

    def run_geom_gen(self, gen):
        gen.generate_final_geometry()

    def run_mesh_gen(self, gen):
        gen.generate_mesh_file()

    def save_processed_model(self):
        self.model.save_to_file('model_proc.pmfm', meshfile_relativepath=False)

    def fill_empty_diag(self, A):
        '''
        A is ScipyCoo (this one fully supports complex)
        '''
        csr = A.tocsr()
        csr.eliminate_zeros()
        zerorows = np.where(np.diff(csr.indptr) == 0)[0]
        if len(zerorows) == csr.shape[0]:
            dprint1(
                "!!! skipping fill_empty_diag: this diagonal block is compltely zero")
            return
        lil = A.tolil()
        lil[zerorows, zerorows] = 1.0
        coo = lil.tocoo()
        A.data = coo.data
        A.row = coo.row
        A.col = coo.col


class ParallelEngine(Engine):
    def __init__(self, modelfile='', model=None):
        super(ParallelEngine, self).__init__(modelfile=modelfile, model=model)
        self.isParallel = True

    def run_mesh(self, meshmodel=None):
        from mpi4py import MPI
        from petram.mesh.mesh_model import MeshFile, MFEMMesh
        from petram.mesh.mesh_extension import MeshExt
        from petram.mesh.mesh_utils import get_extended_connectivity

        dprint1("Loading mesh (parallel)")

        self.base_meshes = []
        self.meshes = []
        self.emeshes = []
        if self.emesh_data is None:
            assert False, "emesh data must be generated before parallel mesh generation"

        if meshmodel is None:
            parent = self.model['Mesh']
            children = [parent[g] for g in parent.keys()
                        if isinstance(parent[g], MFEMMesh) and parent[g].enabled]

            for idx, child in enumerate(children):
                self.meshes.append(None)
                self.base_meshes.append(None)
                if not child.enabled:
                    continue
                target = None

                srefines = [child[x]
                            for x in child if child[x].isSerialRefinement]
                for k in child.keys():
                    o = child[k]
                    if not o.enabled:
                        continue
                    # dprint1(k)
                    if o.isMeshGenerator:
                        smesh = o.run()
                        if len(smesh.GetBdrAttributeArray()) > 0:
                            self.max_bdrattr = np.max([self.max_bdrattr,
                                                       max(smesh.GetBdrAttributeArray())])
                        if len(smesh.GetAttributeArray()) > 0:
                            self.max_attr = np.max([self.max_attr,
                                                    max(smesh.GetAttributeArray())])

                        p_method = self.get_partitiong_method()

                        if p_method == 'by attribute':
                            attr = list(smesh.GetAttributeArray()-1)
                            attr_array = mfem.intArray(attr)
                            parts = attr_array.GetData()
                        else:
                            if p_method != 'auto':
                                dprint1(
                                    "Unkown partitioning method, fallback to auto !!!")
                            if smesh.GetNE() < MPI.COMM_WORLD.size*3:
                                parts = smesh.GeneratePartitioning(
                                    smesh.GetNE()//1000+1, 1)
                            else:
                                parts = None

                        for srefine in srefines:
                            smesh = srefine.run(smesh)

                        self.base_meshes[idx] = mfem.ParMesh(
                            MPI.COMM_WORLD, smesh, parts)
                        self.meshes[idx] = self.base_meshes[idx]
                        target = self.meshes[idx]

                        #self.base_meshes[idx] = self.meshes[idx]
                    else:
                        if hasattr(o, 'run') and target is not None:
                            if o.isSerialRefinement:
                                continue
                            target = self.new_mesh_from_mesh(target)
                            self.meshes[idx] = o.run(target)
                            target = self.meshes[idx]

        for m in self.meshes:
            # 2021. Nov
            # m.ReorientTetMesh()
            m.GetEdgeVertexTable()
            get_extended_connectivity(m)

    def run_assemble_mat(self, phys_target, phys_range, update=False):
        self.is_matrix_distributed = True
        return super(ParallelEngine, self).run_assemble_mat(phys_target,
                                                            phys_range,
                                                            update=update)

    def new_lf(self, fes):
        return mfem.ParLinearForm(fes)

    def new_bf(self, fes, fes2=None):
        return mfem.ParBilinearForm(fes)

    def new_mixed_bf(self, fes1, fes2):
        bf = mfem.ParMixedBilinearForm(fes1, fes2)
        bf._finalized = False
        return bf

    def new_gf(self, fes, init=True, gf=None):
        if gf is None:
            gf = mfem.ParGridFunction(fes)
        else:
            assert False, "I don't think this is used..."
            gf = mfem.ParGridFunction(gf.ParFESpace())
        if init:
            gf.Assign(0.0)

        idx = self.fespaces.get_fes_emesh_idx(fes)
        if idx is not None:
            gf._emesh_idx = idx
        else:
            assert False, "new gf is called with unknonw fes"

        return gf

    def new_fespace(self, mesh, fec, vdim):
        if hasattr(mesh, 'GetComm'):
            return mfem.ParFiniteElementSpace(mesh, fec, vdim)
        else:
            return mfem.FiniteElementSpace(mesh, fec, vdim)

    def new_matrix(self, init=True):
        return mfem.HypreParMatrix()

    def new_blockmatrix(self, shape):
        from petram.helper.block_matrix import BlockMatrix
        return BlockMatrix(shape, kind='hypre')

    def get_true_v_sizes(self, phys):
        fe_sizes = [self.fespaces[name].GlobalTrueVSize()
                    for name in phys.dep_vars]
        from mpi4py import MPI
        myid = MPI.COMM_WORLD.rank
        if (myid == 0):
            dprint1('Number of finite element unknowns: ' + str(fe_sizes))
        return fe_sizes

    def get_point_essential_tdofs(self, fes, ess_point_array):
        fec_name = fes.FEColl().Name()
        if not fec_name.startswith("H1"):
            assert False, "Pointwise Essential supports only H1 element"

        mesh = fes.GetParMesh()

        dofs = []
        for iv in range(mesh.GetNV()):
            if tuple(mesh.GetVertexArray(iv)) in ess_point_array:
                tmp = fes.GetVertexDofs(iv)
                dofs.extend(tmp)

        dofs = [fes.GetLocalTDofNumber(x) for x in dofs]

        return dofs

    #
    #  postprocess
    #
    def store_pp_extra(self, name, data, save_once=False):
        from mpi4py import MPI
        name = name + self._ppname_postfix
        self.model._parameters[name] = data
        if not save_once or MPI.COMM_WORLD.rank == 0:
            self._pp_extra_update.append(name)

    def save_parmesh(self, phys_target):
        from mpi4py import MPI
        num_proc = MPI.COMM_WORLD.size
        myid = MPI.COMM_WORLD.rank
        smyid = '{:0>6d}'.format(myid)

        for phys in phys_target:
            k = phys.emesh_idx
            name = phys.dep_vars[0]

            mesh = self.fespaces.get_mesh(name)

            header = 'solparmesh_' + str(k)
            self.clear_solmesh_files(header)

            mesh_name = header+'.'+smyid
            mesh.ParPrintToFile(mesh_name, 16)

            header = 'solsermesh_' + str(k)
            mesh_name = header+'.mesh'
            mesh.PrintAsSerial(mesh_name)

    def solfile_suffix(self):
        from mpi4py import MPI
        num_proc = MPI.COMM_WORLD.size
        myid = MPI.COMM_WORLD.rank
        smyid = '{:0>6d}'.format(myid)
        return "."+smyid

    def split_sol_array_fespace(self, sol, P):
        sol0 = sol[0, 0]
        if P is not None:
            sol0 = (P.transpose()).dot(sol0)
        return sol0

    def collect_all_ess_tdof(self, M=None):
        from mpi4py import MPI

        #gl_ess_tdofs = []
        # for name in phys.dep_vars:
        #    fes = self.fespaces[name]

        for name in self.ess_tdofs:
            tdof1, tdof2 = self.ess_tdofs[name]
            myoffset = self.fespaces[name].GetMyTDofOffset()
            data1 = (np.array(tdof1) + myoffset).astype(np.int32)
            data2 = (np.array(tdof2) + myoffset).astype(np.int32)

            gl_ess_tdof1 = allgather_vector(data1, MPI.INT)
            gl_ess_tdof2 = allgather_vector(data2, MPI.INT)
            MPI.COMM_WORLD.Barrier()
            #gl_ess_tdofs.append((name, gl_ess_tdof))
            # TO-DO intArray must accept np.int32
            gtdofs1 = [int(x) for x in gl_ess_tdof1]
            gtdofs2 = [int(x) for x in gl_ess_tdof2]
            self.gl_ess_tdofs[name] = (gtdofs1, gtdofs2)

    def mkdir(self, path):
        from mpi4py import MPI
        myid = MPI.COMM_WORLD.rank
        if myid == 0:
            if not os.path.exists(path):
                os.mkdir(path)
        else:
            pass
        MPI.COMM_WORLD.Barrier()

    def symlink(self, target, link):
        from mpi4py import MPI
        myid = MPI.COMM_WORLD.rank
        if myid == 0:
            if not os.path.exists(link):
                os.symlink(target, link)
        else:
            pass
        # MPI.COMM_WORLD.Barrier()

    def open_file(self, *args, **kwargs):
        from mpi4py import MPI
        myid = MPI.COMM_WORLD.rank
        if myid == 0:
            return open(*args, **kwargs)
        return

    def cleancwd(self):
        from mpi4py import MPI
        myid = MPI.COMM_WORLD.rank
        if myid == 0:
            for f in os.listdir("."):
                os.remove(f)
        else:
            pass
        MPI.COMM_WORLD.Barrier()

    def remove_solfiles(self):
        from mpi4py import MPI
        myid = MPI.COMM_WORLD.rank
        if myid == 0:
            super(ParallelEngine, self).remove_solfiles()
        else:
            pass
        MPI.COMM_WORLD.Barrier()

    def remove_case_dirs(self):
        from mpi4py import MPI
        myid = MPI.COMM_WORLD.rank
        if myid == 0:
            super(ParallelEngine, self).remove_case_dirs()
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
        if not a._finalized:
            a.Finalize()
            a._finalized = True

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

    def X2x(self, X, x):  # RecoverFEMSolution
        fes = x.ParFESpace()
        P = fes.GetProlongationMatrix()
        x.SetSize(P.Height())
        P.Mult(X, x)

    def t1_2_T1(self, t1, x):
        '''
        Extra is already restricted (LF2PyVec/BF2PyMat calls ParallelAssemble)
        '''
        return t1

    def t2_2_T2(self, t2, x):
        '''
        Extra is already restricted (LF2PyVec/BF2PyMat calls ParallelAssemble)
        '''
        return t2

    def run_geom_gen(self, gen):
        from mpi4py import MPI
        myid = MPI.COMM_WORLD.rank
        if myid == 0:
            gen.generate_final_geometry()
        else:
            pass
        MPI.COMM_WORLD.Barrier()

    def run_mesh_gen(self, gen):
        '''
        run mesh generator 
        '''
        gen.generate_mesh_file()

    def save_processed_model(self):
        from mpi4py import MPI
        myid = MPI.COMM_WORLD.rank
        if myid == 0:
            self.model.save_to_file(
                'model_proc.pmfm', meshfile_relativepath=False)
        else:
            pass
        MPI.COMM_WORLD.Barrier()

    def fill_empty_diag(self, A):
        '''
        A is CHypre (complex is supported only when imaginary is zero)
        '''
        from mpi4py import MPI
        if A[0] is None:
            return

        nnz0, tnnz0 = A[0].get_local_true_nnz()
        tnnz0 = np.sum(MPI.COMM_WORLD.allgather(tnnz0))

        if A[1] is None:
            if tnnz0 == 0:
                dprint1(
                    "!!! skipping fill_empty_diag: this diagnal block is compltely zero")
                return
            A[0].EliminateZeroRows()
        else:

            nnz, tnnz = A[1].get_local_true_nnz()
            tnnz = np.sum(MPI.COMM_WORLD.allgather(tnnz))
            if tnnz == 0:
                if tnnz0 == 0:
                    dprint1(
                        "!!! skipping fill_empty_diag: this diagnal block is compltely zero")
                    return
                A[0].EliminateZeroRows()
