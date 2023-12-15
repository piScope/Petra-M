'''

    Model Tree to stroe MFEM model parameters

'''
from petram.utils import (check_cluster_access,
                          check_addon_access)
from petram.namespace_mixin import NS_mixin
import numpy as np
from petram.model import Model

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('MFEMModel')


has_addon_access = check_addon_access()
has_cluster_access = check_cluster_access()


class MFEM_GeneralRoot(Model, NS_mixin):
    can_delete = False
    has_2nd_panel = True

    def __init__(self, *args, **kwargs):
        Model.__init__(self, *args, **kwargs)
        NS_mixin.__init__(self, *args, **kwargs)

    def get_info_str(self):
        return NS_mixin.get_info_str(self)

    def attribute_set(self, v):
        v['debug_level'] = 1
        v['dwc_object_name'] = ''
        v['mesh_gen'] = ''
        v['geom_gen'] = ''
        v['diagpolicy'] = 'one'
        v['partitioning'] = 'auto'
        v['submeshpartitioning'] = 'auto'
        v['autofilldiag'] = 'off'
        v['savegz'] = 'on'
        v['allow_fallback_nonjit'] = 'allow'
        v['debug_numba_jit'] = 'off'
        v['trim_debug_print'] = 'on'
        v['warning_control'] = 'once'
        super(MFEM_GeneralRoot, self).attribute_set(v)
        return v

    def panel1_param(self):
        txt = ["0-3: Larger number prints more info.",
               "Negative number: print from all nodes",
               "Speical debug bits",
               " 4: write essentail BC vector",
               " 8: memory check"]
        txt2 = "DWC (direct wrapper call) is for low level API access"
        return [["debug level", self.debug_level, 400, {}],
                ["", "\n".join(txt), 2, None],
                ["geom generator", self.geom_gen, 0, {}],
                ["mesh generator", self.mesh_gen, 0, {}],
                ["DWC object", "", 0, {}],
                ["", txt2, 2, None], ]

    def get_panel1_value(self):
        return (self.debug_level, None,
                self.geom_gen, self.mesh_gen, self.dwc_object_name, None)

    def import_panel1_value(self, v):
        self.debug_level = v[0]
        self.geom_gen = v[2]
        self.mesh_gen = v[3]
        self.dwc_object_name = str(v[4])
        import petram.debug
        petram.debug.debug_default_level = int(self.debug_level)

    def panel2_tabname(self):
        return "Extra."

    def panel2_param(self):
        return [["DiagPolicy", None, 1, {"values": ["one", "keep"]}],
                ["File compression", None, 1, {"values": ["on", "off"]}],
                ["Mesh partitioning", None, 1, {
                    "values": ["auto", "by attribute"]}],
                ["SubMesh partitioning", None, 1, {
                    "values": ["auto", "safe"]}],
                ["Autofill emtpy diag rows", None,
                    1, {"values": ["on", "off"]}],
                ["Fallback Python coefficient", None,
                    1, {"values": ["allow", "warn", "error", "always use Python coeff."]}],
                ["Check numba JIT process", None,
                    1, {"values": ["on", "off"]}],
                ["Trim debug print text", None,
                    1, {"values": ["on", "off"]}],
                ["Warning control", None,
                    1, {"values": ["default", "error", "ignore", "always",
                                   "module", "once"]}],
                ]

    def get_panel2_value(self):
        return (self.diagpolicy, self.savegz, self.partitioning, self.submeshpartitioning,
                self.autofilldiag, self.allow_fallback_nonjit, self.debug_numba_jit,
                self.trim_debug_print, self.warning_control)

    def import_panel2_value(self, v):
        self.diagpolicy = v[0]
        self.savegz = v[1]
        self.partitioning = v[2]
        self.submeshpartitioning = v[3]
        self.autofilldiag = v[4]
        self.allow_fallback_nonjit = v[5]
        self.debug_numba_jit = v[6]
        self.trim_debug_print = v[7]
        self.warning_control = v[8]

    def run(self):
        import petram.debug
        if petram.debug.debug_default_level == 0:
            petram.debug.debug_default_level = int(self.debug_level)

        petram.debug.trim_debug_print = bool(self.trim_debug_print == 'on')

        if not hasattr(self.root(), "_variables"):
            from petram.helper.variables import Variables
            self.root()._variables = Variables()

        if not hasattr(self, "warning_control"):
            self.warning_control = 'once'

        self.root()._parameters = {}
        self.root()._init_done = True

    def get_defualt_local_ns(self):
        '''
        GeneralRoot Namelist knows basic functions
        '''
        import petram.helper.functions
        return petram.helper.functions.f.copy()

    def save_attribute_set(self, skip_def_check):
        ret = Model.save_attribute_set(self, skip_def_check)
        return [x for x in ret if x != '_variable']


class MFEM_PhysRoot(Model):
    can_delete = False
    has_2nd_panel = False

    def get_possible_child(self):
        ans = []
        from petram.helper.phys_module_util import all_phys_models
        models, classes = all_phys_models()

        tmp = sorted([(cls.__name__, cls) for cls in classes])
        classes = [x[1] for x in tmp]

        return classes

    def get_possible_child_menu(self):
        '''
        return hierachial menus
        '''
        ans = []
        from petram.helper.phys_module_util import all_phys_models
        models, classes = all_phys_models()

        name_class = sorted([(cls.__name__, cls) for cls in classes])

        import wx
        petram_model = wx.GetApp().TopWindow.proj.setting.parameters.eval('PetraM')
        mesh = petram_model.variables.eval('mesh')
        if mesh is None:
            return []

        sdim = mesh.SpaceDimension()

        allkeys = ['3D', '2D', '2Da', '1D']
        if sdim == 3:
            keys = ['3D']
        elif sdim == 2:
            keys = ['2D', '2Da']
        else:
            keys = ['1D']

        menus = []

        for n, cls in name_class:
            hitkey = ''
            for k in allkeys:
                if n.find(k) != -1:
                    hitkey = k
                    break
            if hitkey == '':
                menus.append(("", cls))
            else:
                if hitkey in keys:
                    menus.append(("", cls))

        return menus

    def make_solvars(self, solsets, g=None):
        from petram.mesh.mesh_utils import (get_extended_connectivity,
                                            get_reverse_connectivity)
        from petram.helper.variables import Variable
        from petram.helper.variables import add_scalar

        for phys in self.iter_enabled():
            for mm in phys.walk_enabled():
                mm.compile_coeffs()

        solvars = [None] * len(solsets)
        if g is None:
            g = {}
        for k, v in enumerate(solsets):
            mesh, soldict = v

            get_extended_connectivity(mesh[0])
            get_reverse_connectivity(mesh[0])

            solvar = g.copy()
            for phys in self.iter_enabled():
                phys.soldict_to_solvars(soldict, solvar)

            for var in solvar.values():
                if isinstance(var, Variable):
                    var.add_topological_info(mesh[0])

            solvars[k] = solvar

            if 'minSJac_' + str(k) in soldict:
                add_scalar(solvar, "minSJac", "", ['x', 'y', 'z'],
                           soldict['minSJac_' + str(k)][0], None)
        # print "solvars", solvars
        return solvars

    def onItemSelChanged(self, evt):
        '''
        GUI response when model object is selected in
        the dlg_edit_model
        '''
        viewer = evt.GetEventObject().GetTopLevelParent().GetParent()
        viewer.set_view_mode('phys')

    def dependent_values(self, include_disabled=False):
        '''
        return dependent_values
           names: name of values
           pnames: list of physics module
           pindex: index of dependent value in the physics module
        '''
        if include_disabled:
            method = self.get_children
        else:
            method = self.iter_enabled

        names = sum([c.dep_vars for c in method()], [])
        pnames = sum([[c.name()] * len(c.dep_vars)
                      for c in method()], [])
        pindex = sum([list(range(len(c.dep_vars)))
                      for c in method()], [])

        return names, pnames, pindex

    '''
    def get_num_matrix(self, get_matrix_weight, phys_target = None):
        # get_matrix_weight: solver method to evaulate matrix weight
        if phys_target is None:
             phys_target = [self[k] for k in self]

        num_matrix = 0
        for phys in phys_target:
            for mm in phys.walk():
                if not mm.enabled: continue
                mm.set_matrix_weight(get_matrix_weight)

                wt = np.array(mm.get_matrix_weight())
                tmp = int(np.max((wt != 0)*(np.arange(len(wt))+1)))
                num_matrix = max(tmp, num_matrix)
        return num_matrix
    '''

    def get_num_matrix(self, get_matrix_weight, phys_target=None):
        import traceback
        traceback.print_stack()
        assert False, "this should not be called"

    def all_dependent_vars(self, num_matrix, phys_target, phys_range):
        '''
        FES variable + extra variable
        '''
        dep_vars = []
        isFesvars_g = []

        phys_target = phys_target if phys_target is not None else [
            self[k] for k in self]

        for phys in phys_target:
            # if not phys.enabled: continue
            dv = phys.dep_vars
            dep_vars.extend(dv)
            extra_vars = []
            for mm in phys.walk():
                if not mm.enabled:
                    continue
                for j in range(num_matrix):
                    for k in range(len(dv)):
                        for phys2 in phys_range:
                            # if not phys2.enabled: continue
                            if not mm.has_extra_DoF2(k, phys2, j):
                                continue
                            name = mm.extra_DoF_name2(k)
                            if not name in extra_vars:
                                extra_vars.append(name)

            dep_vars.extend(extra_vars)
        return dep_vars


class MFEM_PostProcessRoot(Model):
    can_delete = False
    has_2nd_panel = False

    def get_possible_child(self):
        from petram.postprocess.pp_model import PostProcess
        return [PostProcess]

    def onItemSelChanged(self, evt):
        '''
        GUI response when model object is selected in
        the dlg_edit_model
        '''
        viewer = evt.GetEventObject().GetTopLevelParent().GetParent()
        viewer.set_view_mode('phys', self)

    def is_viewmode_grouphead(self):
        return True

    def add_solvars(self, solsets, solvars):
        for k, v in enumerate(solsets):
            mesh, soldict = v
            solvar = solvars[k]
            for p in self.iter_enabled():
                for pp in p.iter_enabled():
                    pp.soldict_to_solvars(soldict, solvar)
            #solvars[k] = solvar
        return solvars


class MFEM_InitRoot(Model):
    can_delete = False
    has_2nd_panel = False

    def get_possible_child(self):
        from petram.init_model import InitSetting
        return [InitSetting]

    def onItemSelChanged(self, evt):
        '''
        GUI response when model object is selected in
        the dlg_edit_model
        '''
        viewer = evt.GetEventObject().GetTopLevelParent().GetParent()
        viewer.set_view_mode('phys', self)

    def is_viewmode_grouphead(self):
        return True


class MFEM_GeomRoot(Model):
    can_delete = False
    has_2nd_panel = False

    def get_possible_child(self):
        ret = []

        try:
            from petram.geom.occ_geom_model import OCCGeom
            ret.append(OCCGeom)
        except ImportError:
            pass

        try:
            from petram.geom.gmsh_geom_model import GmshGeom
            ret.append(GmshGeom)
        except ImportError:
            pass

        return ret


class MFEM_MeshRoot(Model):
    can_delete = False
    has_2nd_panel = False

    def get_possible_child(self):
        from petram.mesh.mesh_model import MFEMMesh, MeshGroup
        try:
            from petram.mesh.gmsh_mesh_model import GmshMesh
            return [MFEMMesh, GmshMesh]
        except BaseException:
            return [MFEMMesh]

    def onItemSelChanged(self, evt):
        '''
        GUI response when model object is selected in
        the dlg_edit_model
        '''
        viewer = evt.GetEventObject().GetTopLevelParent().GetParent()
        viewer.set_view_mode('mesh')

    @property
    def sdim(self):
        from petram.mesh.mesh_model import MeshGroup
        for obj in self.walk():
            if isinstance(obj, MeshGroup) and obj.enabled:
                return obj.sdim
        return 1


class MFEM_SolverRoot(Model):
    can_delete = False
    has_2nd_panel = False

    def get_possible_child(self):
        from petram.solver.solver_model import SolveStep
        from petram.solver.parametric import Parametric
        #from petram.solver.solve_loop import Loop
        from petram.solver.solver_controls import ForLoop, DWCCall
        return [SolveStep, Parametric, ForLoop, DWCCall]

    def get_active_solvers(self, mm=None):
        return [x for x in self.iter_enabled()]

    def onItemSelChanged(self, evt):
        '''
        GUI response when model object is selected in
        the dlg_edit_model
        '''
        viewer = evt.GetEventObject().GetTopLevelParent().GetParent()
        viewer.set_view_mode('phys', self)

    def is_viewmode_grouphead(self):
        return True

    def get_phys(self):
        phys_root = self.root()['Phys']
        ret = []
        for k in self.keys():
            for x in self[k].get_target_phys():
                if not x in ret:
                    ret.append(x)
        return ret

    def get_special_menu(self, evt):
        return [["+Run...", None, None, ],
                ["Serial", self.run_serial, None, ],
                ["Parallel", self.run_parallel, None, ],
                ["Cluster...", self.run_cluster, None, ],
                ["!", None, None, ], ]

    def run_serial(self, evt):
        evt.GetEventObject().GetParent().onSerDriver(evt)

    def run_parallel(self, evt):
        evt.GetEventObject().GetParent().onParDriver(evt)

    def run_cluster(self, evt):
        evt.GetEventObject().GetParent().onServerSolve(evt)


try:
    from petram.geom.geom_model import GeomBase
    has_geom = True
except BaseException:
    import traceback
    traceback.print_exc()
    has_geom = False


class MFEM_ModelRoot(Model):
    def __init__(self, **kwargs):
        super(MFEM_ModelRoot, self).__init__(**kwargs)
        self['General'] = MFEM_GeneralRoot()

        if has_geom:
            self['Geometry'] = MFEM_GeomRoot()
        self['Mesh'] = MFEM_MeshRoot()
        self['Phys'] = MFEM_PhysRoot()
        self['InitialValue'] = MFEM_InitRoot()
        self['PostProcess'] = MFEM_PostProcessRoot()
        self['Solver'] = MFEM_SolverRoot()

        from petram.helper.variables import Variables
        self._variables = Variables()
        self._parameters = {}

    def attribute_set(self, v):
        from petram.helper.variables import Variables
        v['_variables'] = Variables()
        v['_parameters'] = {}
        v['enabled'] = True
        v['root_path'] = ''
        v['model_path'] = ''
        return v

    def set_root_path(self, path):
        self.root_path = path

    def get_root_path(self):
        return self.root_path

    def save_setting(self, filename=''):
        fid = open(fiilename, 'w')
        for od in self.walk():
            od.write_setting(fid)
        fid.close()

    def save_to_file(self, path, meshfile_relativepath=False):
        import petram.helper.pickle_wrapper as pickle

        if meshfile_relativepath:
            for od in self.walk():
                if hasattr(od, 'use_relative_path'):
                    od.use_relative_path()
        fid = open(path, 'wb')
        pickle.dump(self, fid)
        if meshfile_relativepath:
            for od in self.walk():
                if hasattr(od, 'use_relative_path'):
                    od.restore_fullpath()
        fid.close()
