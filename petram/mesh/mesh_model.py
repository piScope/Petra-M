import petram.debug
from petram.mfem_config import use_parallel
from petram.namespace_mixin import NS_mixin
from petram.model import Model
import os
import numpy as np
import mfem
from abc import abstractmethod

PyMFEM_PATH = os.path.dirname(os.path.dirname(mfem.__file__))
PetraM_PATH = os.getenv("PetraM")
HOME = os.path.expanduser("~")


if use_parallel:
    import mfem.par as mfem
    from mpi4py import MPI
    num_proc = MPI.COMM_WORLD.size
    myid = MPI.COMM_WORLD.rank
    from petram.helper.mpi_recipes import *
else:
    import mfem.ser as mfem

dprint1, dprint2, dprint3 = petram.debug.init_dprints('MeshModel')


class Mesh(Model, NS_mixin):
    isMeshGenerator = False
    isRefinement = False         # refinement performed in either serial/parallel
    isSerialRefinement = False   # refinement performed in serial

    def __init__(self, *args, **kwargs):
        super(Mesh, self).__init__(*args, **kwargs)
        NS_mixin.__init__(self, *args, **kwargs)

    def onItemSelChanged(self, evt):
        '''
        GUI response when model object is selected in
        the dlg_edit_model
        '''
        viewer = evt.GetEventObject().GetTopLevelParent().GetParent()
        viewer.set_view_mode('phys', self)

    def get_mesh_root(self):
        from petram.mfem_model import MFEM_MeshRoot

        p = self.parent
        while p is not None:
            if isinstance(p, MFEM_MeshRoot):
                return p
            p = p.parent


class MeshGenerator(Mesh):
    isMeshGenerator = True
    isRefinement = False
    isSerialRefinement = False   # refinement performed in serial

    def attribute_set(self, v):
        v = super(MeshGenerator, self).attribute_set(v)
        v['enforce_ncmesh'] = False
        return v

    def panel1_param(self):
        panels = super(MeshGenerator, self).panel1_param()

        p1 = [None, False, 3, {"text": "EnforceNCMesh"}]

        panels.append(p1)
        return panels

    def get_panel1_value(self):
        values = super(MeshGenerator, self).get_panel1_value()
        values.append(self.enforce_ncmesh)
        return values

    def import_panel1_value(self, v):
        super(MeshGenerator, self).import_panel1_value(v[:-1])
        self.enforce_ncmesh = v[-1]

    def run_serial(self, mesh=None):
        # By default this will call run. Sub-classes can re-implement this.
        m = self.run(mesh=mesh)
        return m

    @abstractmethod
    def run(self, mesh=None):
        pass


class MFEMMesh(Model):
    can_delete = True
    has_2nd_panel = False
    isMeshGroup = True

    def get_possible_child(self):
        try:
            from petram.mesh.pumimesh_model import PumiMesh
            return [MeshFile, PumiMesh, Mesh1D, Mesh2D, Mesh3D,
                    UniformRefinement, DomainRefinement, BoundaryRefinement, Scale]
        except BaseException:
            return [MeshFile, Mesh1D, Mesh2D, Mesh3D, UniformRefinement,
                    DomainRefinement, BoundaryRefinement, Scale]

    def get_possible_child_menu(self):
        try:
            from petram.mesh.pumimesh_model import PumiMesh
            return [("", MeshFile),
                    ("Other Meshes", Mesh1D),
                    ("", Mesh2D),
                    ("", Mesh3D),
                    ("!", PumiMesh),
                    ("", Scale),
                    ("Refinement...", UniformRefinement),
                    ("", DomainRefinement),
                    ("!", BoundaryRefinement)]
        except BaseException:
            return [("", MeshFile),
                    ("Other Meshes", Mesh1D),
                    ("", Mesh2D),
                    ("!", Mesh3D),
                    ("", Scale),
                    ("Refinement...", UniformRefinement),
                    ("", DomainRefinement),
                    ("!", BoundaryRefinement)]

    def panel1_param(self):
        if not hasattr(self, "_topo_check_char"):
            self._topo_check_char = "\n".join([' '*15, ' '*15, ' '*15, ' '*15])
            self._invalid_data = None
        import wx
        return [[None, None, 341, {"label": "Reload mesh",
                                   "func": 'call_reload_mfem_mesh',
                                   "noexpand": True}],
                [None, None, 341, {"label": "Check topology",
                                   "func": 'check_topology',
                                   "noexpand": True}],
                [None, self._topo_check_char, 2, None], ]

    def get_panel1_value(self):
        if not hasattr(self, "_topo_check_char"):
            self._topo_check_char = ''
            self._invalid_data = None

        return [self, self, self._topo_check_char]

    def import_panel1_value(self, v):
        pass

    def onItemSelChanged(self, evt):
        '''
        GUI response when model object is selected in
        the dlg_edit_model
        '''
        viewer = evt.GetEventObject().GetTopLevelParent().GetParent()
        viewer.set_view_mode('phys', self)

    def is_viewmode_grouphead(self):
        return True

    def figure_data_name(self):
        return 'mfem'

    def get_special_menu(self, evt):
        # menu =[["Reload Mesh", self.reload_mfem_mesh, None,],]
        menu = [["+Mesh parameters...", None, None],
                ["Plot low quality elements", self.plot_lowqualities, None], ]

        if (self._invalid_data is not None and
                len(self._invalid_data[0]) > 0):
            menu.append(["Plot invalid faces", self.plot_invalids, None])
        if (self._invalid_data is not None and
                len(self._invalid_data[2]) > 0):
            menu.append(["Plot inverted elements", self.plot_inverted, None])

        menu.append(["Compute minSJac", self.compute_scaled_jac, None])
        menu.append(["!", None, None])
        return menu

    def reload_mfem_mesh(self, evt):
        evt.GetEventObject().GetParent().onLoadMesh(evt)

    def call_reload_mfem_mesh(self, evt):
        editor = evt.GetEventObject().GetTopLevelParent()
        viewer = editor.GetParent()

        viewer.onLoadMesh(evt)
        self._topo_check_char = ''
        self._invalid_data = None
        editor.import_selected_panel_value()

    def check_topology(self, evt):
        from petram.mesh.mesh_inspect import (find_invalid_topology,
                                              format_error)

        editor = evt.GetEventObject().GetTopLevelParent()
        dlg = editor.show_progress_bar('Checking topology...')

        try:
            viewer = editor.GetParent()
            mesh = viewer.model.variables.getvar('mesh')
            if mesh is None:
                out = 'Mesh is not loaded'
            else:
                invalids, invalid_attrs, inverted, sj_min_max = find_invalid_topology(
                    mesh)
                if sj_min_max[0] < 0:
                    out = "\n".join(["Some elements are inverted",
                                     "min(ScaledJac) = " + str(sj_min_max[0]),
                                     "max(ScaledJac) = " + str(sj_min_max[1]), ])
                elif len(invalids) == 0:
                    out = "\n".join(["No error",
                                     "min(ScaledJac) = " + str(sj_min_max[0]),
                                     "max(ScaledJac) = " + str(sj_min_max[1]), ])
                else:
                    out = format_error(invalids, invalid_attrs, inverted)
                self._invalid_data = invalids, invalid_attrs, inverted, sj_min_max
            self._topo_check_char = out

            import wx
            wx.CallAfter(editor.import_selected_panel_value)
        except BaseException:
            import traceback
            traceback.print_exc()
        finally:
            dlg.Destroy()
            evt.Skip()

    def compute_scaled_jac(self, evt):
        from petram.mesh.mesh_inspect import get_scaled_jacobian

        editor = evt.GetEventObject().GetTopLevelParent()
        dlg = editor.show_progress_bar(
            'Computing minimum scaled Jacobian (minSJac)...')

        try:
            viewer = editor.GetParent()
            mesh = viewer.model.variables.getvar('mesh')
            sj = get_scaled_jacobian(mesh, sd=-1)
            fec = mfem.L2_FECollection(0, mesh.Dimension())
            fes = mfem.FiniteElementSpace(mesh, fec)
            vec = mfem.Vector(sj)
            gf = mfem.GridFunction(fes, vec.GetData())

        except BaseException:
            import traceback
            traceback.print_exc()
        finally:
            dlg.Destroy()
            evt.Skip()

        folder = viewer.model.param.eval('sol')
        cwd = os.getcwd()
        os.chdir(folder.owndir())
        viewer.engine.save_solfile_fespace('minSJac', 0, gf, None)
        os.chdir(cwd)

        # this triggers to reload solfile
        viewer.model.variables.setvar('solfiles', None)

    def plot_invalids(self, evt):
        from petram.mesh.mesh_inspect import plot_faces_containing_elements

        editor = evt.GetEventObject().GetTopLevelParent()
        viewer = editor.GetParent()
        mesh = viewer.model.variables.getvar('mesh')

        invalids = self._invalid_data[0]

        from ifigure.interactive import figure
        from petram.mfem_viewer import setup_figure
        win = figure()
        setup_figure(win)
        win.view('noclip')
        plot_faces_containing_elements(mesh, invalids, refine=10,
                                       win=win)

    def plot_inverted(self, evt):
        from petram.mesh.mesh_inspect import plot_elements

        editor = evt.GetEventObject().GetTopLevelParent()
        viewer = editor.GetParent()
        mesh = viewer.model.variables.getvar('mesh')

        inverted = self._invalid_data[2]

        from ifigure.interactive import figure
        from petram.mfem_viewer import setup_figure
        win = figure()
        setup_figure(win)
        win.view('noclip')
        plot_elements(mesh, inverted, refine=10, win=win)

    def plot_lowqualities(self, evt):
        from petram.mesh.mesh_inspect import plot_elements

        editor = evt.GetEventObject().GetTopLevelParent()
        viewer = editor.GetParent()

        mesh = viewer.model.variables.getvar('mesh')
        if mesh is None:
            return
        dim = mesh.Dimension()
        sdim = mesh.SpaceDimension()

        if sdim != dim:
            return

        import wx
        from ifigure.utils.edit_list import DialogEditList

        l = [["minimum kappa :", str(1000),  0, {'noexpand': True}], ]

        value = DialogEditList(l, parent=editor,
                               title="Enter mimimum kappa...",
                               style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        if not value[0]:
            return
        minkappa = float(value[1][0])

        J = mfem.DenseMatrix(sdim, dim)

        def GetElementJacobian(mesh, i):
            bgeom = mesh.GetElementBaseGeometry(i)
            T = mesh.GetElementTransformation(i)
            T.SetIntPoint(mfem.Geometries.GetCenter(bgeom))
            mfem.Geometries.JacToPerfJac(bgeom, T.Jacobian(), J)

        lowq_elements = []
        for i in range(mesh.GetNE()):
            GetElementJacobian(mesh, i)
            kappa = J.CalcSingularvalue(0) / J.CalcSingularvalue(dim-1)
            if kappa > minkappa:
                lowq_elements.append(i)

        print("number of low Q elements: " + str(len(lowq_elements)))
        from ifigure.interactive import figure
        from petram.mfem_viewer import setup_figure
        win = figure()
        setup_figure(win)
        win.view('noclip')
        plot_elements(mesh, lowq_elements, refine=10, win=win)

    @property
    def sdim(self):
        if not hasattr(self, '_sdim'):
            self._sdim = 1
        return self._sdim

    @sdim.setter
    def sdim(self, value):
        self._sdim = value


MeshGroup = MFEMMesh


def format_mesh_characteristic(mesh):
    h_min = mfem.doublep()
    h_max = mfem.doublep()
    kappa_min = mfem.doublep()
    kappa_max = mfem.doublep()
    Vh = mfem.Vector()
    Vk = mfem.Vector()
    mesh.GetCharacteristics(h_min, h_max, kappa_min, kappa_max, Vh, Vk)
    h_min = h_min.value()
    h_max = h_max.value()
    kappa_min = kappa_min.value()
    kappa_max = kappa_max.value()

    out = ["", "=== Mesh Statistics ==="]
    out.append("Dimension          : " + str(mesh.Dimension()))
    out.append("Space dimension    : " + str(mesh.SpaceDimension()))

    if mesh.Dimension() == 0:
        out.append("Number of vertices : " + str(mesh.GetNV()))
        out.append("Number of elements : " + str(mesh.GetNE()))
        out.append("Number of bdr elem : " + str(mesh.GetNBE()))
    elif mesh.Dimension() == 1:
        out.append("Number of vertices : " + str(mesh.GetNV()))
        out.append("Number of elements : " + str(mesh.GetNE()))
        out.append("Number of bdr elem : " + str(mesh.GetNBE()))
        out.append("h_min              : " + str(h_min))
        out.append("h_max              : " + str(h_max))
    elif mesh.Dimension() == 2:
        out.append("Number of vertices : " + str(mesh.GetNV()))
        out.append("Number of edges    : " + str(mesh.GetNEdges()))
        out.append("Number of elements : " + str(mesh.GetNE()))
        out.append("Number of bdr elem : " + str(mesh.GetNBE()))
        out.append("Euler Number       : " + str(mesh.EulerNumber2D()))
        out.append("h_min              : " + str(h_min))
        out.append("h_max              : " + str(h_max))
        out.append("kappa_min              : " + str(kappa_min))
        out.append("kappa_max              : " + str(kappa_max))
    elif mesh.Dimension() == 3:
        out.append("Number of vertices : " + str(mesh.GetNV()))
        out.append("Number of edges    : " + str(mesh.GetNEdges()))
        out.append("Number of faces    : " + str(mesh.GetNFaces()))
        out.append("Number of elements : " + str(mesh.GetNE()))
        out.append("Number of bdr elem : " + str(mesh.GetNBE()))
        out.append("Euler Number       : " + str(mesh.EulerNumber()))
        out.append("h_min              : " + str(h_min))
        out.append("h_max              : " + str(h_max))
        out.append("kappa_min              : " + str(kappa_min))
        out.append("kappa_max              : " + str(kappa_max))
    return '\n'.join(out)


class MeshFile(MeshGenerator):
    has_2nd_panel = False

    def __init__(self, parent=None, **kwargs):
        self.path = kwargs.pop("path", "")
        self.generate_edges = kwargs.pop("generate_edges", 1)
        self.refine = kwargs.pop("refien", 1)
        self.fix_orientation = kwargs.pop("fix_orientation", True)
        super(MeshFile, self).__init__(parent=parent, **kwargs)

    def __repr__(self):
        try:
            return 'MeshFile(' + self.path + ')'
        except BaseException:
            return 'MeshFile(!!!Error!!!)'

    def attribute_set(self, v):
        v = super(MeshFile, self).attribute_set(v)
        v['path'] = ''
        v['generate_edges'] = 1
        v['refine'] = True
        v['fix_orientation'] = True

        return v

    def panel1_param(self):
        if not hasattr(self, "_mesh_char"):
            self._mesh_char = ''
        wc = "ANY|*|MFEM|*.mesh|GMSH|*.gmsh"
        p1 = [["Path", self.path, 45, {'wildcard': wc}],
              ["",
               "note: ~ and environmental variables are expanded. \n    In addition, {petram}=$PetraM, {mfem}=PyMFEM, \n     {home}=~ ,{model}=project file dir.",
               2,
               None],
              [None, self.generate_edges == 1,
               3, {"text": "Generate edges"}],
              [None, self.refine == 1, 3, {"text": "Refine"}],
              [None, self.fix_orientation, 3, {"text": "FixOrientation"}],
              [None, self._mesh_char, 2, None], ]

        p2 = MeshGenerator.panel1_param(self)
        return p1[:-1] + p2 + p1[-1:]

    def get_panel1_value(self):
        v1 = [self.path, None, self.generate_edges,
              self.refine, self.fix_orientation, None]

        v2 = MeshGenerator.get_panel1_value(self)

        return v1[:-1] + v2 + v1[-1:]

    def import_panel1_value(self, v):
        self.path = str(v[0])
        self.generate_edges = 1 if v[2] else 0
        self.refine = 1 if v[3] else 0
        self.fix_orientation = v[4]

        MeshGenerator.import_panel1_value(self, v[4:-1])

    def use_relative_path(self):
        self._path_bk = self.path

        try:
            self.path = os.path.basename(self.get_real_path())
        except AssertionError as error:
            if error.args[0].startswith("Mesh file does not exist :"):
                pass
        except BaseException:
            raise

    def restore_fullpath(self):
        self.path = self._path_bk
        self._path_bk = ''

    def get_real_path(self):
        path = str(self.path)

        if path == '':
            # if path is empty, file is given by internal mesh generator.
            parent = self.get_mesh_root()
            for key in parent.keys():
                if not parent[key].is_enabled():
                    continue
                if hasattr(parent[key], 'get_meshfile_path'):
                    return parent[key].get_meshfile_path()
        import os
        path = os.path.expanduser(path)
        path = os.path.expandvars(path)

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
                if (not os.path.exists(path1) and "__main__" in globals()
                        and hasattr(__main__, '__file__')):
                    from __main__ import __file__ as mainfile
                    path1 = os.path.join(os.path.dirname(
                        os.path.realpath(mainfile)), path)
                    dprint1("trying :", path1)
                if not os.path.exists(path1) and os.getenv(
                        'PetraM_MeshDir') is not None:
                    path1 = os.path.join(os.getenv('PetraM_MeshDir'), path)
                    dprint1("trying :", path1)
            if os.path.exists(path1):
                path = path1
            else:
                assert False, "can not find mesh file from relative path: " + path
        return path

    def run(self, mesh=None):
        path = self.get_real_path()
        if not os.path.exists(path):
            print("mesh file does not exists : " + path + " in " + os.getcwd())
            return None
        args = (path, self.generate_edges, self.refine, self.fix_orientation)

        mesh = mfem.Mesh(*args)

        if self.enforce_ncmesh:
            mesh.EnsureNCMesh()

        self.parent.sdim = mesh.SpaceDimension()
        self._mesh_char = format_mesh_characteristic(mesh)
        try:
            mesh.GetNBE()
            return mesh
        except BaseException:
            return None


class Mesh1D(MeshGenerator):
    has_2nd_panel = False
    unique_child = True

    def attribute_set(self, v):
        v = super(Mesh1D, self).attribute_set(v)
        v['length'] = [1, ]
        v['nsegs'] = [100, ]
        v['length_txt'] = "1"
        v['nsegs_txt'] = "100"
        v['refine'] = 1
        v['fix_orientation'] = True
        v['mesh_x0_txt'] = "0.0"
        v['mesh_x0'] = 0.0
        return v

    def panel1_param(self):
        if not hasattr(self, "_mesh_char"):
            self._mesh_char = ''

        def check_int_array(txt, param, w):
            try:
                val = [int(x) for x in txt.split(',')]
                return True
            except BaseException:
                return False

        def check_float_array(txt, param, w):
            try:
                val = [float(x) for x in txt.split(',')]
                return True
            except BaseException:
                return False

        def check_float(txt, param, w):
            try:
                val = float(txt)
                return True
            except BaseException:
                return False

        p1 = [["Length", self.length_txt, 0, {"validator": check_float_array}],
              ["N segments", self.nsegs_txt, 0, {
                  "validator": check_int_array}],
              ["x0", self.mesh_x0_txt, 0, {"validator": check_float}],
              [None, "Note: use comma separated float/integer for a multisegments mesh", 2, {}],
              [None, self._mesh_char, 2, None], ]

        return p1

    def get_panel1_value(self):
        v1 = [self.length_txt, self.nsegs_txt, self.mesh_x0_txt, None, None]
        return v1

    def import_panel1_value(self, v):
        self.length_txt = str(v[0])
        self.nsegs_txt = str(v[1])
        self.mesh_x0_txt = str(v[2])

    def eval_strings(self):
        g = self._global_ns.copy()
        l = {}

        try:
            self.length = [float(eval(x, g, l))
                           for x in self.length_txt.split(',')]
            self.nsegs = [int(eval(x, g, l))
                          for x in self.nsegs_txt.split(',')]
            self.mesh_x0 = float(eval(self.mesh_x0_txt, g, l))
            return True
        except BaseException:
            import traceback
            traceback.print_exc()

        return False

    def run(self, mesh=None):

        from petram.mesh.make_simplemesh import straight_line_mesh

        success = self.eval_strings()
        assert success, "Conversion error of input parameter"

        mesh = straight_line_mesh(self.length, self.nsegs,
                                  filename='',
                                  refine=self.refine == 1,
                                  fix_orientation=self.fix_orientation,
                                  sdim=1, x0=self.mesh_x0)
        self.parent.sdim = mesh.SpaceDimension()
        self._mesh_char = format_mesh_characteristic(mesh)
        try:
            mesh.GetNBE()
            return mesh
        except BaseException:
            return None


class Mesh2D(MeshGenerator):
    isRefinement = False
    has_2nd_panel = False
    unique_child = True

    def attribute_set(self, v):
        v = super(Mesh2D, self).attribute_set(v)
        v['length'] = [1, ]
        v['nsegs'] = [100, ]
        v['xlength_txt'] = "1"
        v['ylength_txt'] = "1"
        v['xnsegs_txt'] = "30"
        v['ynsegs_txt'] = "20"
        v['refine'] = 1
        v['fix_orientation'] = True
        v['mesh_x0_txt'] = "0.0, 0.0"
        v['mesh_x0'] = (0.0, 0.0, )
        return v

    def panel1_param(self):
        if not hasattr(self, "_mesh_char"):
            self._mesh_char = ''

        def check_int_array(txt, param, w):
            try:
                val = [int(x) for x in txt.split(',')]
                return True
            except BaseException:
                return False

        def check_float_array(txt, param, w):
            try:
                val = [float(x) for x in txt.split(',')]
                return True
            except BaseException:
                return False

        def check_float(txt, param, w):
            try:
                val = float(txt)
                return True
            except BaseException:
                return False

        p1 = [["Length(x)", self.xlength_txt, 0, {"validator": check_float_array}],
              ["N segments(x)", self.xnsegs_txt, 0, {
                  "validator": check_int_array}],
              ["Length(y)", self.ylength_txt, 0, {
                  "validator": check_float_array}],
              ["N segments(y)", self.ynsegs_txt, 0, {
                  "validator": check_int_array}],
              ["x0", self.mesh_x0_txt, 0, {"validator": check_float_array}],
              [None, "Note: use comma separated float/integer for a multisegments mesh", 2, {}],
              [None, self._mesh_char, 2, None], ]

        p2 = MeshGenerator.panel1_param(self)
        return p1[:-2] + p2 + p1[-2:]

    def get_panel1_value(self):
        v1 = [self.xlength_txt, self.xnsegs_txt, self.ylength_txt, self.ynsegs_txt,
              self.mesh_x0_txt, None, None]
        v2 = MeshGenerator.get_panel1_value(self)

        return v1[:-2] + v2 + v1[-2:]

    def import_panel1_value(self, v):
        self.xlength_txt = str(v[0])
        self.xnsegs_txt = str(v[1])
        self.ylength_txt = str(v[2])
        self.ynsegs_txt = str(v[3])
        self.mesh_x0_txt = str(v[4])

        MeshGenerator.import_panel1_value(self, v[4:-2])

    def eval_strings(self):
        g = self._global_ns.copy()
        l = {}

        try:
            self.xlength = [float(eval(x, g, l))
                            for x in self.xlength_txt.split(',')]
            self.xnsegs = [int(eval(x, g, l))
                           for x in self.xnsegs_txt.split(',')]
            self.ylength = [float(eval(x, g, l))
                            for x in self.ylength_txt.split(',')]
            self.ynsegs = [int(eval(x, g, l))
                           for x in self.ynsegs_txt.split(',')]
            self.mesh_x0 = [float(eval(x, g, l))
                            for x in self.mesh_x0_txt.split(',')]
            return True
        except BaseException:
            import traceback
            traceback.print_exc()

        return False

    def run(self, mesh=None):

        from petram.mesh.make_simplemesh import quad_rectangle_mesh

        success = self.eval_strings()
        assert success, "Conversion error of input parameter"

        mesh = quad_rectangle_mesh(self.xlength, self.xnsegs, self.ylength, self.ynsegs,
                                   filename='', refine=self.refine == 1,
                                   fix_orientation=self.fix_orientation,
                                   sdim=2, x0=self.mesh_x0)

        if self.enforce_ncmesh:
            mesh.EnsureNCMesh()
        self.parent.sdim = mesh.SpaceDimension()
        self._mesh_char = format_mesh_characteristic(mesh)

        try:
            mesh.GetNBE()
            return mesh
        except BaseException:
            return None


class Mesh3D(MeshGenerator):
    has_2nd_panel = False
    unique_child = True

    def attribute_set(self, v):
        v = super(Mesh3D, self).attribute_set(v)
        v['length'] = [1, ]
        v['nsegs'] = [100, ]
        v['xlength_txt'] = "1"
        v['ylength_txt'] = "1"
        v['zlength_txt'] = "1"
        v['xnsegs_txt'] = "10"
        v['ynsegs_txt'] = "10"
        v['znsegs_txt'] = "10"
        v['refine'] = 1
        v['fix_orientation'] = True
        v['mesh_x0_txt'] = "0.0, 0.0, 0.0"
        v['mesh_x0'] = (0.0, 0.0, 0.0)
        return v

    def panel1_param(self):
        if not hasattr(self, "_mesh_char"):
            self._mesh_char = ''

        def check_int_array(txt, param, w):
            try:
                val = [int(x) for x in txt.split(',')]
                return True
            except BaseException:
                return False

        def check_float_array(txt, param, w):
            try:
                val = [float(x) for x in txt.split(',')]
                return True
            except BaseException:
                return False

        def check_float(txt, param, w):
            try:
                val = float(txt)
                return True
            except BaseException:
                return False

        p1 = [["Length(x)", self.xlength_txt, 0, {"validator": check_float_array}],
              ["N segments(x)", self.xnsegs_txt, 0, {
                  "validator": check_int_array}],
              ["Length(y)", self.ylength_txt, 0, {
                  "validator": check_float_array}],
              ["N segments(y)", self.ynsegs_txt, 0, {
                  "validator": check_int_array}],
              ["Length(z)", self.zlength_txt, 0, {
                  "validator": check_float_array}],
              ["N segments(z)", self.znsegs_txt, 0, {
                  "validator": check_int_array}],
              ["x0", self.mesh_x0_txt, 0, {"validator": check_float_array}],
              [None, "Note: use comma separated float/integer for a multisegments mesh", 2, {}],
              [None, self._mesh_char, 2, None], ]

        p2 = MeshGenerator.panel1_param(self)

        return p1[:-2] + p2 + p1[-2:]

    def get_panel1_value(self):
        v1 = [self.xlength_txt, self.xnsegs_txt, self.ylength_txt, self.ynsegs_txt,
              self.zlength_txt, self.znsegs_txt, self.mesh_x0_txt, None, None]
        v2 = MeshGenerator.get_panel1_value(self)

        return v1[:-2] + v2 + v1[-2:]

    def import_panel1_value(self, v):
        self.xlength_txt = str(v[0])
        self.xnsegs_txt = str(v[1])
        self.ylength_txt = str(v[2])
        self.ynsegs_txt = str(v[3])
        self.zlength_txt = str(v[4])
        self.znsegs_txt = str(v[5])
        self.mesh_x0_txt = str(v[6])

        MeshGenerator.import_panel1_value(self, v[6:-2])

    def eval_strings(self):
        g = self._global_ns.copy()
        l = {}

        try:
            self.xlength = [float(eval(x, g, l))
                            for x in self.xlength_txt.split(',')]
            self.xnsegs = [int(eval(x, g, l))
                           for x in self.xnsegs_txt.split(',')]
            self.ylength = [float(eval(x, g, l))
                            for x in self.ylength_txt.split(',')]
            self.ynsegs = [int(eval(x, g, l))
                           for x in self.ynsegs_txt.split(',')]
            self.zlength = [float(eval(x, g, l))
                            for x in self.zlength_txt.split(',')]
            self.znsegs = [int(eval(x, g, l))
                           for x in self.znsegs_txt.split(',')]
            self.mesh_x0 = [float(eval(x, g, l))
                            for x in self.mesh_x0_txt.split(',')]
            return True
        except BaseException:
            import traceback
            traceback.print_exc()

        return False

    def run(self, mesh=None):
        from petram.mesh.make_simplemesh import hex_box_mesh

        success = self.eval_strings()
        assert success, "Conversion error of input parameter"

        mesh = hex_box_mesh(self.xlength, self.xnsegs, self.ylength, self.ynsegs, self.zlength, self.znsegs,
                            filename='', refine=self.refine == 1, fix_orientation=self.fix_orientation,
                            sdim=3, x0=self.mesh_x0)

        if self.enforce_ncmesh:
            mesh.EnsureNCMesh()

        self.parent.sdim = mesh.SpaceDimension()
        self._mesh_char = format_mesh_characteristic(mesh)

        try:
            mesh.GetNBE()
            return mesh
        except BaseException:
            return None


class UniformRefinement(Mesh):
    isRefinement = True
    has_2nd_panel = False

    def __init__(self, parent=None, **kwargs):
        self.num_refine = kwargs.pop("num_refine", "0")
        super(UniformRefinement, self).__init__(parent=parent, **kwargs)

    def __repr__(self):
        try:
            return 'MeshUniformRefinement(' + self.num_refine + ')'
        except BaseException:
            return 'MeshUniformRefinement(!!!Error!!!)'

    def attribute_set(self, v):
        v = super(UniformRefinement, self).attribute_set(v)
        v['num_refine'] = '0'
        return v

    def panel1_param(self):
        return [["Number", str(self.num_refine), 0, {}], ]

    def import_panel1_value(self, v):
        self.num_refine = str(v[0])

    def get_panel1_value(self):
        return (str(self.num_refine),)

    def run(self, mesh):
        gtype = np.unique([mesh.GetElementBaseGeometry(i)
                           for i in range(mesh.GetNE())])
        if use_parallel:
            from mpi4py import MPI
            gtype = gtype.astype(np.int32)
            gtype = np.unique(allgather_vector(gtype, MPI.INT))

        if len(gtype) > 1:
            dprint1(
                "(Warning) Element Geometry Type is mixed. Cannot perform UniformRefinement")
            return mesh
        for i in range(int(self.num_refine)):
            mesh.UniformRefinement()  # this is parallel refinement
        return mesh


class Scale(Mesh):
    def attribute_set(self, v):
        v = super(Scale, self).attribute_set(v)
        v['scale'] = '1.0, 1.0, 1.0'
        v['scale_ns'] = 'global'
        return v

    def panel1_param(self):
        return [["scale", self.scale, 0, {}, ],
                ["NS for expr.", self.scale_ns, 0, {}], ]

    def import_panel1_value(self, v):
        self.scale = str(v[0])
        self.scale_ns = str(v[1])

    def get_panel1_value(self):
        return (str(self.scale),
                str(self.scale_ns), )

    def run(self, mesh):
        if self.scale != '':
            code = compile(self.scale, '<string>', 'eval')
            names = list(code.co_names)
            ns_obj, ns_g = self.find_ns_by_name(self.scale_ns)

        ll = {}
        value = eval(code, ns_g, ll)

        sdim = mesh.SpaceDimension()
        # nicePrint("refining elements domain choice", domains)
        for v in mesh.GetVertexArray():
            for i in range(sdim):
                v[i] = v[i]*value[i]

        return mesh


class DomainRefinement(Mesh):
    isRefinement = True
    has_2nd_panel = False

    def __init__(self, parent=None, **kwargs):
        self.num_refine = kwargs.pop("num_refine", "0")
        self.expression = kwargs.pop("expression", "")
        super(DomainRefinement, self).__init__(parent=parent, **kwargs)

    def __repr__(self):
        try:
            return 'MeshUniformRefinement(' + self.num_refine + ')'
        except BaseException:
            return 'MeshUniformRefinement(!!!Error!!!)'

    def attribute_set(self, v):
        v = super(DomainRefinement, self).attribute_set(v)
        v['num_refine'] = '0'
        v['expression'] = ''
        v['expression_ns'] = 'global'
        return v

    def panel1_param(self):
        from petram.model import validate_sel

        return [["Number", str(self.num_refine), 0, {}],
                ["Domains", self.sel_index_txt, 0, {'changing_event': True,
                                                    'setfocus_event': True,
                                                    'validator': validate_sel,
                                                    'validator_param': self}, ],
                ["Expr.", self.expression, 0, {}, ],
                ["NS for expr.", self.expression_ns, 0, {}], ]

    def import_panel1_value(self, v):
        self.num_refine = str(v[0])
        self.sel_index_txt = str(v[1])
        self.expression = str(v[2])
        self.expression_ns = str(v[3])

    def get_panel1_value(self):
        return (str(self.num_refine),
                str(self.sel_index_txt),
                str(self.expression),
                str(self.expression_ns), )

    def run(self, mesh):
        gtype = np.unique([mesh.GetElementBaseGeometry(i)
                           for i in range(mesh.GetNE())])
        if use_parallel:
            from mpi4py import MPI
            gtype = gtype.astype(np.int32)
            gtype = np.unique(allgather_vector(gtype, MPI.INT))

        if len(gtype) > 1:
            dprint1(
                "(Warning) Element Geometry Type is mixed. Cannot perform UniformRefinement")
            return mesh

        domains = self.process_sel_index()
        if len(domains) == 0:
            return mesh

        if self.expression != '':
            code = compile(self.expression, '<string>', 'eval')
            names = list(code.co_names)
            ns_obj, ns_g = self.find_ns_by_name(self.expression_ns)

        v = mfem.Vector()
        coords = ['x', 'y', 'z']
        # nicePrint("refining elements domain choice", domains)
        for i in range(int(self.num_refine)):
            attr = mesh.GetAttributeArray()
            idx = list(np.where(np.in1d(attr, domains))[0])

            if self.expression != '':
                idx2 = []

                for ii in idx:
                    ll = {}
                    vv = mesh.GetElementCenterArray(ii)
                    for i in range(len(vv)):
                        ll[coords[i]] = vv[i]
                    if eval(code, ns_g, ll):
                        idx2.append(ii)

                idx = idx2
            # nicePrint("number of refined element: ", len(idx))
            idx0 = mfem.intArray(idx)
            mesh.GeneralRefinement(idx0)  # this is parallel refinement
        return mesh


class BoundaryRefinement(Mesh):
    isRefinement = True
    isSerialRefinement = True
    has_2nd_panel = False

    def __init__(self, parent=None, **kwargs):
        self.num_refine = kwargs.pop("num_refine", "0")
        self.num_layer = kwargs.pop("num_layer", "4")
        super(BoundaryRefinement, self).__init__(parent=parent, **kwargs)

    def __repr__(self):
        try:
            return 'MeshBoundaryRefinement(' + self.num_refine + ')'
        except BaseException:
            return 'MeshBoundaryRefinement(!!!Error!!!)'

    def attribute_set(self, v):
        v = super(BoundaryRefinement, self).attribute_set(v)
        v['num_refine'] = '0'
        v['num_layer'] = '4'
        return v

    def panel1_param(self):
        from petram.model import validate_sel

        return [["Number", str(self.num_refine), 0, {}],
                ["Boundaries", self.sel_index_txt, 0, {'changing_event': True,
                                                       'setfocus_event': True,
                                                       'validator': validate_sel,
                                                       'validator_param': self}, ],
                ["#Layers", self.num_layer, 0, {}], ]

    def import_panel1_value(self, v):
        self.num_refine = str(v[0])
        self.sel_index_txt = str(v[1])
        self.num_layer = str(v[2])

    def get_panel1_value(self):
        return (str(self.num_refine),
                str(self.sel_index_txt),
                str(self.num_layer),)

    def run(self, mesh):

        from petram.helper.boundary_refinement import apply_boundary_refinement

        nlayers = int(self.num_layer)
        sels = self.process_sel_index()

        ne0 = mesh.GetNE()

        krefine = int(self.num_refine)
        for i in range(krefine):
            mesh = apply_boundary_refinement(mesh, sels, nlayers=nlayers)

        ne1 = mesh.GetNE()
        dprint1("Number of element before/after boundary refinementmesh: " +
                str(ne0) + " -->> " + str(ne1))
        return mesh
