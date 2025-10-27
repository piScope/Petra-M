from __future__ import print_function
from petram.mfem_config import use_parallel
from petram.phys.vtable import VtableElement, Vtable, Vtable_mixin
from petram.mesh.mesh_model import Mesh

import tempfile
import os
import subprocess
import tempfile
import weakref
import numpy as np
import traceback
import time

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('GmshMeshModel')


if use_parallel:
    import mfem.par as mfem
    from mpi4py import MPI
    num_proc = MPI.COMM_WORLD.size
    myid = MPI.COMM_WORLD.rank
    from petram.helper.mpi_recipes import *
else:
    import mfem.ser as mfem
    myid = 0


class GMesh(Mesh):
    def onItemSelChanged(self, evt):
        '''
        GUI response when model object is selected in
        the dlg_edit_model
        '''
        geom_root = self.geom_root
        if not geom_root.is_finalized:
            geom_root.onBuildAll(evt)
        if geom_root.is_finalized:
            if geom_root.geom_timestamp != self.geom_timestamp:
                self.onClearMesh(evt)
                self.geom_timestamp = geom_root.geom_timestamp
                evt.Skip()
                return
        viewer = evt.GetEventObject().GetTopLevelParent().GetParent()
        viewer.set_view_mode('mesh', self)
        evt.Skip()

    @property
    def geom_timestamp(self):
        return self.parent.geom_timestamp

    @geom_timestamp.setter
    def geom_timestamp(self, value):
        self.parent.geom_timestamp = value


class GMeshTop(Mesh):
    def attribute_set(self, v):
        v = super(GMeshTop, self).attribute_set(v)
        v['geom_timestamp'] = -1
        return v

    def onItemSelChanged(self, evt):
        '''
        GUI response when model object is selected in
        the dlg_edit_model
        '''
        geom_root = self.geom_root
        if not geom_root.is_finalized:
            geom_root.onBuildAll(evt)
        if geom_root.is_finalized:
            if geom_root.geom_timestamp != self.geom_timestamp:
                self.onClearMesh(evt)
                self.geom_timestamp = geom_root.geom_timestamp
                evt.Skip()
                return
        viewer = evt.GetEventObject().GetTopLevelParent().GetParent()
        viewer.set_view_mode('mesh', self)
        evt.Skip()

    def get_default_ns(self):
        '''
        this method is overwriten when model wants to
        set its own default namespace. For example, when
        RF module set freq and omega
        '''
        return {"remaining": "remaining", "all": "all", "auto": "auto"}


class GmshMeshActionBase(GMesh, Vtable_mixin):
    hide_ns_menu = True
    has_2nd_panel = False
    isGmshMesh = True
    dim = -1

    def attribute_set(self, v):
        v = super(GmshMeshActionBase, self).attribute_set(v)
        self.vt.attribute_set(v)
        return v

    @property
    def geom_root(self):
        return self.root()['Geometry'][self.parent.geom_group]

    def panel1_param(self):
        from wx import BU_EXACTFIT
        # b1 = {"label": "S", "func": self.onBuildBefore,
        #      "noexpand": True, "style": BU_EXACTFIT}
        b2 = {"label": "R", "func": self.onBuildAfter,
              "noexpand": True, "style": BU_EXACTFIT}

        ll = [[None, None, 241, {'buttons': [b2, ],  # b2],
                                 'alignright':True,
                                 'noexpand': True}, ], ]
        ll.extend(self.vt.panel_param(self))
        return ll

    def get_panel1_value(self):
        return [None] + list(self.vt.get_panel_value(self))

    def preprocess_params(self, engine):
        self.vt.preprocess_params(self)
        return

    def import_panel1_value(self, v):
        self.vt.import_panel_value(self, v[1:])
        return True

    def panel1_tip(self):
        return [None] + self.vt.panel_tip()

    def add_meshcommand(self):
        raise NotImplementedError(
            "you must specify this method in subclass")

    def _onBuildThis(self, evt, **kwargs):
        dlg = evt.GetEventObject().GetTopLevelParent()
        viewer = dlg.GetParent()
        engine = viewer.engine
        engine.build_ns()
        geom_root = self.root()['Geometry'][self.parent.geom_group]

        do_clear = True
        if not geom_root.is_finalized:
            geom_root.onBuildAll(evt)

        try:
            filename = os.path.join(
                viewer.model.owndir(),
                self.name()) + '.msh'
            kwargs['gui_parent'] = dlg
            kwargs['filename'] = filename

            count = self.parent.build_mesh(geom_root, **kwargs)
            do_clear = (count == 0)
        except BaseException:
            import ifigure.widgets.dialog as dialog
            dialog.showtraceback(parent=dlg,
                                 txt='Failed to generate meshing script',
                                 title='Error',
                                 traceback=traceback.format_exc())
        dlg.OnRefreshTree()
        self.parent.update_meshview(dlg, viewer, clear=do_clear)

    # def onBuildBefore(self, evt):
    #    dlg = evt.GetEventObject().GetTopLevelParent()
    #
    #    mm = dlg.get_selected_mm()
    #    self._onBuildThis(evt, stop1=mm)
    #    evt.Skip()

    def onBuildAfter(self, evt):
        dlg = evt.GetEventObject().GetTopLevelParent()
        _ = dlg.import_selected_panel_value()

        mm = dlg.get_selected_mm()
        self._onBuildThis(evt, stop2=mm)
        dlg = evt.GetEventObject().GetTopLevelParent()

        import wx
        wx.CallAfter(dlg.select_next_enabled)
        evt.Skip()

    def onClearMesh(self, evt):
        self.parent.onClearMesh(evt)

    def element_selection_empty(self):
        return {'volume': [],
                'face': [],
                'edge': [],
                'point': [], }, None

    def get_element_selection(self):
        # this is default..will be overwitten.
        return self.element_selection_empty()

    def onItemSelChanged(self, evt):
        super(GmshMeshActionBase, self).onItemSelChanged(evt)
        dlg = evt.GetEventObject().GetTopLevelParent()
        self.update_viewer_selection(dlg)

    def update_after_ELChanged(self, dlg):
        self.update_viewer_selection(dlg)

    def update_viewer_selection(self, dlg):
        viewer = dlg.GetParent()
        sel, mode = self.get_element_selection()

        if mode == 'volume':
            viewer.set_toolbar_mode('volume')
            viewer.highlight_domain(sel["volume"])
            viewer._dom_bdr_sel = (sel["volume"], [], [], [])
            status_txt = 'Volume :' + ','.join([str(x) for x in sel["volume"]])
            viewer.set_status_text(status_txt, timeout=60000)
        else:
            viewer.set_toolbar_mode(mode)
            figobjs = viewer.highlight_element(sel)
            if len(figobjs) > 0:
                import ifigure.events
                sel = [weakref.ref(x._artists[0]) for x in figobjs]
                ifigure.events.SendSelectionEvent(
                    figobjs[0], dlg, sel, multi_figobj=figobjs)

    def get_embed(self):
        return [], [], []

    def _eval_choices(self, mode):
        mesh_base = self.parent
        data = mesh_base.geom_root.geom_data
        if data is None:
            return []

        if mode == 3:
            choices = list(data[5])
        elif mode == 2:
            choices = list(data[4])
        elif mode == 1:
            choices = list(data[3])
        else:
            choices = list(range(1, len(data[0])+1))
        return np.array(choices)

    def _eval_entity_id(self, text):
        '''
        "remaining" -> "remaining"
        "all" -> "all"
        "auto" -> "auto"
        something else  -> global vaiable

        failure -> pass thorough
        '''
        if len(text.strip()) == 0:
            return ''
        try:
            # try to inteprete as integer numbers naively...
            values = list(set([int(x) for x in text.split(',')]))
            values = ','.join([str(x) for x in values])
            return values
        except BaseException:
            pass

        # then convert it using namespace
        g, l = self.namespace
        ll = {}
        try:
            values = eval(text, g, ll)
        except BaseException:
            assert False, "can not interpret entity number : " + text

        if not isinstance(values, str):
            try:
                values = ",".join([str(int(x)) for x in values])
            except BaseException:
                assert False, "entity id field must be text or arrays convertible to text"

        return values

    def eval_entity_id(self, *text):
        if len(text) == 1:
            return self._eval_entity_id(text[0])

        return [self._eval_entity_id(x) for x in text]

    def eval_entity_id2(self, text):
        '''
        similar to eval_entity_id but hanldes 'all' and 'remainig'
        used only for GUI. 
        note all/remaining are handled in mesh_wrapper separately
        '''
        if self.dim == -1:
            self.eval_entity_id(text)
        modes = ['point', 'edge', 'face', 'volume']
        mode = modes[self.dim]

        if text == 'all':
            choices = self._eval_choices(self.dim)
            choices = ",".join([str(int(x)) for x in choices])
            return choices

        elif text == 'remaining':
            choices = self._eval_choices(self.dim)
            for child in self.parent.get_children():
                if child.dim == -1:
                    continue
                if child == self:
                    break
                sel, mode = child.get_element_selection()
                choices = choices[np.in1d(choices, sel[mode], invert=True)]
            choices = ",".join([str(int(x)) for x in choices])
            return choices

        else:
            return self.eval_entity_id(text)


data = (('clmax', VtableElement('clmax', type='float',
                                guilabel='Max size(def)',
                                default_txt='',
                                default=1e20,
                                tip="CharacteristicLengthMax")),
        ('clmin', VtableElement('clmin', type='float',
                                guilabel='Min size(def)',
                                default_txt='',
                                default=0.0,
                                tip="CharacteristicLengthMin")),)


class GmshMesh(GMeshTop, Vtable_mixin):
    has_2nd_panel = False
    isMeshGroup = True
    vt = Vtable(data)

    @property
    def geom_root(self):
        return self.root()['Geometry'][self.geom_group]

    @property
    def mesher_data(self):
        if hasattr(self, "_mesher_data"):
            return self._mesher_data
        return None

    @property
    def mesh_output(self):
        if not hasattr(self, '_mesh_output'):
            return ''
        return self._mesh_output

    def attribute_set(self, v):
        v['geom_group'] = 'GmshGeom1'
        v['algorithm'] = 'default'
        v['algorithm3d'] = 'default'
        v['algorithmr'] = 'default'
        v['gen_all_phys_entity'] = False
        v['use_profiler'] = False
        v['use_expert_mode'] = False
        v['use_ho'] = False
        v['optimize_ho'] = 'none'
        v['optimize_dom'] = 'all'
        v['optimize_lim'] = "0.1, 2"
        v['ho_order'] = 2

        super(GmshMesh, self).attribute_set(v)
        self.vt.attribute_set(v)
        return v

    def panel1_param(self):
        ll = [["Geometry", self.geom_group, 0, {}, ], ]
        ll.extend(self.vt.panel_param(self))

        from petram.mesh.gmsh_mesh_wrapper import (Algorithm2D,
                                                   Algorithm3D,
                                                   AlgorithmR,
                                                   HighOrderOptimize)

        c1 = list(Algorithm2D)
        c2 = list(Algorithm3D)
        c3 = list(HighOrderOptimize)
        c4 = list(AlgorithmR)

        from wx import CB_READONLY
        setting1 = {"style": CB_READONLY, "choices": c1}
        setting2 = {"style": CB_READONLY, "choices": c2}
        setting3 = {"style": CB_READONLY, "choices": c3}
        setting4 = {"style": CB_READONLY, "choices": c4}
        ll_ho = [None, [True, [1, c3[0], 'all', '0.1, 2']],
                 27, [{'text': 'use high order (in dev, upto order 3, tet only)'},
                      {'elp': [["Order", self.ho_order, 400],
                               ["HighOrder optimize",
                                c3[-1], 4, setting3],
                               ["Optimize domain", 'all', 0, None],
                               ["SJac limits", '0.1, 2', 0, None], ]}
                      ]]

        ll.extend([["2D Algorithm", c1[-1], 4, setting1],
                   ["3D Algorithm", c2[-1], 4, setting2],
                   ["Recombine Alg.", c4[-1], 4, setting4],
                   [None, self.gen_all_phys_entity == 1, 3,
                    {"text": "Write physical entities for all dimensions."}],
                   [None, self.use_profiler, 3, {"text": "use profiler"}],
                   [None, self.use_expert_mode, 3, {
                       "text": "use GMSH expert mode"}],
                   ll_ho,
                   #[None, self.use_2nd_order,  3, {"text": "use 2nd order mesh (in dev order 3, tet only)"}],
                   #["HighOrder optimize", c3[-1], 4, setting3],
                   [None, None, 341, {"label": "Use default size",
                                      "func": 'onSetDefSize',
                                      "noexpand": True}],
                   [None, None, 341, {"label": "Finalize Mesh",
                                      "func": 'onBuildAll',
                                      "noexpand": True}], ])

        return ll

    def get_panel1_value(self):
        return ([self.geom_group, ] + list(self.vt.get_panel_value(self)) +
                [self.algorithm,
                 self.algorithm3d,
                 self.algorithmr,
                 self.gen_all_phys_entity,
                 self.use_profiler, self.use_expert_mode,
                 [self.use_ho, [self.ho_order, self.optimize_ho, self.optimize_dom,
                                self.optimize_lim], ],
                 self, self, ])

    def preprocess_params(self, engine):
        self.vt.preprocess_params(self)
        return

    def import_panel1_value(self, v):
        viewer_update = False
        if self.geom_group != str(v[0]):
            viewer_update = True
        self.geom_group = str(v[0])
        self.vt.import_panel_value(self, v[1:-9])

        self.algorithm = str(v[-9])
        self.algorithm3d = str(v[-8])
        self.algorithmr = str(v[-7])
        self.gen_all_phys_entity = v[-6]
        self.use_profiler = bool(v[-5])
        self.use_expert_mode = bool(v[-4])
        self.use_ho = bool(v[-3][0])
        self.ho_order = int(v[-3][1][0])
        self.optimize_ho = str(v[-3][1][1])
        self.optimize_dom = str(v[-3][1][2])
        self.optimize_lim = str(v[-3][1][3])

        return viewer_update

    def panel1_tip(self):
        return ([None] +
                self.vt.panel_tip() +
                ["Alogirth for 2D mesh",
                 "Algoirthm for 3D mesh",
                 "Algoirthm for recombine",
                 "Write lower dimensional physical entity. This may take a long time",
                 "Use cProfiler",
                 "Enable GMSH expert mode to suppress some warning",
                 "Generate high order mesh",
                 None, None])

    def get_possible_child(self):
        from .gmsh_mesh_actions import (TransfiniteLine, TransfiniteSurface,
                                        TransfiniteVolume, FreeFace,
                                        FreeVolume, FreeEdge, CharacteristicLength,
                                        CopyFace, CopyFaceRotate, RecombineSurface,
                                        ExtrudeMesh, RevolveMesh, MergeText, CompoundCurve,
                                        CompoundSurface, BoundaryLayer)

        return [FreeVolume, FreeFace, FreeEdge, TransfiniteLine,
                TransfiniteSurface, TransfiniteVolume,
                CharacteristicLength, CopyFace, CopyFaceRotate, RecombineSurface,
                ExtrudeMesh, RevolveMesh, CompoundCurve, CompoundSurface, BoundaryLayer,
                MergeText]

    def get_special_menu(self, evt):
        from petram.geom.gmsh_geom_model import use_gmsh_api

        return [('Build all', self.onBuildAll, None),
                ('+Export', None, None),
                ('Msh', self.onExportMsh, None),
                ('STL', self.onExportSTL, None),
                ('!', None, None),
                ('Clear mesh', self.onClearMesh, None),
                ('Clear mesh sequense...', self.onClearMeshSq, None)]

    def on_created_in_tree(self):
        check = self.geom_group in self.root()['Geometry']
        if not check:
            if len(self.root()['Geometry'].get_children()) == 0:
                assert False, "No geometry is found in the model"
            self.geom_group = self.root()['Geometry'].get_children()[0].name()

    def update_after_ELChanged(self, dlg):
        pass

    def update_after_ELChanged2(self, evt):
        dlg = evt.GetEventObject().GetTopLevelParent()
        viewer = dlg.GetParent()

        geom_root = self.geom_root
        geom_root.update_figure_data(viewer)

        self.onItemSelChanged(evt)

    def onSetDefSize(self, evt):
        geom_root = self.geom_root
        clmax_root, clmin_root = geom_root._clmax_guess
        self.clmax_txt = str(clmax_root)
        self.clmin_txt = str(clmin_root)
        dlg = evt.GetEventObject().GetTopLevelParent()
        dlg.OnItemSelChanged()

    def onExportMsh(self, evt):
        dlg = evt.GetEventObject().GetTopLevelParent()
        viewer = dlg.GetParent()

        src = self.mesh_output
        if src == '':
            return

        ext = src.split('.')[-1]

        from ifigure.widgets.dialog import write
        parent = evt.GetEventObject()
        dst = write(parent,
                    defaultfile='Untitled.' + ext,
                    message='Enter mesh file name')

        if dst == '':
            return
        try:
            import shutil
            dext = dst.split('.')[-1]
            if dext != ext:
                dst = dst + '.' + ext
            shutil.copyfile(src, dst)
        except BaseException:
            import ifigure.widgets.dialog as dialog
            dialog.showtraceback(parent=dlg,
                                 txt='Failed to export msh file',
                                 title='Error',
                                 traceback=traceback.format_exc())

    def onExportSTL(self, evt):
        src = self.mesh_output

        if src == '':
            import wx
            trash = wx.GetApp().GetTopWindow().proj.get_trash()
            src = os.path.join(trash, 'tmp0.msh')

        if not os.path.exists(src):
            import ifigure.widgets.dialog as dialog
            ret = dialog.message(parent=dlg,
                                 message='Mesh must be created first',
                                 title='Can not export STL.',
                                 style=0)
            return

        from ifigure.widgets.dialog import write
        parent = evt.GetEventObject()
        dst = write(parent,
                    defaultfile='Untitled.stl',
                    message='Enter STL file name')
        if dst == '':
            return
        if not dst.endswith('.stl'):
            dst = dst + '.dst'
        try:
            import gmsh
            gmsh.open(src)
            gmsh.write(dst)

        except BaseException:
            import ifigure.widgets.dialog as dialog
            dialog.showtraceback(parent=dlg,
                                 txt='Failed to export STL file',
                                 title='Error',
                                 traceback=traceback.format_exc())

    def update_meshview(self, dlg, viewer, clear=False):
        import gmsh
        from petram.geom.read_gmsh import read_pts_groups, read_loops

        if clear:
            viewer.del_figure_data('mesh', self.name())
        elif self.mesher_data is None:
            viewer.del_figure_data('mesh', self.name())
        else:
            print("number of meshed face/line",
                  len(self._mesh_fface), len(self._mesh_fline))
            viewer.set_figure_data('mesh', self.name(), self.mesher_data)

        if 'geom' in viewer._s_v_loop:
            viewer._s_v_loop['mesh'] = viewer._s_v_loop['geom']

        viewer.update_figure('mesh', self.figure_data_name())

    def onClearMesh(self, evt):
        dlg = evt.GetEventObject().GetTopLevelParent()
        viewer = dlg.GetParent()
        engine = viewer.engine
        engine.build_ns()
        geom_root = self.geom_root

        if not geom_root.is_finalized:
            geom_root.onBuildAll(evt)

        do_clear = True
        try:
            count = self.build_mesh(geom_root, nochild=True,
                                    gui_parent=dlg)
            do_clear = (count == 0)
        except BaseException:
            import ifigure.widgets.dialog as dialog
            dialog.showtraceback(parent=dlg,
                                 txt='Failed to generate meshing script',
                                 title='Error',
                                 traceback=traceback.format_exc())
        dlg.OnRefreshTree()
        self.update_meshview(dlg, viewer, clear=do_clear)

        viewer._view_mode_group = ''
        viewer.set_view_mode('mesh', self)

    def onClearMeshSq(self, evt):
        dlg = evt.GetEventObject().GetTopLevelParent()
        viewer = dlg.GetParent()
        engine = viewer.engine

        import ifigure.widgets.dialog as dialog
        ret = dialog.message(parent=dlg,
                             message='Are you sure to delete all Mesh sequence',
                             title='Mesh Sequence Delete',
                             style=2)
        if ret == 'ok':
            dialog.showtraceback(parent=dlg,
                                 txt='Failed to generate meshing script',
                                 title='Error',
                                 traceback=traceback.format_exc())

            for x in self.keys():
                del self[x]
            dlg.tree.RefreshItems()

    def onBuildAll(self, evt):
        dlg = evt.GetEventObject().GetTopLevelParent()
        dlg.import_selected_panel_value()

        viewer = dlg.GetParent()
        engine = viewer.engine
        engine.build_ns()

        geom_root = self.geom_root
        if not geom_root.is_finalized:
            geom_root.onBuildAll(evt)

        do_clear = True
        try:
            filename = os.path.join(
                viewer.model.owndir(),
                self.name()) + '.msh'
            count = self.build_mesh(geom_root, finalize=True, filename=filename,
                                    gui_parent=dlg)
            do_clear = count == 0
        except BaseException:
            import ifigure.widgets.dialog as dialog
            dialog.showtraceback(parent=dlg,
                                 txt='Failed to generate mesh script',
                                 title='Error',
                                 traceback=traceback.format_exc())
        dlg.OnRefreshTree()
        self.update_meshview(dlg, viewer, clear=do_clear)
        evt.Skip()

    def gather_embed(self):
        children = [x for x in self.walk()]
        children = children[1:]

        embed = [[], [], []]
        for child in children:
            s, l, p = child.get_embed()
            embed[0].extend(s)
            embed[1].extend(l)
            embed[2].extend(p)
        print("embed", embed)
        return embed

    def build_mesh(self, geom_root, stop1=None, stop2=None, filename='',
                   nochild=False, finalize=False, gui_parent=None):
        import gmsh
        from petram.geom.read_gmsh import read_pts_groups, read_loops

        self.vt.preprocess_params(self)
        clmax, clmin = self.vt.make_value_or_expression(self)
        dprint1("calling build mesh with", clmax, clmin)
        geom_root = self.geom_root

        if not geom_root.is_finalized:
            geom_root.build_geom4(no_mesh=True, finalize=True)

        if gui_parent is not None:
            import wx
            trash = wx.GetApp().GetTopWindow().proj.get_trash()
        else:
            cwd = os.getcwd()
            trash = os.path.join(cwd, '.trash')
            if not os.path.exists(trash):
                os.mkdir(trash)

        from petram.mesh.mesh_sequence_operator import MeshSequenceOperator

        mso = MeshSequenceOperator()

        children = [x for x in self.walk()]
        children = children[1:]

        if not nochild:
            for child in children:
                if not child.enabled:
                    continue
                if child is stop1:
                    break            # for build before
                child.vt.preprocess_params(child)
                # child.check_master_slave(mesher)
                child.add_meshcommand(mso)
                if child is stop2:
                    break            # for build after

        if mso.count_sequence() > 0:
            # set None since mesher may die...
            self._mesher_data = None

            L = mso.count_sequence() * 4 + 3

            if gui_parent is not None:
                import wx
                #gui_parent = wx.GetApp().TopWindow
                pgb = wx.ProgressDialog("Generating mesh...",
                                        "", L, parent=gui_parent,
                                        style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_CAN_ABORT)

                def close_dlg(evt, dlg=pgb):
                    pgb.Destroy()
                pgb.Bind(wx.EVT_CLOSE, close_dlg)
            else:
                pgb = None

            # collect edge tesselation from geometry
            ptx = self.geom_root.geom_data[0]
            idx = self.geom_root.geom_data[1]['line']
            a, b = np.unique(idx, return_inverse=True)
            idx2 = b.reshape(idx.shape)
            ptx2 = ptx[a]
            line_idx = self.geom_root.geom_data[2]['line']['geometrical']
            edge_tss = (ptx2, idx2, line_idx)

            import multiprocessing as mp
            num_th = max([mp.cpu_count()-1, 1])
            kwargs = {'CharacteristicLengthMax': clmax,
                      'CharacteristicLengthMin': clmin,
                      'EdgeResolution': 3,
                      'MeshAlgorithm': self.algorithm,
                      'MeshAlgorithm3D': self.algorithm3d,
                      'MeshAlgorithmR': self.algorithmr,
                      'use_profiler': self.use_profiler,
                      'use_expert_mode': self.use_expert_mode,
                      'use_ho': self.use_ho,
                      'ho_order': self.ho_order,
                      'optimize_ho': self.optimize_ho,
                      'optimize_dom': self.optimize_dom,
                      'optimize_lim': [float(x) for x in self.optimize_lim.split(',')],
                      'trash': trash,
                      'gen_all_phys_entity': self.gen_all_phys_entity,
                      'meshformat': 2.2,
                      'MaxThreads': [num_th, num_th, num_th, num_th],
                      'edge_tss': edge_tss}

            if self.mesh_output != '':
                if os.path.exists(self.mesh_output):
                    os.remove(self.mesh_output)

            max_mdim, done, data, msh_output = mso.run_generater(geom_root.geom_brep,
                                                                 filename,
                                                                 kwargs,
                                                                 finalize=finalize,
                                                                 progressbar=pgb)
            self._mesher_data = data
            self._max_mdim = max_mdim
            if finalize:
                '''
                if self.use_ho:
                    fname = msh_output[:-3] + 'mesh'

                    from petram.mesh.gmsh2mfem import Translator
                    t = Translator(msh_output, verbose=True)
                    t.write(fname)

                    #os.remove(msh_output)
                    self._mesh_output = fname
                else:
                    self._mesh_output = msh_output
                '''
                self._mesh_output = msh_output

            else:
                self._mesh_output = ''
        else:
            self._max_mdim = 0
            done = [], [], [], []

        self._mesh_fface = done[2]  # finished surfaces
        self._mesh_fline = done[1]  # finished lines

        return mso.count_sequence() > 0

    def generate_mesh_file(self):
        '''
        called from solver_model
        '''
        cwd = os.getcwd()
        dprint1("Generating Mesh in " + cwd)
        self._mesh_output = ''

        if myid == 0:
            geom_root = self.geom_root
            filename = os.path.join(cwd, self.name()) + '.msh'

            # reset this value so that it does not delete a
            # file in parametric scan

            count = self.build_mesh(geom_root,
                                    finalize=True,
                                    filename=filename,
                                    gui_parent=None,)
        else:
            count = 0

        if use_parallel:
            count = MPI.COMM_WORLD.bcast(count)
        if count == 0:
            assert False, "Failed to generate mesh"

        if use_parallel:
            self._mesh_output = MPI.COMM_WORLD.bcast(self._mesh_output)

        dprint1("Generating Mesh ... Done")

    def load_gui_figure_data(self, viewer):
        return 'mesh', self.name(), None

    def is_viewmode_grouphead(self):
        return True

    def figure_data_name(self):
        try:
            geom_root = self.geom_root
        except BaseException:
            return
        if geom_root.is_finalized:
            return self.name(), self.geom_group.strip()
        else:
            print("Geometry not finalized")
            return '', self.geom_group.strip()

    def get_meshfile_path(self):
        '''
        '''
        if hasattr(self, '_mesh_output') and self._mesh_output != '':
            path = self._mesh_output
            if os.path.exists(path):
                dprint1("gmsh file path", path)
                return path

        #ext = '.mesh' if self.use_ho else '.msh'
        ext = '.msh'

        path = os.path.join(self.root().get_root_path(), self.name() + ext)
        if os.path.exists(path):
            dprint1("gmsh file path", path)
            return path

        path = os.path.abspath(self.name() + ext)
        if os.path.exists(path):
            dprint1("gmsh file path", path)
            return path

        assert False, "Mesh file does not exist : " + path
