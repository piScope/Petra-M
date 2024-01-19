from __future__ import print_function
from ifigure.widgets.canvas.ifigure_canvas import ifigure_canvas
'''
   Viewer/Editor of MFEM model

   * this modules is designed to avoid importing mfem directly here

'''
import numpy as np
import os
import wx
import traceback
import weakref
from ifigure.interactive import figure
from ifigure.widgets.book_viewer import BookViewer
from ifigure.utils.cbook import BuildMenu
import ifigure.widgets.dialog as dialog
import ifigure.events
from petram.pi.dlg_progressbar import progressbar

import petram
try:
    import petram.geom
    hasGeom = True
except ImportError:
    hasGeom = False


def setup_figure(fig):
    fig.nsec(1)
    fig.property(fig.get_axes(0), 'axis', False)
    fig.get_page(0).set_nomargin(True)
    fig.property(fig.get_page(0), 'bgcolor', 'white')


ID_SOL_FOLDER = wx.NewIdRef(count=1)


def MFEM_menus(parent):
    self = parent
    menu1 = [("+Open Model...", None, None),
             ("Binary...", self.onOpenPMFEM, None, None),
             ("Script/Data Files...", self.onOpenModelS, None),
             ("!", None, None),
             ("+Mesh", None, None),
             ("New Mesh File...",  self.onNewMesh, None),

             ("Reload Mesh",  self.onLoadMesh, None),
             ("Mesh visualization +",  self.onMeshVisPlus, None),
             ("Mesh visualization -",  self.onMeshVisMinus, None),
             ("!", None, None),
             ("+Namespace", None, None),
             ("New...", self.onNewNS, None),
             ("Load...", self.onLoadNS, None),
             ("Export...", self.onExportNS, None),
             ("Rebuild", self.onRebuildNS, None),
             ("!", None, None),
             ("Edit Model...", self.onEditModel, None),
             ("+Solve", None, None),
             ("Serial",    self.onSerDriver, None),
             ("Parallel",  self.onParDriver, None),
             ("+Extra", None, None),
             ("+Store solution to", None, None, None, ID_SOL_FOLDER),
             ("!", None, None),
             ("---", None, None),
             ("New sol...",   self.onNewLocalSol, None),
             ("Clear sol", self.onClearSol, None),
             ("Preprocess data",   self.onRunPreprocess, None),
             ("!", None, None),
             ("!", None, None), ]

    if petram.mfem_model.has_cluster_access:
        menu2 = [("+Cluster", None, None),
                 ("Setting...", self.onServerSetting, None),
                 ("Solve...", self.onServerSolve, None),
                 ("Retrieve File", self.onServerRetrieve, None),
                 ("!", None, None), ]
    else:
        menu2 = []

    menu3 = [("+Plot", None, None),
             ("Function...",    self.onPlotExpr, None),
             ("Solution ...",    self.onDlgPlotSol, None),
             ("!", None, None),
             #("+Solution", None, None, None, ID_SOL_FOLDER),
             #("Reload Sol", None, None,),
             #("Clear...",    self.onClearSol, None),
             #("!", None, None),
             ("+Export Model...", self.onSaveModel, None),
             ("Binary...", self.onSaveModel, None),
             ("Script/Data Files...", self.onSaveModelS, None),
             ("!", None, None),
             ("---", None, None),
             ("Reset Model", self.onResetModel, None), ]
    return menu1 + menu2 + menu3


class MFEMViewerCanvas(ifigure_canvas):
    def __init__(self, *args, **kwags):
        self._mfemviewer_spacehit = False
        return ifigure_canvas.__init__(self, *args, **kwags)

    def unselect_all(self):
        ifigure_canvas.unselect_all(self)
        self.TopLevelParent._dom_bdr_sel = ([], [], [], [])

    def onKey(self, evt):
        keycode = evt.guiEvent.GetKeyCode()
        if keycode == wx.WXK_SPACE:
            self._mfemviewer_spacehit = True

        return ifigure_canvas.onKey(self, evt)

    def onKey2(self, evt):
        keycode = evt.guiEvent.GetKeyCode()
        if keycode == wx.WXK_SPACE:
            if self._mfemviewer_spacehit:
                self.TopLevelParent.handle_shifthit()
        self._mfemviewer_spacehit = False
        return ifigure_canvas.onKey2(self, evt)


class MFEMViewer(BookViewer):
    def __init__(self, *args, **kargs):
        kargs['isattachable'] = False
        kargs['isinteractivetarget'] = False
        kargs['ismultipage'] = False
        BookViewer.__init__(self, *args, **kargs)
        extra_menu = wx.Menu()
        self.menuBar.Insert(self.menuBar.GetMenuCount()-1,
                            extra_menu, "PetraM")
        menus = MFEM_menus(self)
        ret = BuildMenu(extra_menu, menus)

        data = ret[ID_SOL_FOLDER]
        if len(data) == 2:
            self._solmenu, item = data
            self._ID_SOL_FOLDER = item.GetId()
        else:
            self._solmenu = data
            self._ID_SOL_FOLDER = ID_SOL_FOLDER

        self._hidemesh = True
        self._sel_mode = ''  # selecting particular geomgetry element
        self._view_mode = ''  # ('geom', 'mesh', 'phys')
        self._s_v_loop = {}
        self._selected_volume = []    # store selected volume
        self._figure_data = {}

        # extra refinement of mesh for visualization
        self._mesh_vis = 0
        # hidden element in MFEM mode
        self._hidden_volume = []

        # hidden element in mesh mode
        # for mesh we need to keep track face and edge too.
        self._mhidden_volume = []
        self._mhidden_face = []
        self._mhidden_edge = []

        self._view_mode_group = ''
        self._is_mfem_geom_fig = False
        self._dom_bdr_sel = ([], [], [], [])
        self._palette_focus = ''
        self.model = self.book.get_parent()
        self.editdlg = None
        self.selection_palette = None
        self.geom_info_palette = None
        self.plotsoldlg = None
        self.plotexprdlg = None
        self.engine = None
        self.dombdr = None

        from petram.pi.sel_buttons import btask, refresh

        self.install_toolbar_palette('petram_phys',
                                     btask,
                                     mode='3D',
                                     refresh=refresh)
        from petram.mesh.mesh_sel_buttons import btask
        self.install_toolbar_palette('petram_mesh',
                                     btask,
                                     mode='3D',
                                     refresh=refresh)
        self.use_toolbar_palette('petram_phys',
                                 mode='3D')
        if hasGeom:
            from petram.geom.geom_sel_buttons import btask
            self.install_toolbar_palette('petram_geom',
                                         btask,
                                         mode='3D',
                                         refresh=refresh)

        od = self.model.param.getvar('mfem_model')
        od.set_root_path(self.model.owndir())

        if od is None:
            self.model.scripts.helpers.reset_model()

        self.cla()
        setup_figure(self)
        self.start_engine()

        for child in od.walk():
            view_mode, name, data = child.load_gui_figure_data(self)
            if data is not None:
                self.set_figure_data(view_mode, name, data)

        self.model.scripts.helpers.rebuild_ns()

        if self.model.variables.getvar('mesh') is None:
            try:
                self.load_mesh()
                self.engine.run_mesh_extension_prep(reset=True)
            except:
                dialog.showtraceback(parent=self,
                                     txt='mesh file load error',
                                     title='Error',
                                     traceback=traceback.format_exc())
        self.plot_mfem_geom()

        self.Bind(ifigure.events.TD_EVT_ARTIST_DRAGSELECTION,
                  self.onTD_DragSelectionInFigure)

        self.canvas._popup_style = 1  # popup_skip_2d
        self.canvas.__class__ = MFEMViewerCanvas

        # make an empty sol folder if it is not defined.
        if self.model.param.eval('sol') is None:
            self.model.scripts.helpers.make_new_sol()

        import logging
        numba_logger = logging.getLogger('numba')
        numba_logger.setLevel(logging.WARNING)
        print("numba debug logging is suppressed")

    @property
    def view_mode_group(self):
        return self._view_mode_group

    @property
    def dom_bdr_sel(self):
        return self._dom_bdr_sel

    def set_view_mode(self, mode, mm=None):
        p = mm
        while p is not None:
            if p.is_viewmode_grouphead():
                break
            p = p.parent
        group = p.name() if p is not None else 'root'

        do_palette = (self._view_mode != mode)
        do_plot = (self._view_mode != mode) or (self._view_mode_group != group)

        self._view_mode = mode
        self._view_mode_group = p.name() if p is not None else 'root'

        if do_palette:
            if mode == '':
                self.use_toolbar_std_palette()
            else:
                self.use_toolbar_palette('petram_'+mode, mode='3D')
        if do_plot:
            if p is not None:
                # update figure data
                p.update_figure_data(self)
                #print("calling do_plot", self._view_mode, p.figure_data_name())
                self.update_figure(self._view_mode, p.figure_data_name(),
                                   updateall=True)

    def set_figure_data(self, view_mode, name, data):
        if not view_mode in self._figure_data:
            self._figure_data[view_mode] = {}
        self._figure_data[view_mode][name] = data

    def del_figure_data(self, view_mode, name):
        if not view_mode in self._figure_data:
            self._figure_data[view_mode] = {}
        if name in self._figure_data[view_mode]:
            del self._figure_data[view_mode][name]

    def update_figure(self, view_mode, name, updateall=False):
        from petram.mesh.geo_plot import plot_geometry, oplot_meshed
        from petram.mesh.geo_plot import hide_face_meshmode, hide_edge_meshmode

        if self._is_mfem_geom_fig and view_mode == 'phys':
            self.onHideMesh()
        elif self._is_mfem_geom_fig and view_mode == 'mesh' and name == 'mfem':
            self.onShowMesh()
        else:
            self._is_mfem_geom_fig = False
            if not view_mode in self._figure_data:
                self.cls()
                return
            if view_mode == 'geom':
                d = self._figure_data['geom']
                self._hidden_volume = []
                if name in d:
                    ret = d[name]
                    plot_geometry(self,  ret)
                else:
                    #print('Geometry figure data not found :' + name)
                    self.cls()
                    return

            elif view_mode == 'mesh':
                if name == 'mfem':
                    if not 'mfem' in self._figure_data['mesh']:
                        self.cls()
                    else:
                        d = self._figure_data['mesh']['mfem']
                        plot_geometry(self, d, geo_phys='physical', lw=1.0)
                        self._is_mfem_geom_fig = True
                else:
                    d = self._figure_data['mesh']
                    if updateall or not name[0] in d:
                        if not 'geom' in self._figure_data:
                            # geom is not yet run
                            self.cls()
                            return
                        d = self._figure_data['geom']
                        plot_geometry(self,  d[name[1]])
                    if name[0] in d:
                        #print("calling oplot")
                        oplot_meshed(self,  d[name[0]])
                        self._hidemesh = False
                    else:
                        hide_face_meshmode(self, [])
                        hide_edge_meshmode(self, [])

            elif view_mode == 'phys':
                ret = self._figure_data['phys']
                plot_geometry(self,  ret, geo_phys='physical')
                self._is_mfem_geom_fig = True
                self._hidemesh = True

    def onUpdateUI(self, evt):
        if evt.GetId() == self._ID_SOL_FOLDER:
            m = self._solmenu
            for item in m.GetMenuItems():
                m.DestroyItem(item)
            try:
                if not self.model.solutions.has_owndir():
                    return
                dir = self.model.solutions.owndir()
                sol_names = [x for x in os.listdir(
                    dir) if os.path.isdir(os.path.join(dir, x))]
                sol_names = sorted(sol_names)
                menus = []
                for n in sol_names:
                    menus.append((n, n))
            except:
                import traceback
                traceback.print_exc()
                evt.Enable(False)
                return
            mm = []
            from petram.sol.solsets import read_solsets, find_solfiles
            for m0, m2 in menus:
                def handler(evt, dir0=m0):
                    # self.model.scripts.helpers.rebuild_ns()
                    # self.engine.assign_sel_index()
                    #path = os.path.join(dir, dir0)
                    #print('loading sol from ' + path)
                    model = self.model
                    folder = model.solutions.get_child(name=str(dir0))
                    param = model.param
                    param.setvar('sol', '='+folder.get_full_path())
                    m = self.model.param.getvar('mfem_model')
                    m.set_root_path(self.model.owndir())
                    evt.Skip()

                mm.append((m2, 'Store solution in ' + m0, handler))

            # mm.append(('Other...', 'Load from ohter place (FileDialog will open)',
            #           handler2))
            param = self.model.param
            sol = param.eval('sol')
            choice = sol.name if sol is not None else None

            if len(mm) > 0:
                for a, b, c in mm:
                    mmi = self.add_menu(m, wx.ID_ANY, a, b,
                                        c, kind=wx.ITEM_CHECK)
                    if a == choice:
                        mmi.Check(True)
            evt.Enable(True)
        else:
            super(MFEMViewer, self).onUpdateUI(evt)

    def start_engine(self):
        self.engine = self.model.scripts.helpers.start_engine()
        # if self.model.variables.hasvar('engine')
        #from engine import SerialEngine
        #self.engine = SerialEngine()
        #self.model.variables.setvar('engine', self.engine)

    def onOpenPMFEM(self, evt):
        import petram.helper.pickle_wrapper as pickle
        from ifigure.mto.py_code import PyData
        from ifigure.mto.py_script import PyScript

        from petram.mesh.mesh_model import MeshFile
        path = dialog.read(
            message='Select model file to read', wildcard='*.pmfm')
        if path == '':
            return

        od = pickle.load(open(path, 'rb'))
        self.model.param.setvar('mfem_model', od)

        # clean up namespaces/datasets
        for name, child in self.model.namespaces.get_children():
            child.destroy()
        for name, child in self.model.datasets.get_children():
            child.destroy()

        # first expand all namelist under namespaces
        dir = self.model.namespaces.owndir()

        ns_names = []
        for node in od.walk():
            if node.has_ns():
                if node.ns_name is None:
                    continue
                if not node.ns_name in ns_names:
                    ns_names.append(node.ns_name)
                    node.write_ns_script_data(dir=dir)

        for file in os.listdir(dir):
            print("processing file", file)
            if file == os.path.basename(path):
                continue
            if file.endswith('.py'):
                #shutil.copy(os.path.join(dir, file), self.model.namespaces.owndir())
                sc = self.model.namespaces.add_childobject(PyScript, file[:-3])
                sc.load_script(os.path.join(
                    self.model.namespaces.owndir(), file))
            if file.endswith('.dat'):
                fid = open(os.path.join(dir, file), 'rb')
                data = pickle.load(fid)
                fid.close()
                obj = self.model.datasets.add_childobject(
                    PyData, file[:-6]+'data')
                obj.setvar(data)

        self.cla()
        self.load_mesh()

        if self.editdlg is not None:
            od = self.model.param.getvar('mfem_model')
            self.editdlg.set_model(od)
        self.model.variables.setvar('modelfile_path', path)
        evt.Skip()

    def onOpenModelS(self, evt):
        import imp
        import shutil
        import ifigure.utils.pickle_wrapper as pickle
        from ifigure.mto.py_code import PyData
        from ifigure.mto.py_script import PyScript
        path = dialog.read(
            message='Select model file to read', wildcard='*.py')
        try:
            print("loading", path)
            m = imp.load_source('petram.user_model', path)
            model = m.make_model()
        except:
            print(path)
            dialog.showtraceback(parent=self,
                                 txt='Model file load error',
                                 title='Error',
                                 traceback=traceback.format_exc())
            return
        dir = os.path.dirname(path)
        c = [child for name, child in self.model.datasets.get_children()]
        for child in c:
            child.destroy()
        c = [child for name, child in self.model.namespaces.get_children()]
        for child in c:
            child.destroy()

        for file in os.listdir(dir):
            if file == os.path.basename(path):
                continue
            if file.endswith('.py'):
                shutil.copy(os.path.join(dir, file),
                            self.model.namespaces.owndir())
                sc = self.model.namespaces.add_childobject(PyScript, file[:-3])
                sc.load_script(os.path.join(
                    self.model.namespaces.owndir(), file))
            if file.endswith('.dat'):
                fid = open(os.path.join(dir, file), 'rb')
                data = pickle.load(fid)
                fid.close()
                obj = self.model.datasets.add_childobject(
                    PyData, file[:-6]+'data')
                obj.setvar(data)

        self.model.param.setvar('mfem_model', model)
        self.cla()
        self.load_mesh()

        if self.editdlg is not None:
            od = self.model.param.getvar('mfem_model')
            self.editdlg.set_model(od)

        evt.Skip()

    def _getSelectedIndexVolume(self, already_selected_surface):
        objs = [o().figobj for o in self.canvas.selection
                if o() is not None and o().figobj.name.startswith('face')]

        unselected = []
        selected = []
        for o in objs:
            idx = set(o.getSelectedIndex())
            unselected.extend(list(already_selected_surface.difference(idx)))
            selected.extend(list(idx.difference(already_selected_surface)))

        unselected = list(set(unselected))
        selected = list(set(selected))

        return selected, unselected, objs

    def _getSelectedIndex(self, mode='face'):
        objs = [o().figobj for o in self.canvas.selection
                if o() is not None]

        sel = []
        oo = []
        for obj in objs:
            if obj.name.startswith(mode):
                sel.extend(obj.getSelectedIndex())
                oo.append(obj)
        return sel, oo

    def _getFigObjs(self, mode='face'):
        objs = []
        ax = self.get_axes()
        for name, obj in ax.get_children():
            if name.startswith(mode):
                objs.append(obj)
        return objs

    def onTD_DragSelectionInFigure(self, evt):
        # print("onTD_DragSelectionInFigure")
        self.onTD_SelectionInFigure(evt)

    def onTD_SelectionInFigure(self, evt=None):
        if len(self.canvas.selection) == 0:
            self._dom_bdr_sel = ([], [], [], [])
        #    return

        status_txt = ''
        if not self._view_mode in self._s_v_loop:
            self.set_status_text('', timeout=600000)
            evt.selections = self.canvas.selection
            self.property_editor.onTD_Selection(evt)
            return

        #print("canvas sel",  self.canvas.selection)
        _s_v_loop = self._s_v_loop[self._view_mode]
        sf, sv, se, sp = [], [], [], []
        if self._sel_mode == 'volume':
            if _s_v_loop[1] is None:
                return

            sl = _s_v_loop[1]
            already_selected_surface = set(
                sum([sl[k] for k in self._dom_bdr_sel[0]], []))

            already_selected = self._dom_bdr_sel[0]
            already_selected = [k for k in already_selected if k in sl]

            selected, unselected, objs = self._getSelectedIndexVolume(
                already_selected_surface)

            selected_volume = already_selected[:]

            if len(unselected) > 0:
                for i in unselected:
                    for k in sl.keys():
                        if i in sl[k] and k in selected_volume:
                            selected_volume.remove(k)
            else:
                for i in selected:
                    for k in sl.keys():
                        if i in sl[k]:
                            selected_volume.append(k)

            selected_volume = list(set(selected_volume))

            hidden = self._mhidden_volume if self._view_mode == 'mesh' else self._hidden_volume
            hidden = list(set(hidden))

            for x in hidden:
                if x in selected_volume:
                    selected_volume.remove(x)

            status_txt = 'Volume :' + \
                ','.join([str(x) for x in selected_volume])

            surf_idx = []
            for kk in selected_volume:
                surf_idx.extend(sl[kk])
            surf_idx = list(set(surf_idx))

            objs = self._getFigObjs(mode='face')

            for o in objs:
                o.setSelectedIndex(surf_idx)
                if len(surf_idx) > 0:
                    if len(self.canvas.selection) > 0 and len(o._artists) > 0:
                        self.canvas.add_selection(o._artists[0])
                else:
                    self.canvas.unselect_all()
            sv = selected_volume

        elif self._sel_mode == 'face':
            idx, objs = self._getSelectedIndex(mode='face')
            status_txt = 'Face: ' + ','.join([str(x) for x in idx])
            v = _s_v_loop[1]
            connected_vol = []
            if v is not None:  # in case surface loop is defined (3D)

                for i in idx:
                    for k in v.keys():
                        if i in v[k]:
                            connected_vol.append(k)
                connected_vol = list(set(connected_vol))
                status_txt += ('(Volume: ' +
                               ','.join([str(x) for x in connected_vol]) + ')')
            sv = connected_vol
            sf = idx

        elif self._sel_mode == 'edge':
            idx, objs = self._getSelectedIndex(mode='edge')
            s = _s_v_loop[0]
            connected_surf = []
            if s is not None:  # in case line loop is defined...(2D/3D)
                for i in idx:
                    for k in s.keys():
                        if i in s[k]:
                            connected_surf.append(k)
                connected_suf = list(set(connected_surf))
                sf = connected_suf
                status_txt = ('Edge: ' + ','.join([str(x) for x in idx]) + '(Face: ' +
                              ','.join([str(x) for x in connected_suf]) + ')')
            else:
                status_txt = 'Edge: ' + ','.join([str(x) for x in idx])

            se = idx

        elif self._sel_mode == 'point':
            if (self.get_axes().has_child('point') and
                    not self.get_axes().point.isempty()):
                point = self.get_axes().point
                idx = point.getSelectedIndex()
                aidx = point.getvar('array_idx')

                status_txt = 'Vertex: ' + ','.join([str(x) for x in idx])
                if len(idx) == 1:
                    ii = np.where(aidx == idx[0])[0][0]
                    t = (" (" + str(point.getvar('x')[ii]) + ", " +
                         str(point.getvar('y')[ii]) + ", " +
                         str(point.getvar('z')[ii]) + ")")
                    status_txt = status_txt + t
                elif len(idx) == 2:
                    x, y, z = point.getvar('x', 'y', 'z')
                    ii1 = np.where(aidx == idx[0])[0][0]
                    ii2 = np.where(aidx == idx[1])[0][0]
                    dx = x[ii1] - x[ii2]
                    dy = y[ii1] - y[ii2]
                    dz = z[ii1] - z[ii2]
                    dd = np.sqrt(dx**2+dy**2+dz**2)
                    t = (" (delta = " + str(dx) + ", " + str(dy) + ", "+str(dz) +
                         ", dist. = " + str(dd))
                    status_txt = status_txt + t
                else:
                    pass
            else:
                idx = []
            sp = idx
        else:
            pass

        self.set_status_text(status_txt, timeout=600000)

        self._dom_bdr_sel = (sv, sf, se, sp)
        evt.selections = self.canvas.selection
        self.property_editor.onTD_Selection(evt)

    def onNewMesh(self, evt):
        from ifigure.widgets.dialog import read
        from petram.mesh.mesh_model import MeshFile, MFEMMesh
        path = read(message='Select mesh file to read',
                    wildcard='MFEM|*.mesh|Gmsh|*.msh')
        if path == '':
            return
        od = self.model.param.getvar('mfem_model')

        mg = MFEMMesh()
        data = MeshFile(path=path)

        nameg = od['Mesh'].add_itemobj('MFEMMesh', mg)
        name = od['Mesh'][nameg].add_itemobj('MeshFile', data)
        for key in od['Mesh']:
            if key != nameg:
                od['Mesh'][key].enabled = False
        self.load_mesh()

    def onLoadMesh(self, evt):
        win = evt.GetEventObject()
        if hasattr(win, 'GetTopLevelParent'):
            top_win = win.GetTopLevelParent()
        else:
            top_win = self
        dlg = dialog.progressbar(top_win, 'Loading mesh...',
                                 'In progress', 5)
        dlg.Show()
        wx.GetApp().Yield()

        self.load_mesh()
        self._hidemesh = True

        dlg.Destroy()

    def load_mesh(self):
        if self.engine is None:
            self.start_engine()
        od = self.model.param.getvar('mfem_model')

        import wx
        projfile = wx.GetApp().TopWindow.proj.getvar('filename')
        if projfile is not None:
            od.model_path = os.path.dirname(projfile)
        self.engine.set_model(od)
        try:
            cdir = os.getcwd()
        except:
            from os.path import expanduser
            cdir = expanduser("~")

        try:
            err = -1
            os.chdir(self.model.owndir())
            try:
                self.engine.run_mesh(skip_refine=True)
            except:
                exception = traceback.format_exc()
                assert False, "error in run_mesh"
            mesh = self.engine.get_mesh()
            self.model.variables.setvar('mesh', mesh)
            os.chdir(cdir)

        except:
            os.chdir(cdir)
            dialog.showtraceback(parent=self,
                                 txt='Mesh load error',
                                 title='Error (Load Mesh)',
                                 traceback=exception)
        try:
            err, exception = self.engine.run_config()
            if err != 0:
                assert False, "error in run_config"
        except:
            dialog.showtraceback(parent=self,
                                 txt='Error during run_config after loading mesh',
                                 title='Error (run_config)',
                                 traceback=exception)

        if err != -1:
            self.plot_mfem_geom()
            self.use_toolbar_palette('petram_phys', mode='3D')

    def onMeshVisPlus(self, evt):
        self._mesh_vis = self._mesh_vis + 1
        self.onLoadMesh(evt)

    def onMeshVisMinus(self, evt):
        self._mesh_vis = self._mesh_vis - 1
        self.onLoadMesh(evt)

    def plot_mfem_geom(self):
        from petram.mesh.geo_plot import plot_geometry
        from petram.mesh.read_mfemmesh import extract_mesh_data

        mesh = self.model.variables.getvar('mesh')
        if mesh is not None:
            from petram.mesh.refined_mfem_geom import default_refine as refine

            if mesh.GetNodalFESpace() is not None:
                refine = mesh.GetNodalFESpace().GetOrder(0)
            else:
                refine = 1
            refine = refine + self._mesh_vis
            X, cells, cell_data, sl, iedge2bb = extract_mesh_data(mesh, refine)
            self._s_v_loop['phys'] = sl
            self._s_v_loop['mesh'] = sl
            ret = (X, cells, None, cell_data, None)
            self._ret_ret = ret, iedge2bb
            plot_geometry(self, ret, geo_phys='physical')
            self._figure_data['phys'] = ret
            if not 'mesh' in self._figure_data:
                self._figure_data['mesh'] = {}
            self._figure_data['mesh']['mfem'] = ret
            self._view_mode = 'phys'
            self._is_mfem_geom_fig = True
        else:
            self.cls()

    def highlight_element(self, sel):
        if not self._view_mode in self._s_v_loop:
            return None

        _s_v_loop = self._s_v_loop[self._view_mode]
        ax = self.get_axes()

        if len(sel['volume']) != 0:
            if _s_v_loop[1] is None:
                return None
            sl = _s_v_loop[1]
            vfaces = []
            for i in sel['volume']:
                if i in sl:
                    vfaces.extend(sl[i])
                else:
                    print('Volume: ' + str(i) + " not found")
            vfaces = list(set(vfaces))
        else:
            vfaces = []

        obj = None
        robj = []
        for name, obj in ax.get_children():
            if name.startswith('point'):
                obj.setSelectedIndex(sel['point'])
                if len(sel['point']) != 0:
                    robj.append(obj)

            if name.startswith('edge'):
                obj.setSelectedIndex(sel['edge'])
                if len(sel['edge']) != 0:
                    robj.append(obj)

            if name.startswith('face'):
                obj.setSelectedIndex(sel['face'])
                if len(sel['face']) != 0:
                    robj.append(obj)
                if len(vfaces) > 0:
                    obj.setSelectedIndex(vfaces)
                    robj.append(obj)

        return robj

    def highlight_domain(self, i):
        '''
        i is 1-based index
        '''
        if not self._view_mode in self._s_v_loop:
            return

        _s_v_loop = self._s_v_loop[self._view_mode]
        ax = self.get_axes()
        try:
            x = len(i)
        except:
            i = list(i)

        hidden = (self._mhidden_volume if self._view_mode == 'mesh'
                  else self._hidden_volume)
        hidden = list(set(hidden))

        i = [x for x in i if not x in hidden]

        self.canvas.unselect_all()
        for name, obj in ax.get_children():
            if not name.startswith('face'):
                continue
            if _s_v_loop[1] is None:
                return
            if len(i) > 0:
                sl = _s_v_loop[1]
                faces = []
                for key in i:
                    if key in sl:
                        faces.extend(sl[key])
                    else:
                        print('Volume: ' + str(key) + " not found")
                obj.setSelectedIndex(faces)
                if len(obj._artists) > 0:
                    self.canvas.add_selection(obj._artists[0])
            else:
                obj.setSelectedIndex([])
        wx.CallAfter(self.canvas.refresh_hl)

    def highlight_face(self, i):
        '''
        i is 1-based index
        '''
        try:
            x = len(i)
        except:
            i = list(i)
        ax = self.get_axes()

        self.canvas.unselect_all()

        for name, obj in ax.get_children():
            if not name.startswith('face'):
                continue
            if len(i) > 0:
                obj.setSelectedIndex(i)
                #print("add_selection", obj, obj._artists[0])
                if len(obj._artists) > 0:
                    self.canvas.add_selection(obj._artists[0])
            else:
                obj.setSelectedIndex([])

        self._dom_bdr_sel = (self._dom_bdr_sel[0],
                             tuple(i),
                             self._dom_bdr_sel[2],
                             self._dom_bdr_sel[3],)
        wx.CallAfter(self.canvas.refresh_hl)

    def highlight_edge(self, i, unselect=True):
        '''
        i is 1-based index
        '''
        try:
            x = len(i)
        except:
            i = list(i)
        ax = self.get_axes()

        if unselect:
            self.canvas.unselect_all()

        for name, obj in ax.get_children():
            if not name.startswith('edge'):
                continue
            if len(i) > 0:
                obj.setSelectedIndex(i)
                if len(obj._artists) > 0:
                    self.canvas.add_selection(obj._artists[0])
            else:
                obj.setSelectedIndex([])

        self._dom_bdr_sel = (self._dom_bdr_sel[0],
                             self._dom_bdr_sel[1],
                             tuple(i),
                             self._dom_bdr_sel[3],)

        wx.CallAfter(self.canvas.refresh_hl)

    def highlight_point(self, i, unselect=True):
        '''
        i is 1-based index
        '''
        try:
            x = len(i)
        except:
            i = list(i)
        ax = self.get_axes()

        if unselect:
            self.canvas.unselect_all()

        for name, obj in ax.get_children():
            if not name.startswith('point'):
                continue
            if len(i) > 0:
                obj.setSelectedIndex(i)
                if len(obj._artists) > 0:
                    self.canvas.add_selection(obj._artists[0])
            else:
                obj.setSelectedIndex([])

        self._dom_bdr_sel = (self._dom_bdr_sel[0],
                             self._dom_bdr_sel[1],
                             self._dom_bdr_sel[2],
                             tuple(i),)

        wx.CallAfter(self.canvas.refresh_hl)

    def highlight_none(self):
        self.canvas.unselect_all()

        ax = self.get_axes()
        for name, obj in ax.get_children():
            if hasattr(obj, 'setSelectedIndex'):
                obj.setSelectedIndex([])
        self.canvas.refresh_hl()

    def change_panel_button(self, kind):
        # kind = ('domain', 'face', 'edge', 'dot')
        if self._view_mode == 'geom':
            prefix = 'g'
        elif self._view_mode == 'mesh':
            prefix = 'm'
        else:
            prefix = ''
        self.canvas.toolbar.ClickP1Button(prefix+kind)

    def onResetModel(self, evt):
        ans = dialog.message(self,
                             "Do you want to delete all model setting?",
                             style=2)
        if ans == 'ok':
            self.model.scripts.helpers.reset_model()
            self.model.scripts.helpers.rebuild_ns()
            self.cla()
            if self.editdlg is not None:
                od = self.model.param.getvar('mfem_model')
                self.editdlg.set_model(od)

    def onEditModel(self, evt):
        from petram.pi.dlg_edit_model import DlgEditModel
        try:
            self.engine.assign_phys_pp_sel_index()
            self.engine.assign_sel_index()
        except:
            traceback.print_exc()

        model = self.model.param.getvar('mfem_model')
        if self.editdlg is None:
            self.model.scripts.helpers.rebuild_ns()
            self.editdlg = DlgEditModel(self, wx.ID_ANY, 'Model Tree',
                                        model=model)
            self.editdlg.Show()
        self.editdlg.Raise()

    def onSelectionPanel(self, evt):
        from petram.pi.selection_palette import SelectionPalette

        if self.selection_palette is None:
            self.selection_palette = SelectionPalette(
                self, wx.ID_ANY, 'Selection')
            self.selection_palette.Show()
        self.selection_palette.Raise()
        evt.Skip()

    def onGeomInfo(self, evt):
        try:
            from petram.geom.geom_info_palette import GeomInfoPalette

            if self.geom_info_palette is None:
                self.geom_info_palette = GeomInfoPalette(self,
                                                         wx.ID_ANY,
                                                         'Geometry')
                self.geom_info_palette.Show()
            self.geom_info_palette.Raise()
        except ImportError:
            pass
        evt.Skip()

    def onSaveModel(self, evt):
        from ifigure.widgets.dialog import write
        from petram.mesh.mesh_model import MeshFile
        path = write(parent=self,
                     message='Enter model file name',
                     wildcard='*.pmfm')
        if path == '':
            return
        self.model.scripts.helpers.save_model(path)

    def onSaveModelS(self, evt):
        from ifigure.widgets.dialog import writedir
        path = writedir(parent=self,
                        message='Directory to write')

        m = self.model.param.getvar('mfem_model')
        try:
            m.generate_script(dir=path)
        except:
            dialog.showtraceback(parent=self,
                                 txt='Failed to evauate expression',
                                 title='Error',
                                 traceback=traceback.format_exc())

    def onRunPreprocess(self, evt):
        try:
            self.run_preprocess()
        except:
            dialog.showtraceback(parent=self,
                                 txt='Failed to during pre-processing model data',
                                 title='Error',
                                 traceback=traceback.format_exc())

    def set_num_threads(self, nproc=1):
        import os
        import multiprocessing

        num_threads1 = self.model.param.getvar('openmp_num_threads')
        if num_threads1 is None:
            self.model.param.setvar('openmp_num_threads', 'auto')
            num_threads1 = 1
        if num_threads1 == 'auto':
            ncpus = multiprocessing.cpu_count()
            num_threads1 = ncpus//nproc

        os.environ["OMP_NUM_THREADS"] = str(num_threads1)

        num_threads2 = self.model.param.getvar('openblas_num_threads')
        if num_threads2 is None:
            self.model.param.setvar('openblas_num_threads', 1)
            num_threads2 = 1
        if num_threads2 != 'auto':
            os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads2)

    def onSerDriver(self, evt):
        m = self.model.param.getvar('mfem_model')
        m.set_root_path(self.model.owndir())
        debug_level = m['General'].debug_level
        odir = os.getcwd()

        try:
            self.engine.run_verify_setting()
        except:
            os.chdir(odir)
            dialog.showtraceback(parent=self,
                                 txt='Failed to verify setting',
                                 title='Error',
                                 traceback=traceback.format_exc())
            return

        try:
            self.run_preprocess()
        except:
            os.chdir(odir)
            dialog.showtraceback(parent=self,
                                 txt='Failed to during pre-processing model data',
                                 title='Error',
                                 traceback=traceback.format_exc())
            return

        self.set_num_threads()
        self.model.scripts.run_serial.RunT(debug=debug_level)

        os.chdir(odir)

    def onParDriver(self, evt):
        m = self.model.param.getvar('mfem_model')
        m.set_root_path(self.model.owndir())
        debug_level = m['General'].debug_level
        odir = os.getcwd()

        try:
            self.engine.run_verify_setting()
        except:
            os.chdir(odir)
            dialog.showtraceback(parent=self,
                                 txt='Failed to verify setting',
                                 title='Error',
                                 traceback=traceback.format_exc())
            return

        try:
            self.run_preprocess()
        except:
            os.chdir(odir)
            dialog.showtraceback(parent=self,
                                 txt='Failed to during pre-processing model data',
                                 title='Error',
                                 traceback=traceback.format_exc())
            return

        nproc = self.model.param.getvar('nproc')
        if nproc is None:
            nproc = 2
        self.set_num_threads(nproc)

        self.model.scripts.run_parallel.RunT(nproc=nproc, debug=debug_level)

        os.chdir(odir)

    def viewer_canvasmenu(self):
        menus = [("+Petra-M", None, None), ]
        if self._hidemesh:
            menus.append(("Show Mesh",  self.onShowMesh, None))
        else:
            menus.append(("Hide Mesh",  self.onHideMesh, None))

        selmode = self.get_sel_mode()

        menus.append(("Copy " + selmode + " selection",
                      self.onCopySelection1, None))
        if self._view_mode == 'geom':
            menus.append(
                ("Copy " + selmode + " selection with prefix", self.onCopySelection2, None))

        # if len(self.canvas.selection) > 0:
        #    if self._view_mode == 'mesh':
        #        menus.append(("Show meshed " + selmode, self.onShowMeshedEntity, None))
        #        menus.append(("Hide meshed " + selmode, self.onHideMeshedEntity, None))

        if self.editdlg is not None and self._palette_focus == 'edit':
            check, kind, cidxs, labels = self.editdlg.isSelectionPanelOpen()
            if check:
                if kind == 'domain':
                    idx = self._dom_bdr_sel[0]
                elif kind == 'bdry':
                    idx = self._dom_bdr_sel[1]
                elif kind == 'edge':
                    idx = self._dom_bdr_sel[2]
                elif kind == 'point':
                    idx = self._dom_bdr_sel[3]
                else:
                    idx = None
                k = 0
                for cidx, label in zip(cidxs, labels):
                    if label == '':
                        continue
                    show_rm = any([x in cidx for x in idx])
                    show_add = any([not x in cidx for x in idx])

                    if show_add:
                        m = getattr(self, 'onAddSelection'+str(k))
                        txt = "Add to " + label
                        menus.append((txt, m, None))
                    if show_rm:
                        m = getattr(self, 'onRmSelection'+str(k))
                        txt = "Remove from " + label
                        menus.append((txt, m, None))
                    k = k + 1

        elif self.plotsoldlg is not None and self._palette_focus == 'plot':
            kind, cidx = self.plotsoldlg.get_selected_plotmode(kind=True)
            if kind == 'domain':
                idx = self._dom_bdr_sel[0]
            elif kind == 'bdry':
                idx = self._dom_bdr_sel[1]
            elif kind == 'edge':
                idx = self._dom_bdr_sel[2]
            elif kind == 'point':
                idx = self._dom_bdr_sel[3]
            else:
                idx = None
            if idx is not None:
                if cidx != 'all':
                    show_rm = any([x in cidx for x in idx])
                    show_add = any([not x in cidx for x in idx])
                else:
                    show_rm = False
                    show_add = False

                m1 = self.plotsoldlg.add_selection
                m2 = self.plotsoldlg.rm_selection
                m3 = self.plotsoldlg.set_selection

                def onAddPlotSel(evt, idx=idx):
                    m1(idx)
                    evt.Skip()

                def onRmPlotSel(evt, idx=idx):
                    m2(idx)
                    evt.Skip()

                def onSetPlotSel(evt, idx=idx):
                    m3(idx)
                    evt.Skip()
                if show_add:
                    menus.append(("Add to plot " + kind +
                                  " selection", onAddPlotSel, None))
                if show_rm:
                    menus.append(("Remove from plot " + kind + " selection", onRmPlotSel,
                                  None))
                menus.append(("Set to plot " + kind +
                              " selection", onSetPlotSel, None))
        elif self.selection_palette is not None and self._palette_focus == 'selection':
            pass
        else:
            pass

        palette_menu = []
        if self.selection_palette is None:
            palette_menu.append(
                ("Selection palette...", self.onSelectionPanel, None),)

        if (self._view_mode == 'geom' and
            self._view_mode_group.startswith('OCC') and
                self.geom_info_palette is None):
            palette_menu.append(("Geomtry info ...", self.onGeomInfo,  None),)

        if len(palette_menu) > 0:
            menus.append(('---', None, None),)
            menus.extend(palette_menu)

        menus.extend([("!", None, None),
                      ("---", None, None), ])
        return menus

    def _on_cp_selection1(self, use_prefix=False):

        kind = self.get_sel_mode()
        if kind == 'volume':
            idx = self._dom_bdr_sel[0]
            prefix = 'v'

        elif kind == 'face':
            idx = self._dom_bdr_sel[1]
            prefix = 'f'

        elif kind == 'edge':
            idx = self._dom_bdr_sel[2]
            prefix = 'l'

        elif kind == 'point':
            idx = self._dom_bdr_sel[3]
            prefix = 'p'

        else:
            pass

        prefix = prefix if use_prefix else ''

        return ", ".join([prefix + str(x) for x in idx])

    def onCopySelection1(self, evt):
        txt = self._on_cp_selection1(use_prefix=False)
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(wx.TextDataObject(txt))
            wx.TheClipboard.Close()

    def onCopySelection2(self, evt):
        txt = self._on_cp_selection1(use_prefix=True)
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(wx.TextDataObject(txt))
            wx.TheClipboard.Close()

    def onAddSelection(self, evt, flag=0):
        check, kind, cidx, labels = self.editdlg.isSelectionPanelOpen()
        if check:
            if kind == 'domain':
                idx = self._dom_bdr_sel[0]
            elif kind == 'bdry':
                idx = self._dom_bdr_sel[1]
            elif kind == 'edge':
                idx = self._dom_bdr_sel[2]
            elif kind == 'point':
                idx = self._dom_bdr_sel[3]
            else:
                idx = None
            if idx is not None:
                self.editdlg.add_remove_AreaSelection(idx, flag=flag)

    def onAddSelection0(self, evt):
        self.onAddSelection(evt, flag=0)

    def onAddSelection1(self, evt):
        self.onAddSelection(evt, flag=1)

    def onAddSelection2(self, evt):
        self.onAddSelection(evt, flag=2)

    def onRmSelection(self, evt, flag=0):
        check, kind, cidx, labels = self.editdlg.isSelectionPanelOpen()
        if check:
            if kind == 'domain':
                idx = self._dom_bdr_sel[0]
            elif kind == 'bdry':
                idx = self._dom_bdr_sel[1]
            elif kind == 'pair':
                idx = self._dom_bdr_sel[1]
            else:
                idx = None
            if idx is not None:
                self.editdlg.add_remove_AreaSelection(idx, rm=True, flag=flag)

    def onRmSelection0(self, evt):
        self.onRmSelection(evt, flag=0)

    def onRmSelection1(self, evt):
        self.onRmSelection(evt, flag=1)

    def onRmSelection2(self, evt):
        self.onRmSelection(evt, flag=2)

    def onSelVolume(self, evt):
        self.set_sel_mode('volume')

    def onSelFace(self, evt):
        self.set_sel_mode('face')

    def onSelEdge(self, evt):
        self.set_sel_mode('edge')

    def onSelPoint(self, evt):
        self.set_sel_mode('point')

    def onSelAny(self, evt):
        self.set_picker_mask('')
        self._sel_mode = ''

    def set_sel_mode(self, mode=None):
        if mode is None:
            self.refresh_toolbar_buttons()
            return
        self._sel_mode = mode
        mask = mode if mode != 'volume' else 'face'
        self.set_picker_mask(mask)

    def get_sel_mode(self):
        return self._sel_mode

    def set_picker_mask(self, key):
        for name, child in self.get_axes().get_children():
            child.set_pickmask(not name.startswith(key))

    def onShowMesh(self, evt=None):
        from petram.mesh.plot_mesh import plot_domainmesh
        mesh = self.engine.get_mesh()
        self._hidemesh = False
        self.update(False)

        children = [child
                    for name, child in self.get_axes().get_children()
                    if name.startswith('face')]
        childrene = [child
                     for name, child in self.get_axes().get_children()
                     if name.startswith('face') and name.endswith('e')]
        if len(childrene) > 0:
            for child in childrene:
                if not child.isempty():
                    child.set_linewidth(1.0, child._artists[0])
        else:
            for child in children:
                if not child.isempty():
                    child.set_linewidth(1.0, child._artists[0])
        self.update(True)
        self.draw_all()

    def onHideMesh(self, evt=None):
        self._hidemesh = True
        self.update(False)
        mesh = self.engine.get_mesh()
        children = [child
                    for name, child in self.get_axes().get_children()
                    if name.startswith('face')]
        for child in children:
            child.set_linewidth(0.0, child._artists[0])
        self.update(True)
        self.draw_all()

    def onShowAll(self, evt):
        for obj in self.book.walk_tree():
            if obj.is_suppress():
                obj.set_suppress(False)
        self.draw()

    def onPlotExpr(self, evt):
        try:
            self.plotexprdlg.Raise()
            return
        except AttributeError:
            pass
        except:
            traceback.print_exc()
            pass

        choices = self.model.param.getvar('mfem_model')['Phys'].keys()
        if len(choices) == 0:
            ans = dialog.message(self,
                                 "No physics is defined",
                                 style=0)
            return

        iattr = [x().figobj.name.split('_')[1]
                 for x in self.canvas.selection
                 if x().figobj.name.startswith('bdry')]

        from petram.pi.dlg_plot_expr import DlgPlotExpr
        # try:
        if self.plotexprdlg is not None:
            self.plotexprdlg.Raise()
        else:
            self.plotexprdlg = DlgPlotExpr(
                self, close_cb=self.onDlgPlotExprClose)

        evt.Skip()

    def onDlgPlotExprClose(self, evt):
        self.plotexprdlg = None
        evt.Skip()

    def onDlgPlotSol(self, evt):
        m = self.model.param.getvar('mfem_model')
        m.set_root_path(self.model.owndir())

        from petram.pi.dlg_plot_sol import DlgPlotSol
        # try:
        if self.plotsoldlg is not None:
            self.plotsoldlg.Raise()
        else:
            self.plotsoldlg = DlgPlotSol(self, close_cb=self.onDlgPlotSolClose)
        # except:
        #    self.plotsoldlg = DlgPlotSol(self)
        self.plotsoldlg.load_sol_if_needed()

    def onDlgPlotSolClose(self, evt):
        self.plotsoldlg = None
        evt.Skip()

    def onNewNS(self, evt):
        ret, txt = dialog.textentry(self,
                                    "Enter namespace name", "New NS...", '')
        if not ret:
            return
        self.model.scripts.helpers.create_ns(txt)
        from ifigure.events import SendChangedEvent
        SendChangedEvent(self.model, w=self)
        evt.Skip()

    def onLoadNS(self, evt):
        from petram.pi.ns_utils import import_ns
        import_ns(self.model)
        from ifigure.events import SendChangedEvent
        SendChangedEvent(self.model, w=self)
        evt.Skip()

    def onExportNS(self, evt):
        choices = ['_'.join(name.split('_')[:-1])
                   for name, child in self.model.namespaces.get_children()]
        if len(choices) == 0:
            return

        ll = [
            [None, choices[0], 4, {'style': wx.CB_READONLY,
                                   'choices': choices}], ]
        from ifigure.utils.edit_list import DialogEditList
        ret = DialogEditList(ll, modal=True, parent=self,
                             title='Select Namespace')
        if not ret[0]:
            return
        name = str(ret[1][0])

        from petram.pi.ns_utils import export_ns
        export_ns(self.model, name)
        from ifigure.events import SendChangedEvent
        SendChangedEvent(self.model, w=self)
        evt.Skip()

    def onRebuildNS(self, evt):
        try:
            self.rebuild_ns()
        except:
            dialog.showtraceback(parent=self,
                                 txt='Failed to rebuild namespace',
                                 title='Error',
                                 traceback=traceback.format_exc())
            return

        dialog.message(parent=self,
                       message='Namespace is built successfully. ',
                       title='Passed',
                       center_on_screen=True)

        evt.Skip()

    def onClearSol(self, evt):
        self.model.scripts.helpers.clear_sol(w=self)
        self.model.param.setvar('sol', None)
        evt.Skip()

    def onNewLocalSol(self, evt):
        import re

        model = self.model
        if not model.has_child('solutions'):
            model.add_folder('solutions')

        param = model.param
        sol = param.eval('sol')
        if sol is None:
            names = sorted([x[0] for x in model.solutions.get_children()])
        else:
            names = [sol.name]

        if len(names) > 0:
            numbers = [''.join(re.findall('\d+$', n)) for n in names]
            numbers = [int(x if len(x) > 0 else '0') for x in numbers]
            basename = [n.rstrip('0123456789') for n in names]
            txt = basename[-1] + str(numbers[-1]+1)
        else:
            txt = 'sol'

        from ifigure.widgets.dialog import textentry
        f, txt = textentry(self, message='Enter local sol name',
                           title='Creating sol directory',
                           def_string=txt, center=True)
        if not f:
            return
        if not txt in names:
            folder = model.solutions.add_folder(txt)
            folder.mk_owndir()
        else:
            folder = model.solutions.get_child(name=txt)

        param.setvar('sol', '='+folder.get_full_path())

    # def onActivate(self, evt):
    #    windows = [self.editdlg, self.plotsoldlg, self.plotexprdlg]
    #    for w in windows:
    #        if w is not None:
    #            if not w.IsShown():
    #                w.Show()
    #    evt.Skip()

    def onWindowClose(self, evt=None):
        if self.editdlg is not None:
            try:
                self.editdlg.Destroy()
            except:
                pass
            self.editdlg = None
        if self.plotsoldlg is not None:
            try:
                self.plotsoldlg.Destroy()
            except:
                pass
            self.plotsoldlg = None

        super(MFEMViewer, self).onWindowClose(evt)

    def onServerSetting(self, evt):
        '''
        param needs to have two parameters
        remote = {'name': 'eofe7',
                  'rwdir': remote working directory,
                  'sol', None)
        host : points host setting object
        '''
        remote = self.model.param.eval('remote')
        if remote is not None:
            hostname = remote['name']
        else:
            hostname = ''

        proj = self.model.get_root_parent()
        p = proj.setting.parameters
        no_existing_c = True

        if p.hasvar('connection'):
            c = p.eval('connection')
            if c is not None:
                names = c.get_childnames()
                no_existing_c = (len(names) == 0)

        if no_existing_c:
            remote = {'name': '',
                      'rwdir': '',
                      'sol': ''}
            ret, new_name = dialog.textentry(self,
                                             "Enter the name of new connection",
                                             "Add Connection",
                                             hostname, center=False,
                                             center_on_screen=True)
        else:
            c = p.eval('connection')
            names = c.get_childnames()
            names.append("New...")
            ret, new_name = dialog.textselect(self,
                                              "Select the name of connection or New...",
                                              "Add Connection",
                                              def_string=hostname,
                                              center=False,
                                              center_on_screen=True,
                                              choices=names,
                                              endmodal_on_lastvalue=names[-1])
            if not ret:
                return
            if (ret and new_name == "New...") or c.get_child(name=new_name) is None:
                remote = {'name': '',
                          'rwdir': '',
                          'sol': ''}
                if new_name == "New...":
                    ret, new_name = dialog.textentry(self,
                                                     "Enter the name of new connection",
                                                     "Add Connection",
                                                     "enter-hostname.pppl.gov",
                                                     center=False,
                                                     center_on_screen=True)
                else:
                    pass
            else:
                child = c.get_child(name=new_name)
                child.onSetting()
                remote['name'] = new_name
                self.model.param.setvar('host', '='+child.get_full_path())
                return

        if ret:
            self.model.param.setvar('remote', remote)
            remote['name'] = new_name
            from petram.remote.client_script import make_remote_connection
            obj = make_remote_connection(self.model, new_name)
            self.model.param.setvar('host', '='+obj.get_full_path())

    def onServerSolve(self, evt):
        m = self.model.param.getvar('mfem_model')
        m.set_root_path(self.model.owndir())
        odir = os.getcwd()

        try:
            self.engine.run_verify_setting()
        except:
            os.chdir(odir)
            dialog.showtraceback(parent=self,
                                 txt='Failed to verify setting',
                                 title='Error',
                                 traceback=traceback.format_exc())
            return

        try:
            self.run_preprocess()
        except:
            os.chdir(odir)
            dialog.showtraceback(parent=self,
                                 txt='Failed to during pre-processing model data',
                                 title='Error',
                                 traceback=traceback.format_exc())
            return

        remote = self.model.param.eval('remote')
        if remote is None:
            return

        from petram.pi.dlg_submit_job import get_defaults
        values, keys = get_defaults()

        for i, key in enumerate(keys):
            if remote.get(key, None) is not None:
                values[i] = remote.get(key, None)

        from petram.pi.dlg_submit_job import get_job_submisson_setting
        from petram.remote.client_script import get_job_queue

        dlg = progressbar(self, 'Checking queue config...',
                          'In progress', 5,
                          can_abort=True)
        dlg.Show()
        wx.GetApp().Yield()

        success, q = get_job_queue(self.model, progdlg=dlg)
        if dlg:
            dlg.Destroy()

        if not success:
            q = {'type': '',
                 'queues': [{'name': 'failed to read queue config'}, ]}

        if q["queues"][0]['name'] == ' no queue for a user':
            from ifigure.widgets.dialog import message
            ret = message(parent=self,
                          message="You do not have permission to access use to Petra-M on this computer system.",
                          title="No queue is available",
                          icon=wx.ICON_EXCLAMATION,
                          center_on_screen=False,
                          center_on_parent=True,)
            return

        setting = get_job_submisson_setting(self, remote['name'].upper(),
                                            value=values,
                                            queues=q)
        if len(setting) == 0:
            return

        dlg = progressbar(self, 'Preparing...', 'Job submission',
                          4, can_abort=True)
        dlg.Show()
        wx.GetApp().Yield()

        from petram.remote.client_script import prepare_remote_dir
        # if remote['rwdir'] != setting['rwdir']:
        cancelled = prepare_remote_dir(self.model,
                                       setting['rwdir'],
                                       dirbase='',
                                       progdlg=dlg)
        if not dlg:
            return
        if cancelled:
            dlg.Destroy()
            return

        for k in setting.keys():
            remote[k] = setting[k]
        if self.model.param.eval('sol') is None:
            sol = self.model.scripts.helpers.make_new_sol()
        else:
            sol = self.model.param.eval('sol')

        dlg.Update(1, newmsg="Preparing Data...")
        wx.GetApp().Yield()
        # remote.scripts.clean_remote_dir()
        sol.clean_owndir()
        self.model.scripts.helpers.save_model(os.path.join(sol.owndir(),
                                                           'model.pmfm'),
                                              meshfile_relativepath=True)

        from petram.remote.client_script import send_file, submit_job

        dlg.Update(2, newmsg="Sending file")
        wx.GetApp().Yield()

        cancelled = send_file(self.model, dlg,
                              skip_mesh=setting['skip_mesh'])
        if not dlg:
            return
        if cancelled:
            dlg.Destroy()
            return

        dlg.Update(3, newmsg="Submitting a job...")
        wx.GetApp().Yield()

        cancelled = submit_job(self.model, dlg)
        if not dlg:
            return
        if cancelled:
            dlg.Destroy()
            return

        dlg.Update(3, newmsg="Done...")
        wx.GetApp().Yield()
        wx.Sleep(1)

        dlg.Destroy()

    def onServerRetrieve(self, evt):
        from petram.remote.client_script import retrieve_files
        sol = self.model.param.eval('sol')
        if sol is None:
            sol = self.model.scripts.helpers.make_new_sol()
        sol_dir = sol.owndir()
        retrieve_files(self.model,
                       sol_dir=sol_dir)

    def run_preprocess(self):
        model = self.model
        engine = self.engine
        engine.preprocess_ns(model.namespaces, model.datasets)
        engine.build_ns()
        engine.run_preprocess(model.namespaces, model.datasets)

    def rebuild_ns(self):
        engine = self.engine
        model = self.model
        engine.preprocess_ns(model.namespaces, model.datasets)
        engine.build_ns()

    def get_internal_bc(self):
        d = self._s_v_loop['phys'][1]
        dd = []
        for k in d:
            dd.extend(d[k])

        from collections import defaultdict
        seen = defaultdict(int)
        for k in dd:
            seen[k] += 1
        return [k for k in seen if seen[k] > 1]

    def handle_shifthit(self):
        '''
        when space is hit in canvas
        '''
        if self._sel_mode == 'volume':
            mode = 'point'
        if self._sel_mode == 'face':
            mode = 'volume'
        if self._sel_mode == 'edge':
            mode = 'face'
        if self._sel_mode == 'point':
            mode = 'edge'
        self.set_toolbar_mode(mode)
        '''
        self._sel_mode = mode
        
        toolbarname = self.canvas.toolbar.p1_choice[1]
        buttonname = bmode + '_' + toolbarname

        if toolbarname == 'petram_mesh':
            buttonname = "m"+buttonname
        if toolbarname == 'petram_geom':
            buttonname = "g"+buttonname

        self.canvas.toolbar.ClickP1Button(buttonname)
        self.canvas.toolbar.ClickP1Button('select')
        '''

    def set_toolbar_mode(self, mode):
        bmodes = {'volume': 'domain',
                  'face': 'face',
                  'edge': 'edge',
                  'point': 'dot'}
        if not mode in bmodes:
            return
        bmode = bmodes[mode]

        self.set_sel_mode(mode)

        toolbarname = self.canvas.toolbar.p1_choice[1]
        buttonname = bmode + '_' + toolbarname

        if toolbarname == 'petram_mesh':
            buttonname = "m"+buttonname
        if toolbarname == 'petram_geom':
            buttonname = "g"+buttonname

        self.canvas.toolbar.ClickP1Button(buttonname)
        self.canvas.toolbar.ClickP1Button('select')
