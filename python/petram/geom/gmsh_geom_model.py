from __future__ import print_function
from petram.phys.vtable import VtableElement, Vtable, Vtable_mixin
from petram.namespace_mixin import NS_mixin
from petram.geom.geom_model import GeomBase, GeomTopBase
import petram.geom.gmsh_config
from threading import Thread
from petram.model import Model

import tempfile
import os
import subprocess
import traceback
import sys
import re
import time
import multiprocessing as mp
from collections import defaultdict

import numpy as np
import warnings

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('GmshGeomModel')

try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty  # python 3.x


debug = True
geom_key_dict = {'SurfaceBase': 'sb',
                 'PlaneSurface': 'sp',
                 'Point': 'pt',
                 'Line': 'ln',
                 'Spline': 'sp',
                 'SurfaceID': 's',
                 'VolumeID': 'v',
                 'LineID': 'l',
                 'VertexID': 'p'}


def get_gmsh_exe():
    from shutil import which

    gmsh_executable = which('gmsh')
    if gmsh_executable is not None:
        return gmsh_executable
    else:
        macos_gmsh_location = '/Applications/Gmsh.app/Contents/MacOS/gmsh'
        if os.path.isfile(macos_gmsh_location):
            return macos_gmsh_location
    return 'gmsh'


def get_gmsh_major_version():
    gmsh_exe = get_gmsh_exe()
    try:
        out = subprocess.check_output(
            [gmsh_exe, '--version'],
            stderr=subprocess.STDOUT
        ).strip().decode('utf8')
    except BaseException:
        return -1
    ex = out.split('.')
    return int(ex[0])


use_gmsh_api = True


def get_geom_key(obj):
    if obj.__class__.__name__ in geom_key_dict:
        return geom_key_dict[obj.__class__.__name__]

    assert False, " name not found for " + \
        obj.__class__.__name__ + " in " + str(geom_key_dict)

    # it should not come here...
    name = obj.__class__.__name__
    key = ''.join([i.lower() for i in name if not i.isupper()])
    for k in geom_key_dict.keys():
        if geom_key_dict[k] == key:
            assert False, key + " is used for " + k.__name__

    geom_key_dict[obj.__class__] = key
    if debug:
        print(geom_key_dict)
    return key


def get_twoletter_keys(t):
    if t == 'p':
        return 'pt'
    if t == 'l':
        return 'ln'
    if t == 'f':
        return 'fs'
    if t == 'v':
        return 'vl'
    return t


class GeomObjs(dict):
    def duplicate(self):
        if not hasattr(self, "_past_keys"):
            self._past_keys = []
        obj = GeomObjs(self)
        obj._past_keys = self._past_keys
        return obj

    def addobj(self, obj, name):
        key = ''.join([i for i in name if not i.isdigit()])
        key = get_twoletter_keys(key)
        if not hasattr(self, "_past_keys"):
            self._past_keys = []
        keys = self._past_keys
        nums = []
        for k in keys:
            t = ''.join([i for i in k if not i.isdigit()])
            if t == key:
                n = int(''.join([i for i in k if i.isdigit()]))
                nums.append(n)

        if len(nums) == 0:
            newkey = key + str(1)
        else:
            newkey = key + str(max(nums) + 1)
        self[newkey] = obj
        self._past_keys.append(newkey)
        return newkey


class GmshPrimitiveBase(GeomBase, Vtable_mixin):
    hide_ns_menu = True
    has_2nd_panel = False
    isGeom = True
    isWP = False

    def __init__(self, *args, **kwargs):
        super(GmshPrimitiveBase, self).__init__(*args, **kwargs)
        NS_mixin.__init__(self, *args, **kwargs)

    def attribute_set(self, v):
        v = super(GmshPrimitiveBase, self).attribute_set(v)
        self.vt.attribute_set(v)
        return v

    def panel1_param(self):
        from wx import BU_EXACTFIT
        # b1 = {"label": "S", "func": self.onBuildBefore,
        #      "noexpand": True, "style": BU_EXACTFIT}
        b2 = {"label": "R", "func": self.onBuildAfter,
              "noexpand": True, "style": BU_EXACTFIT}

        ll = [[None, None, 241, {'buttons': [b2],  # b2],
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
        return self.vt.import_panel_value(self, v[1:])

    def panel1_tip(self):
        return [None] + list(self.vt.panel_tip())

    # def build_geom(self, geom, objs):
    #    self._newobjs = []
    #    warnings.warn("Not implemented: " + self.__class__.__name__, Warning)

    def gsize_hint(self, geom, objs):
        '''
        return quick estimate of geometry size min and max
        '''
        warnings.warn("Not implemented", Warning)
        return -1, -1

    def get_special_menu(self, evt):
        return [('Build this step', self.onBuildAfter, None)]

    def _onBuildThis(self, evt, **kwargs):
        dlg = evt.GetEventObject().GetTopLevelParent()
        viewer = dlg.GetParent()
        engine = viewer.engine
        engine.build_ns()
        kwargs['gui_parent'] = dlg

        try:
            p = self.parent
            while True:
                if isinstance(p, GmshGeom):
                    rootg = p
                    break
                p = p.parent

            rootg._geom_finalized = False

            od = os.getcwd()
            os.chdir(viewer.model.owndir())
            rootg.build_geom(**kwargs)
            os.chdir(od)
            success = True
        except BaseException:
            os.chdir(od)

            import ifigure.widgets.dialog as dialog
            dialog.showtraceback(parent=dlg,
                                 txt='Failed to build geometry',
                                 title='Error',
                                 traceback=traceback.format_exc())
            success = False

        dlg.OnRefreshTree()
        rootg.onUpdateGeoView(evt)
        return success

    def onBuildBefore(self, evt):
        dlg = evt.GetEventObject().GetTopLevelParent()
        dlg.import_selected_panel_value()

        mm = dlg.get_selected_mm()
        if mm is None:
            return

        self._onBuildThis(evt, stop1=mm)
        evt.Skip()

    def onBuildAfter(self, evt):
        dlg = evt.GetEventObject().GetTopLevelParent()
        dlg.import_selected_panel_value()
        mm = dlg.get_selected_mm()
        if mm is None:
            return

        success = self._onBuildThis(evt, stop2=mm)

        if success:
            dlg = evt.GetEventObject().GetTopLevelParent()
            import wx
            wx.CallAfter(dlg.select_next_enabled)

        evt.Skip()

    def add_geom_sequence(self, geom):
        gui_name = self.fullname()
        self.vt.preprocess_params(self)
        gui_param = self.vt.make_value_or_expression(self)
        geom_name = self.__class__.__name__
        geom.add_sequence(gui_name, gui_param, geom_name)


class GmshGeom(GeomTopBase):
    has_2nd_panel = False

    @classmethod
    def fancy_menu_name(self):
        return 'Gmsh Geometry'

    @classmethod
    def fancy_tree_name(self):
        return 'GmshSequence'

    def __init__(self, *args, **kwargs):
        super(GmshGeom, self).__init__(*args, **kwargs)
        NS_mixin.__init__(self, *args, **kwargs)

    '''
    def __del__(self):
        self.terminate_child()
    '''
    '''
    def terminate_child(self):
        if (hasattr(self, "_p") and self._p[0].is_alive()):
            self._p[1].close()
            self._p[2].close()
            self._p[1].cancel_join_thread()
            self._p[2].cancel_join_thread()
            self._p[0].terminate()
            del self._p
    '''
    @property
    def is_finalized(self):
        if not hasattr(self, "_geom_finalized"):
            self._geom_finalized = False
        return self._geom_finalized

    @property
    def geom_finalized(self):
        if not hasattr(self, "_geom_finalized"):
            self._geom_finalized = False
        return self._geom_finalized

    @geom_finalized.setter
    def geom_finalized(self, value):
        self._geom_finalized = value

    @property
    def geom_data(self):
        return self._gmsh4_data

    @property
    def build_stop(self):
        if not hasattr(self, "_build_stop"):
            self._build_stop = (None, None)
        return self._build_stop

    @property
    def geom_brep(self):
        if not hasattr(self, "_geom_brep"):
            return ''
        base = self._geom_brep

        if self._geom_brep_dir == '':
            import wx
            trash = wx.GetApp().GetTopWindow().proj.get_trash()
        else:
            trash = self._geom_brep_dir

        return os.path.join(trash, base)

    def attribute_set(self, v):
        v = super(GmshGeom, self).attribute_set(v)
        v['geom_timestamp'] = 0
        v['geom_prev_algorithm'] = 2
        v['geom_prev_res'] = 3
        v['occ_parallel'] = False
        v['maxthreads'] = 1
        v['skip_final_frag'] = False
        v['use_1d_preview'] = False
        v['use_occ_preview'] = False
        v['use_curvature'] = False
        v['long_edge_thr'] = 0.3
        v['small_edge_thr'] = 0.001
        v['small_edge_seg'] = 1
        v['max_seg'] = 30
        return v

    def get_possible_child(self):
        from petram.geom.geom_primitives import (Point, PointCenter, PointByUV, PointOnEdge,
                                                 PointCircleCenter, Line, Spline,
                                                 Circle, CircleByAxisPoint, CircleBy3Points,
                                                 Rect, Polygon, Box, Ball,
                                                 Cone, Wedge, Cylinder, Torus, Extrude, Revolve, Sweep,
                                                 LineLoop, CreateLine, CreateSurface, CreateVolume,
                                                 SurfaceLoop, Union, Union2, Intersection, Difference, Fragments,
                                                 SplitByPlane, Copy, Remove, Remove, Remove2, RemoveFaces,
                                                 Move, Rotate, Flip, Scale, WorkPlane,
                                                 WorkPlaneByPoints, healCAD, CADImport, BrepImport,
                                                 Fillet, Chamfer,
                                                 Array, ArrayRot, ArrayByPoints, ArrayRotByPoints,
                                                 ThruSection, RotateCenterPoints, MoveByPoints)
        return [Point, PointCenter, PointOnEdge, PointByUV, PointCircleCenter,
                Line, Circle, CircleByAxisPoint, CircleBy3Points,
                Rect, Polygon, Spline, Box,
                Ball, Cone, Wedge, Cylinder,
                Torus, CreateLine, CreateSurface, CreateVolume, LineLoop, SurfaceLoop,
                Extrude, Revolve, Sweep, Union, Union2,
                Intersection, Difference, Fragments, SplitByPlane, Copy, Remove,
                Remove2, RemoveFaces, Move, Rotate,
                Flip, Scale, WorkPlane, WorkPlaneByPoints, healCAD, CADImport, BrepImport,
                Fillet, Chamfer, Array, ArrayRot, ArrayByPoints, ArrayRotByPoints,
                ThruSection, RotateCenterPoints, MoveByPoints]

    def get_possible_child_menu(self):
        from petram.geom.geom_primitives import (Point, PointCenter, PointCircleCenter,
                                                 PointOnEdge, PointByUV,  Line, Spline,
                                                 Circle, CircleByAxisPoint, CircleBy3Points,
                                                 Rect, Polygon, Box, Ball,
                                                 Cone, Wedge, Cylinder, Torus, Extrude, Revolve, Sweep,
                                                 LineLoop, CreateLine, CreateSurface, CreateVolume,
                                                 SurfaceLoop, Union, Union2, Intersection, Difference, Fragments,
                                                 SplitByPlane, Copy, Remove, Remove2, RemoveFaces,
                                                 Move, Rotate, Flip, Scale,
                                                 WorkPlane, WorkPlaneByPoints, healCAD, CADImport, BrepImport,
                                                 Fillet, Chamfer,
                                                 Array, ArrayRot, ArrayByPoints, ArrayRotByPoints,
                                                 ThruSection, RotateCenterPoints, MoveByPoints)

        return [("", Point), ("", Line), ("", Circle), ("", Rect), ("", Polygon),
                ("", Spline), ("", Fillet), ("", Chamfer),
                ("3D shape...", Box),
                ("", Ball), ("", Cone), ("", Wedge), ("", Cylinder),
                ("!", Torus),
                ("Create...", CreateLine), ("", CreateSurface), ("!", CreateVolume),
                #("", LineLoop), ("", SurfaceLoop),
                ("Protrude...", Extrude), ("", Revolve), ("!", Sweep),
                ("", Copy), ("", Remove),
                ("Translate...", Move,), ("", Rotate), ("", Flip), ("", Scale),
                ("", Array), ("!", ArrayRot),
                ("Boolean...", Union), ("", Intersection), ("",
                                                            Difference), ("", Fragments), ("!", SplitByPlane),
                ("WorkPlane...", WorkPlane), ("!", WorkPlaneByPoints),
                ("Import...", BrepImport), ("", CADImport), ("!", healCAD),
                ]

    def get_special_menu(self, evt):
        menu = [('Build All', self.onBuildAll, None),
                ('Export .brep', self.onExportBrep, None)]

        if hasattr(self, '_gso'):
            print('here')
            if self._gso.child_alive():
                m2 = [('---', None, None),
                      ('Terminate geometry process.',
                       self.onTerminateChild, None),
                      ('---', None, None), ]
                print(m2)
                menu.extend(m2)
        return menu

    def panel1_param(self):
        import wx
        return [["", "Geometry model using GMSH", 2, None],
                ["PreviewAlgorith", "Automatic", 4, {"style": wx.CB_READONLY,
                                                     "choices": ["Auto", "MeshAdpat",
                                                                 "Delaunay", "Frontal"]}],
                ["Preview Resolution", 30, 400, None],
                ["Long  Edge Thr.", self.long_edge_thr, 300, None],
                ["Small Edge Thr.", self.small_edge_thr, 300, None],
                ["Small Edge #Seg.", self.small_edge_seg, 400, None],
                ["Max #seg in Preview", self.max_seg, 400, None],
                ["Preview #threads", self.maxthreads, 400, None],
                [None, self.occ_parallel, 3, {"text": "OCC parallel boolean"}],
                [None, self.skip_final_frag, 3, {
                    "text": "Skip fragmentationn"}],
                [None, self.use_1d_preview, 3, {"text": "Use line preview"}],
                #                [None, self.use_occ_preview, 3, {
                #                    "text": "OCC preview (in dev.)"}],
                [None, self.use_curvature, 3, {
                    "text": "Consider curvature in preview generation"}],
                [None, None, 341, {"label": "Finalize Geom",
                                   "func": 'onBuildAll',
                                   "noexpand": True}], ]

    def get_panel1_value(self):
        aname = {2: "Auto", 1: "MeshAdpat", 5: "Delaunay", 6: "Frontal"}
        txt = aname[self.geom_prev_algorithm]
        return [None, txt, self.geom_prev_res, self.long_edge_thr,
                self.small_edge_thr, self.small_edge_seg, self.max_seg,
                self.maxthreads, self.occ_parallel,
                self.skip_final_frag, self.use_1d_preview, self.use_curvature, self]

    def import_panel1_value(self, v):
        aname = {2: "Auto", 1: "MeshAdpat", 5: "Delaunay", 6: "Frontal"}
        for k in aname:
            if v[1] == aname[k]:
                self.geom_prev_algorithm = k

        self.geom_prev_res = int(v[2])
        self.long_edge_thr = float(v[3])
        self.small_edge_thr = float(v[4])
        self.small_edge_seg = int(v[5])
        self.max_seg = int(v[6])
        self.maxthreads = int(v[7])
        self.occ_parallel = v[8]
        self.skip_final_frag = v[9]
        self.use_1d_preview = v[10]
        #self.use_occ_preview = v[11]
        self.use_curvature = v[11]
        self.use_occ_preview = False

    def onBuildAll(self, evt):
        dlg = evt.GetEventObject().GetTopLevelParent()
        viewer = dlg.GetParent()
        engine = viewer.engine
        engine.build_ns()

        try:
            od = os.getcwd()
            os.chdir(viewer.model.owndir())
            self.build_geom(finalize=True, gui_parent=dlg)
            os.chdir(od)
        except BaseException:
            os.chdir(od)

            import ifigure.widgets.dialog as dialog
            dialog.showtraceback(parent=dlg,
                                 txt='Failed to build geometry',
                                 title='Error',
                                 traceback=traceback.format_exc())
        dlg.OnRefreshTree()

        filename = os.path.join(viewer.model.owndir(), self.name())
        self.onUpdateGeoView(evt, filename=filename)

        '''
        if not use_gmsh_api:
            fid = open(filename + '.geo_unrolled', 'w')
            fid.write('\n'.join(self._txt_unrolled))
            fid.close()
        '''
        self.geom_finalized = True
        self.geom_timestamp = time.ctime()
        evt.Skip()

    def onUpdateGeoView4(self, evt, filename=None):
        if self._gmsh4_data is None:
            return

        dlg = evt.GetEventObject().GetTopLevelParent()
        viewer = dlg.GetParent()
        self.update_figure_data(viewer)

    def update_figure_data(self, viewer):
        if not hasattr(self, "_gmsh4_data"):
            return
        if self._gmsh4_data is None:
            return

        ptx, cells, cell_data, l, s, v, geom = self._gmsh4_data
        ret = ptx, cells, {}, cell_data, {}

        self._geom_coords = ret
        viewer.set_figure_data('geom', self.name(), ret)
        viewer.set_figure_data('mesh', self.name(), ret)
        viewer.update_figure('geom', self.name())

        viewer._s_v_loop['geom'] = s, v
        viewer._s_v_loop['mesh'] = s, v

    def onUpdateGeoView(self, evt, filename=None):
        return self.onUpdateGeoView4(evt, filename=filename)

    def walk_over_geom_chidlren(self, geom, stop1=None, stop2=None):
        geom.clear_sequence()

        self._build_stop = (None, None)

        children = [x for x in self.walk()]
        children = children[1:]
        for child in children:
            if hasattr(child, "_newobjs"):
                del child._newobjs

        def proccess_children(children, local_ns=None):
            if local_ns is None:
                local_ns = {}

            do_break = False

            for child in children:
                if not child.enabled:
                    continue

                if len(child.get_children()) > 0 and not child.isWP:
                    children2 = child.get_children()
                    ll = local_ns.copy()
                    ll.update(child.seq_values)
                    do_break = proccess_children(children2, local_ns=ll)
                    if child is stop2:
                        do_break = True
                        break
                    if do_break:
                        do_break = True
                        break            # for build after

                elif len(child.get_children()) == 0 and not child.isWP:
                    child.vt.preprocess_params(child)
                    if child is stop1:
                        break            # for build before
                    child.add_geom_sequence(geom)
                    if child is stop2:
                        do_break = True
                        break            # for build after

                elif child.isWP:
                    children2 = child.get_children()
                    child.vt.preprocess_params(child)
                    if child is stop1:
                        break            # for build before

                    geom.add_sequence('WP_Start', 'WP_Start', 'WP_Start')
                    child.add_geom_sequence_wp_start(geom)
                    for child2 in children2:
                        if not child2.enabled:
                            continue
                        child2.vt.preprocess_params(child2)
                        if child2 is stop1:
                            do_break = True
                            break            # for build before
                        child2.add_geom_sequence(geom)
                        if child2 is stop2:
                            do_break = True
                            break            # for build after
                    else:
                        if self.use_occ_preview:
                            geom.add_sequence(
                                'WP_End_OCC', 'WP_End_OCC', 'WP_End_OCC')
                        child.add_geom_sequence_wp_end(geom)

                    geom.add_sequence('WP_End', 'WP_End', 'WP_End')

                    if do_break:
                        do_break = True
                        break
                    if child is stop2:
                        do_break = True
                        break

                else:
                    assert False, "Should not come here"
            return do_break

        children = self.get_children()
        proccess_children(children)

        if stop1 is not None:
            self._build_stop = (stop1, None)
            return stop1.name()
        if stop2 is not None:
            self._build_stop = (None, stop2)
            return stop2.name()
        return self.name()

    def update_GUI_after_geom(self, data, objs):
        children = [x for x in self.walk()]
        children = children[1:]

        for child in children:
            if hasattr(child, "_newobjs"):
                del child._newobjs

        for child in children:
            if child.fullname() in data:
                dd = data[child.fullname()]
                child._objkeys = dd[0]
                child._newobjs = dd[1]

        self._objs = objs

    def do_build_geom4(self, stop1=None, stop2=None, filename=None,
                       finalize=False, no_mesh=False, gui_parent=None,
                       cwd=None):

        if not hasattr(self, "_gmsh4_data"):
            self._gmsh4_data = None

        if not hasattr(self, '_gso'):
            from petram.geom.geom_sequence_operator import GeomSequenceOperator
            self._gso = GeomSequenceOperator()

        stopname = self.walk_over_geom_chidlren(
            self._gso, stop1=stop1, stop2=stop2)

        L = len(self._gso.geom_sequence) + 3

        if gui_parent is not None:
            import wx
            #gui_parent = wx.GetApp().TopWindow
            pgb = wx.ProgressDialog("Generating geometry...",
                                    "", L, parent=gui_parent,
                                    style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_CAN_ABORT)

            def close_dlg(evt, dlg=pgb):
                pgb.Destroy()
            pgb.Bind(wx.EVT_CLOSE, close_dlg)
            trash = wx.GetApp().GetTopWindow().proj.get_trash()
            self._geom_brep_dir = ''
        else:
            pgb = None
            trash = os.path.join(cwd, '.trash')
            if not os.path.exists(trash):
                os.mkdir(trash)
            self._geom_brep_dir = trash

        success, dataset = self._gso.run_generator(self, no_mesh=no_mesh, finalize=finalize,
                                                   filename=stopname, progressbar=pgb,
                                                   trash=trash,)
        if not success:
            assert False, dataset
            return

        gui_data, objs, brep_file, data, vcl, esize = dataset

        self._geom_brep = os.path.basename(brep_file)

        self.update_GUI_after_geom(gui_data, objs)

        if data is None:  # if no_mesh = True
            self._gso.terminate_child()
            return

        # if finalize:
        #    self._gso.terminate_child()
        # for the readablity I expend data here, do we need geom?
        ptx, cells, cell_data, l, s, v = data
        self._gmsh4_data = (ptx, cells, cell_data, l, s, v, self._gso)

        values = vcl.values()
        if len(values) > 0:
            self._clmax_guess = (max(values), min(values))
        else:
            self._clmax_guess = 1e20, 0

        self._vcl = vcl
        self._esize = esize
        return

    def build_geom4(self, stop1=None, stop2=None, filename=None,
                    finalize=False, no_mesh=False, gui_parent=None,
                    cwd=None):

        self.use_occ_preview = False
        self.do_build_geom4(stop1=stop1, stop2=stop2, filename=filename,
                            finalize=finalize, no_mesh=no_mesh,
                            gui_parent=gui_parent, cwd=cwd)

    def onTerminateChild(self, evt):
        self._gso.terminate_child()
        evt.Skip()

    def generate_final_geometry(self):
        cwd = os.getcwd()
        dprint1("Generating Geometry in " + cwd)
        bk = self.use_1d_preview
        self.use_1d_preview = True
        self.build_geom4(finalize=True, gui_parent=None, cwd=cwd)
        self.use_1d_preview = bk
        self.geom_finalized = True
        self.geom_timestamp = time.ctime()
        self._gso.terminate_child()
        dprint1("Generating Geometry ... Done")

    def build_geom(self, stop1=None, stop2=None, filename=None,
                   finalize=False, gui_parent=None):
        self.do_build_geom4(stop1=stop1, stop2=stop2,
                            filename=filename,
                            finalize=finalize,
                            gui_parent=gui_parent)

    def onExportGeom(self, evt):
        if not hasattr(self, "_txt_unrolled"):
            evt.Skip()
            return
        from ifigure.widgets.dialog import write
        parent = evt.GetEventObject()
        path = write(parent,
                     message='Enter .geo file name',
                     wildcard='*.geo')
        if path != '':
            fid = open(path, 'w')
            fid.write('\n'.join(self._txt_rolled))
            fid.close()

    def onExportBrep(self, evt):
        if self.geom_brep == '':
            evt.Skip()
            return

        from ifigure.widgets.dialog import write
        parent = evt.GetEventObject()
        path = write(parent,
                     message='Enter .brep file name',
                     wildcard='*.brep')
        if path != '':
            if not path.endswith('.brep'):
                path = path + '.brep'
            from shutil import copyfile
            copyfile(self.geom_brep, path)

    def is_viewmode_grouphead(self):
        return True

    def get_faceedges(self, faces):
        s2l = self._gmsh4_data[4]
        edges = [s2l[k] for k in faces]
        return edges

    def get_facevertices(self, faces):
        edges = self.get_faceedges(faces)
        l2p = self._gmsh4_data[3]
        points = [self.get_edgevertices(k) for k in edges]
        return points

    def get_edgevertices(self, edges):
        l2p = self._gmsh4_data[3]
        points = sum([l2p[kk] for kk in edges], [])
        return points

    def get_edgecontaining_faces(self, edge):
        from petram.mesh.mesh_utils import line2surf
        l2s = line2surf(self._gmsh4_data[4])

        return l2s[edge]

    def get_vertexcontaining_edges(self, vertex):
        from petram.mesh.mesh_utils import line2surf
        p2l = line2surf(self._gmsh4_data[3])

        return p2l[vertex]

    def find_tinyedges(self, thr=1e-5):
        '''
        find shot edges
        '''
        if not hasattr(self, '_esize'):
            return None
        el_max = max(list(self._esize.values()))
        edges = np.array(
            [x for x in self._esize if self._esize[x] < el_max * thr])
        ratio = np.array([self._esize[x] / el_max for x in edges])

        idx = np.argsort(ratio)
        return list(edges[idx]), list(ratio[idx])

    def find_tinyfaces(self, thr=1e-5):
        '''
        find faces those all edges are small
        '''
        if not hasattr(self, '_esize'):
            return None
        if not hasattr(self, '_gmsh4_data'):
            return None

        el_limit = max(list(self._esize.values())) * thr

        s2l = self._gmsh4_data[4]
        l2p = self._gmsh4_data[3]
        tinyfaces = [k for k in s2l if np.all(
            [self._esize[x] < el_limit for x in s2l[k]])]
        edges = self.get_faceedges(tinyfaces)
        points = self.get_facevertices(tinyfaces)

        return tinyfaces, edges, points

# This is meant only to load a old project which uses BrepFile;
# We don't use it anymore.
# It is not quite correct in the sense BrepFile was meant to load a BrepFile.
# This alias at least allows for loading  an old project file...


BrepFile = GmshGeom


def check_dim(unrolled):
    for line in unrolled:
        if line.startswith('Volume'):
            return 3
    for line in unrolled:
        if line.find('Surface'):
            return 2
    return 1


def guess_geom_size(unrolled):
    points = []
    for line in unrolled:
        if line.startswith("Point("):
            try:
                coords = line.split("=")[1]
                coords = coords[coords.find("{") + 1:coords.find("}")]
                xyz = np.array([float(x) for x in coords.split(",")[:3]])
                points.append(xyz)
            except BaseException:
                pass
    points = np.vstack(points)
    return points


def read_loops(unrolled):
    ll = {}  # line loop
    sl = {}  # surface loop
    v = {}
    s = {}

    def split_line(line):
        a, b = line.split("=")
        k = int(a[a.find("(") + 1:a.find(")")])
        idx = [abs(int(x)) for x in b[b.find("{") + 1:b.find("}")].split(",")]
        return k, idx

    for line in unrolled:
        line = line.strip()
        if line.startswith("Surface Loop("):
            k, idx = split_line(line)
            sl[k] = idx
        elif line.startswith("Line Loop("):
            k, idx = split_line(line)
            ll[k] = idx
        elif line.startswith("Volume("):
            k, idx = split_line(line)
            v[k] = idx
        elif line.startswith("Plane Surface("):
            k, idx = split_line(line)
            s[k] = idx
        elif line.startswith("Surface("):
            k, idx = split_line(line)
            s[k] = idx
        else:
            pass

    for kv in v.keys():
        tmp = []
        for k in v[kv]:
            tmp.extend(sl[k])
        v[kv] = list(set(tmp))
    for ks in s.keys():
        tmp = []
        for k in s[ks]:
            tmp.extend(ll[k])
        s[ks] = list(set(tmp))
    return s, v
