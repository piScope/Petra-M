from __future__ import print_function
import petram.geom.occ_inspect
from threading import Thread
from queue import Queue

from petram.geom.occ_cbook import *
from petram.geom.geom_id import (
    GeomIDBase,
    VertexID,
    LineID,
    SurfaceID,
    VolumeID,
    LineLoopID,
    SurfaceLoopID,
    WorkPlaneParam)
import os
import numpy as np
import time
import tempfile
import traceback
from collections import defaultdict
import multiprocessing as mp
from six.moves.queue import Empty as QueueEmpty

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('OCCGeomWrapper')


class Counter():
    def __init__(self):
        self.value = 0

    def increment(self, x):
        self.value = self.value + x

    def __call__(self):
        return self.value


class trans_delta(list):
    def __init__(self, xyz):
        list.__init__(self)
        self.append(xyz[0])
        self.append(xyz[1])
        self.append(xyz[2])


class topo2id():
    def __init__(self, dd, mapper):
        self.mapper = mapper
        self.mapperout2k = {mapper.FindIndex(dd[k]): k for k in dd}
        #self._d = [(dd[k], k) for k in dd]

    def __getitem__(self, val):
        out = self.mapper.FindIndex(val)
        return self.mapperout2k[out]
        #assert False, "ID is not found in ap from Topo to ID"


class Geometry():
    def __init__(self, **kwargs):
        self.process_kwargs(kwargs)

        self.builder = BRep_Builder()
        self.bt = BRep_Tool()

        self.geom_sequence = []

        write_log = kwargs.pop('write_log', False)
        if write_log:
            self.logfile = tempfile.NamedTemporaryFile('w', delete=True)
        else:
            self.logfile = None

        self.queue = kwargs.pop("queue", None)
        self.p = None
        self._shape_bk = None

    def process_kwargs(self, kwargs):
        self.geom_prev_res = kwargs.pop('PreviewResolution', 30)
        self.geom_prev_algorithm = kwargs.pop('PreviewAlgorithm', 2)
        self.occ_parallel = kwargs.pop('OCCParallel', 0)
        #self.occ_parallel = False
        self.occ_boolean_tolerance = kwargs.pop('OCCBooleanTol', 1e-5)
        #self.occ_boolean_tolerance = kwargs.pop('OCCBooleanTol', 0)
        self.occ_geom_tolerance = kwargs.pop('OCCGeomTol', 1e-6)
        self.maxthreads = kwargs.pop('Maxthreads', 1)
        self.skip_final_frag = kwargs.pop('SkipFrag', False)
        self.use_1d_preview = kwargs.pop('Use1DPreview', False)
        self.use_occ_preview = kwargs.pop('UseOCCPreview', False)
        self.long_edge_thr = kwargs.pop('LongEdgeThr', 0.1)
        self.small_edge_thr = kwargs.pop('SmallEdgeThr', 0.001)
        self.small_edge_seg = kwargs.pop('SmallEdgeSeg', 3)
        self.max_seg = kwargs.pop('MaxSeg', 30)
        self.occ_angle_deflection = kwargs.pop('AngleDeflection', 0.05)
        self.occ_linear_deflection = kwargs.pop('LinearDeflection', 0.01)

        self.occ_angle_deflection = self.long_edge_thr
        self.occ_linear_deflection = self.small_edge_thr

    def prep_topo_list(self):
        self.vertices = topo_list_vertex()
        self.edges = topo_list_edge()
        self.wires = topo_list_wire()
        self.faces = topo_list_face()
        self.shells = topo_list_shell()
        self.solids = topo_list_solid()

        self.edge_2_wire = {}

    def register_wire(self, edges, wire):
        idx = tuple(sorted([int(x) for x in edges]))
        self.edge_2_wire[idx] = wire

    def check_wire_registered(self, edges):
        idx = tuple(sorted([int(x) for x in edges]))
        if idx in self.edge_2_wire:
            return self.edge_2_wire[idx]
        return None

    def new_compound(self, gids=None):
        comp = TopoDS_Compound()
        self.builder.MakeCompound(comp)

        gids = gids if gids is not None else []
        for gid in gids:
            topolist = self.get_topo_list_for_gid(gid)
            shape = topolist[gid]
            self.builder.Add(comp, shape)
        return comp

    def store_shape_and_topolist(self):
        self._shape_bk = self.shape

        # self._topo_bk = (self.vertices, self.edges, self.wires,
        #                 self.faces, self.shells, self.solids)
        self.shape = self.new_compound()
        self.vertices.new_group()
        self.edges.new_group()
        self.wires.new_group()
        self.faces.new_group()
        self.shells.new_group()
        group = self.solids.new_group()
        return group

    def current_topolist_group(self):
        return self.vertices.current_group()

    def set_topolist_group(self, val):
        self.vertices.set_group(val)
        self.edges.set_group(val)
        self.wires.set_group(val)
        self.faces.set_group(val)
        self.shells.set_group(val)
        self.solids.set_group(val)

    def pop_shape_and_topolist(self):
        self.set_topolist_group(0)
        self.shape = self._shape_bk

    def get_topo_list_for_gid(self, gid, child=0):
        ll = [self.vertices, self.edges, self.wires,
              self.faces, self.shells, self.solids]
        idx = gid.idx
        idx = idx - child
        if idx < 0:
            return None
        return ll[idx]

    def get_topolist_for_shape(self, shape):
        if isinstance(shape, TopoDS_Solid):
            return self.solids
        elif isinstance(shape, TopoDS_Shell):
            return self.shells
        elif isinstance(shape, TopoDS_Face):
            return self.faces
        elif isinstance(shape, TopoDS_Wire):
            return self.wires
        elif isinstance(shape, TopoDS_Edge):
            return self.edges
        elif isinstance(shape, TopoDS_Vertex):
            return self.vertices
        else:
            assert False, "Unkown shape type: " + type(shape)
        return None

    def get_shape_for_gid(self, gid, group=0):
        topolist = self.get_topo_list_for_gid(gid)
        return topolist.get_item_from_group(gid, group=group)

    def get_gid_for_shape(self, shape):
        '''
        add shpae those kind is not known
        '''
        topolist = self.get_topolist_for_shape(shape)
        gid = topolist.find_gid(shape)
        return gid

    def add_to_topo_list(self, shape):
        '''
        add shpae those kind is not known
        '''
        topolist = self.get_topolist_for_shape(shape)
        gid = topolist.add(shape)
        return gid

        '''
        if isinstance(shape, TopoDS_Solid):
            gid = self.solids.add(shape)
        elif isinstance(shape, TopoDS_Shell):
            gid =self.shells.add(shape)
        elif isinstance(shape, TopoDS_Face):
            gid = self.faces.add(shape)
        elif isinstance(shape, TopoDS_Wire):
            gid = self.wires.add(shape)
        elif isinstance(shape, TopoDS_Edge):
            gid =self.edges.add(shape)
        elif isinstance(shape, TopoDS_Vertex):
            gid = self.vertices.add(shape)
        else:
            assert False, "Unkown shape type: " + type(shape)
        return gid
        '''

    def gid2shape(self, gid):
        d = self.get_topo_list_for_gid(gid)
        return d[int(gid)]

    def print_number_of_topo_objects(self, shape=None):
        if shape is None:
            shape = self.shape
        maps = prep_maps(shape, return_all=True)

        dprint1("Entity counts: solid/face/edge/vert : ",
                maps['solid'].Size(), maps['face'].Size(),
                maps['edge'].Size(), maps['vertex'].Size(),
                "  shell/wire:",
                maps['shell'].Size(), maps['wire'].Size(),)

    def count_topos(self):
        maps = prep_maps(self.shape, return_all=False)
        return (maps['solid'].Size(), maps['face'].Size(),
                maps['edge'].Size(), maps['vertex'].Size())

    def bounding_box(self, shape=None, tolerance=1e-5):
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepBndLib import brepbndlib

        shape = self.shape if shape is None else shape

        bbox = Bnd_Box()
        bbox.SetGap(tolerance)
        brepbndlib.Add(shape, bbox)
        if bbox.IsOpen():
            return (0, 0, 0, 0, 0, 0)
        if bbox.IsVoid():
            return (0, 0, 0, 0, 0, 0)
        values = bbox.Get()
        return [values[i] for i in range(6)]

    def get_esize(self):
        esize = {}
        for iedge in self.edges:
            x1, y1, z1, x2, y2, z2 = self.bounding_box(self.edges[iedge])
            s = ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)**0.5

            esize[iedge] = s
        return esize

    def get_vcl(self, l, esize):
        lcar = defaultdict(lambda: np.inf)
        for iedge in esize:
            if iedge not in l:
                continue
            iverts = l[iedge]
            for ivert in iverts:
                lcar[ivert] = min(lcar[ivert], esize[iedge])

        return dict(lcar)

    def get_target1(self, objs, targets, cls):
        # this is when target type is given
        if cls == 'l':
            cc = LineID
        elif cls == 'v':
            cc = VolumeID
        elif cls == 'f':
            cc = SurfaceID
        elif cls == 'p':
            cc = VertexID
        else:
            pass
        gids = [objs[t] if t in objs else cc(t) for t in targets]
        return gids

    def get_target2(self, objs, targets):
        # this is when target type is not given
        ret = []

        for t in targets:
            if t in objs:
                ret.append(objs[t])
            else:
                if t.startswith("p"):
                    ret.append(VertexID(int(t[1:])))
                if t.startswith("l"):
                    ret.append(LineID(int(t[1:])))
                if t.startswith("f"):
                    ret.append(SurfaceID(int(t[1:])))
                if t.startswith("v"):
                    ret.append(VolumeID(int(t[1:])))

        if len(ret) == 0:
            assert False, "empty imput objects: " + ','.join(targets)

        return ret

    def get_point_coord(self, gid):
        if gid not in self.vertices:
            assert False, "can not find point: " + str(int(gid))
        shape = self.vertices[gid]
        pnt = self.bt.Pnt(shape)
        return np.array((pnt.X(), pnt.Y(), pnt.Z(),))

    def get_line_center(self, gid):
        if gid not in self.edges:
            assert False, "can not find edge: " + str(int(gid))
        shape = self.edges[gid]
        curve, first, last = self.bt.Curve(shape)
        pnt1 = gp_Pnt()
        pnt2 = gp_Pnt()
        curve.D0(first, pnt1)
        curve.D0(last, pnt2)
        p1 = np.array((pnt1.X(), pnt1.Y(), pnt1.Z(),))
        p2 = np.array((pnt2.X(), pnt2.Y(), pnt2.Z(),))
        return (p1 + p2) / 2.0

    def get_line_direction(self, gid, u_n=None):
        if gid not in self.edges:
            assert False, "can not find edge: " + str(int(gid))
        shape = self.edges[gid]
        curve, first, last = self.bt.Curve(shape)
        pnt1 = gp_Pnt()
        pnt2 = gp_Pnt()

        if u_n is None:
            curve.D0(first, pnt1)
            curve.D0(last, pnt2)
            p1 = np.array((pnt1.X(), pnt1.Y(), pnt1.Z(),))
            p2 = np.array((pnt2.X(), pnt2.Y(), pnt2.Z(),))
            p = p2 - p1
        else:
            uu = first + (last-first)*u_n
            gvec = gp_Vec()
            #print(uu, type(uu))
            curve.D1(uu, pnt1, gvec)
            p = np.array((gvec.X(), gvec.Y(), gvec.Z(),))
        p = p / np.sqrt(np.sum(p**2))
        return p

    def get_line_point(self, gid, u_n):
        if gid not in self.edges:
            assert False, "can not find edge: " + str(int(gid))
        shape = self.edges[gid]
        curve, first, last = self.bt.Curve(shape)
        pnt1 = gp_Pnt()

        uu = first + (last-first)*u_n
        curve.D0(uu, pnt1)
        p = np.array((pnt1.X(), pnt1.Y(), pnt1.Z(),))
        return p

    def get_face_normal(self, gid, check_flat=True):
        '''
        return normal vector of flat surface and a representative point on
        the plane
        '''
        if gid not in self.faces:
            assert False, "can not find surface: " + str(int(gid))

        shape = self.faces[gid]
        surface = self.bt.Surface(shape)

        uMin, uMax, vMin, vMax = surface.Bounds()

        dirc = gp_Dir()
        tool = BOPTools_AlgoTools3D()

        if check_flat:
            tool.GetNormalToSurface(surface, uMin, vMin, dirc)
            n1 = (dirc.X(), dirc.Y(), dirc.Z())
            tool.GetNormalToSurface(surface, uMin, vMax, dirc)
            n2 = (dirc.X(), dirc.Y(), dirc.Z())
            tool.GetNormalToSurface(surface, uMax, vMin, dirc)
            n3 = (dirc.X(), dirc.Y(), dirc.Z())

            if (np.abs(np.sum(np.array(n1) * np.array(n2))) -
                    1) > self.occ_geom_tolerance:
                assert False, "surface is not flat"
            if (np.abs(np.sum(np.array(n1) * np.array(n3))) -
                    1) > self.occ_geom_tolerance:
                assert False, "surface is not flat"
        else:
            tool.GetNormalToSurface(surface, 0., 0., dirc)
            n1 = (dirc.X(), dirc.Y(), dirc.Z())

        ptx = gp_Pnt()
        surface.D0(0.0, 0.0, ptx)
        ptx = (ptx.X(), ptx.Y(), ptx.Z())

        return np.array(n1), np.array(ptx)

    def get_point_on_face(self, gid):
        if gid not in self.faces:
            assert False, "can not find surface: " + str(int(gid))

        face = self.faces[gid]
        for x in iter_shape(face, 'vertex'):
            break
        pnt = self.bt.Pnt(x)
        return np.array((pnt.X(), pnt.Y(), pnt.Z(),))

    def get_circle_center(self, gid):

        from OCC.Core.GeomLProp import GeomLProp_CLProps

        shape = self.edges[gid]

        curve, first, last = self.bt.Curve(shape)
        pnt1 = gp_Pnt()

        uarr = np.linspace(0, 1, 10)

        pt = np.zeros(3)
        for uu in uarr:
            uuu = first * (1 - uu) + last * uu
            prop = GeomLProp_CLProps(curve, uuu, 2, self.occ_geom_tolerance)
            prop.CentreOfCurvature(pnt1)
            x = np.array((pnt1.X(), pnt1.Y(), pnt1.Z()))
            pt = pt + x

        pt = pt / len(uarr)
        return pt

    def get_conic_focus(self, gid):
        shape = self.edges[gid]
        curve, first, last = self.bt.Curve(shape)

        curve, kind = downcast_curve(curve)

        pnt = []
        if hasattr(curve, 'Focus'):
            pnt.append(curve.Focus())
        if hasattr(curve, 'Focus1'):
            pnt.append(curve.Focus1())
        if hasattr(curve, 'Focus2'):
            pnt.append(curve.Focus2())

        if len(pnt) == 0:
            assert False, "Curve does not have Focus/Focus1/Focus2"

        return np.vstack([(p.X(), p.Y(). p.Z()) for p in pnt])

    def write_brep(self, filename, shape=None):
        if shape is None:
            shape = self.shape

        comp = TopoDS_Compound()
        b = self.builder
        b.MakeCompound(comp)
        ex1 = TopExp_Explorer(shape, TopAbs_SOLID)
        while ex1.More():
            b.Add(comp, ex1.Current())
            ex1.Next()
        ex1 = TopExp_Explorer(shape, TopAbs_FACE, TopAbs_SHELL)
        while ex1.More():
            b.Add(comp, ex1.Current())
            ex1.Next()
        ex1 = TopExp_Explorer(shape, TopAbs_EDGE, TopAbs_WIRE)
        while ex1.More():
            b.Add(comp, ex1.Current())
            ex1.Next()
        ex1 = TopExp_Explorer(shape, TopAbs_VERTEX, TopAbs_EDGE)
        while ex1.More():
            b.Add(comp, ex1.Current())
            ex1.Next()

        dprint1("exproted brep file:", filename)
        breptools.Write(comp, filename)

    def inspect_shape(self, shape, verbose=False, return_all=False):
        maps = prep_maps(shape)

        if return_all:
            names = ['solid', 'shell', 'face', 'wire', 'edge', 'vertex']
            #all_maps = solidMap, shellMap, faceMap, wireMap, edgeMap, vertMap
        else:
            names = ['solid', 'face', 'edge', 'vertex']
            #all_maps = solidMap, faceMap, edgeMap, vertMap

        all_maps = [maps[x] for x in names]
        if verbose:
            dprint1("--------- Shape inspection ---------")

        self.print_number_of_topo_objects()

        if not verbose:
            return all_maps

        usolids = topo_seen(mapping=maps['solid'])
        ufaces = topo_seen(mapping=maps['face'])
        uedges = topo_seen(mapping=maps['edge'])
        uvertices = topo_seen(mapping=maps['vertex'])

        ex1 = TopExp_Explorer(shape, TopAbs_SOLID)
        while ex1.More():
            s = ex1.Current()
            if usolids.not_seen(s):
                dprint1(s)
            ex1.Next()
        ex1 = TopExp_Explorer(shape, TopAbs_FACE)
        while ex1.More():
            s = ex1.Current()
            if ufaces.not_seen(s):
                surf = self.bt.Surface(s)
                surf, kind = downcast(surf)
                dprint1("Surface", kind)
            ex1.Next()
        ex1 = TopExp_Explorer(shape, TopAbs_EDGE)
        while ex1.More():
            s = ex1.Current()
            if uedges.not_seen(s):
                curve, first, last = self.bt.Curve(s)
                curve, kind = downcast_curve(curve)
                dprint1("Curve", kind)
            ex1.Next()
        ex1 = TopExp_Explorer(shape, TopAbs_VERTEX)
        while ex1.More():
            s = ex1.Current()
            if uvertices.not_seen(s):
                pnt = self.bt.Pnt(s)
                dprint1("Point", pnt.X(), pnt.Y(), pnt.Z())
            ex1.Next()

        if verbose:
            dprint1("--------- Shape inspection ---------")
        return all_maps

    def synchronize_topo_list(self, **kwargs):
        maps = prep_maps(self.shape)
        self.solids.synchronize(maps['solid'], **kwargs)
        self.shells.synchronize(maps['shell'], **kwargs)
        self.faces.synchronize(maps['face'], **kwargs)
        self.wires.synchronize(maps['wire'], **kwargs)
        self.edges.synchronize(maps['edge'], **kwargs)
        self.vertices.synchronize(maps['vertex'], **kwargs)

    @property
    def dim(self):
        if len(self.model.getEntities(3)) > 0:
            return 3
        if len(self.model.getEntities(2)) > 0:
            return 2
        if len(self.model.getEntities(1)) > 0:
            return 1
        return 0

    def add_point(self, p):
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        p = BRepBuilderAPI_MakeVertex(gp_Pnt(x, y, z)).Shape()
        return self.vertices.add(p)

    def add_point_on_face(self, gid, uarr, varr):
        face = self.faces[gid]

        location = TopLoc_Location()
        surf = self.bt.Surface(face, location)
        uMin, uMax, vMin, vMax = surf.Bounds()
        value = [surf.Value(u, v) for u, v in zip(uarr, varr)]

        if not location.IsIdentity():
            trans = location.Transformation()
            xyz = [v.XYZ() for v in value]
            for x in xyz:
                trans.Transforms(x)
            value = [gp_Pnt(x) for x in xyz]

        ret = []
        for pnt in value:
            p = BRepBuilderAPI_MakeVertex(pnt).Shape()
            ret.append(self.vertices.add(p))

        return ret

    def add_point_on_edge(self, gid, uarr):
        shape = self.edges[gid]

        curve, first, last = self.bt.Curve(shape)
        pnt1 = gp_Pnt()

        ret = []
        for uu in uarr:
            uuu = first * (1 - uu) + last * uu
            curve.D0(uuu, pnt1)
            p = BRepBuilderAPI_MakeVertex(pnt1).Shape()
            ret.append(self.vertices.add(p))

        return ret

    def add_line(self, p1, p2):
        if not isinstance(p1, gp_Pnt):
            p1 = self.vertices[p1]
            p2 = self.vertices[p2]

        edgeMaker = BRepBuilderAPI_MakeEdge(p1, p2)
#            self.vertices[p1], self.vertices[p2])
        edgeMaker.Build()
        if not edgeMaker.IsDone():
            assert False, "Can not make line"
        edge = edgeMaker.Edge()
        return self.edges.add(edge)

    def add_reversed_line(self, gid):
        shape = self.edges[gid]
        shape2 = shape.Reversed()
        return self.edges.add(shape2)
        '''
        curve, first, last = self.bt.Curve(shape)
        rcurve = curve.Reversed()
        edgeMaker = BRepBuilderAPI_MakeEdge(rcurve, last, first)
        edgeMaker.Build()
        if not edgeMaker.IsDone():
            assert False, "Can not make line"
        edge = edgeMaker.Edge()
        return self.edges.add(edge)
        '''

    def add_extended_line(self, gid, ratio, resample):
        shape = self.edges[gid]
        curve, first, last = self.bt.Curve(shape)

        if first < last:
            first1 = first - ratio[0] * (last - first)
            last1 = last + ratio[1] * (last - first)
        else:
            first1 = first + ratio[0] * (first - last)
            last1 = last - ratio[1] * (first - last)

        pts = []
        for x in np.linspace(first1, last1):
            pnt1 = gp_Pnt()
            curve.D0(x, pnt1)
            pts.append(pnt1)

        array = TColgp_HArray1OfPnt(1, len(pts))
        for i, p in enumerate(pts):
            array.SetValue(i + 1, p)

        itp = GeomAPI_Interpolate(array, False, self.occ_geom_tolerance)
        itp.Perform()
        if not itp.IsDone():
            assert False, "Can not interpolate points (add_extend)"

        start = pts[0]
        end = pts[-1]

        edgeMaker = BRepBuilderAPI_MakeEdge(itp.Curve(), start, end)
        edgeMaker.Build()
        if not edgeMaker.IsDone():
            assert False, "Can not make line"
        edge = edgeMaker.Edge()
        # return self.edges.add(edge)
        new_objs = self.register_shapes_balk(edge)
        return new_objs

    def add_circle_arc(self, pnt1, pnt3, pnt2):
        if not isinstance(pnt1, gp_Pnt):
            bt = self.bt
            pnt1 = bt.Pnt(self.vertices[pnt1])
            pnt2 = bt.Pnt(self.vertices[pnt2])
            pnt3 = bt.Pnt(self.vertices[pnt3])

        arc = GC_MakeArcOfCircle(pnt1, pnt3, pnt2)

        edgeMaker = BRepBuilderAPI_MakeEdge(arc.Value())
        edgeMaker.Build()
        if not edgeMaker.IsDone():
            assert False, "Can not make circle arc"
        edge = edgeMaker.Edge()

        return self.edges.add(edge)

    def add_circle_by_axis_radius(self, center, dirct, radius, npts=0,
                                  pts=None):
        x, y, z = center
        dx, dy, dz = dirct
        pnt = gp_Pnt(x, y, z)
        vec = gp_Dir(dx, dy, dz)
        axis = gp_Ax1(pnt, vec)

        if npts == 0 or npts == 1:
            cl = GC_MakeCircle(axis, radius)
            edgeMaker = BRepBuilderAPI_MakeEdge(cl.Value())
            edgeMaker.Build()
            if not edgeMaker.IsDone():
                assert False, "Can not make circle"
            edge = edgeMaker.Edge()
            if npts == 0:
                return self.edges.add(edge)
            else:
                return [self.edges.add(edge)]
        else:
            cir = gp_Circ()
            cir.SetAxis(axis)
            cir.SetRadius(radius)

            ret = []
            for i in range(npts):
                # for i in range(1):
                a1 = np.pi*2/npts*i
                a2 = np.pi*2/npts*(i+1)
                if pts is None:
                    arc = GC_MakeArcOfCircle(cir, a1, a2, True)
                else:
                    e1 = pts-center
                    e2 = np.cross(dirct, e1)
                    e1 = e1/np.linalg.norm(e1)
                    e2 = e2/np.linalg.norm(e2)
                    p1 = (np.cos(a1)*e1 + np.sin(a1)*e2)*radius + center
                    p2 = (np.cos(a2)*e1 + np.sin(a2)*e2)*radius + center
                    p1 = gp_Pnt(*[x for x in p1])
                    p2 = gp_Pnt(*[x for x in p2])
                    arc = GC_MakeArcOfCircle(cir, p1, p2, True)
                edgeMaker = BRepBuilderAPI_MakeEdge(arc.Value())
                edgeMaker.Build()
                if not edgeMaker.IsDone():
                    assert False, "Can not make circle arc"
                edge = edgeMaker.Edge()
                ret.append(self.edges.add(edge))
            return ret

    def add_circle_by_3points(self, p1, p2, p3):
        bt = self.bt
        pnt1 = bt.Pnt(self.vertices[p1])
        pnt2 = bt.Pnt(self.vertices[p2])
        pnt3 = bt.Pnt(self.vertices[p3])

        cl = GC_MakeCircle(pnt1, pnt3, pnt2)

        edgeMaker = BRepBuilderAPI_MakeEdge(cl.Value())
        edgeMaker.Build()
        if not edgeMaker.IsDone():
            assert False, "Can not make circle"
        edge = edgeMaker.Edge()

        return self.edges.add(edge)

    def add_ellipse_arc(self, startTag, centerTag, endTag):
        a = self._point[startTag] - self._point[centerTag]
        b = self._point[endTag] - self._point[centerTag]
        if np.sum(a * a) > np.sum(b * b):
            l = self.factory.addEllipseArc(startTag, centerTag, endTag)
        else:
            l = self.factory.addEllipseArc(endTag, centerTag, startTag)
        return LineID(l)

    def add_polygon(self, gids):
        L = len(gids)
        gids = list(gids) + [gids[0]]

        lines = []
        for i in range(L):
            l1 = self.add_line(gids[i], gids[i + 1])
            lines.append(l1)

        ll1 = self.add_curve_loop(lines)
        s1 = self.add_plane_surface(ll1)

        return s1

    def add_spline(self, pos, tolerance=1e-5, periodic=False):
        bt = self.bt

        pts = [BRepBuilderAPI_MakeVertex(gp_Pnt(p[0], p[1], p[2])).Shape()
               for p in pos]

        array = TColgp_HArray1OfPnt(1, len(pts))
        for i, p in enumerate(pts):
            array.SetValue(i + 1, bt.Pnt(p))

        itp = GeomAPI_Interpolate(array, periodic, tolerance)
        itp.Perform()
        if not itp.IsDone():
            assert False, "Can not interpolate points (add_spline)"

        start = pts[0]
        end = pts[-1]
        if periodic:
            edgeMaker = BRepBuilderAPI_MakeEdge(itp.Curve(), start, start)
        else:
            edgeMaker = BRepBuilderAPI_MakeEdge(itp.Curve(), start, end)
        edgeMaker.Build()
        if not edgeMaker.IsDone():
            assert False, "Can not make spline"
        edge = edgeMaker.Edge()

        l_id = self.edges.add(edge)
        if periodic:
            self.vertices.add(start)
        else:
            self.vertices.add(start)
            self.vertices.add(end)
        return l_id

    def add_plane_surface(self, tag):

        wire = self.wires[tag]
        faceMaker = BRepBuilderAPI_MakeFace(wire)
        faceMaker.Build()

        if not faceMaker.IsDone():
            assert False, "can not create face"

        face = faceMaker.Face()

        fixer = ShapeFix_Face(face)
        fixer.Perform()
        face = fixer.Face()
        f_id = self.faces.add(face)

        return f_id

    '''
    def add_plate_surface(self, gids_edge, gids_vertex):
        bt = BRep_Tool()
        BPSurf = GeomPlate_BuildPlateSurface(2, 150, 10)

        # make wire first
        wireMaker = BRepBuilderAPI_MakeWire()
        for gid in gids_edge:
            edge = self.edges[gid]
            wireMaker.Add(edge)
        wireMaker.Build()

        if not wireMaker.IsDone():
            assert False, "Failed to make wire"
        wire = wireMaker.Wire()

        # make wire constraints
        ex1 = Breptools.WireExplorer(wire)
        while ex1.More():
            edge = ex1.Current()
            C = BRepAdaptor_HCurve()
            C.ChangeCurve().Initialize(edge)
            Cont = BRepFill_CurveConstraint(C, 0)
            BPSurf.Add(Cont)
            ex1.Next()

        # make point constraints
        for gid in gids_vertex:
            vertex = self.vertices[gid]
            Pcont = GeomPlate_PointConstraint(bt.Pnt(vertex), 0)
            BPSurf.Add(Pcont)

        BPSurf.Perform()

        MaxSeg = 9
        MaxDegree = 8
        CritOrder = 0

        PSurf = BPSurf.Surface()

        dmax = max(0.0001, 10 * BPSurf.G0Error())
        Tol = 0.0001

        Mapp = GeomPlate_MakeApprox(PSurf, Tol, MaxSeg, MaxDegree,
                                    dmax, CritOrder)
        Surf = Mapp.Surface()
        uMin, uMax, vMin, vMax = Surf.Bounds()

        faceMaker = BRepBuilderAPI_MakeFace(Surf, uMin, uMax, vMin, vMax, 1e-6)
        result = faceMaker.Face()

        fix = ShapeFix_Face(result)
        fix.SetPrecision(self.occ_geom_tolerance)
        fix.Perform()
        fix.FixOrientation()
        result = fix.Face()

        result = self.select_highest_dim(result)
        new_objs = self.register_shapes_balk(result)

        return new_objs
    '''

    def add_surface_filling(self, gids_edge, gids_vertex):
        from OCC.Core.GeomAbs import GeomAbs_C0

        bt = BRep_Tool()
        f = BRepOffsetAPI_MakeFilling()

        # make wire first
        wireMaker = BRepBuilderAPI_MakeWire()
        for gid in gids_edge:
            edge = self.edges[gid]
            wireMaker.Add(edge)
        wireMaker.Build()

        if not wireMaker.IsDone():
            assert False, "Failed to make wire"
        wire = wireMaker.Wire()

        # make wire constraints
        ex1 = Breptools_WireExplorer(wire)
        while ex1.More():
            edge = ex1.Current()
            f.Add(edge, GeomAbs_C0)
            ex1.Next()

        for gid in gids_vertex:
            vertex = self.vertices[gid]
            pnt = bt.Pnt(vertex)
            f.Add(pnt)

        f.Build()

        if not f.IsDone():
            assert False, "Cannot make filling"

        face = f.Shape()
        s = bt.Surface(face)

        faceMaker = BRepBuilderAPI_MakeFace(s, wire)
        self.wires.add(wire)

        result = faceMaker.Face()

        fix = ShapeFix_Face(result)
        fix.SetPrecision(self.occ_geom_tolerance)
        fix.Perform()
        fix.FixOrientation()
        result = fix.Face()

        face_id = self.faces.add(result)

        return face_id

    def add_thrusection(self, gid_wire1, gid_wire2,
                        makeSolid=False, makeRuled=False):

        wire1 = self.wires[gid_wire1]
        wire2 = self.wires[gid_wire2]
        maker = BRepOffsetAPI_ThruSections(makeSolid, makeRuled)
        maker.AddWire(wire1)
        maker.AddWire(wire2)
        maker.CheckCompatibility(False)
        maker.Build()
        if not maker.IsDone():
            assert False, "Could not create ThruSection"
        result = maker.Shape()

        if makeSolid:
            gid_new = self.solids.add(result)
        else:
            gid_new = self.shells.add(result)

        return gid_new, result

    def add_line_loop(self, pts, sign=None):
        edges = list(np.atleast_1d(pts))

        w_id = self.check_wire_registered(edges)
        if w_id is not None:
            return w_id

        wireMaker = BRepBuilderAPI_MakeWire()
        for t in edges:
            edge = self.edges[t]
            wireMaker.Add(edge)
        wireMaker.Build()

        if not wireMaker.IsDone():
            assert False, "Failed to make wire"
        wire = wireMaker.Wire()

        w_id = self.wires.add(wire)
        self.register_wire(edges, w_id)

        return w_id

    def add_curve_loop(self, pts):
        return self.add_line_loop(pts)

    def add_surface_loop(self, sl):
        tags = list(np.atleast_1d(sl))

        # first sew the surfaces.
        try:
            sewingMaker = BRepBuilderAPI_Sewing()
            for t in tags:
                face = self.faces[t]
                sewingMaker.Add(face)
            sewingMaker.Perform()
            result = sewingMaker.SewedShape()
        except BaseException:
            assert False, "Failed to sew faces"

        # remove input if it is not used anymore.
        # (1) find tags which is not in other existing shells
        tags2 = tags[:]
        for shell in iter_shape(self.shape, 'shell'):
            mapper = get_mapper(shell, 'face')
            for t in tags:
                face = self.faces[t]
                if mapper.Contains(face):
                    tags2.remove(t)
        # (2) find tags which are used in sewed shape
        remove = []
        mapper = get_mapper(result, 'face')
        for t in tags2:
            face = self.faces[t]
            if not mapper.Contains(face):
                remove.append(face)

        # (3) find tags which are used in sewed shape
        if len(remove) > 0:
            rebuild = ShapeBuild_ReShape()
            for shape in remove:
                rebuild.Remove(shape)
            self.shape = rebuild.Apply(self.shape)

        ex1 = TopExp_Explorer(result, TopAbs_SHELL)
        while ex1.More():
            shell = ex1.Current()
            fixer = ShapeFix_Shell(shell)
            fixer.Perform()
            shell = fixer.Shell()
            break
            ex.Next()

        shell_id = self.shells.add(shell)

        return shell_id
    '''
    def check_shape_generated(self, operator, top_shape, top_abs):
        shapes = TopTools_ListOfShape()
        ex1 = TopExp_Explorer(top_shape, top_abs)
        while ex1.More():
            shape = ex1.Current()
            shape_new = operator.Generated(shape)
            iterator2 = TopTools_ListIteratorOfListOfShape(shapes_new)
            while iterator2.More():
                shape_new = iterator2.Value()
                iterator2.Next()
                print("shape new", shape_new)
    '''

    def add_volume(self, shells):
        tags = list(np.atleast_1d(shells))

        solidMaker = BRepBuilderAPI_MakeSolid()
        for t in tags:
            shell = self.shells[t]
            solidMaker.Add(shell)
        result = solidMaker.Solid()

        if not solidMaker.IsDone():
            assert False, "Failed to make solid"

        fixer = ShapeFix_Solid(result)
        fixer.Perform()

        assert isinstance(fixer.Solid(), TopoDS_Solid), "wrong TopoDS type is detected"
        result = fixer.Solid()

        solid_id = self.solids.add(result)
        return solid_id

    def add_sphere(self, xyzc, radius, angle1, angle2, angle3):
        if radius <= 0:
            assert False, "Sphere radius should be > 0"

        if (angle3 <= 0 or angle3 > 2 * np.pi):
            assert False, "Cannot build sphere with angle <= 0 or angle > 2*pi"

        pnt = gp_Pnt(xyzc[0], xyzc[1], xyzc[2])
        s = BRepPrimAPI_MakeSphere(pnt, radius, angle1, angle2, angle3)

        s.Build()
        if not s.IsDone():
            assert False, "Could not create sphere"

        assert isinstance(s.Shape(), TopoDS_Solid), "wrong TopoDS type is detected"
        result = s.Shape()
        result = self.select_highest_dim(result)
        new_objs = self.register_shapes_balk(result)

        return new_objs

    def add_cone(self, x, y, z, dx, dy, dz, r1, r2, angle):
        H = np.sqrt(dx * dx + dy * dy + dz * dz)
        if H == 0:
            assert False, "Cone hight must be > 0"
        if angle <= 0:
            assert False, "Cone angle should be positive"

        pnt = gp_Pnt(x, y, z)
        vec = gp_Dir(dx / H, dy / H, dz / H)
        axis = gp_Ax2(pnt, vec)

        c = BRepPrimAPI_MakeCone(axis, r1, r2, H, angle)
        c.Build()
        if not c.IsDone():
            assert False, "Could not create cone"

        assert isinstance(c.Shape(), TopoDS_Solid), "wrong TopoDS type is detected"
        result = c.Shape()
        result = self.select_highest_dim(result)
        new_objs = self.register_shapes_balk(result)

        return new_objs

    def add_wedge(self, xyz, dxyz, ltx):
        x, y, z = xyz
        dx, dy, dz = dxyz
        pnt = gp_Pnt(x, y, z)
        vec = gp_Dir(0, 0, 1)
        axis = gp_Ax2(pnt, vec)

        w = BRepPrimAPI_MakeWedge(axis, dx, dy, dz, ltx)
        w.Build()
        if not w.IsDone():
            assert False, "Could not create wedge"

        assert isinstance(w.Shape(), TopoDS_Solid), "wrong TopoDS type is detected"
        result = w.Shape()
        result = self.select_highest_dim(result)
        new_objs = self.register_shapes_balk(result)

        return new_objs

    def add_cylinder(self, xyz, dxyz, r, angle):
        x, y, z = xyz
        dx, dy, dz = dxyz
        H = np.sqrt(dx * dx + dy * dy + dz * dz)
        if H == 0:
            assert False, "Cylinder height must be > 0"
        if r <= 0:
            assert False, "Cylinder radius must be > 0"
        if (angle <= 0 or angle > 2 * np.pi):
            assert False, "Cannot build a cylinder with angle <= 0 or angle > 2*pi"

        pnt = gp_Pnt(x, y, z)
        vec = gp_Dir(dx / H, dy / H, dz / H)
        axis = gp_Ax2(pnt, vec)

        cl = BRepPrimAPI_MakeCylinder(axis, r, H, angle)
        cl.Build()
        if not cl.IsDone():
            assert False, "Can not create cylinder"

        assert isinstance(cl.Shape(), TopoDS_Solid), "wrong TopoDS type is detected"
        result = cl.Shape()
        result = self.select_highest_dim(result)
        new_objs = self.register_shapes_balk(result)

        return new_objs

    def add_torus(self, xyz, r1, r2, angle):
        x, y, z = xyz
        pnt = gp_Pnt(x, y, z)
        vec = gp_Dir(0, 0, 1)
        axis = gp_Ax2(pnt, vec)

        if r1 <= 0:
            assert False, "Torus major radius must be > 0"
        if r2 <= 0:
            assert False, "Torus minor radius must be > 0"
        if (angle <= 0 or angle > 2 * np.pi):
            assert False, "Torus angle must be between 0, and 2*pi"

        t = BRepPrimAPI_MakeTorus(axis, r1, r2, angle)
        t.Build()
        if not t.IsDone():
            assert False, "Could not create torus"

        assert isinstance(t.Shape(), TopoDS_Solid), "wrong TopoDS type is detected"
        result = t.Shape()
        result = self.select_highest_dim(result)
        new_objs = self.register_shapes_balk(result)

        return new_objs

    def fillet(self, gid_vols, gid_curves, radii):
        radii = [float(x) for x in radii]

        comp = TopoDS_Compound()
        self.builder.MakeCompound(comp)

        for gid in gid_vols:
            topolist = self.get_topo_list_for_gid(gid)
            shape = topolist[gid]
            self.builder.Add(comp, shape)
            self.remove(gid, recursive=True)

        f = BRepFilletAPI_MakeFillet(comp)

        for kk, gid in enumerate(gid_curves):
            topolist = self.get_topo_list_for_gid(gid)
            shape = topolist[gid]

            if len(radii) == 1:
                f.Add(radii[0], shape)
            elif len(gid_curves) == len(radii):
                f.Add(radii[kk], shape)
            elif len(gid_curves) + 1 == len(radii):
                f.Add(radii[kk], radii[kk + 1], shape)
            else:
                assert False, "Wrong radius setting"

        f.Build()
        if not f.IsDone():
            assert False, "Can not make fillet"

        result = f.Shape()

        result = self.select_highest_dim(result)
        new_objs = self.register_shapes_balk(result, self.shape)

        return new_objs

    def fillet2d(self, gid_face, gid_corners, radius):
        comp = TopoDS_Compound()
        self.builder.MakeCompound(comp)

        topolist = self.get_topo_list_for_gid(gid_face)
        shape = topolist[gid_face]
        self.builder.Add(comp, shape)
        self.remove(gid_face, recursive=True)

        f = BRepFilletAPI_MakeFillet2d(shape)

        for kk, gid in enumerate(gid_corners):
            topolist = self.get_topo_list_for_gid(gid)
            ptx = topolist[gid]
            f.AddFillet(ptx, radius)

        f.Build()
        if not f.IsDone():
            assert False, "Can not make fillet"

        result = f.Shape()

        result = self.select_highest_dim(result)
        new_objs = self.register_shapes_balk(result, self.shape)

        return new_objs

    def chamfer(self, gid_vols, gid_curves, gid_faces, distances):

        distances = [float(x) for x in distances]

        comp = TopoDS_Compound()
        self.builder.MakeCompound(comp)

        for gid in gid_vols:
            topolist = self.get_topo_list_for_gid(gid)
            shape = topolist[gid]
            self.builder.Add(comp, shape)
            self.remove(gid, recursive=True)

        f = BRepFilletAPI_MakeChamfer(comp)

        kk = 0
        for gid_c, gid_f in zip(gid_curves, gid_faces):
            topolist = self.get_topo_list_for_gid(gid_c)
            edge = topolist[gid_c]
            topolist = self.get_topo_list_for_gid(gid_f)
            face = topolist[gid_f]

            if len(distances) == 1:
                f.Add(distances[0], distances[0], edge, face)
            elif len(distances) == len(gid_curves):
                f.Add(distances[kk], edge, face)
            elif len(distances) == len(gid_curves) * 2:
                f.Add(distances[2 * kk], distances[2 * kk + 1], edge, face)
            kk = kk + 1

        f.Build()
        if not f.IsDone():
            assert False, "Can not make chamfer"

        result = f.Shape()

        result = self.select_highest_dim(result)
        new_objs = self.register_shapes_balk(result)

        return new_objs

    def chamfer2d(self, gid_face, gid_edges, e1, e2):
        comp = TopoDS_Compound()
        self.builder.MakeCompound(comp)

        topolist = self.get_topo_list_for_gid(gid_face)
        shape = topolist[gid_face]
        self.builder.Add(comp, shape)
        self.remove(gid_face, recursive=True)

        f = BRepFilletAPI_MakeFillet2d(shape)

        L = len(gid_edges)
        for kk in range(L//2):
            gid1 = gid_edges[2*kk]
            gid2 = gid_edges[2*kk+1]
            topolist = self.get_topo_list_for_gid(gid1)
            l1 = topolist[gid1]
            l2 = topolist[gid2]
            f.AddChamfer(l1, l2, e1, e2)

        f.Build()
        if not f.IsDone():
            assert False, "Can not make fillet"

        result = f.Shape()

        result = self.select_highest_dim(result)
        new_objs = self.register_shapes_balk(result, self.shape)

        return new_objs

    def add_box(self, points):
        p1, p2, p3, p4, p5, p6, p7, p8 = points
        lcar = 0.0

        p1 = self.add_point(p1)
        p2 = self.add_point(p2)
        p3 = self.add_point(p3)
        p4 = self.add_point(p4)
        p5 = self.add_point(p5)
        p6 = self.add_point(p6)
        p7 = self.add_point(p7)
        p8 = self.add_point(p8)

        l1 = self.add_line(p1, p2)
        l2 = self.add_line(p2, p5)
        l3 = self.add_line(p5, p3)
        l4 = self.add_line(p3, p1)
        l5 = self.add_line(p1, p4)
        l6 = self.add_line(p2, p7)
        l7 = self.add_line(p5, p8)
        l8 = self.add_line(p3, p6)
        l9 = self.add_line(p4, p7)
        l10 = self.add_line(p7, p8)
        l11 = self.add_line(p8, p6)
        l12 = self.add_line(p6, p4)

        ll1 = self.add_curve_loop([l1, l2, l3, l4])
        ll2 = self.add_curve_loop([l5, l9, l6, l1])
        ll3 = self.add_curve_loop([l6, l10, l7, l2])
        ll4 = self.add_curve_loop([l7, l11, l8, l3])
        ll5 = self.add_curve_loop([l8, l12, l5, l4])
        ll6 = self.add_curve_loop([l9, l10, l11, l12])

        rec1 = self.add_plane_surface(ll1)
        rec2 = self.add_plane_surface(ll2)
        rec3 = self.add_plane_surface(ll3)
        rec4 = self.add_plane_surface(ll4)
        rec5 = self.add_plane_surface(ll5)
        rec6 = self.add_plane_surface(ll6)

        sl = self.add_surface_loop([rec1, rec2, rec3, rec4, rec5, rec6])

        v1 = self.add_volume(sl)

        return v1

    def update_topo_list_from_history(self, operator, list_of_shapes,
                                      verbose=False):
        iterator = TopTools_ListIteratorOfListOfShape(list_of_shapes)
        while iterator.More():
            shape = iterator.Value()
            iterator.Next()
            # Do I need to do something with modified?
            # operator.Modified(shape)

            shape_gone = operator.IsDeleted(shape)
            if shape_gone:
                if verbose:
                    print("shape gone", shape_gone)

            shapes_new = operator.Generated(shape)
            iterator2 = TopTools_ListIteratorOfListOfShape(shapes_new)
            while iterator2.More():
                shape_new = iterator2.Value()
                iterator2.Next()
                if verbose:
                    print("shape new", shape_new)

    def do_boolean(self, operation, gid_objs, gid_tools,
                   remove_tool=True, remove_obj=True,
                   keep_highest=False, upgrade=False,
                   tol_scale=-1.0):

        if operation == 'fuse':
            operator = BRepAlgoAPI_Fuse()
        elif operation == 'cut':
            operator = BRepAlgoAPI_Cut()
        elif operation == 'common':
            operator = BRepAlgoAPI_Common()
        elif operation == 'fragments':
            operator = BRepAlgoAPI_BuilderAlgo()
        else:
            assert False, "Unknown boolean operation"

        assert tol_scale >= 0., "Possible bug tolerance scalar is negative"

        objs = TopTools_ListOfShape()
        for gid in gid_objs:
            topolist = self.get_topo_list_for_gid(gid)
            shape = topolist[gid]
            objs.Append(shape)

        operator.SetRunParallel(self.occ_parallel)
        operator.SetArguments(objs)

        if operation == 'fragments':
            pass
        else:
            tools = TopTools_ListOfShape()
            for gid in gid_tools:
                topolist = self.get_topo_list_for_gid(gid)
                shape = topolist[gid]
                tools.Append(shape)

            operator.SetTools(tools)

        if self.occ_boolean_tolerance > 0:
            operator.SetFuzzyValue(self.occ_boolean_tolerance * tol_scale)

        operator.Build()
        if not operator.IsDone():
            assert False, "boolean operation failed:" + operation

        result = operator.Shape()
        self.update_topo_list_from_history(operator, objs)

        if operation != 'fragments':
            self.update_topo_list_from_history(operator, tools)

        if remove_tool:
            for gid in gid_tools:
                self.remove(gid)
        if remove_obj:
            for gid in gid_objs:
                self.remove(gid)

        # apparenlty I have to do it just after removing shape...
        self.synchronize_topo_list(action='both')

        if upgrade:
            unifier = ShapeUpgrade_UnifySameDomain(result)
            unifier.Build()
            result = unifier.Shape()

        if keep_highest:
            result = self.select_highest_dim(result)

        nsmall, smax, faces = check_shape_area(result, 1e-6)
        if nsmall > 0:
            dprint1(
                "!!!!! after boolean " +
                str(nsmall) +
                " faces are found too small")

        new_objs = self.register_shapes_balk(result, check_this=self.shape)

        return new_objs

    def union(self, gid_objs, gid_tools, remove_tool=True, remove_obj=True,
              keep_highest=False, upgrade=False, tol_scale=1.0):

        return self.do_boolean('fuse', gid_objs, gid_tools,
                               remove_tool=remove_tool,
                               remove_obj=remove_obj,
                               keep_highest=keep_highest,
                               upgrade=upgrade,
                               tol_scale=tol_scale)

    def intersection(
            self,
            gid_objs,
            gid_tools,
            remove_tool=True,
            remove_obj=True,
            keep_highest=False,
            upgrade=False,
            tol_scale=1.0):

        return self.do_boolean('common', gid_objs, gid_tools,
                               remove_tool=remove_tool,
                               remove_obj=remove_obj,
                               keep_highest=keep_highest,
                               upgrade=upgrade,
                               tol_scale=tol_scale)

    def difference(
            self,
            gid_objs,
            gid_tools,
            remove_tool=True,
            remove_obj=True,
            keep_highest=False,
            upgrade=False, tol_scale=1.0):

        return self.do_boolean('cut', gid_objs, gid_tools,
                               remove_tool=remove_tool,
                               remove_obj=remove_obj,
                               keep_highest=keep_highest,
                               upgrade=upgrade,
                               tol_scale=tol_scale)

    def fragments(self, gid_objs, gid_tools, remove_tool=True, remove_obj=True,
                  keep_highest=False, tol_scale=1.0):

        gid_objs = gid_objs + gid_tools
        return self.do_boolean('fragments', gid_objs, gid_tools,
                               remove_tool=remove_tool,
                               remove_obj=remove_obj,
                               keep_highest=keep_highest,
                               tol_scale=tol_scale)

    def merge_face(
            self,
            gid_objs,
            gid_tools,
            remove_tool=True,
            remove_obj=True,
            keep_highest=False,
            use_upgrade=True):
        '''
        merge faces on the same plane by operationg two cut

        1) works only for the planer surface

        '''
        gid_all = gid_objs + gid_tools

        n1, p1 = self.get_face_normal(gid_objs[0], check_flat=True)
        for gid in gid_all:
            self.get_face_normal(gid, check_flat=True)

        comp = self.new_compound(gid_all)
        xmin, ymin, zmin, xmax, ymax, zmax = self.bounding_box(comp)

        rect = rect_by_bbox_projection(n1, p1, xmin, ymin, zmin,
                                       xmax, ymax, zmax, scale=1.5)

        verts = [BRepBuilderAPI_MakeVertex(gp_Pnt(pt[0], pt[1], pt[2])).Shape()
                 for pt in rect]
        idx = [0, 1, 2, 3, 0]
        edges = []
        for i in range(4):
            edgeMaker = BRepBuilderAPI_MakeEdge(
                verts[idx[i]], verts[idx[i + 1]])
            edgeMaker.Build()
            if not edgeMaker.IsDone():
                assert False, "Can not make line"
            edges.append(edgeMaker.Edge())

        wireMaker = BRepBuilderAPI_MakeWire()
        for edge in edges:
            wireMaker.Add(edge)
        wireMaker.Build()
        if not wireMaker.IsDone():
            assert False, "Failed to make wire"
        wire = wireMaker.Wire()

        faceMaker = BRepBuilderAPI_MakeFace(wire)
        faceMaker.Build()

        if not faceMaker.IsDone():
            assert False, "can not create face"

        face = faceMaker.Face()
        fixer = ShapeFix_Face(face)
        fixer.Perform()
        face = fixer.Face()

        #face_id = self.faces.add(face)
        #self.builder.Add(self.shape, face)
        # return [face_id,]

        operator = BRepAlgoAPI_Cut()

        objs = TopTools_ListOfShape()
        tools = TopTools_ListOfShape()

        for gid in gid_all:
            topolist = self.get_topo_list_for_gid(gid)
            shape = topolist[gid]
            tools.Append(shape)

        objs.Append(face)

        operator.SetRunParallel(self.occ_parallel)
        operator.SetArguments(objs)
        operator.SetTools(tools)

        if self.occ_boolean_tolerance > 0:
            operator.SetFuzzyValue(self.occ_boolean_tolerance)

        operator.Build()
        if not operator.IsDone():
            assert False, "boolean operation failed:" + operation

        result = operator.Shape()

        operator = BRepAlgoAPI_Cut()

        objs = TopTools_ListOfShape()
        tools = TopTools_ListOfShape()

        tools.Append(result)
        objs.Append(face)

        operator.SetRunParallel(self.occ_parallel)
        operator.SetArguments(objs)
        operator.SetTools(tools)

        if self.occ_boolean_tolerance > 0:
            operator.SetFuzzyValue(self.occ_boolean_tolerance)

        operator.Build()
        if not operator.IsDone():
            assert False, "boolean operation failed:" + operation

        result = operator.Shape()

        if use_upgrade:
            unifier = ShapeUpgrade_UnifySameDomain(result)
            unifier.Build()
            result = unifier.Shape()

        if remove_tool:
            for gid in gid_tools:
                self.remove(gid)
        if remove_obj:
            for gid in gid_objs:
                self.remove(gid)

        self.synchronize_topo_list()

        if keep_highest:
            result = self.select_highest_dim(result)
        new_objs = self.register_shapes_balk(result)

        return new_objs

    def union2d(self, gid_objs, gid_tools, remove_tool=True, remove_obj=True,
                keep_highest=False, tol_scale=1.0):
        return self.do_boolean('fuse', gid_objs, gid_tools,
                               remove_tool=remove_tool,
                               remove_obj=remove_obj,
                               keep_highest=keep_highest,
                               upgrade=True,
                               tol_scale=tol_scale)

    def apply_fragments(self):
        if len(self.solids) > 1:
            keys = self.solids.keys()
        elif len(self.solids) == 1:
            keys = []
        elif len(self.faces) > 1:
            keys = self.faces.keys()
        elif len(self.faces) == 1:
            keys = []
        elif len(self.edges) > 1:
            keys = self.edges.keys()
        else:
            keys = []

        if len(keys) > 1:
            gid_objs = keys[:1]
            gid_tools = keys[1:]
            self.fragments(gid_objs, gid_tools,
                           remove_obj=True, remove_tool=True)

    def remove(self, gid, recursive=True):
        akind = {VolumeID: 'face',
                 SurfaceLoopID: 'face',
                 SurfaceID: 'edge',
                 LineLoopID: 'edge',
                 LineID: 'vertex'}

        topolist = self.get_topo_list_for_gid(gid)
        shape = topolist[gid]

        # self.print_number_of_topo_objects()

        if not recursive:
            anc = list(topolist.get_ancestors(gid, akind[gid.__class__]))
            anc_id = [self.get_gid_for_shape(i) for i in anc]
            copier = BRepBuilderAPI_Copy()
            sub_shapes = []
            for s in anc:
                copier.Perform(s)
                assert copier.IsDone(), "Can not copy sub-shape"
                sub_shapes.append(copier.Shape())
            org_subshapes = anc

        if not isinstance(gid, VertexID):
            mapper = get_mapper(self.shape, akind[gid.__class__])

        # this may work, too?
        # self.builder.Remove(self.shape, shape)
        rebuild = ShapeBuild_ReShape()
        rebuild.Remove(shape)
        new_shape = rebuild.Apply(self.shape,  TopAbs_EDGE)

        if not isinstance(gid, VertexID):
            mapper2 = get_mapper(new_shape, akind[gid.__class__])

        # self.print_number_of_topo_objects(new_shape)

        # note we dont put back shell/wire.
        if not recursive and not isinstance(gid, VertexID):
            shape_added = []
            for s, s_org, gid_org in zip(sub_shapes, org_subshapes, anc_id):
                if mapper.Contains(s_org) and not mapper2.Contains(s_org):
                    shape_added.append((s, s_org, gid_org))
            for s, s_org, gid_org in shape_added:
                self.builder.Add(new_shape, s)
                topolist = self.get_topolist_for_shape(s)
                topolist[gid_org] = s

            '''
            for s in anc:
                flag = mapper.Contains(s)
                #print("status",  mapper.Contains(s), mapper2.Contains(s))
                if mapper.Contains(s) and not mapper2.Contains(s):
                    shape_added.append(s)
            print("putting back", shape_added)
            for s in shape_added:
                location = s.Location()
                if not location.IsIdentity():
                    location = TopLoc_Location()
                ss = s.Located(location)
                print('putting back', ss)
                self.builder.Add(new_shape, ss)
            '''

        self.shape = new_shape

        '''
        elif recursive:

        #    When face is deleted. Sometime edge remains.
        #    Why?
        #    Do I need to do it really recursively??

            shape_removed = []
            for s in anc:
                flag = mapper2.Contains(s)
                if flag:
                    shape_removed.append(s)

            for s in shape_removed:
                self.builder.Remove(new_shape, s)
        '''

    def inverse_remove(self, gids):
        comp = TopoDS_Compound()
        b = self.builder
        b.MakeCompound(comp)

        for gid in gids:
            topolist = self.get_topo_list_for_gid(gid)
            shape = topolist[gid]
            self.builder.Add(comp, shape)

        self.shape = comp

        return gids

    def copy(self, gid):
        topolist = self.get_topo_list_for_gid(gid)
        shape = topolist[gid]

        copier = BRepBuilderAPI_Copy()
        copier.Perform(shape)
        if not copier.IsDone():
            assert False, "Can not copy shape"
        shape = copier.Shape()

        self.builder.Add(self.shape, shape)

        gid = topolist.add(shape)
        return gid

    def _perform_transform(self, gid, transformer, copy, transformer2=None):
        return_list = False
        if isinstance(gid, (tuple, list)):
            return_list = True
            if len(gid) > 1:
                shape = self.new_compound(gids=gid)
                use_compound = True
            else:
                topolist = self.get_topo_list_for_gid(gid[0])
                gid = gid[0]
                shape = topolist[gid]
                use_compound = False
        else:
            topolist = self.get_topo_list_for_gid(gid)
            shape = topolist[gid]
            use_compound = False

        transformer.Perform(shape, True)

        if not transformer.IsDone():
            assert False, "can not translate"

        new_shape = transformer.ModifiedShape(shape)

        if transformer2 is not None:
            transformer2.Perform(new_shape, True)
            new_shape = transformer2.ModifiedShape(new_shape)

        isNew = not new_shape.IsSame(shape)

        if use_compound:
            gids_new = self.register_shapes_balk(new_shape)
            if not copy:
                for g in gid:
                    self.remove(g)
            self.synchronize_topo_list()
            return gids_new
        elif isNew:
            if not copy:
                self.remove(gid)
            self.builder.Add(self.shape, new_shape)

            if not copy:
                topolist[gid] = new_shape
                new_gid = None
            else:
                new_gid = topolist.add(new_shape)
        else:
            new_gid = None

        if isinstance(gid, (tuple, list)):
            return [new_gid]
        if return_list:
            return [new_gid]
        return new_gid

    def translate(self, gid, delta, copy=False):
        trans = gp_Trsf()
        trans.SetTranslation(gp_Vec(delta[0], delta[1], delta[2]))
        transformer = BRepBuilderAPI_Transform(trans)

        return self._perform_transform(gid, transformer, copy)

    def rotate(self, gid, point_on_axis, axis_dir, angle, copy=False):
        trans = gp_Trsf()

        x, y, z = point_on_axis
        ax, ay, az = axis_dir
        axis_revolution = gp_Ax1(gp_Pnt(x, y, z), gp_Dir(ax, ay, az))
        trans.SetRotation(axis_revolution, angle)

        transformer = BRepBuilderAPI_Transform(trans)

        return self._perform_transform(gid, transformer, copy)

    def translate_rot(self, gid, delta, point_on_axis, axis_dir, angle,
                      copy=False):
        trans = gp_Trsf()
        trans.SetTranslation(gp_Vec(delta[0], delta[1], delta[2]))
        transformer = BRepBuilderAPI_Transform(trans)

        trans2 = gp_Trsf()
        x, y, z = point_on_axis
        ax, ay, az = axis_dir
        axis_revolution = gp_Ax1(gp_Pnt(x, y, z), gp_Dir(ax, ay, az))
        trans2.SetRotation(axis_revolution, angle)
        transformer2 = BRepBuilderAPI_Transform(trans2)

        return self._perform_transform(gid, transformer, copy,
                                       transformer2=transformer2)

    def dilate(self, gids, xyz, abc, copy=False):
        x, y, z = xyz
        a, b, c = abc

        if a == b and b == c:
            gt = gp_Trsf()
            pts = gp_Pnt(x, y, z)
            gt.SetScale(pts, float(a))
            transformer = BRepBuilderAPI_Transform(gt)
        else:
            gt = gp_GTrsf()
            gt.SetVectorialPart(gp_Mat(a, 0, 0, 0, b, 0, 0, 0, c))
            gt.SetTranslationPart(
                gp_XYZ(x * (1 - a), y * (1 - b), z * (1 - c)))
            transformer = BRepBuilderAPI_GTransform(gt)

        return self._perform_transform(gids, transformer, copy)

    def symmetrize(self, gid, abcd, copy=False):

        a, b, c, d = abcd

        '''
        p = max((a * a + b * b + c * c), 1e-12)
        f = -2.0 / p
        vec = (a * d * f, b * d * f, c * d * f)
        mat = (1 + a * a * f,
               a * b * f,
               a * c * f,
               a * b * f,
               1. + b * b * f,
               b * c * f,
               a * c * f,
               b * c * f,
               1. + c * c * f)
        gt = gp_GTrsf()
        gt.SetVectorialPart(gp_Mat(*mat))
        gt.SetTranslationPart(gp_XYZ(*vec))
        transformer = BRepBuilderAPI_GTransform(gt)
        '''
        a, b, c, d = abcd
        gt = gp_Trsf()
        dirt = gp_Dir(a, b, c)
        if (abs(a) >= abs(b) and abs(a) >= abs(c)):
            pts = gp_Pnt(d/a, 0, 0)
        elif (abs(b) >= abs(a) and abs(b) >= abs(c)):
            pts = gp_Pnt(0, d/b, 0)
        elif (abs(c) >= abs(a) and abs(c) >= abs(a)):
            pts = gp_Pnt(0, 0, d/c)
        ax2 = gp_Ax2(pts, dirt)
        gt.SetMirror(ax2)
        transformer = BRepBuilderAPI_Transform(gt)
        return self._perform_transform(gid, transformer, copy)

    def extrude(self, gids, translation=None, rotation=None, wire=None):
        '''
        comp = TopoDS_Compound()
        self.builder.MakeCompound(comp);

        for gid in gids:
           topolist = self.get_topo_list_for_gid(gid)
           shape = topolist[gid]
           self.builder.Add(comp, shape)
        '''
        ret = []
        for kk, gid in enumerate(gids):
            topolist = self.get_topo_list_for_gid(gid)
            shape = topolist[gid]
            delete_input = topolist.is_toplevel(gid, self.shape)

            if translation is not None:
                if isinstance(translation, trans_delta):
                    dx, dy, dz = translation
                else:
                    dx, dy, dz = translation[kk]
                p = BRepPrimAPI_MakePrism(shape, gp_Vec(dx, dy, dz), False)

            elif rotation is not None:
                x, y, z = rotation[0]
                ax, ay, az = rotation[1]
                angle = rotation[2]
                pnt = gp_Pnt(x, y, z)
                dr = gp_Dir(ax, ay, az)
                ax = gp_Ax1(pnt, dr)
                p = BRepPrimAPI_MakeRevol(shape, ax, angle, False)

            elif wire is not None:
                from OCC.Core.GeomFill import GeomFill_IsDiscreteTrihedron
                p = BRepOffsetAPI_MakePipe(
                    wire, shape, GeomFill_IsDiscreteTrihedron)

            else:
                assert False, "unknonw option"

            p.Build()
            if not p.IsDone():
                assert False, "can not extrude : " + str(gid)

            if delete_input:
                self.builder.Remove(self.shape, shape)

            last = p.LastShape()

            if translation is not None:
                result = p.Prism().Shape()
            else:
                result = p.Shape()

            gid_last = self.add_to_topo_list(last)
            gid_extruded = self.add_to_topo_list(result)

            ret.append((gid_last, gid_extruded, result))

        return ret

    def defeature(self, gid, gids_face):

        aSolid = self.solids[gid]

        features = TopTools_ListOfShape()
        for tmp in gids_face:
            topolist = self.get_topo_list_for_gid(tmp)
            shape = topolist[tmp]
            features.Append(shape)

        self.remove(gid, recursive=True)
        self.synchronize_topo_list(action='both')

        aDF = BRepAlgoAPI_Defeaturing()
        aDF.SetShape(aSolid)
        aDF.AddFacesToRemove(features)
        aDF.SetRunParallel(self.occ_parallel)
        aDF.SetToFillHistory(False)
        aDF.Build()

        if not aDF.IsDone():
            assert False, "Can not defeature"
        result = aDF.Shape()

#        self.remove(gid)
#
        result = self.select_highest_dim(result)
        new_objs = self.register_shapes_balk(result, check_this=self.shape)

        return new_objs

    def project_shape_on_wp(self, gids, c1, n1, ptol=-1, fill=False, base_shape=None):
        '''
        Project shapes using Offset_NormalProjection

           1) Collect unique edges and make list of wires which makes surfaces
           2) Project edges one-by-one so that we can track which edges are projected to where.
              if project does not create edges. the edge is normal to the projection, and avoid
              trying to create a face for it.
           3) Create wire from projected edges and fill inside using MakeFace -> FixShape
           4) Check if any vertex is missing (for isolatee vertices)

           (note 1) this subroutine is meant to handle the projection of surface.
           (note 2) if a face is made from several edges and if one of edges are perfectly.
                  normal to the destination plane. it does not create the surface.
        '''
        shapes = [self.get_shape_for_gid(gid, group=0) for gid in gids]

        base_shape = self.shape if base_shape is None else base_shape

        mapper = get_mapper(base_shape, 'edge')
        seen = topo_seen(mapping=mapper)

        unique_edges = []
        edge_index = {}
        wires = []

        # find unique_edges + insolated vertex
        for s in shapes:
            wires.append([])
            for p in iter_shape_once(s, 'edge'):
                if seen.check_shape(p) == 0:
                    unique_edges.append(p)
                    edge_index[seen.index(p)] = len(unique_edges)-1
                idx = seen.index(p)
                if isinstance(s, TopoDS_Face):
                    wires[-1].append(idx)

        pnt = gp_Pnt(c1[0], c1[1], c1[2])
        dr = gp_Dir(n1[0], n1[1], n1[2])

        pl = Geom_Plane(pnt, dr)

        maker = BRepBuilderAPI_MakeFace(pl, self.occ_geom_tolerance)
        maker.Build()
        if not maker.IsDone():
            assert False, "Faild to generate plane"
        plane = maker.Face()

        projected_edges = []
        result = self.new_compound()

        mapping = [-1]*len(unique_edges)

        for k, s in enumerate(unique_edges):
            proj = BRepOffsetAPI_NormalProjection(plane)
            proj.Add(s)
            proj.Build()
            if not proj.IsDone():
                assert False, "Failed to perform projection"
            tmp = proj.Projection()
            count = 0
            for p in iter_shape_once(tmp, 'edge'):
                projected_edges.append(p)
                self.builder.Add(result, p)
                count = count + 1
            if count == 1:
                mapping[k] = len(projected_edges)-1

        if fill:
            for wire in wires:
                if len(wire) == 0:
                    continue
                if any([mapping[edge_index[w]] == -1 for w in wire]):
                    continue

                idx = [mapping[edge_index[w]] for w in wire]

                wireMaker = BRepBuilderAPI_MakeWire()

                for i in idx:
                    wireMaker.Add(projected_edges[i])
                wireMaker.Build()
                if not wireMaker.IsDone():
                    assert False, "Failed to make wire"
                wire = wireMaker.Wire()

                faceMaker = BRepBuilderAPI_MakeFace(wire)
                faceMaker.Build()

                if not faceMaker.IsDone():
                    assert False, "can not create face"

                face = faceMaker.Face()
                fixer = ShapeFix_Face(face)
                fixer.Perform()
                self.builder.Add(result, fixer.Face())

        ###
        # somehow some points are not projected. we check if all points
        # are projected if not we add it to results.
        ###

        point_shapes = [p for p in iter_shape_once(result, 'vertex')]
        pnts = [self.bt.Pnt(p) for p in point_shapes]
        ptx = np.array([(pnt.X(), pnt.Y(), pnt.Z(),) for pnt in pnts])

        xmin, ymin, zmin, xmax, ymax, zmax = self.bounding_box(self._shape_bk)
        size = np.sqrt((xmin - xmax)**2 + (ymax - ymin)**2 + (zmax - zmin)**2)
        size = (size * self.occ_geom_tolerance if ptol == -1 else
                size * ptol)

        for gid in gids:
            if not isinstance(gid, VertexID):
                continue

            shape = self.get_shape_for_gid(gid, group=0)

            pnt = self.bt.Pnt(shape)
            p = np.array((pnt.X(), pnt.Y(), pnt.Z(),))
            p2 = project_ptx_2_plain(n1, c1, p)
            if len(ptx) != 0:
                dist = np.min(np.sqrt(np.sum((ptx - p2)**2, -1)))
                if dist < size:
                    continue
            #print("adding ",p2)
            x, y, z = float(p2[0]), float(p2[1]), float(p2[2])
            p = BRepBuilderAPI_MakeVertex(gp_Pnt(x, y, z)).Shape()
            self.builder.Add(result, p)

        return result

    def apply_fixshape_shell(self, gids):
        for gid in gids:
            print(self.get_face_normal(gid))

        rebuild = ShapeBuild_ReShape()
        for t in gids:
            face = self.faces[t]
            sff = ShapeFix_Face(face)
            sff.SetFixAddNaturalBoundMode(True)
            sff.SetFixSmallAreaWireMode(True)

            sff.Perform()
            if sff.Status(ShapeExtend_DONE1):
                print(" . Some wires are fixed")
            elif sff.Status(ShapeExtend_DONE2):
                print(" . Orientation of wires fixed")
            elif sff.Status(ShapeExtend_DONE3):
                print(" . Missing seam added")
            elif sff.Status(ShapeExtend_DONE4):
                print(" . Small area wire removed")
            elif sff.Status(ShapeExtend_DONE5):
                print(" . Natural bounds added")

            self.builder.Remove(self.shape, face)
            newface = sff.Face()
            self.faces[t] = newface

        for gid in gids:
            print(self.get_face_normal(gid))

        '''
        try:
            shellMaker = BRepBuilderAPI_MakeShell()
            shellMaker.Perform()
            result = shellMaker.SewedShape()
        except BaseException:
            assert False, "Failed to make shells"

        ex1 = TopExp_Explorer(result, TopAbs_SHELL)
        while ex1.More():
            print("fixing shell")
            shell = ex1.Current()
            fixer = ShapeFix_Shell(shell)
            fixer.Perform()
            shell = fixer.Shell()
            break
            ex.Next()
        '''

    def process_normal_parameters(self, tax, c1, objs):
        p0 = self.get_point_coord(c1)

        if tax[0] == 'normal':
            if tax[1].startswith('wp'):
                wp = objs[tax[1]]
                n1 = wp.get_norm()
                #p0 = wp.get_center()
            else:
                gid = self.get_target2(objs, [tax[1]])[0]
                n1, _void = self.get_face_normal(gid, check_flat=True)
            if tax[2]:
                tt = -n1
            else:
                tt = n1
        elif tax[0] == 'fromto_points':
            dests1 = [x.strip() for x in tax[1].split(',')]
            dests2 = [x.strip() for x in tax[2].split(',')]

            gid_dests1 = self.get_target1(objs, dests1, 'p')
            gid_dests2 = self.get_target1(objs, dests2, 'p')

            assert len(gid_dests1) == 1, "Incorrect destination setting"
            assert len(gid_dests2) == 1, "Incorrect destination setting"

            p1 = self.get_point_coord(gid_dests1[0])
            p2 = self.get_point_coord(gid_dests2[0])

            n1 = p2 - p1
            n1 /= np.sqrt(np.sum(n1**2))

            if tax[4]:
                n1 *= -1

        elif tax[0] == 'radial':
            axis = np.array(eval(tax[1]))
            axis = axis / np.sqrt(np.sum(axis**2))
            point_on_axis = np.array(eval(tax[2]))

            n1 = p0 - axis * np.sum((p0 - point_on_axis) * axis)
            n1 = n1 / np.sqrt(np.sum(n1**2))

            if tax[3]:
                n1 *= -1

        elif tax[0] == 'polar':
            center = np.array(eval(tax[1]))
            n1 = p0 - center
            n1 = n1 / np.sqrt(np.sum(n1**2))

            if tax[2]:
                n1 *= -1

        else:
            tax = np.array(tax).flatten()
            n1 = tax / np.sqrt(np.sum(np.array(tax)**2))

        return n1, p0

    def process_plane_parameters(self, args, objs, gids=None):
        comp = None if gids is None else self.new_compound(gids)
        xmin, ymin, zmin, xmax, ymax, zmax = self.bounding_box(comp)

        if args[0] == '3_points':
            # args[1] = ['3_points', '1', '7', '8']

            gid_ptx1 = self.get_target1(objs, [args[1], ], 'p')[0]
            gid_ptx2 = self.get_target1(objs, [args[2], ], 'p')[0]
            gid_ptx3 = self.get_target1(objs, [args[3], ], 'p')[0]
            ptx1 = self.get_point_coord(gid_ptx1)
            ptx2 = self.get_point_coord(gid_ptx2)
            ptx3 = self.get_point_coord(gid_ptx3)

            n = np.cross(ptx1 - ptx2, ptx1 - ptx3)
            if np.sum(n**2) == 0:
                assert False, "three points does not span a surface."
            normal = n / np.sqrt(np.sum(n**2))
            cptx = (ptx1 + ptx2 + ptx3) / 3.0

        elif args[0] == 'by_abc':
            data = np.array(args[1]).flatten()
            normal = np.array(data[:3])
            normal = normal/np.sqrt(np.sum(normal**2))
            xx = np.array(
                [(xmin + xmax) / 2, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0])
            s = data[-1] - np.sum(normal * xx)
            cptx = xx + s * normal

        elif args[0] == 'face_parallel':
            gid_face = self.get_target1(objs, [args[1], ], 'f')[0]

            if args[2].strip() != '':
                gid_ptx = self.get_target1(objs, [args[2], ], 'p')[0]
                cptx = self.get_point_coord(gid_ptx)
            else:
                cptx = self.get_point_on_face(gid_face)

            normal, _void = self.get_face_normal(gid_face, check_flat=True)

        elif args[0] == 'face_normal':
            gid_face = self.get_target1(objs, [args[1], ], 'f')[0]
            tmp = [x.strip() for x in args[2].split(',')]
            gid_ptx = self.get_target1(objs, tmp, 'p')
            ptx1 = self.get_point_coord(gid_ptx[0])
            ptx2 = self.get_point_coord(gid_ptx[1])

            cptx = ptx1

            n1, _void = self.get_face_normal(gid_face, check_flat=True)
            n2 = ptx2 - ptx1
            n2 = n2 / np.sqrt(np.sum(n2**2))

            normal = np.cross(n1, n2)
        elif args[0] == 'workplane':
            wp = objs['wp' + args[1]]
            normal = wp.get_norm()
            cptx = wp.get_center()
        else:
            assert False, "unknown option:" + args
        return cptx, normal

    def add_sequence(self, gui_name, gui_param, geom_name):
        self.geom_sequence.append((gui_name, gui_param, geom_name))

    '''
    high level interface:
       methods below directroy corresponds to GUI interface
       these routine should
        1) call builder.Add to add a shape to Compound
        2) register the name of shape
    '''

    # 0D vertices
    def Point_build_geom(self, objs, *args):
        xarr, yarr, zarr = args

        try:
            pos = np.vstack((xarr, yarr, zarr)).transpose()
        except BaseException:
            assert False, "can not make proper input array"

        PTs = [self.add_point(p) for p in pos]

        _newobjs = []
        for p in PTs:
            shape = self.vertices[p]
            self.builder.Add(self.shape, shape)
            newkey = objs.addobj(p, 'pt')
            _newobjs.append(newkey)

        return list(objs), _newobjs

    def PointOCC_build_geom(self, objs, *args):
        return self.Point_build_geom(objs, *args)

    def PointCenter_build_geom(self, objs, *args):
        targets1, targets2 = args
        targets1 = [x.strip() for x in targets1.split(',')]
        targets2 = [x.strip() for x in targets2.split(',')]

        gids_1 = self.get_target1(objs, targets1, 'p')
        gids_2 = self.get_target1(objs, targets2, 'p')

        PTs = []
        for g1, g2 in zip(gids_1, gids_2):
            ptx1 = self.get_point_coord(g1)
            ptx2 = self.get_point_coord(g2)
            PTs.append(self.add_point((ptx1 + ptx2) / 2.0))

        _newobjs = []
        for p in PTs:
            shape = self.vertices[p]
            self.builder.Add(self.shape, shape)
            newkey = objs.addobj(p, 'pt')
            _newobjs.append(newkey)

        return list(objs), _newobjs

    def PointOnEdge_build_geom(self, objs, *args):
        targets, uarr = args
        targets = [x.strip() for x in targets.split(',')]
        gids = self.get_target1(objs, targets, 'l')

        uarr = np.array(uarr, dtype=float)

        PTs = []
        for gid in gids:
            PTs.extend(self.add_point_on_edge(gid, uarr))

        _newobjs = []
        for p in PTs:
            shape = self.vertices[p]
            self.builder.Add(self.shape, shape)
            newkey = objs.addobj(p, 'pt')
            _newobjs.append(newkey)

        return list(objs), _newobjs

    def PointCircleCenter_build_geom(self, objs, *args):
        targets, use_focus = args
        targets = [x.strip() for x in targets.split(',')]
        gids = self.get_target1(objs, targets, 'l')

        if use_focus:
            ptx = np.vstack([self.get_conic_focus(gid) for gid in gids])
        else:
            ptx = np.vstack([self.get_circle_center(gid) for gid in gids])

        newobjs = []
        for p in ptx:
            p = self.add_point(p)
            shape = self.vertices[p]
            self.builder.Add(self.shape, shape)
            newobjs.append(objs.addobj(p, 'pt'))

        return list(objs), newobjs

    def PointByUV_build_geom(self, objs, *args):
        targets, uarr, varr = args
        targets = [x.strip() for x in targets.split(',')]
        gids = self.get_target1(objs, targets, 'f')

        uarr = np.array(uarr, dtype=float)
        varr = np.array(varr, dtype=float)

        assert len(uarr) == len(varr), "u and v should be the same length"
        PTs = []
        for gid in gids:
            PTs.extend(self.add_point_on_face(gid, uarr, varr))

        _newobjs = []
        for p in PTs:
            shape = self.vertices[p]
            self.builder.Add(self.shape, shape)
            newkey = objs.addobj(p, 'pt')
            _newobjs.append(newkey)

        return list(objs), _newobjs

    # 1D edges
    def Line_build_geom(self, objs, *args):
        xarr, yarr, zarr, make_spline, periodic = args
        lcar = 0.0
        if len(xarr) < 2:
            return
        try:
            pos = np.vstack((xarr, yarr, zarr)).transpose()
        except BaseException:
            assert False, "can not make proper input array"

        dist = np.sqrt(np.sum((pos[:-1, :] - pos[1:, :])**2, 1))

        if min(dist) == 0.0:
            assert False, "minimum distance between point is 0.0"
        if max(dist) > min(dist) * 1e4:
            assert False, "some points are too close (d_max > d_min*1e4)"

        if not make_spline:
            pts = [self.add_point(p) for ii, p in enumerate(pos)]
            pts1 = pts[:-1]
            pts2 = pts[1:]

            newkeys = []
            for p1, p2 in zip(pts1, pts2):
                ln = self.add_line(p1, p2)
                shape = self.edges[ln]
                self.builder.Add(self.shape, shape)
                newkeys.append(objs.addobj(ln, 'ln'))
            if periodic:
                ln = self.add_line(pts[-1], pts[0])
                shape = self.edges[ln]
                self.builder.Add(self.shape, shape)
                newkeys.append(objs.addobj(ln, 'ln'))

            _newobjs = newkeys
            if not periodic:
                newobj1 = objs.addobj(pts[0], 'pt')
                newobj2 = objs.addobj(pts[-1], 'pt')
                _newobjs.append(newobj1)
                _newobjs.append(newobj2)
        else:
            spline = self.add_spline(pos, periodic=periodic)
            shape = self.edges[spline]
            self.builder.Add(self.shape, shape)

            newobj = objs.addobj(spline, 'sp')
            _newobjs = [newobj]

        return list(objs), _newobjs

    def LineOCC_build_geom(self, objs, *args):
        return self.Line_build_geom(objs, *args)

    def ExtendedLine_build_geom(self, objs, *args):
        lines, ratio, resample = args
        lines = [x.strip() for x in lines.split(',')]
        gids = self.get_target1(objs, lines, 'p')

        lines = [self.add_extended_line(gid, ratio, resample)
                 for gid in gids]

        newobjs = []
        newkeys = []
        for l in lines:
            for gid in l:
                newkeys.append(objs.addobj(gid, 'ln'))
        return list(objs), newkeys
#            shape = self.edges[l]
#            self.builder.Add(self.shape, shape)
#            newobj = objs.addobj(l, 'sp')
#            newobjs.append(newobj)
#        return list(objs), newobjs

    def Polygon_build_geom(self, objs, *args):
        xarr, yarr, zarr = args
        lcar = 0.0
        if len(xarr) < 2:
            return
        try:
            pos = np.vstack((xarr, yarr, zarr)).transpose()
        except BaseException:
            print("can not make proper input array")
            return
        # check if data is already closed...
        if np.abs(np.sum((pos[0] - pos[-1])**2)) < 1e-17:
            pos = pos[:-1]

        gids = [self.add_point(p) for p in pos]
        polygon = self.add_polygon(gids)
        shape = self.faces[polygon]
        self.builder.Add(self.shape, shape)

        newobj = objs.addobj(polygon, 'plg')
        return list(objs), [newobj]

    def Polygon2_build_geom(self, objs, *args):
        return self.Polygon_build_geom(objs, *args)

    def OCCPolygon_build_geom(self, objs, *args):
        pts = args
        pts = [x.strip() for x in pts[0].split(',')]
        gids = self.get_target1(objs, pts, 'p')

        if len(gids) < 3:
            assert False, "Polygon requires more than 2 guide points"

        polygon = self.add_polygon(gids)
        shape = self.faces[polygon]
        self.builder.Add(self.shape, shape)

        newobj = objs.addobj(polygon, 'plg')
        return list(objs), [newobj]

    def Spline_build_geom(self, objs, *args):
        pts = args
        pts = [x.strip() for x in pts[0].split(',')]
        gids = self.get_target1(objs, pts, 'p')

        if len(gids) < 3:
            assert False, "Spline requires more than 2 guide points"
        if len(gids) == 2 and gids[0] == gids[-1]:
            assert False, "Spline loop requires more than 3 guide points"

        if len(gids) > 3 and gids[0] == gids[-1]:
            periodic = True
            gids = gids[:-1]
        else:
            periodic = False

        pos = np.vstack([self.get_point_coord(gid) for gid in gids])

        spline = self.add_spline(pos, periodic=periodic)
        shape = self.edges[spline]
        self.builder.Add(self.shape, shape)

        newobj = objs.addobj(spline, 'sp')
        return list(objs), [newobj]

    def Spline2D_build_geom(self, objs, *args):
        return self.Spline_build_geom(objs, *args)

    def MovePoint_build_geom(self, objs, *args):
        target, p1, p2 = args

        target = [x.strip() for x in target.split(',')]
        target = self.get_target2(objs, target)[0]
        shape = self.get_shape_for_gid(target)

        p1 = [x.strip() for x in p1.split(',')]
        p1 = self.get_target1(objs, p1, 'p')[0]
        p2 = [x.strip() for x in p2.split(',')]
        p2 = self.get_target1(objs, p2, 'p')[0]

        p1 = self.vertices[p1]

        p = self.get_point_coord(p2)
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        p2 = BRepBuilderAPI_MakeVertex(gp_Pnt(x, y, z)).Shape()

        rebuild = ShapeBuild_ReShape()
        rebuild.Apply(shape)

        rebuild.SetModeConsiderLocation(False)
        rebuild.Replace(p1, p2)

        result = rebuild.Apply(shape)

        # for edge in iter_shape_once(result, 'edge'):
        #    if BRep_Tool.Degenerated(edge):
        #        rebuild.Remove(edge)
        #result = rebuild.Apply(result)

        self.remove(target)
        self.synchronize_topo_list(action='both')

        gids_new = self.register_shapes_balk(result, check_this=self.shape)
        newkeys = []
        for gid in gids_new:
            newkeys.append(objs.addobj(gid, 'obj'))

        #self.shape =result
        self.synchronize_topo_list()

        return list(objs), newkeys

    def Simplifiers_build_geom(self, objs, *args):
        targets, use_merge, use_remover, rm_size = args

        #print(targets, use_merge, use_remover, rm_size)

        targets = [x.strip() for x in targets.split(',')]
        vols = self.get_target1(objs, targets, 'v')

        from OCC.Core.Precision import precision_Confusion, precision_Angular

        tol1 = precision_Confusion()
        tol2 = precision_Angular()

        newkeys = []

        for v in vols:
            shape = self.solids[v]
            good = False
            if use_merge:
                unifier = ShapeUpgrade_UnifySameDomain(shape)
                unifier.SetLinearTolerance(tol1)
                unifier.SetAngularTolerance(tol2)
                unifier.Build()
                shape = unifier.Shape()
                good = True
            if use_remover:
                remover = ShapeUpgrade_RemoveInternalWires(shape)
                remover.SetMinArea(rm_size)
                remover.SetRemoveFaceMode(True)
                flag = remover.Perform()

                if (remover.Status(ShapeExtend_OK) or
                    remover.Status(ShapeExtend_DONE1) or
                        remover.Status(ShapeExtend_DONE2)):
                    shape = remover.GetResult()
                    if remover.Status(ShapeExtend_DONE1):
                        print("internal wire are removed")
                    if remover.Status(ShapeExtend_DONE2):
                        print("small faces are removed")
                    good = True
            if good:
                self.remove(v)
                gids_new = self.register_shapes_balk(
                    shape, check_this=self.shape)
                for gid in gids_new:
                    newkeys.append(objs.addobj(gid, 'uni'))

        return list(objs), newkeys

    def SplitHairlineFace_build_geom(self, objs, *args):
        vols, thr1, thr2, targets, recursive = args
        targets = [x.strip() for x in targets.split(',') if len(x.strip()) > 0]
        faces = self.get_target1(objs, targets, 'f')
        vols = [x.strip() for x in vols.split(',')]
        vols = self.get_target1(objs, vols, 'v')

        thr1 = float(thr1)
        thr2 = float(thr2)

        from .occ_heal_extra import (split_hairlineface,
                                     replace_faces)
        from .occ_inspect import find_min_distance_in_face

        newobj = []

        for v in vols:
            vol = self.solids[v]
            orig_vol = vol
            while True:
                mapper = get_mapper(vol, 'face')

                new_faces = []
                nsmall, smax, sfaces, areas = check_shape_area(vol, thr1,
                                                               return_area=True)
                small_faces = [f for f, a in zip(sfaces, areas)
                               if find_min_distance_in_face(self.bt, f) < thr2]

                if len(small_faces) != 0:
                    data = [(mapper.FindIndex(f), a, find_min_distance_in_face(self.bt, f))
                            for f, a in zip(sfaces, areas)
                            if find_min_distance_in_face(self.bt, f) < thr2]

                    txt = '\n'.join([str(int(gid)) + "\t(area = "+str(a) +
                                     ",\tmin D= " + str(dist) + ")"
                                     for gid, a, dist in data])

                    print("Working on following faces: value=" + str(v))
                    print(txt)

                faces = [self.faces[f] for f in faces]
                for f in faces:
                    if mapper.Contains(f):
                        small_faces.append(f)
                    else:
                        print("not contained!!")

                print(small_faces)
                if len(small_faces) == 0:
                    break

                replaced_faces = []
                for f in small_faces:
                    print("processing face:" + str(mapper.FindIndex(f)))
                    replacements = split_hairlineface(f, limit=thr2)
                    if replacements is None:
                        print("skipping...." + str(mapper.FindIndex(f)))
                        continue
                    new_faces.extend(replacements)
                    replaced_faces.append(f)

                vol = replace_faces(vol, replaced_faces, new_faces)
                if not recursive:
                    break

            solid_id = self.solids.add(vol)
            self.builder.Add(self.shape, vol)

            newobj.append(objs.addobj(solid_id, 'vol'))

            self.builder.Remove(self.shape, orig_vol)

        self.synchronize_topo_list(action='both', verbose=False)
        return list(objs), newobj

    def CapFaces_build_geom(self, objs, *args):
        vols, targets, use_filling = args
        targets = [x.strip() for x in targets.split(',')]
        faces = self.get_target1(objs, targets, 'f')
        vols = [x.strip() for x in vols.split(',')]
        vols = self.get_target1(objs, vols, 'v')

        from .occ_heal_extra import (create_cap_face,
                                     replace_faces)
        newobj = []

        vol = self.solids[vols[0]]
        faces = [self.faces[f] for f in faces]

        new_face = create_cap_face(
            vol, faces, use_filling, self.occ_geom_tolerance)

        for v in vols:
            vol = self.solids[v]
            orig_vol = vol
            vol = replace_faces(vol, faces, [new_face])

            self.builder.Add(self.shape, vol)
            self.builder.Remove(self.shape, orig_vol)

            solid_id = self.solids.add(vol)
            newobj.append(objs.addobj(solid_id, 'vol'))

        self.synchronize_topo_list(action='both', verbose=False)
        return list(objs), newobj

    def ReplaceFaces_build_geom(self, objs, *args):
        vols, targets, ntargets = args
        targets = [x.strip() for x in targets.split(',')]
        faces = self.get_target1(objs, targets, 'f')
        ntargets = [x.strip() for x in ntargets.split(',')]
        faces2 = self.get_target1(objs, ntargets, 'f')

        vols = [x.strip() for x in vols.split(',')]
        vols = self.get_target1(objs, vols, 'v')

        from .occ_heal_extra import replace_faces

        faces = [self.faces[f] for f in faces]
        faces2 = [self.faces[f] for f in faces2]

        newobj = []

        for v in vols:
            vol = self.solids[v]
            orig_vol = vol
            vol = replace_faces(vol, faces, faces2)

            self.builder.Add(self.shape, vol)
            self.builder.Remove(self.shape, orig_vol)

            solid_id = self.solids.add(vol)
            newobj.append(objs.addobj(solid_id, 'vol'))

        self.synchronize_topo_list(action='both', verbose=False)
        return list(objs), newobj

    def CreateLine_build_geom(self, objs, *args):
        pts = args
        pts = [x.strip() for x in pts[0].split(',')]
        gids = self.get_target1(objs, pts, 'p')
        newkeys = []

        for i in range(len(gids) - 1):
            p0 = gids[i]
            p1 = gids[i + 1]
            ln = self.add_line(p0, p1)
            shape = self.edges[ln]
            self.builder.Add(self.shape, shape)
            newkeys.append(objs.addobj(ln, 'ln'))

        return list(objs), newkeys

    def LineLoop_build_geom(self, objs, *args):
        assert False, "We don't support this"

    # 2D faces
    def Rect_build_geom(self, objs, *args):
        c1, e1, e2 = args
        lcar = 0.0

        c1 = np.array(c1)
        e1 = np.array(e1)
        e2 = np.array(e2)
        p1 = self.add_point(c1)
        p2 = self.add_point(c1 + e1)
        p3 = self.add_point(c1 + e1 + e2)
        p4 = self.add_point(c1 + e2)
        l1 = self.add_line(p1, p2)
        l2 = self.add_line(p2, p3)
        l3 = self.add_line(p3, p4)
        l4 = self.add_line(p4, p1)
        ll1 = self.add_line_loop([l1, l2, l3, l4])
        rec1 = self.add_plane_surface(ll1)

        shape = self.faces[rec1]
        self.builder.Add(self.shape, shape)

        newkey = objs.addobj(rec1, 'rec')
        return list(objs), [newkey]

    def Circle_build_geom(self, objs, *args):
        if len(args) == 6:
            center, ax1, ax2, radius, npts, make_face = args
        else:
            center, ax1, ax2, radius, make_face = args
            npts = 4
        assert radius != 0, "Circle radius must be >0"

        a1 = np.array(ax1)
        a2 = np.array(ax2)

        a1 = a1 / np.sqrt(np.sum(a1**2))
        a2 = a2 / np.sqrt(np.sum(a2**2))
        a2 = np.cross(np.cross(a1, a2), a1)
        dirct = np.cross(a1, a2)

        a1 = a1 * radius
        a2 = a2 * radius

        c = np.array(center)
        #print(a1, a2, c)
        if npts == 0:
            edges = [self.add_circle_by_axis_radius(c, dirct, radius)]
        else:
            da = 2 * np.pi / npts
            da_h = np.pi / npts
            points = []
            hpoints = []
            for i in range(npts):
                angle = i * da
                points.append(self.add_point(c + a1 * np.cos(angle) +
                                             a2 * np.sin(angle)))
                angle = i * da + da_h
                hpoints.append(self.add_point(c + a1 * np.cos(angle) +
                                              a2 * np.sin(angle)))
            edges = []
            for i in range(npts - 1):
                edges.append(self.add_circle_arc(points[i], hpoints[i],
                                                 points[i + 1]))

            edges.append(self.add_circle_arc(points[-1], hpoints[-1],
                                             points[0]))
        if make_face:
            ll1 = self.add_line_loop(edges)
            ps1 = self.add_plane_surface(ll1)
            shape = self.faces[ps1]
            self.builder.Add(self.shape, shape)
            newkey = [objs.addobj(ps1, 'ps')]
        else:
            newkey = []
            for e in edges:
                shape = self.edges[e]
                self.builder.Add(self.shape, shape)
                newkey1 = objs.addobj(shape, 'cl')
                newkey.append(newkey1)

        self.synchronize_topo_list(action='both')

        return list(objs), newkey

    def CircleOCC_build_geom(self, objs, *args):
        return self.Circle_build_geom(objs, *args)

    def CircleByAxisPoint_build_geom(self, objs, *args):
        pts, pt_on_cl, npts, make_face = args

        pts = [x.strip() for x in pts.split(',')]
        gids_vert = self.get_target1(objs, pts, 'p')

        pt_on_cl = [x.strip() for x in pt_on_cl.split(',')]
        gid = self.get_target1(objs, pt_on_cl, 'p')[0]

        ptx1 = self.get_point_coord(gids_vert[0])
        ptx2 = self.get_point_coord(gids_vert[1])
        ptx3 = self.get_point_coord(gid)

        dirct = ptx2 - ptx1
        dirct = dirct / np.sqrt(np.sum(dirct**2))

        d = ptx3 - ptx1

        center = np.array(ptx1) + np.sum(d * dirct) * dirct
        radius = np.sqrt(np.sum(d**2) - np.sum(d * dirct)**2)

        edges = self.add_circle_by_axis_radius(center,
                                               dirct,
                                               radius,
                                               npts=npts,
                                               pts=ptx3)

        if make_face:
            ll1 = self.add_line_loop(edges)
            ps1 = self.add_plane_surface(ll1)
            shape = self.faces[ps1]
            self.builder.Add(self.shape, shape)
            newkey = objs.addobj(ps1, 'ps')
        else:
            for edge in edges:
                shape = self.edges[edge]
                self.builder.Add(self.shape, shape)
                newkey = objs.addobj(edge, 'cl')

        self.synchronize_topo_list(action='both')
        return list(objs), [newkey]

    def CircleByAxisCenterRadius_build_geom(self, objs, *args):
        ax, pt_on_cl, radius, npts, make_face = args

        pt_on_cl = [x.strip() for x in pt_on_cl.split(',')]
        ax = [x.strip() for x in ax.split(',')]
        radius = [float(x.strip()) for x in radius.split(',')]

        gids_c = self.get_target1(objs, pt_on_cl, 'p')
        if len(ax) == 1:
            gid = self.get_target1(objs, ax, 'l')[0]
            shape = self.edges[gid]
            pnt1 = gp_Pnt()
            pnt2 = gp_Pnt()
            curve, first, last = self.bt.Curve(shape)
            curve.D0(first, pnt1)
            curve.D0(last, pnt2)
            ptx1 = np.array((pnt1.X(), pnt1.Y(), pnt1.Z()))
            ptx2 = np.array((pnt2.X(), pnt2.Y(), pnt2.Z()))
        elif len(ax) == 2:
            gids_vert = self.get_target1(objs, ax, 'p')
            ptx1 = self.get_point_coord(gids_vert[0])
            ptx2 = self.get_point_coord(gids_vert[1])

        dirct = ptx2 - ptx1
        dirct = dirct / np.sqrt(np.sum(dirct**2))

        centers = [self.get_point_coord(gid) for gid in gids_c]
        if len(radius) == 1:
            radius = radius * len(centers)

        edges = [self.add_circle_by_axis_radius(c, dirct, r, npts=npts)
                 for c, r in zip(centers, radius)]
        # print(edges)

        newkeys = []
        for e in edges:
            if make_face:
                ll1 = self.add_line_loop(e)
                ps1 = self.add_plane_surface(ll1)
                shape = self.faces[ps1]
                self.builder.Add(self.shape, shape)
                newkey = objs.addobj(ps1, 'ps')
            else:
                for ee in e:
                    shape = self.edges[ee]
                    self.builder.Add(self.shape, shape)
                    newkey = objs.addobj(e, 'cl')
            newkeys.append(newkey)

        self.synchronize_topo_list(action='both')

        return list(objs), newkeys

    def CircleBy3Points_build_geom(self, objs, *args):
        pts,  make_face = args
        pts = [x.strip() for x in pts.split(',')]
        gids_vertex = self.get_target1(objs, pts, 'p')

        assert len(gids_vertex) == 3, "Need 3 points to define circle"
        edge = self.add_circle_by_3points(*gids_vertex)

        if make_face:
            ll1 = self.add_line_loop([edge])
            ps1 = self.add_plane_surface(ll1)
            shape = self.faces[ps1]
            self.builder.Add(self.shape, shape)
            newkey = objs.addobj(ps1, 'ps')
        else:
            shape = self.edges[edge]
            self.builder.Add(self.shape, shape)
            newkey = objs.addobj(edge, 'cl')

        self.synchronize_topo_list(action='both')
        return list(objs), [newkey]

    def CreateSurface_build_geom(self, objs, *args):
        pts, points, isFilling = args
        pts = [x.strip() for x in pts.split(',')]
        gids_edge = self.get_target1(objs, pts, 'l')

        if isFilling:
            pts = [x.strip() for x in points.split(',') if len(x.strip()) > 0]
            gids_vertex = self.get_target1(objs, pts, 'p')

            face_id = self.add_surface_filling(gids_edge, gids_vertex)
            shape = self.faces[face_id]
            self.builder.Add(self.shape, shape)
            newobj2 = objs.addobj(face_id, 'sf')
            newkeys = [newobj2]
            '''
            gids_new = self.add_plate_surface(gids_edge, gids_vertex)

            newkeys = []
            for gid in gids_new:
                newkeys.append(objs.addobj(gid, 'sf'))
            '''

        else:
            ill = self.add_line_loop(gids_edge)
            ips = self.add_plane_surface(ill)

            shape = self.faces[ips]
            self.builder.Add(self.shape, shape)
            newobj2 = objs.addobj(ips, 'ps')
            newkeys = [newobj2]

        self.synchronize_topo_list(action='both')
        return list(objs), newkeys

    def ThruSection_build_geom(self, objs, *args):
        loop1, loop2, makeSolid, makeRuled, rev1, rev2 = args
        loop1 = [x.strip() for x in loop1.split(',')]
        loop2 = [x.strip() for x in loop2.split(',')]

        gids_loop1 = self.get_target1(objs, loop1, 'l')
        gids_loop2 = self.get_target1(objs, loop2, 'l')

        if rev1:
            gids_loop1 = [self.add_reversed_line(gid) for gid in gids_loop1]

        if rev2:
            gids_loop2 = [self.add_reversed_line(gid) for gid in gids_loop2]

        gid_wire1 = self.add_line_loop(gids_loop1)
        gid_wire2 = self.add_line_loop(gids_loop2)

        gid_new, shape_new = self.add_thrusection(gid_wire1, gid_wire2,
                                                  makeSolid=makeSolid,
                                                  makeRuled=makeRuled)

        self.builder.Add(self.shape, shape_new)

        self.synchronize_topo_list(action='both')

        if makeSolid:
            newobj2 = objs.addobj(gid_new, 'vol')
            newkeys = [newobj2]
        else:
            newkeys = []

        return list(objs), newkeys

    def SurfaceLoop_build_geom(self, objs, *args):
        assert False, "We don't support this"

    # 3D solids
    def CreateVolume_build_geom(self, objs, *args):

        pts = args
        pts = [x.strip() for x in pts[0].split(',')]

        gids = self.get_target1(objs, pts, 'f')
        sl = self.add_surface_loop(gids)
        v1 = self.add_volume(sl)

        shape = self.solids[v1]
        self.builder.Add(self.shape, shape)

        newobj2 = objs.addobj(v1, 'vol')
        self.synchronize_topo_list(action='both', verbose=False)
        return list(objs), [newobj2]

    def CreateShell_build_geom(self, objs, *args):
        vols, openings, thick, conner = args
        vols = [x.strip() for x in vols.split(',')]
        vols = self.get_target1(objs, vols, 'v')

        if len(vols) != 1:
            assert False, "Choose only one volume to make shell"

        solid = self.solids[vols[0]]

        openings = [x.strip() for x in openings.split(',')]
        openings = self.get_target1(objs, openings, 'f')

        LCF = TopTools_ListOfShape()
        for tmp in openings:
            topolist = self.get_topo_list_for_gid(tmp)
            shape = topolist[tmp]
            LCF.Append(shape)
        from OCC.Core.Precision import precision_Confusion
        tol = precision_Confusion()

        SolidMaker = BRepOffsetAPI_MakeThickSolid()

        from OCC.Core.GeomAbs import GeomAbs_Arc, GeomAbs_Intersection
        from OCC.Core.BRepOffset import BRepOffset_Skin

        if conner:
            jointype = GeomAbs_Arc
        else:
            jointype = GeomAbs_Intersection

        SolidMaker.MakeThickSolidByJoin(solid,
                                        LCF,
                                        thick,
                                        tol,
                                        BRepOffset_Skin,
                                        False,
                                        False,
                                        jointype)

        if not SolidMaker.IsDone():
            assert False, "Faile to make shell"

        result = SolidMaker.Shape()

        gids_new = self.register_shapes_balk(result)

        newkeys = []
        for gid in gids_new:
            newkeys.append(objs.addobj(gid, 'sh'))

        self.synchronize_topo_list()
        return list(objs), newkeys

    def CreateOffsetFace_build_geom(self, objs, *args):
        v_l, offsets, conner, fill = args
        v_l = [x.strip() for x in v_l.split(',')]

        gids = self.get_target2(objs, v_l)

        from OCC.Core.GeomAbs import GeomAbs_Arc, GeomAbs_Intersection

        if v_l[0].startswith('f'):
            # first sew the surfaces.
            try:
                sewingMaker = BRepBuilderAPI_Sewing()
                for t in gids:
                    face = self.faces[t]
                    sewingMaker.Add(face)
                sewingMaker.Perform()
                target = sewingMaker.SewedShape()
            except BaseException:
                assert False, "Failed to sew faces"
        elif v_l[0].startswith('v'):
            target = self.solids[gids[0]]
        else:
            assert False, "input must be eihter volume or faces"

        if conner:
            jointype = GeomAbs_Arc
        else:
            jointype = GeomAbs_Intersection

        from OCC.Core.BRepOffset import BRepOffset_Skin
        from OCC.Core.Precision import precision_Confusion
        tol = precision_Confusion()

        OM = BRepOffsetAPI_MakeOffsetShape()

        #OM.PerformBySimple(target, offsets[0])

        intersection = False
        selfinter = False
        removeintedge = False

        results = []
        for offset in offsets:
            OM.PerformByJoin(target, offset, tol, BRepOffset_Skin,
                             intersection,
                             selfinter,
                             jointype,
                             removeintedge,)
            if not OM.IsDone():
                assert False, "Faile to make offset"
            result = OM.Shape()
            shells = [p for p in iter_shape_once(result, 'shell')]

            if fill:
                for shell in shells:
                    solidMaker = BRepBuilderAPI_MakeSolid()
                    solidMaker.Add(shell)
                    result = solidMaker.Solid()
                    if not solidMaker.IsDone():
                        assert False, "Failed to make solid"

                    fixer = ShapeFix_Solid(result)
                    fixer.Perform()

                    assert isinstance(fixer.Solid(), TopoDS_Solid), "wrong TopoDS type is detected"
                    result = fixer.Solid()
                    results.append(result)
            else:
                results.extend(shells)

        newkeys = []
        for r in results:
            gids_new = self.register_shapes_balk(r)
            for gid in gids_new:
                newkeys.append(objs.addobj(gid, 'ofst'))

        return list(objs), newkeys

    def CreateOffset_build_geom(self, objs, *args):
        f_l, offsets, altitudes, conner, close, fill = args

        if len(altitudes) == 1:
            altitudes = np.array([altitudes[0]]*len(offsets))

        f_l = [x.strip() for x in f_l.split(',')]
        gids = self.get_target2(objs, f_l)

        from OCC.Core.GeomAbs import GeomAbs_Arc, GeomAbs_Intersection

        if conner:
            jointype = GeomAbs_Arc
        else:
            jointype = GeomAbs_Intersection

        mode = ''
        if f_l[0].startswith('l'):
            mode = 'wire'
            wireMaker = BRepBuilderAPI_MakeWire()
            for e in gids:
                wireMaker.Add(self.edges[e])
            wireMaker.Build()
            if not wireMaker.IsDone():
                assert False, "Failed to make wire"
            target = wireMaker.Wire()
        elif f_l[0].startswith('f'):
            mode = 'face'
            target = self.faces[gids[0]]

        else:
            assert False, "input must be edges or face"

        isopen = not close
        if fill:
            isopen = False
        OM = BRepOffsetAPI_MakeOffset(target, jointype, isopen)

        results = []

        for offset, altitude in zip(offsets, altitudes):
            # print(offset)
            OM.Perform(float(offset), float(altitude))
            if not OM.IsDone():
                assert False, "Faile to make offset"

            result = OM.Shape()
            results.append(result)

            wires = [p for p in iter_shape_once(result, 'wire')]

            if fill:
                for wire in wires:
                    faceMaker = BRepBuilderAPI_MakeFace(wire)
                    faceMaker.Build()

                    if not faceMaker.IsDone():
                        assert False, "can not create face"

                    face = faceMaker.Face()
                    fixer = ShapeFix_Face(face)
                    fixer.Perform()
                    face = fixer.Face()
                    results.append(face)
            else:
                results.extend(wires)

        newkeys = []
        for r in results:
            gids_new = self.register_shapes_balk(r)
            for gid in gids_new:
                newkeys.append(objs.addobj(gid, 'ofst'))

        return list(objs), newkeys

    def CreateProjection_build_geom(self, objs, *args):
        targets, plane_params, offset, fill = args
        targets = [x.strip() for x in targets.split(',')]
        targets = self.get_target2(objs, targets)

        cptx, normal = self.process_plane_parameters(plane_params, objs)
        if offset != 0:
            cptx = cptx + normal * offset

        # splitter alogrithm
        normal = -normal

        result = self.project_shape_on_wp(targets, cptx, normal, fill=fill)

        newkeys = []
        gids_new = self.register_shapes_balk(result)

        for gid in gids_new:
            newkeys.append(objs.addobj(gid, 'prj'))

        return list(objs), newkeys

    def Box_build_geom(self, objs, *args):
        c1, e1, e2, e3 = args
        lcar = 0.0
        c1 = np.array(c1)
        e1 = np.array(e1)
        e2 = np.array(e2)
        p1 = c1
        p2 = c1 + e1
        p3 = c1 + e2
        p4 = c1 + e3
        p5 = c1 + e1 + e2
        p6 = c1 + e2 + e3
        p7 = c1 + e3 + e1
        p8 = c1 + e3 + e2 + e1

        v1 = self.add_box((p1, p2, p3, p4, p5, p6, p7, p8,))
        shape = self.solids[v1]
        self.builder.Add(self.shape, shape)

        newkey = objs.addobj(v1, 'bx')
        return list(objs), [newkey]

    def Ball_build_geom(self, objs, *args):

        x0, l1, l2, l3, a1, a2, a3 = args
        radii = [l1, l2, l3]
        rr = min(radii)

        gids_new = self.add_sphere(
            x0,
            rr,
            a1 / 180 * np.pi,
            a2 / 180 * np.pi,
            a3 / 180 * np.pi)
        newkeys = []

        ss = (l1 / rr, l2 / rr, l3 / rr)
        if ss[0] != ss[1] or ss[1] != ss[2]:
            gids_new = [self.dilate(gids_new[0], x0, ss, copy=False)]

        for gid_new in gids_new:
            newkeys.append(objs.addobj(gid_new, 'bl'))

        self.synchronize_topo_list(action='both')

        return list(objs), newkeys

    def Cone_build_geom(self, objs, *args):
        x0, d0, r1, r2, angle = args

        gids_new = self.add_cone(x0[0], x0[1], x0[2], d0[0], d0[1], d0[2],
                                 r1, r2, angle / 180 * np.pi)

        newkeys = []
        for gid in gids_new:
            newkeys.append(objs.addobj(gid, 'cn'))

        return list(objs), newkeys

    def Cylinder_build_geom(self, objs, *args):
        x0, d0, r1, angle = args

        gids_new = self.add_cylinder(x0, d0, r1, angle / 180 * np.pi)

        newkeys = []
        for gid in gids_new:
            newkeys.append(objs.addobj(gid, 'cyl'))

        return list(objs), newkeys

    def Wedge_build_geom(self, objs, *args):
        x0, d0, ltx = args
        gids_new = self.add_wedge(x0, d0, ltx)

        newkeys = []
        for gid in gids_new:
            newkeys.append(objs.addobj(gid, 'wg'))

        return list(objs), newkeys

    def Torus_build_geom(self, objs, *args):
        x0, r1, r2, angle, keep_interior = args

        gids_new = self.add_torus(x0, r1, r2, angle * np.pi / 180)

        newkeys = []
        for gid in gids_new:
            newkeys.append(objs.addobj(gid, 'trs'))

        return list(objs), newkeys

    # prutrusions
    def Extrude_build_geom(self, objs, *args):
        targets, tax, lengths = args

        targets = [x.strip() for x in targets.split(',')]
        gids = self.get_target2(objs, targets)

        offset = 0
        if lengths[0] < 0:
            offset = lengths[0]
            lengths = lengths[1:]
        trans = []
        if tax[0] == 'normal':
            for length in lengths:
                trans2 = []
                for gid in gids:
                    if tax[1] == '':
                        assert isinstance(
                            gid, SurfaceID), "target must be surface"
                        n1, p0 = self.get_face_normal(gid, check_flat=False)
                    else:
                        wp = objs['wp' + tax[1]]
                        n1 = wp.get_norm()
                        p0 = wp.get_center()
                    if tax[2]:
                        tt = -n1 * length
                    else:
                        tt = n1 * length
                    trans2.append(trans_delta(tt))
                trans.append(trans2)

        elif tax[0] == 'normalp':
            assert len(lengths) == 1, "length should have one element"
            if tax[2] == '':
                assert isinstance(gids[0], SurfaceID), "target must be surface"
                n1, p0 = self.get_face_normal(gids[0], check_flat=False)
            else:
                wp = objs['wp' + tax[2]]
                n1 = wp.get_norm()
                p0 = wp.get_center()

            dests = [x.strip() for x in tax[1].split(',')]
            gid_dests = self.get_target1(objs, dests, 'p')
            length = lengths[0]
            for gid_dest in gid_dests:
                p1 = self.get_point_coord(gid_dest)
                if tax[3]:
                    tt = -n1 * np.sum((p1 - p0) * n1) * length
                else:
                    tt = n1 * np.sum((p1 - p0) * n1) * length
                trans.append(trans_delta(tt))

        elif tax[0] == 'fromto_points':
            dests1 = [x.strip() for x in tax[1].split(',')]
            dests2 = [x.strip() for x in tax[2].split(',')]

            gid_dests1 = self.get_target1(objs, dests1, 'p')
            gid_dests2 = self.get_target1(objs, dests2, 'p')

            assert len(gid_dests1) == 1, "Incorrect destination setting"
            assert len(gid_dests2) == 1, "Incorrect destination setting"

            p1 = self.get_point_coord(gid_dests1[0])
            p2 = self.get_point_coord(gid_dests2[0])

            n1 = p2 - p1
            if not tax[3]:
                n1 /= np.sqrt(np.sum(n1**2))
            if tax[4]:
                n1 *= -1

            for length in lengths:
                trans.append(trans_delta(length * n1))
        elif tax[0] == 'radial':
            axis = np.array(eval(tax[1]))
            axis = axis / np.sqrt(np.sum(axis**2))
            point_on_axis = np.array(eval(tax[2]))

            for length in lengths:
                trans2 = []
                for gid in gids:
                    if isinstance(gid, SurfaceID):
                        n1, p0 = self.get_face_normal(gid, check_flat=False)
                    elif isinstance(gid, LineID):
                        p0 = self.get_line_center(gid)
                    elif isinstance(gid, VertexID):
                        p0 = self.get_point_coord(gid)
                    else:
                        assert False, "unsupported input (polar extrude)"

                    n1 = p0 - axis * np.sum((p0 - point_on_axis) * axis)
                    n1 = n1 / np.sqrt(np.sum(n1**2))
                    if tax[3]:
                        tt = -n1 * length
                    else:
                        tt = n1 * length
                    trans2.append(trans_delta(tt))
                trans.append(trans2)
        elif tax[0] == 'polar':
            center = np.array(eval(tax[1]))
            for length in lengths:
                trans2 = []
                for gid in gids:
                    if isinstance(gid, SurfaceID):
                        n1, p0 = self.get_face_normal(gid, check_flat=False)
                    elif isinstance(gid, LineID):
                        p0 = self.get_line_center(gid)
                    elif isinstance(gid, VertexID):
                        p0 = self.get_point_coord(gid)
                    else:
                        assert False, "unsupported input (polar extrude)"
                    n1 = p0 - center
                    n1 = n1 / np.sqrt(np.sum(n1**2))
                    if tax[2]:
                        tt = -n1 * length
                    else:
                        tt = n1 * length
                    trans2.append(trans_delta(tt))
                trans.append(trans2)

        else:
            tax = np.array(tax).flatten()
            tax = tax / np.sqrt(np.sum(np.array(tax)**2))
            for length in lengths:
                trans.append(trans_delta(length * tax))
        newkeys = []

        if offset != 0:
            tt0 = trans[0]

            new_gids = []
            for kk, gid in enumerate(gids):
                if not isinstance(tt, trans_delta):
                    tt = tt0[kk]
                else:
                    tt = tt0
                tt = np.array(tt)
                tt = tt * offset / np.sqrt(np.sum(tt**2))

                new_gids.append(self.translate(gid, tt, copy=False))
            gids = new_gids

        for tt in trans:
            new_shapes = self.extrude(gids, translation=tt)

            gids = []
            for t, ret in zip(targets, new_shapes):
                gid_last, gid_extruded, shape = ret
                newkeys.append(objs.addobj(gid_last, t))
                newkeys.append(objs.addobj(gid_extruded, 'ex'))
                self.builder.Add(self.shape, shape)

                gids.append(gid_last)

        self.synchronize_topo_list(action='add')
        return list(objs), newkeys

    def Revolve_build_geom(self, objs, *args):

        targets, params, angles = args

        if params[0] == 'xyz':
            rax = [float(x) for x in params[1]]
            pax = [float(x) for x in params[2]]
        elif params[0] == 'fromto_points':
            param1 = [x.strip() for x in params[1].split(',')]
            param2 = [x.strip() for x in params[2].split(',')]
            gid1 = self.get_target1(objs, param1, 'p')[0]
            gid2 = self.get_target1(objs, param2, 'p')[0]

            p1 = self.get_point_coord(gid1)
            p2 = self.get_point_coord(gid2)
            rax = p2 - p1
            pax = p1
        elif params[0] == 'normalp':
            param1 = [x.strip() for x in params[1].split(',')]
            param2 = [x.strip() for x in params[2].split(',')]
            gid1 = self.get_target1(objs, param1, 'f')[0]
            gid2 = self.get_target1(objs, param2, 'p')[0]

            n1 = self.get_face_normal(gid1)
            p1 = self.get_point_coord(gid2)

            rax = n1
            pax = p1
        elif params[0] == 'edgep':
            param1 = [x.strip() for x in params[1].split(',')]
            param2 = [x.strip() for x in params[2].split(',')]
            gid1 = self.get_target1(objs, param1, 'l')[0]
            gid2 = self.get_target1(objs, param2, 'p')[0]

            n1 = self.get_line_direction(gid1)
            p1 = self.get_point_coord(gid2)

            rax = n1
            pax = p1
        else:
            assert False, "Unknonw parameter" + str(params)

        targets = [x.strip() for x in targets.split(',')]
        gids = self.get_target2(objs, targets)

        newkeys = []

        for angle in angles:
            rot = (pax, rax, angle * np.pi / 180)
            new_shapes = self.extrude(gids, rotation=rot)

            gids = []
            for t, ret in zip(targets, new_shapes):
                gid_last, gid_extruded, shape = ret
                newkeys.append(objs.addobj(gid_last, t))
                newkeys.append(objs.addobj(gid_extruded, 'ex'))
                self.builder.Add(self.shape, shape)

                gids.append(gid_last)

        self.synchronize_topo_list(action='add')
        return list(objs), newkeys

    def Sweep_build_geom(self, objs, *args):
        targets, lines = args
        targets = [x.strip() for x in targets.split(',')]

        gids = self.get_target2(objs, targets)

        lines = [x.strip() for x in lines.split(',')]
        gid_lines = self.get_target1(objs, lines, 'l')

        wire_id = self.add_line_loop(gid_lines)
        wire = self.wires[wire_id]

        new_shapes = self.extrude(gids, wire=wire)
        newkeys = []

        for t, ret in zip(targets, new_shapes):
            gid_last, gid_extruded, shape = ret
            newkeys.append(objs.addobj(gid_last, t))
            newkeys.append(objs.addobj(gid_extruded, 'swp'))
            self.builder.Add(self.shape, shape)

        self.synchronize_topo_list(action='add')

        return list(objs), newkeys

    # translation
    def Move_build_geom(self, objs, *args):
        targets, dx, dy, dz, keep = args
        targets = [x.strip() for x in targets.split(',')]

        newkeys = []
        gids = self.get_target2(objs, targets)

        delta = (dx, dy, dz)

        for gid in gids:
            new_gid = self.translate(gid, delta, copy=keep)
            if new_gid is not None:
                newkeys.append(objs.addobj(new_gid, 'mv'))

        self.synchronize_topo_list(action='both')

        return list(objs), newkeys

    def MoveByPoints_build_geom(self, objs, *args):
        targets, point1, point2, dist, scale_d, keep = args

        targets = [x.strip() for x in targets.split(',')]
        gids = self.get_target2(objs, targets)

        point1 = [x.strip() for x in point1.split(',')]
        point2 = [x.strip() for x in point2.split(',')]
        gids_1 = self.get_target1(objs, point1, 'p')[0]
        gids_2 = self.get_target1(objs, point2, 'p')[0]

        p1 = self.get_point_coord(gids_1)
        p2 = self.get_point_coord(gids_2)

        d = p2 - p1
        if scale_d:
            d = d * dist
        else:
            d = d / np.sqrt(np.sum(d**2)) * dist
        dx, dy, dz = d

        newkeys = []
        delta = (dx, dy, dz)

        for gid in gids:
            new_gid = self.translate(gid, delta, copy=keep)
            if new_gid is not None:
                newkeys.append(objs.addobj(new_gid, 'mv'))

        self.synchronize_topo_list(action='both')

        return list(objs), newkeys

    def Rotate_build_geom(self, objs, *args):
        targets, point_on_axis, axis_dir, angle, keep = args

        newkeys = []
        targets = [x.strip() for x in targets.split(',')]

        gids = self.get_target2(objs, targets)

        '''
        for gid in gids:
            new_gid = self.rotate(gid, point_on_axis, axis_dir,
                                  np.pi * angle / 180., copy=keep)
            if new_gid is not None:
                newkeys.append(objs.addobj(new_gid, 'mv'))
        '''
        new_gids = self.rotate(gids, point_on_axis, axis_dir,
                               np.pi * angle / 180., copy=keep)
        for new_gid in new_gids:
            newkeys.append(objs.addobj(new_gid, 'mv'))

        self.synchronize_topo_list(action='both')

        return list(objs), newkeys

    def RotateCenterPoints_build_geom(self, objs, *args):
        targets, center, points, use_sup, keep = args

        targets = [x.strip() for x in targets.split(',')]
        gids = self.get_target2(objs, targets)

        center = [x.strip() for x in center.split(',')]
        points = [x.strip() for x in points.split(',')]
        gids_1 = self.get_target1(objs, center, 'p')
        gids_2 = self.get_target1(objs, points, 'p')

        c1 = self.get_point_coord(gids_1[0])

        p1 = self.get_point_coord(gids_2[0])
        p2 = self.get_point_coord(gids_2[1])

        d1 = p1 - c1
        d2 = p2 - c1

        arm1 = d1 / np.sqrt(np.sum(d1**2))
        arm2 = d2 / np.sqrt(np.sum(d2**2))

        dirct = np.cross(arm1, arm2)
        d2 = np.cross(dirct, arm1)
        d2 = d2 / np.sqrt(np.sum(d2**2))
        yy = np.sum(arm2 * d2)
        xx = np.sum(arm2 * arm1)

        angle = np.arctan2(yy, xx)

        if use_sup:
            angle = angle - np.pi

        newkeys = []
        for gid in gids:
            new_gid = self.rotate(gid, c1, dirct, angle, copy=keep)
            if new_gid is not None:
                newkeys.append(objs.addobj(new_gid, 'mv'))

        self.synchronize_topo_list(action='both')
        return list(objs), newkeys

    def Scale_build_geom(self, objs, *args):
        targets, cc, ss, keep = args
        targets = [x.strip() for x in targets.split(',')]

        newkeys = []
        gids = self.get_target2(objs, targets)

        new_gids = self.dilate(gids, cc, ss, copy=keep)
        for new_gid in new_gids:
            newkeys.append(objs.addobj(new_gid, 'sc'))
        self.synchronize_topo_list(action='both')

        return list(objs), newkeys

    def Flip_build_geom(self, objs, *args):
        targets, plane_param, keep = args

        newkeys = []
        targets = [x.strip() for x in targets.split(',')]

        gids = self.get_target2(objs, targets)

        cptx, normal = self.process_plane_parameters(plane_param, objs, gids)

        d = np.sum(normal * cptx)
        abcd = (normal[0], normal[1], normal[2], d)

        for gid in gids:
            new_gid = self.symmetrize(gid, abcd, copy=keep)
            if new_gid is not None:
                newkeys.append(objs.addobj(new_gid, 'mv'))

        self.synchronize_topo_list(action='both')

        return list(objs), newkeys

    def Array_build_geom(self, objs, *args):
        targets, count, displacement = args
        dx, dy, dz = displacement
        targets = [x.strip() for x in targets.split(',')]

        newkeys = []
        gids = self.get_target2(objs, targets)

        i = 1
        while i < count:
            delta = (dx * i, dy * i, dz * i)

            for gid in gids:
                new_gid = self.translate(gid, delta, True)
                if new_gid is not None:
                    newkeys.append(objs.addobj(new_gid, 'cp'))
            i = i + 1

        self.synchronize_topo_list(action='add')

        return list(objs), newkeys

    def ArrayByPoints_build_geom(self, objs, *args):
        targets, count, ref_ptx = args
        targets = [x.strip() for x in targets.split(',')]
        ref_ptx = [x.strip() for x in ref_ptx.split(',')]

        newkeys = []
        gids = self.get_target2(objs, targets)
        gids_ref = self.get_target1(objs, ref_ptx, 'p')

        delta_arr = []
        if count == 1:
            count = len(gids_ref) - 1
            for i in range(count):
                delta = (self.get_point_coord(gids_ref[i + 1]) -
                         self.get_point_coord(gids_ref[0]))
                delta_arr.append(delta)
        else:
            delta = (self.get_point_coord(gids_ref[1]) -
                     self.get_point_coord(gids_ref[0]))
            i = 1
            while i < count:
                delta_arr.append(delta * i)
                i = i + 1

        for delta in delta_arr:
            for gid in gids:
                new_gid = self.translate(gid, delta, True)
                if new_gid is not None:
                    newkeys.append(objs.addobj(new_gid, 'cp'))

        self.synchronize_topo_list(action='add')

        return list(objs), newkeys

    def ArrayRot_build_geom(self, objs, *args):
        targets, count, point_on_axis, axis_dir, angle = args

        newkeys = []

        targets = [x.strip() for x in targets.split(',')]
        gids = self.get_target2(objs, targets)

        i = 1
        while i < count:
            angle1 = angle * i

            for gid in gids:
                new_gid = self.rotate(gid, point_on_axis, axis_dir,
                                      np.pi * angle1 / 180., True)

                if new_gid is not None:
                    newkeys.append(objs.addobj(new_gid, 'cp'))
            i = i + 1

        self.synchronize_topo_list(action='add')

        return list(objs), newkeys

    def ArrayRotByPoints_build_geom(self, objs, *args):
        targets, count, point_on_axis, axis_dir, ref_ptx = args

        targets = [x.strip() for x in targets.split(',')]
        ref_ptx = [x.strip() for x in ref_ptx.split(',')]

        newkeys = []
        gids = self.get_target2(objs, targets)
        gids_ref = self.get_target1(objs, ref_ptx, 'p')

        def get_angle(p1, p2):
            dirct = np.array(axis_dir)
            dirct = dirct / np.sqrt(np.sum(dirct**2))
            d = p1 - np.array(point_on_axis)
            arm1 = p1 - (np.array(point_on_axis) + np.sum(d * dirct) * dirct)
            d = p2 - np.array(point_on_axis)
            arm2 = p2 - (np.array(point_on_axis) + np.sum(d * dirct) * dirct)
            d2 = np.cross(dirct, arm1)
            d2 = d2 / np.sqrt(np.sum(d2**2))
            arm1 = arm1 / np.sqrt(np.sum(arm1**2))
            arm2 = arm2 / np.sqrt(np.sum(arm2**2))
            yy = np.sum(arm2 * d2)
            xx = np.sum(arm2 * arm1)
            return np.arctan2(yy, xx)

        angle_arr = []
        if count == 1:
            count = len(gids_ref) - 1
            for i in range(count):
                angle = get_angle(self.get_point_coord(gids_ref[0]),
                                  self.get_point_coord(gids_ref[i + 1]))
                angle_arr.append(angle)
        else:
            angle = get_angle(self.get_point_coord(gids_ref[0]),
                              self.get_point_coord(gids_ref[1]))
            #print("angle", angle * 180 / np.pi)
            i = 1
            while i < count:
                angle_arr.append(angle * i)
                i = i + 1

        for angle1 in angle_arr:
            for gid in gids:
                new_gid = self.rotate(gid, point_on_axis, axis_dir,
                                      angle1, True)

                if new_gid is not None:
                    newkeys.append(objs.addobj(new_gid, 'cp'))

        self.synchronize_topo_list(action='add')

        return list(objs), newkeys

    def ArrayPath_build_geom(self, objs, *args):
        newkeys = []
        targets, rpnt, lines, count, margin1, margin2, ignore_rot = args
        targets = [x.strip() for x in targets.split(',')]
        rpnt = [x.strip() for x in rpnt.split(',')]
        lines = [x.strip() for x in lines.split(',')]

        gids = self.get_target2(objs, targets)
        gids_l = self.get_target1(objs, lines, 'l')
        gid_p = self.get_target1(objs, rpnt, 'p')[0]

        # t (length along the path)
        lines = [self.edges[gid] for gid in gids_l]
        l = [measure_edge_length(x) for x in lines]

        intvls = np.array([margin1] + [1] * (count - 1) + [margin2])

        t = np.cumsum(intvls)
        t = (t / t[-1])[:-1] * np.sum(l)

        if len(lines) > 1:
            curve, first, last = self.bt.Curve(lines[0])
            pnt = curve.Value(first)
            p11 = np.array((pnt.X(), pnt.Y(), pnt.Z()))
            pnt = curve.Value(last)
            p12 = np.array((pnt.X(), pnt.Y(), pnt.Z()))

            curve, first, last = self.bt.Curve(lines[1])
            pnt = curve.Value(first)
            p21 = np.array((pnt.X(), pnt.Y(), pnt.Z()))
            pnt = curve.Value(last)
            p22 = np.array((pnt.X(), pnt.Y(), pnt.Z()))

            d1 = np.min((np.sum((p11 - p21)**2),
                         np.sum((p11 - p22)**2)))
            d2 = np.min((np.sum((p12 - p21)**2),
                         np.sum((p12 - p22)**2)))
            if d2 < d1:
                flip = False
                endpoint = p12
            else:
                flip = True
                endpoint = p11
        else:
            curve, _first, last = self.bt.Curve(lines[0])
            pnt = curve.Value(last)
            endpoint = np.array((pnt.X(), pnt.Y(), pnt.Z()))

        curve, _first, last = self.bt.Curve(lines[0])
        pnt = curve.Value(last)
        endpoint = np.array((pnt.X(), pnt.Y(), pnt.Z()))

        k = 0
        sols = []
        flip = False

        while len(t) > 0:
            line = lines[k]

            if k > 0:
                curve, first, last = self.bt.Curve(lines[0])
                pnt = curve.Value(last)
                p1 = np.array((pnt.X(), pnt.Y(), pnt.Z()))
                pnt = curve.Value(last)
                p2 = np.array((pnt.X(), pnt.Y(), pnt.Z()))
                if (np.sum((p1 - endpoint)**2) >
                        np.sum((p2 - endpoint)**2)):
                    flip = True
                    endpoint = p1
                else:
                    endpoint = p2

            ufit = find_point_on_curve(line, t, tol=1e-4, flip=flip)
            sols.append([line, ufit])
            t = t[len(ufit):] - measure_edge_length(line)
            if len(t) == 0:
                break
            k = k + 1

        transforms = []
        gpnt = gp_Pnt()
        gvec = gp_Vec()
        for sol in sols:
            l, ufit = sol
            curve, _first, _last = self.bt.Curve(l)
            for u in ufit:
                curve.D1(u, gpnt, gvec)
                p = np.array((gpnt.X(), gpnt.Y(), gpnt.Z()))
                v = np.array((gvec.X(), gvec.Y(), gvec.Z()))
                transforms.append((p, v))

        p0 = self.get_point_coord(gid_p)
        gids_new = []
        for i, trans in enumerate(transforms):
            if i == 0:
                v0 = trans[1]
                v0 = v0 / np.linalg.norm(v0)

            p1, v1 = trans
            v1 = v1 / np.linalg.norm(v0)
            tt = p1 - p0

            v2 = np.cross(v0, v1)

            if ignore_rot or np.sum(v2**2) < 1e-10:
                gids_new.extend(self.translate(gids, tt, copy=True))

            else:
                poa = p1
                cs = np.sum(v1 * v0)
                ss = np.linalg.norm(v2)
                v2 = v2 / ss
                ang = np.arctan2(ss, cs)

                gids_new.extend(self.translate_rot(gids, tt, poa,
                                                   v2, ang, copy=True))

        # print(gids_new)
        for gid in gids_new:
            newkeys.append(objs.addobj(gid, 'arr'))

        self.synchronize_topo_list()
        return list(objs), newkeys

    # fillet/chamfer
    def Fillet_build_geom(self, objs, *args):

        volumes, curves, radii = args
        volumes = [x.strip() for x in volumes.split(',')]
        curves = [x.strip() for x in curves.split(',')]

        gid_vols = self.get_target1(objs, volumes, 'v')
        gid_curves = self.get_target1(objs, curves, 'l')

        gids_new = self.fillet(gid_vols, gid_curves, radii)

        newkeys = []
        for gid in gids_new:
            newkeys.append(objs.addobj(gid, 'vol'))

        self.synchronize_topo_list()
        return list(objs), newkeys

    def Chamfer_build_geom(self, objs, *args):
        volumes, curves, distances, surfaces = args

        volumes = [x.strip() for x in volumes.split(',')]
        curves = [x.strip() for x in curves.split(',')]
        surfaces = [x.strip() for x in surfaces.split(',')]

        gid_vols = self.get_target1(objs, volumes, 'v')
        gid_curves = self.get_target1(objs, curves, 'l')
        gid_faces = self.get_target1(objs, surfaces, 'f')

        gids_new = self.chamfer(gid_vols, gid_curves, gid_faces, distances)

        newkeys = []
        for gid in gids_new:
            newkeys.append(objs.addobj(gid, 'vol'))

        self.synchronize_topo_list()
        return list(objs), newkeys

    def Fillet2D_build_geom(self, objs, *args):

        faces, corners, radius = args
        faces = [x.strip() for x in faces.split(',')]
        corners = [x.strip() for x in corners.split(',')]

        gid_face = self.get_target1(objs, faces, 'f')[0]
        gid_corners = self.get_target1(objs, corners, 'p')

        gids_new = self.fillet2d(gid_face, gid_corners, float(radius))

        newkeys = []
        for gid in gids_new:
            newkeys.append(objs.addobj(gid, 'ps'))

        self.synchronize_topo_list()
        return list(objs), newkeys

    def Chamfer2D_build_geom(self, objs, *args):

        faces, edges, e1, e2 = args
        if len(e2.strip()) == 0:
            e2 = e1
        else:
            e2 = float(e1)
        faces = [x.strip() for x in faces.split(',')]
        edges = [x.strip() for x in edges.split(',')]

        gid_face = self.get_target1(objs, faces, 'f')[0]
        gid_edges = self.get_target1(objs, edges, 'l')

        # print(gid_edges)

        gids_new = self.chamfer2d(gid_face, gid_edges, e1, e2)

        newkeys = []
        for gid in gids_new:
            newkeys.append(objs.addobj(gid, 'ps'))

        self.synchronize_topo_list()
        return list(objs), newkeys

    # copy/remove

    def Copy_build_geom(self, objs, *args):
        targets = args[0]
        targets = [x.strip() for x in targets.split(',')]

        newkeys = []
        gids = self.get_target2(objs, targets)

        for gid in gids:
            copied_gid = self.copy(gid)
            newkeys.append(objs.addobj(copied_gid, 'cp'))

        self.synchronize_topo_list(action='add')
        return list(objs), newkeys

    def Remove_build_geom(self, objs, *args):
        targets, recursive = args
        targets = [x.strip() for x in targets.split(',')]

        newkeys = []
        gids = self.get_target2(objs, targets)
        if len(gids) == 0:
            assert False, "empty imput objects: " + ','.join(targets)

        for gid in gids:
            self.remove(gid, recursive=recursive)
        self.synchronize_topo_list(action='both')

        for t in targets:
            if t in objs:
                del objs[t]

        return list(objs), newkeys

    def Remove2_build_geom(self, objs, *args):
        targets = args[0]
        targets = [x.strip() for x in targets.split(',')]

        newkeys = []
        gids = self.get_target2(objs, targets)

        ret = self.inverse_remove(gids)
        self.synchronize_topo_list()

        for t in list(objs):
            del objs[t]
        for rr in ret:
            newkeys.append(objs.addobj(rr, 'kpt'))

        return list(objs), newkeys

    def RemoveFaces_build_geom(self, objs, *args):
        targets, faces = args

        targets = [x.strip() for x in targets.split(',')]
        faces = [x.strip() for x in faces.split(',')]

        if len(targets) != 1:
            assert False, "Chose one volume"

        gid = self.get_target1(objs, targets, 'v')[0]
        gids_face = self.get_target1(objs, faces, 'f')

        gids_new = self.defeature(gid, gids_face)

        newkeys = []
        for gid in gids_new:
            newkeys.append(objs.addobj(gid, 'dftr'))

        self.synchronize_topo_list()
        return list(objs), newkeys

    def _Union_build_geom(self, objs, *args, **kwargs):
        #print("args here", args)
        tp, tm, delete_input, delete_tool, keep_highest, do_upgrade, tol_scale = args

        tp = [x.strip() for x in tp.split(',')]
        tm = [x.strip() for x in tm.split(',')]

        gid_objs = self.get_target2(objs, tp)
        gid_tools = self.get_target2(objs, tm)

        if (all([isinstance(x, SurfaceID) for x in gid_objs]) and
            all([isinstance(x, SurfaceID) for x in gid_tools]) and
                do_upgrade):
            print("atttempting face orientation fix")
            self.apply_fixshape_shell(gid_objs + gid_tools)

        gids_new = self.union(gid_objs, gid_tools,
                              remove_obj=delete_input,
                              remove_tool=delete_tool,
                              keep_highest=keep_highest,
                              upgrade=do_upgrade,
                              tol_scale=tol_scale)
        newkeys = []
        for gid in gids_new:
            newkeys.append(objs.addobj(gid, 'uni'))

        self.synchronize_topo_list()

        if delete_input:
            for x in tp:
                if x in objs:
                    del objs[x]
        if delete_tool:
            for x in tm:
                if x in objs:
                    del objs[x]

        return list(objs), newkeys

    def Union_build_geom(self, objs, *args):
        return self._Union_build_geom(objs, *args)

    def Union2_build_geom(self, objs, *args):
        kwargs = {"upgrade": True}
        return self._Union_build_geom(objs, *args, **kwargs)

    def Difference_build_geom(self, objs, *args):
        tp, tm, delete_input, delete_tool, keep_highest, do_upgrade, tol_scale = args
        tp = [x.strip() for x in tp.split(',')]
        tm = [x.strip() for x in tm.split(',')]

        gid_objs = self.get_target2(objs, tp)
        gid_tools = self.get_target2(objs, tm)

        gids_new = self.difference(gid_objs, gid_tools,
                                   remove_obj=delete_input,
                                   remove_tool=delete_tool,
                                   keep_highest=keep_highest,
                                   upgrade=do_upgrade,
                                   tol_scale=tol_scale)

        newkeys = []
        for gid in gids_new:
            #topolist = self.get_topo_list_for_gid(gid)
            #shape = topolist[gid]
            newkeys.append(objs.addobj(gid, 'diff'))

        self.synchronize_topo_list()
        return list(objs), newkeys

    def Intersection_build_geom(self, objs, *args):
        tp, tm, delete_input, delete_tool, keep_highest, do_upgrade, tol_scale = args
        tp = [x.strip() for x in tp.split(',')]
        tm = [x.strip() for x in tm.split(',')]

        gid_objs = self.get_target2(objs, tp)
        gid_tools = self.get_target2(objs, tm)

        gids_new = self.intersection(gid_objs, gid_tools,
                                     remove_obj=delete_input,
                                     remove_tool=delete_tool,
                                     keep_highest=keep_highest,
                                     upgrade=do_upgrade,
                                     tol_scale=tol_scale)
        newkeys = []
        for gid in gids_new:
            #topolist = self.get_topo_list_for_gid(gid)
            #shape = topolist[gid]
            newkeys.append(objs.addobj(gid, 'diff'))

        self.synchronize_topo_list()
        return list(objs), newkeys

    def Fragments_build_geom(self, objs, *args):
        tp, tm, delete_input, delete_tool, keep_highest, tol_scale = args
        tp = [x.strip() for x in tp.split(',')]
        tm = [x.strip() for x in tm.split(',')]

        gid_objs = self.get_target2(objs, tp)
        gid_tools = self.get_target2(objs, tm)

        gids_new = self.fragments(gid_objs, gid_tools,
                                  remove_obj=delete_input,
                                  remove_tool=delete_tool,
                                  keep_highest=keep_highest,
                                  tol_scale=tol_scale)
        newkeys = []
        for gid in gids_new:
            #topolist = self.get_topo_list_for_gid(gid)
            #shape = topolist[gid]
            newkeys.append(objs.addobj(gid, 'diff'))

        self.synchronize_topo_list()
        return list(objs), newkeys

    '''
    2D elements
    '''

    def Point2D_build_geom(self, objs, *args):
        xarr, yarr = args
        xarr = np.atleast_1d(xarr)
        yarr = np.atleast_1d(yarr)
        zarr = xarr * 0.0
        try:
            pos = np.vstack((xarr, yarr, zarr)).transpose()
        except BaseException:
            assert False, "can not make proper input array"

        PTs = [self.add_point(p) for p in pos]

        newobjs = []
        for p in PTs:
            shape = self.vertices[p]
            self.builder.Add(self.shape, shape)
            newkey = objs.addobj(p, 'pt')
            newobjs.append(newkey)

        return list(objs), newobjs

    # Define 2D version the same as 3D
    Line2D_build_geom = Line_build_geom

    def NormalLine2D_build_geom(self, objs, *args):
        lines, u_n, length, reverse = args
        lines = [x.strip() for x in lines.split(',')]
        gids = self.get_target1(objs, lines, 'l')

        newobjs = []
        for gid in gids:
            n = self.get_line_direction(gid, u_n=u_n)
            p0 = self.get_line_point(gid, u_n)
            if reverse:
                n = -n
            # on 2D normal is always (0,0,1)
            n = np.cross(n, (0, 0, 1))

            if len(length) == 1:
                p1 = p0
                p2 = p0 + length[0]*n
            elif len(length) == 2:
                p1 = p0 + length[0]*n
                p2 = p0 + length[1]*n
            else:
                assert False, "Length should have either one or two elements"
            p1 = self.add_point(p1)
            p2 = self.add_point(p2)
            ln = self.add_line(p1, p2)
            shape = self.edges[ln]
            self.builder.Add(self.shape, shape)
            newobjs.append(objs.addobj(ln, 'ln'))

        return list(objs), newobjs

    def Circle2D_build_geom(self, objs, *args):
        center, ax1, ax2, radius = args

        assert radius > 0, "Radius must be > 0"

        a1 = np.array(ax1 + [0])
        a2 = np.array(ax2 + [0])
        a2 = np.cross(np.cross(a1, a2), a1)
        a1 = a1 / np.sqrt(np.sum(a1**2)) * radius
        a2 = a2 / np.sqrt(np.sum(a2**2)) * radius

        c = np.array(center + [0])
        p1 = self.add_point(c + a1)
        p2 = self.add_point(c + a2)
        p3 = self.add_point(c - a1)
        p4 = self.add_point(c - a2)
        ca1 = self.add_circle_arc(p1, p2, p3)
        ca2 = self.add_circle_arc(p3, p4, p1)
        ll1 = self.add_line_loop([ca1, ca2])

        ps1 = self.add_plane_surface(ll1)

        shape = self.faces[ps1]
        self.builder.Add(self.shape, shape)

        self.synchronize_topo_list(action='both')
        newkey = objs.addobj(ps1, 'ps')

        return list(objs), [newkey]

    def Circle2DRadiusTwoTangentCurve_build_geom(self, objs, *args):
        tlines, radius, make_face = args
        #print(tlines, radius, make_face)

        tlines = [x.strip() for x in tlines.split(',')]
        gids = self.get_target1(objs, tlines, 'l')

        e1 = self.edges[gids[0]]
        e2 = self.edges[gids[1]]

        pnt = gp_Pnt(0, 0, 0)
        dr = gp_Dir(0, 0, 1)

        pl = Geom_Plane(pnt, dr)

        from OCC.Core.GccAna import GccAna_Circ2d2TanRad

        loc = TopLoc_Location()
        e1_2d, first1, _last1 = self.bt.CurveOnPlane(e1, pl, loc)
        e2_2d, first2, _last2 = self.bt.CurveOnPlane(e2, pl, loc)

        p1, v1 = gp_Pnt2d(), gp_Vec2d()
        e1_2d.D1(first1, p1, v1)
        p2, v2 = gp_Pnt2d(), gp_Vec2d()
        e2_2d.D1(first2, p2, v2)

        l1 = gp_Lin2d(p1, gp_Dir2d(v1))
        l2 = gp_Lin2d(p2, gp_Dir2d(v2))
        #print(l1, l2)

        from OCC.Core.GccEnt import gccent_Unqualified, GccEnt_QualifiedLin
        from OCC.Core.Geom2d import Geom2d_Circle

        l1_q = gccent_Unqualified(l1)
        l2_q = gccent_Unqualified(l2)

        #l1_q = GccEnt_QualifiedLin(l1, gccent_Unqualified())
        #l2_q = GccEnt_QualifiedLin(l1, gccent_Unqualified())

        c_solver = GccAna_Circ2d2TanRad(l1_q, l2_q,
                                        radius, self.occ_geom_tolerance)
        # print(c_solver.IsDone())
        # print(c_solver.NbSolutions())
        for i in range(c_solver.NbSolutions()):
            c_sol = c_solver.ThisSolution(i + 1)
            c = Geom2d_Circle(c_sol)

            edgeMaker = BRepBuilderAPI_MakeEdge(c, pl)
            edgeMaker.Build()
            if not edgeMaker.IsDone():
                assert False, "Can not make circle"
            edge = edgeMaker.Edge()

            eid = self.edges.add(edge)
            shape = self.edges[eid]
            self.builder.Add(self.shape, shape)

        self.synchronize_topo_list(action='both')

        return list(objs), []

        '''
        #gp_Lin2d (const gp_Pnt2d &P, const gp_Dir2d &V)
        print(e1_2d)
        curve.IsKind(':
         handle = OCC.Core.Geom.__dict__[kind]


        print(circle)
        '''

        newkeys = []
        return list(objs), newkeys

    def Circle2DCenterOnePoint_build_geom(self, objs, *args):
        center, pts, make_face = args

        center = [x.strip() for x in center.split(',')]
        gids_1 = self.get_target1(objs, center, 'p')

        pts = [x.strip() for x in pts.split(',')]
        gids_2 = self.get_target1(objs, pts, 'p')

        c1 = self.get_point_coord(gids_1[0])
        p1 = self.get_point_coord(gids_2[0])

        r1 = np.sqrt(np.sum((c1 - p1)**2))
        n1 = (0.0, 0.0, 1.0)

        edge = self.add_circle_by_axis_radius(c1, n1, r1)

        if make_face:
            ll1 = self.add_line_loop([edge])
            ps1 = self.add_plane_surface(ll1)
            shape = self.faces[ps1]
            self.builder.Add(self.shape, shape)
            newkey = objs.addobj(ps1, 'ps')
        else:
            shape = self.edges[edge]
            self.builder.Add(self.shape, shape)
            newkey = objs.addobj(edge, 'cl')

        self.synchronize_topo_list(action='both')
        return list(objs), [newkey]

    def Circle2DByDiameter_build_geom(self, objs, *args):
        pts, make_face = args

        pts = [x.strip() for x in pts.split(',')]
        gids_1 = self.get_target1(objs, pts, 'p')

        p1 = self.get_point_coord(gids_1[0])
        p2 = self.get_point_coord(gids_1[1])

        c1 = (p1 + p2) / 2.0
        r1 = np.sqrt(np.sum((p2 - p1)**2)) / 2.0
        n1 = (0.0, 0.0, 1.0)

        edge = self.add_circle_by_axis_radius(c1, n1, r1)

        if make_face:
            ll1 = self.add_line_loop([edge])
            ps1 = self.add_plane_surface(ll1)
            shape = self.faces[ps1]
            self.builder.Add(self.shape, shape)
            newkey = objs.addobj(ps1, 'ps')
        else:
            shape = self.edges[edge]
            self.builder.Add(self.shape, shape)
            newkey = objs.addobj(edge, 'cl')

        self.synchronize_topo_list(action='both')
        return list(objs), [newkey]

    def Arc2D_build_geom(self, objs, *args):
        center, ax1, ax2, radius, an1, an2, do_fill = args
        a1 = np.array(ax1 + [0])
        a2 = np.array(ax2 + [0])
        a2 = np.cross(np.cross(a1, a2), a1)
        a1 = a1 / np.sqrt(np.sum(a1**2)) * radius
        a2 = a2 / np.sqrt(np.sum(a2**2)) * radius
        if an1 > an2:
            tmp = an2
            an2 = an1
            an1 = tmp

        print('before arc2d', [int(x) for x in self.vertices.keys()])
        assert an2 - an1 < 360, "angle must be less than 360"
        assert radius > 0, "radius must be positive"

        an3 = (an1 + an2) / 2.0
        pt1 = a1 * np.cos(an1 * np.pi / 180.) + a2 * np.sin(an1 * np.pi / 180.)
        pt2 = a1 * np.cos(an2 * np.pi / 180.) + a2 * np.sin(an2 * np.pi / 180.)
        pt3 = a1 * np.cos(an3 * np.pi / 180.) + a2 * np.sin(an3 * np.pi / 180.)

        c = np.array(center + [0])
        p1 = self.add_point(c + pt1)
        p2 = self.add_point(c + pt2)
        p3 = self.add_point(c + pt3)

        ca1 = self.add_circle_arc(p1, p3, p2)

        if not do_fill:
            newkey1 = objs.addobj(ca1, 'ln')
            shape1 = self.edges[ca1]
            #new_objs = self.register_shapes_balk(shape1)
            self.builder.Add(self.shape, shape1)
            newkeys = [newkey1, ]

        else:
            l1 = self.add_line(p2, p1)
            ll1 = self.add_line_loop([l1, ca1])
            ps1 = self.add_plane_surface(ll1)
            shape1 = self.faces[ps1]
            self.builder.Add(self.shape, shape1)
            newkeys = [objs.addobj(ps1, 'ps')]

        self.synchronize_topo_list(action='both')
        return list(objs), newkeys

    def Arc2DBy3Points_build_geom(self, objs, *args):
        pts, do_fill = args
        targets = [x.strip() for x in pts.split(',')]
        gids = self.get_target1(objs, targets, 'p')

        ca1 = self.add_circle_arc(gids[0], gids[1], gids[2])

        if not do_fill:
            newkey1 = objs.addobj(ca1, 'ln')
            shape1 = self.edges[ca1]
            self.builder.Add(self.shape, shape1)
            newkeys = [newkey1, ]

        else:
            l1 = self.add_line(gids[2], gids[0])
            ll1 = self.add_line_loop([l1, ca1])
            ps1 = self.add_plane_surface(ll1)
            shape1 = self.faces[ps1]
            self.builder.Add(self.shape, shape1)
            newkeys = [objs.addobj(ps1, 'ps')]

        return list(objs), newkeys

    def Arc2DBy2PointsAngle_build_geom(self, objs, *args):
        pts, angle, do_fill = args
        targets = [x.strip() for x in pts.split(',')]
        gids = self.get_target1(objs, targets, 'p')

        p1 = self.get_point_coord(gids[0])
        p2 = self.get_point_coord(gids[1])

        d1 = p2 - p1
        d1[2] = 0.0
        L = np.sqrt(np.sum(d1**2))

        d1 = d1 / L
        d2 = np.cross(d1, [0, 0, 1])

        pm = (p1 + p2) / 2.0
        c1 = pm - L / 2 * d2 / np.tan(angle * np.pi / 180 / 2)
        r = L / 2. / np.sin(angle * np.pi / 180 / 2)
        p3 = c1 + d2 * r
        p3 = self.add_point(p3)

        ca1 = self.add_circle_arc(gids[0], p3, gids[1])

        if not do_fill:
            newkey1 = objs.addobj(ca1, 'ln')
            shape1 = self.edges[ca1]
            self.builder.Add(self.shape, shape1)
            newkeys = [newkey1, ]

        else:
            l1 = self.add_line(gids[1], gids[0])
            ll1 = self.add_line_loop([l1, ca1])
            ps1 = self.add_plane_surface(ll1)
            shape1 = self.faces[ps1]
            self.builder.Add(self.shape, shape1)
            newkeys = [objs.addobj(ps1, 'ps')]

        return list(objs), newkeys

    def Rect2D_build_geom(self, objs, *args):
        c1, e1, e2 = args

        c1 = np.array(c1 + [0])
        e1 = np.array(e1 + [0])
        e2 = np.array(e2 + [0])
        p1 = self.add_point(c1)
        p2 = self.add_point(c1 + e1)
        p3 = self.add_point(c1 + e1 + e2)
        p4 = self.add_point(c1 + e2)
        l1 = self.add_line(p1, p2)
        l2 = self.add_line(p2, p3)
        l3 = self.add_line(p3, p4)
        l4 = self.add_line(p4, p1)
        ll1 = self.add_line_loop([l1, l2, l3, l4])
        rec1 = self.add_plane_surface(ll1)

        shape = self.faces[rec1]
        self.builder.Add(self.shape, shape)

        newkey = objs.addobj(rec1, 'rec')
        return list(objs), [newkey]

    def Rect2DByCorners_build_geom(self, objs, *args):
        pts = args[0]

        targets = [x.strip() for x in pts.split(',')]
        gids = self.get_target1(objs, targets, 'p')

        p1 = self.get_point_coord(gids[0])
        p2 = self.get_point_coord(gids[1])

        c1, d1, d2 = self._last_wp_param

        x1, y1 = p1[:2]
        x2, y2 = p2[:2]

        p1 = gids[0]
        p2 = self.add_point([x1, y2, 0])
        p3 = gids[1]
        p4 = self.add_point([x2, y1, 0])
        l1 = self.add_line(p1, p2)
        l2 = self.add_line(p2, p3)
        l3 = self.add_line(p3, p4)
        l4 = self.add_line(p4, p1)
        ll1 = self.add_line_loop([l1, l2, l3, l4])
        rec1 = self.add_plane_surface(ll1)

        shape = self.faces[rec1]
        self.builder.Add(self.shape, shape)

        newkey = objs.addobj(rec1, 'rec')
        return list(objs), [newkey]

    def Polygon2D_build_geom(self, objs, *args):
        del objs
        del args
        assert False, "We dont support this"

    def Move2D_build_geom(self, objs, *args):
        targets, dx, dy, keep = args
        dz = 0.0

        targets = [x.strip() for x in targets.split(',')]

        newkeys = []
        gids = self.get_target2(objs, targets)

        delta = (dx, dy, dz)
        for gid in gids:
            new_gid = self.translate(gid, delta, keep)
            if new_gid is not None:
                newkeys.append(objs.addobj(new_gid, 'mv'))

        self.synchronize_topo_list(action='both')

        return list(objs), newkeys

    def Rotate2D_build_geom(self, objs, *args):
        targets, cc, angle, keep = args

        point_on_axis = cc[0], cc[1], 0.0
        axis_dir = 0.0, 0.0, 1.0

        newkeys = []
        targets = [x.strip() for x in targets.split(',')]

        gids = self.get_target2(objs, targets)

        for gid in gids:
            new_gid = self.rotate(gid, point_on_axis, axis_dir,
                                  np.pi * angle / 180., copy=keep)
            if new_gid is not None:
                newkeys.append(objs.addobj(new_gid, 'mv'))

        self.synchronize_topo_list(action='both')

        return list(objs), newkeys

    def Scale2D_build_geom(self, objs, *args):
        targets, cc, ss, keep = args

        cc = (cc[0], cc[1], 0.0)
        ss = (ss[0], ss[1], 1.0)
        targets = [x.strip() for x in targets.split(',')]

        newkeys = []
        gids = self.get_target2(objs, targets)

        # for gid in gids:
        #    new_gid = self.dilate(gid, cc, ss, copy=keep)
        #    if new_gid is not None:
        #        newkeys.append(objs.addobj(new_gid, 'sc'))
        new_gids = self.dilate(gids, cc, ss, copy=keep)
        for new_gid in new_gids:
            newkeys.append(objs.addobj(new_gid, 'sc'))

        self.synchronize_topo_list(action='both')

        return list(objs), newkeys

    def Flip2D_build_geom(self, objs, *args):

        targets, a, b, d, keep = args

        abcd = (a, b, 0.0, d)
        targets = [x.strip() for x in targets.split(',')]

        newkeys = []
        gids = self.get_target2(objs, targets)

        for gid in gids:
            new_gid = self.symmetrize(gid, abcd, copy=keep)
            if new_gid is not None:
                newkeys.append(objs.addobj(new_gid, 'mv'))

        self.synchronize_topo_list(action='both')

        return list(objs), newkeys

    def Array2D_build_geom(self, objs, *args):

        targets, count, displacement = args

        dx, dy, dz = (displacement[0], displacement[0], 0.0)
        targets = [x.strip() for x in targets.split(',')]

        newkeys = []
        gids = self.get_target2(objs, targets)

        i = 1
        while i < count:
            delta = (dx * i, dy * i, dz * i)

            for gid in gids:
                new_gid = self.translate(gid, delta, True)
                if new_gid is not None:
                    newkeys.append(objs.addobj(new_gid, 'cp'))
            i = i + 1

        self.synchronize_topo_list(action='add')

        return list(objs), newkeys

    def Union2D_build_geom(self, objs, *args):
        tp, tm, delete_input, delete_tool, keep_highest, tol_scale = args
        tp = [x.strip() for x in tp.split(',')]
        tm = [x.strip() for x in tm.split(',')]

        gid_objs = self.get_target2(objs, tp)
        gid_tools = self.get_target2(objs, tm)

        gids_new = self.union2d(gid_objs, gid_tools,
                                remove_obj=delete_input,
                                remove_tool=delete_tool,
                                keep_highest=keep_highest,
                                tol_scale=tol_scale)

        newkeys = []
        for gid in gids_new:
            newkeys.append(objs.addobj(gid, 'uni'))

        self.synchronize_topo_list()

        if delete_input:
            for x in tp:
                if x in objs:
                    del objs[x]
        if delete_tool:
            for x in tm:
                if x in objs:
                    del objs[x]
        return list(objs), newkeys

    def MergeFace_build_geom(self, objs, *args):
        tp, use_upgrade = args
        tp = [x.strip() for x in tp.split(',')]
        gid_all = self.get_target1(objs, tp, 'f')

        gid_objs = gid_all[:1]
        gid_tools = gid_all[1:]

        gids_new = self.merge_face(gid_objs, gid_tools,
                                   remove_obj=True,
                                   remove_tool=True,
                                   keep_highest=True,
                                   use_upgrade=use_upgrade)

        newkeys = []
        for gid in gids_new:
            newkeys.append(objs.addobj(gid, 'uni'))
        return list(objs), newkeys
        self.synchronize_topo_list()

        for x in tp:
            if x in objs:
                del objs[x]

        return list(objs), newkeys

    def SplitByPlane_build_geom(self, objs, *args):
        targets = [x.strip() for x in args[0].split(',')]
        gids = self.get_target2(objs, targets)

        cptx, normal = self.process_plane_parameters(args[1], objs, gids)
        offset = args[-1]
        if offset != 0:
            cptx = cptx + normal * offset

        # splitter alogrithm
        normal = -normal
        pnt = gp_Pnt(*cptx)
        dr = gp_Dir(*normal)
        pl = Geom_Plane(pnt, dr)
        maker = BRepBuilderAPI_MakeFace(pl, self.occ_geom_tolerance)
        maker.Build()
        if not maker.IsDone():
            assert False, "Faild to generate plane"
        plane = maker.Face()
        operator = BOPAlgo_Splitter()

        sp_objs = TopTools_ListOfShape()
        sp_tools = TopTools_ListOfShape()

        for gid in gids:
            topolist = self.get_topo_list_for_gid(gid)
            shape = topolist[gid]
            sp_objs.Append(shape)
        sp_tools.Append(plane)

        operator.SetArguments(sp_objs)
        operator.SetTools(sp_tools)

        operator.Perform()
        if operator.HasErrors():
            assert False, "Splitter Algorithm failed"

        for gid in gids:
            self.remove(gid)

        result = operator.Shape()
        gids_new = self.register_shapes_balk(result, check_this=self.shape)

        newkeys = []
        for gid in gids_new:
            newkeys.append(objs.addobj(gid, 'splt'))

        self.synchronize_topo_list(action='both')
        return list(objs), newkeys

    def ProjectOnWP_build_geom(self, objs, *args):
        targets, fill = args
        targets = [x.strip() for x in targets.split(',')]
        gids = self.get_target2(objs, targets)

        c1, d1, d2 = self._last_wp_param
        n1 = np.cross(d1, d2)

        result = self.project_shape_on_wp(gids, c1, n1, fill=fill,
                                          base_shape=self._shape_bk)

        ax1, an1, ax2, an2, cxyz = calc_wp_projection(c1, d1, d2)
        if np.sum(c1**2) != 0.0:
            result = do_translate(result, -c1)
        if np.sum(ax2**2) != 0.0 and an2 != 0.0:
            result = do_rotate(result, ax2, -an2, txt='2nd')
        if np.sum(ax1**2) != 0.0 and an1 != 0.0:
            result = do_rotate(result, ax1, -an1, txt='1st')

        gids_new = self.register_shapes_balk(result)

        self.synchronize_topo_list()

        newobjs = []
        for gid in gids_new:
            newkey = objs.addobj(gid, gid.name)
            newobjs.append(newkey)

        return list(objs), newobjs

    def _WorkPlane_build_geom(self, objs, c1, a1, a2):

        ax1, an1, ax2, an2, cxyz = calc_wp_projection(c1, a1, a2)

        if np.sum(ax1**2) != 0.0 and an1 != 0.0:
            self.shape = do_rotate(self.shape, ax1, an1, txt='1st')
        if np.sum(ax2**2) != 0.0 and an2 != 0.0:
            self.shape = do_rotate(self.shape, ax2, an2, txt='2nd')
        if np.sum(c1**2) != 0.0:
            self.shape = do_translate(self.shape, c1)

        shape = self.shape
        self.pop_shape_and_topolist()
        gids_new = self.register_shapes_balk(shape)

        self.synchronize_topo_list(action='both')
        #self.inspect_shape(self.shape, verbose=True)

        newkeys = []
        gid = WorkPlaneParam(c1, a1, a2)
        newkeys.append(objs.addobj(gid, 'wp'))

        return list(objs), newkeys

    def WorkPlaneStart_build_geom(self, objs, *args):
        c1, a1, a2 = args
        c1 = np.array(c1)
        a1 = np.array(a1)
        a1 = a1 / np.sqrt(np.sum(a1**2))
        a2 = np.array(a2)
        a2 = a2 / np.sqrt(np.sum(a2**2))
        self._last_wp_param = c1, a1, a2
        return objs, []

    def WorkPlaneEnd_build_geom(self, objs, *args):
        print('WorkPlaneEnd_build_geom')
        if self.isWP != 0:
            return objs, []
        c1, a1, a2 = self._last_wp_param
        return self._WorkPlane_build_geom(objs, c1, a1, a2)
    '''
    def WorkPlane_build_geom(self, objs, *args):
        if self.isWP != 0:
            return objs, []
        c1, a1, a2 = self._last_wp_param
        return self._WorkPlane_build_geom(objs, c1, a1, a2)
    '''

    def WorkPlaneByPointsStart_build_geom(self, objs, *args):
        c1, a1, a2, flip1, flip2, offset = args

        c1, a1, a2 = self.get_target1(objs, [c1, a1, a2], 'p')

        cgroup = self.vertices.current_group()
        self.set_topolist_group(0)
        c1 = self.get_point_coord(c1)
        a1 = self.get_point_coord(a1)
        a2 = self.get_point_coord(a2)
        self.set_topolist_group(cgroup)

        d1 = np.array(a1) - np.array(c1)
        d1 = d1 / np.sqrt(np.sum(d1**2))
        if flip1:
            d1 = -d1

        d2 = np.array(a2) - np.array(c1)
        d2 = d2 / np.sqrt(np.sum(d2**2))

        d3 = np.cross(d1, d2)
        d3 = d3 / np.sqrt(np.sum(d3**2))
        d2 = np.cross(d3, d1)
        d2 = d2 / np.sqrt(np.sum(d2**2))
        if flip2:
            d2 = -d2

        c1 = c1 + np.cross(d1, d2) * offset
        self._last_wp_param = c1, d1, d2
        return objs, []
        # return self._WorkPlane_build_geom(objs, c1, d1, d2)

    def WPParallelToPlaneStart_build_geom(self, objs, *args):
        s1, c1, p1, flip1, flip2, offset = args

        s1 = self.get_target1(objs, [s1], 'f')[0]
        c1, p1 = self.get_target1(objs, [c1, p1], 'p')

        cgroup = self.vertices.current_group()
        self.set_topolist_group(0)
        n1, _void = self.get_face_normal(s1, check_flat=True)
        c1 = self.get_point_coord(c1)
        p1 = self.get_point_coord(p1)
        self.set_topolist_group(cgroup)

        d1 = np.array(p1) - np.array(c1)
        d1 = d1 / np.sqrt(np.sum(d1**2))

        d2 = np.cross(n1, d1)
        d1 = np.cross(d2, n1)
        if flip1:
            d1 = -d1
        if flip2:
            d2 = -d2

        c1 = c1 + n1 * offset
        self._last_wp_param = c1, d1, d2
        return objs, []

    def WPNormalToPlaneStart_build_geom(self, objs, *args):
        norm_param, c1, p1, flip1, flip2, offset = args

        c1, p1 = self.get_target1(objs, [c1, p1], 'p')
        newkeys = []

        cgroup = self.vertices.current_group()
        self.set_topolist_group(0)

        n1, c1 = self.process_normal_parameters(norm_param, c1, objs)
        p1 = self.get_point_coord(p1)

        self.set_topolist_group(cgroup)

        d1 = np.array(p1) - np.array(c1)
        d1 = d1 / np.sqrt(np.sum(d1**2))

        d2 = np.cross(n1, d1)
        d1 = np.cross(d2, n1)
        if flip1:
            d1 = -d1
        if flip2:
            d2 = -d2

        c1 = c1 + n1 * offset
        self._last_wp_param = c1, d1, d2
        return objs, []

    def Subsequence_build_geom(self, objs, *args):
        return objs, []

    def select_highest_dim(self, shape):
        comp = TopoDS_Compound()
        b = self.builder
        b.MakeCompound(comp)

        mmm = TopTools_IndexedMapOfShape()
        topexp.MapShapes(shape, TopAbs_SOLID, mmm)
        if mmm.Size() == 0:
            topexp.MapShapes(shape, TopAbs_FACE, mmm)
            if mmm.Size() == 0:
                topexp.MapShapes(shape, TopAbs_EDGE, mmm)
                if mmm.Size() == 0:
                    topexp.MapShapes(shape, TopAbs_VERTEX, mmm)
                    ex1 = TopExp_Explorer(shape, TopAbs_VERTEX)
                else:
                    ex1 = TopExp_Explorer(shape, TopAbs_EDGE)
            else:
                ex1 = TopExp_Explorer(shape, TopAbs_FACE)
        else:
            ex1 = TopExp_Explorer(shape, TopAbs_SOLID)

        while ex1.More():
            b.Add(comp, ex1.Current())
            ex1.Next()
        return comp

    def register_shapes_balk(self, shape, check_this=None):
        '''
          register a shape to topo_list.
          check_this : a shape.
              if this is given, we don't register an entity which is already
              contained in this shape. Usually self.shape


        '''
        maps = prep_maps(shape)
        solidMap = maps['solid']
        shellMap = maps['shell']
        faceMap = maps['face']
        wireMap = maps['wire']
        edgeMap = maps['edge']
        vertMap = maps['vertex']

        usolids = topo_seen(mapping=solidMap)
        ushells = topo_seen(mapping=shellMap)
        ufaces = topo_seen(mapping=faceMap)
        uwires = topo_seen(mapping=wireMap)
        uedges = topo_seen(mapping=edgeMap)
        uvertices = topo_seen(mapping=vertMap)

        if check_this is not None:
            maps2 = prep_maps(check_this)
            do_check = True
        else:
            maps2 = None
            do_check = False

        new_objs = []
        # registor solid
        ex1 = TopExp_Explorer(shape, TopAbs_SOLID)
        while ex1.More():
            solid = ex1.Current()
            if (not (do_check and maps2['solid'].Contains(solid)) and
                    usolids.check_shape(solid) == 0):
                solid_id = self.solids.add(solid)
                new_objs.append(solid_id)
            ex1.Next()

        def register_topo(shape, ucounter, topabs, topabs_p,
                          topo_list, dim=-1, map2_name=None):

            if maps2 is not None:
                map2a = maps2[map2_name[0]]
                map2b = maps2[map2_name[1]]

            ex1 = TopExp_Explorer(shape, topabs_p)
            while ex1.More():
                topo_p = ex1.Current()
                ex2 = TopExp_Explorer(topo_p, topabs)
                while ex2.More():
                    topo = ex2.Current()
                    if (not (do_check and map2a.Contains(topo)) and
                            ucounter.check_shape(topo) == 0):
                        topo_id = topo_list.add(topo)
                    ex2.Next()
                ex1.Next()
            ex1.Init(shape, topabs, topabs_p)
            while ex1.More():
                topo = ex1.Current()
                if (not (do_check and map2b.Contains(topo)) and
                        ucounter.check_shape(topo) == 0):
                    topo_id = topo_list.add(topo)
                    if dim != -1:
                        new_objs.append(topo_id)
                ex1.Next()

        register_topo(
            shape,
            ushells,
            TopAbs_SHELL,
            TopAbs_SOLID,
            self.shells,
            map2_name=('shell', 'solid'))
        register_topo(
            shape,
            ufaces,
            TopAbs_FACE,
            TopAbs_SHELL,
            self.faces,
            dim=2,
            map2_name=('face', 'shell'))

        register_topo(
            shape,
            uwires,
            TopAbs_WIRE,
            TopAbs_FACE,
            self.wires,
            map2_name=('wire', 'face'))

        register_topo(
            shape,
            uedges,
            TopAbs_EDGE,
            TopAbs_WIRE,
            self.edges,
            dim=1,
            map2_name=('edge', 'wire'))

        register_topo(shape,
                      uvertices,
                      TopAbs_VERTEX,
                      TopAbs_EDGE,
                      self.vertices,
                      dim=0,
                      map2_name=('vertex', 'edge'))

        b = self.builder
        comp = self.shape
        ex1 = TopExp_Explorer(shape, TopAbs_SOLID)
        while ex1.More():
            b.Add(comp, ex1.Current())
            ex1.Next()
        ex1 = TopExp_Explorer(shape, TopAbs_SHELL, TopAbs_SOLID)
        while ex1.More():
            b.Add(comp, ex1.Current())
            ex1.Next()
        ex1 = TopExp_Explorer(shape, TopAbs_FACE, TopAbs_SHELL)
        while ex1.More():
            b.Add(comp, ex1.Current())
            ex1.Next()
        ex1 = TopExp_Explorer(shape, TopAbs_WIRE, TopAbs_FACE)
        while ex1.More():
            b.Add(comp, ex1.Current())
            ex1.Next()
        ex1 = TopExp_Explorer(shape, TopAbs_EDGE, TopAbs_WIRE)
        while ex1.More():
            b.Add(comp, ex1.Current())
            ex1.Next()
        ex1 = TopExp_Explorer(shape, TopAbs_VERTEX, TopAbs_EDGE)
        while ex1.More():
            b.Add(comp, ex1.Current())
            ex1.Next()

        return new_objs

    def importShape_common(self, shape, highestDimOnly,
                           fix_param, objs):

        from petram.geom.occ_heal_shape import heal_shape

        if fix_param is not None:
            use_fix_param = fix_param[0]
            fixD = use_fix_param[0]
            fixE = use_fix_param[1]
            fixF = use_fix_param[2]
            sewF = use_fix_param[3]
            mSol = use_fix_param[4]
            tol = fix_param[1]
            scaling = fix_param[2]

            # if highestDimOnly:
            #    shape = self.select_highest_dim(shape)
            #new_objs = self.register_shapes_balk(shape)

            shape = heal_shape(shape, scaling=scaling, fixDegenerated=fixD,
                               fixSmallEdges=fixE, fixSmallFaces=fixF,
                               sewFaces=sewF, makeSolids=mSol, tolerance=tol,
                               verbose=True)

            '''
            Note: scaling does not work. OCC get stack during incremental meshing.

            tmp_brep = os.path.join(self.trash, 'tmp.brep')
            self.write_brep(tmp_brep, shape=shape)

            shape = TopoDS_Shape()
            success = breptools.Read(shape, tmp_brep, self.builder)
            if not success:
                assert False, "Failed to read brep"
            '''
        if highestDimOnly:
            shape = self.select_highest_dim(shape)
        new_objs = self.register_shapes_balk(shape)

        self.synchronize_topo_list(action='both')

        newkeys = []
        dim = max([p.idx for p in new_objs])
        for p in new_objs:
            if p.idx == dim:
                newkeys.append(objs.addobj(p, 'impt'))

        return list(objs), newkeys

    def healCAD_build_geom(self, objs, *args):
        fix_entity, fix_param, fix_tol, fix_rescale = args

        fix_param = (fix_param, fix_tol, fix_rescale,)

        targets = fix_entity
        targets = [x.strip() for x in targets.split(',')]
        gids = self.get_target2(objs, targets)

        all_newkeys = []
        for gid in gids:
            topolist = self.get_topo_list_for_gid(gid)
            shape = topolist[gid]

            is_toplevel = topolist.is_toplevel(gid, self.shape)
            if not is_toplevel:
                assert False, "CAD heal should be applied to a top-level entity"

            self.builder.Remove(self.shape, shape)
            self.synchronize_topo_list()

            names, newkeys = self.importShape_common(
                shape, True, fix_param, objs)

            #print("names, newkeys", names, newkeys)

            all_newkeys.extend(newkeys)
            if gid in objs:
                del objs[t]

        self.synchronize_topo_list()

        return list(objs), all_newkeys

    def BrepImport_build_geom(self, objs, *args):
        cad_file, use_fix, use_fix_param, use_fix_tol, use_fix_rescale, highestDimOnly = args

        if use_fix:
            fix_param = (use_fix_param, use_fix_tol, use_fix_rescale,)
        else:
            fix_param = None

        shape = TopoDS_Shape()

        import os
        cad_file = os.path.expanduser(cad_file)
        success = breptools.Read(shape, cad_file, self.builder)

        if not success:
            assert False, "Failed to read brep"

        breptools.Clean(shape)
        return self.importShape_common(shape, highestDimOnly, fix_param, objs)

    def CADImport_build_geom(self, objs, *args):
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.IGESControl import IGESControl_Reader
        from OCC.Core.IFSelect import IFSelect_RetDone, IFSelect_ItemsByEntity

        unit = args[-1]
        cad_file, use_fix, use_fix_param, use_fix_tol, use_fix_rescale, highestDimOnly = args[
            :-1]
        if use_fix:
            fix_param = (use_fix_param, use_fix_tol, use_fix_rescale,)
        else:
            fix_param = None

        import os
        cad_file = os.path.expanduser(cad_file)

        if (cad_file.lower().endswith(".iges") or
                cad_file.lower().endswith(".igs")):
            reader = IGESControl_Reader()
        elif (cad_file.lower().endswith(".step") or
              cad_file.lower().endswith(".stp")):
            reader = STEPControl_Reader()
        else:
            assert False, "unsupported format"

        if unit != '':
            check = Interface_Static.SetCVal("xstep.cascade.unit", unit)
            if not check:
                assert False, "can not set unit"

        status = reader.ReadFile(cad_file)

        if status == IFSelect_RetDone:  # check status
            failsonly = False
            reader.PrintCheckLoad(failsonly, IFSelect_ItemsByEntity)
            reader.PrintCheckTransfer(failsonly, IFSelect_ItemsByEntity)
            reader.NbRootsForTransfer()
            reader.TransferRoots()
            shape = reader.OneShape()
        else:
            assert False, "Error: can't read STEP/IGES file."

        breptools.Clean(shape)
        return self.importShape_common(shape, highestDimOnly, fix_param, objs)

    def make_safe_file(self, filename, trash, ext):
        #map = self.getEntityNumberingInfo()
        # make filename safe
        filename = '_'.join(filename.split("/"))
        filename = '_'.join(filename.split(":"))
        filename = '_'.join(filename.split("\\"))

        return os.path.join(trash, filename + ext)

        # if trash == '':  # when finalizing
        #    return os.path.join(os.getcwd(), filename + ext)
        # else:

    def generate_preview_mesh0(self):

        values = self.bounding_box()
        adeviation = max((values[3] - values[0],
                          values[4] - values[1],
                          values[5] - values[2]))

        if adeviation == 0:
            return None
        else:
            ad = self.occ_angle_deflection
            ld = self.occ_linear_deflection

            breptools.Clean(self.shape)
            # BRepMesh_IncrementalMesh(self.shape, ld * adeviation,
            #                         False, ad, self.occ_parallel)
            prm = IMeshTools_Parameters()
            prm.Deflection = ld
            prm.Angle = ad
            prm.Relative = True
            prm.InParallel = (self.maxthreads > 1)
            prm.MinSize = -1
            prm.InternalVerticesMode = True
            prm.ControlSurfaceDeflection = True
            BRepMesh_IncrementalMesh(self.shape, prm)

        dprint1("Done (IncrementalMesh)")

        bt = BRep_Tool()

        L = 1 if len(self.faces) == 0 else max(list(self.faces)) + 1
        face_vert_offset = [0] * L

        all_ptx = []
        face_idx = {}
        edge_idx = {}
        vert_idx = {}

        # in order to update value from inner functions. this needs to be
        # object
        offset = Counter()
        num_failedface = Counter()
        num_failededge = Counter()

        solidMap, faceMap, edgeMap, vertMap = self.inspect_shape(
            self.shape, verbose=False, return_all=False)
        solid2isolid = topo2id(self.solids, solidMap)
        face2iface = topo2id(self.faces, faceMap)
        edge2iedge = topo2id(self.edges, edgeMap)
        vert2iverte = topo2id(self.vertices, vertMap)

        face2solid = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp.MapShapesAndAncestors(
            self.shape, TopAbs_FACE, TopAbs_SOLID, face2solid)
        edge2face = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp.MapShapesAndAncestors(
            self.shape, TopAbs_EDGE, TopAbs_FACE, edge2face)
        vertex2edge = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp.MapShapesAndAncestors(
            self.shape, TopAbs_VERTEX, TopAbs_EDGE, vertex2edge)

        def value2coord(value, location):
            if not location.IsIdentity():
                trans = location.Transformation()
                xyz = [v.Coord() for v in value]
                xyz = [gp_XYZ(x[0], x[1], x[2]) for x in xyz]
                void = [trans.Transforms(x) for x in xyz]
                ptx = [x.Coord() for x in xyz]
            else:
                ptx = [x.Coord() for x in value]
            return np.vstack(ptx)

        def work_on_face(iface, face):
            face_vert_offset[iface] = offset()

            location = TopLoc_Location()
            facing = (bt.Triangulation(face, location))

            if facing is None:
                num_failedface.increment(1)
                print('tesselation of face is missing, iface=', iface)
                return
            else:
                if not OCC_after_7_6_0:
                    tab = facing.Nodes()
                    tri = facing.Triangles()
                    idx = [tri.Value(i).Get()
                           for i in range(1, facing.NbTriangles() + 1)]
                    values = [tab.Value(i) for i in range(1, tab.Length() + 1)]
                    LL = tab.Length()
                else:
                    values = [facing.Node(i)
                              for i in range(1, facing.NbNodes() + 1)]
                    idx = [facing.Triangle(i).Get()
                           for i in range(1, facing.NbTriangles() + 1)]
                    LL = len(values)

                ptx = value2coord(values, location)

                all_ptx.append(np.vstack(ptx))

                face_idx[iface] = np.vstack(idx) - 1 + offset()
                offset.increment(LL)
                return

        def work_on_edge_on_face(iedge, edge):
            faces = edge2face.FindFromKey(edge)
            topology_iterator = TopTools_ListIteratorOfListOfShape(faces)
            while topology_iterator.More():
                face = topology_iterator.Value()
                topology_iterator.Next()
                location = TopLoc_Location()
                facing = (bt.Triangulation(face, location))
                if facing is not None:
                    break
            else:
                num_failededge.increment(1)
                print('tesselation of edge is missing, iedge=', iedge)
                return

            iface = face2iface[face]
            coffset = face_vert_offset[iface]
            poly = (bt.PolygonOnTriangulation(edge, facing, location))

            if poly is None:
                num_failededge.increment(1)
                print('tesselation of edge is missing, iedge=', iedge)
                return
            else:
                node = poly.Nodes()
                idx = [
                    node.Value(i) +
                    coffset -
                    1 for i in range(
                        1,
                        poly.NbNodes() +
                        1)]
                edge_idx[iedge] = idx

        def work_on_edge(iedge, edge):
            location = TopLoc_Location()
            poly = bt.Polygon3D(edge, location)

            if poly is None:
                work_on_edge_on_face(iedge, edge)
            else:
                nnodes = poly.NbNodes()
                nodes = poly.Nodes()
                values = [nodes.Value(i) for i in range(1, poly.NbNodes() + 1)]
                ptx = value2coord(values, location)
                idx = np.arange(poly.NbNodes())

                all_ptx.append(np.vstack(ptx))
                edge_idx[iedge] = list(idx + offset())
                offset.increment(poly.NbNodes())

        def work_on_vertex(ivert, vertex):
            pnt = bt.Pnt(vertex)
            ptx = [pnt.Coord()]
            idx = [offset()]
            all_ptx.append(ptx)
            vert_idx[ivert] = idx
            offset.increment(1)

        for iface in self.faces:
            work_on_face(iface, self.faces[iface])
        for iedge in self.edges:
            work_on_edge(iedge, self.edges[iedge])
        for ivert in self.vertices:
            work_on_vertex(ivert, self.vertices[ivert])

        def generate_idxmap_from_map(idxmap, parent_imap, child2parents, objs):
            for iobj in objs:
                parents = child2parents.FindFromKey(objs[iobj])
                topology_iterator = TopTools_ListIteratorOfListOfShape(parents)
                while topology_iterator.More():
                    p = topology_iterator.Value()
                    topology_iterator.Next()

                    try:
                        iparent = parent_imap[p]
                    except BaseException:
                        print("parent for ", p, "not found")
                        continue

                        #assert False, "Not found"

                    if iobj not in idxmap[iparent]:
                        idxmap[iparent].append(iobj)

        # make v, s, l
        v = defaultdict(list)
        s = defaultdict(list)
        l = defaultdict(list)

        generate_idxmap_from_map(v, solid2isolid, face2solid, self.faces)
        generate_idxmap_from_map(s, face2iface, edge2face, self.edges)
        generate_idxmap_from_map(l, edge2iedge, vertex2edge, self.vertices)

        v = dict(v)
        s = dict(s)
        l = dict(l)

        shape = {}
        idx = {}

        # vertex
        keys = list(vert_idx)
        if len(keys) > 0:
            shape['vertex'] = np.vstack([vert_idx[k] for k in keys])
            idx['vertex'] = {'geometrical': np.hstack([k for k in keys]),
                             'physical': np.hstack([0 for k in keys])}
        # edge
        keys = list(edge_idx)
        if len(keys) > 0:
            a = [np.vstack([edge_idx[k][:-1], edge_idx[k][1:]]).transpose()
                 for k in keys]
            shape['line'] = np.vstack(a)
            eidx = np.hstack([[k] * (len(edge_idx[k]) - 1) for k in keys])
            idx['line'] = {'geometrical': eidx,
                           'physical': eidx * 0}

        # face
        keys = list(face_idx)
        if len(keys) > 0:
            shape['triangle'] = np.vstack([face_idx[k] for k in keys])
            eidx = np.hstack([[k] * len(face_idx[k]) for k in keys])
            idx['triangle'] = {'geometrical': eidx,
                               'physical': eidx * 0}

        ptx = np.vstack(all_ptx)
        esize = self.get_esize()

        vcl = self.get_vcl(l, esize)
        geom_msh = ''

        dprint1(
            "number of triangulation fails",
            num_failedface(),
            num_failededge())
        return geom_msh, l, s, v, vcl, esize, ptx, shape, idx

    def move_wp_points(self, ptx, c1, a1, a2):
        from petram.geom.geom_utils import rotation_mat

        ax1, an1, ax2, an2, cxyz = calc_wp_projection(c1, a1, a2)
        if np.sum(ax1**2) != 0.0 and an1 != 0.0:
            R = rotation_mat(ax1, an1)
            ptx = np.dot(R, ptx.transpose()).transpose()
        if np.sum(ax2**2) != 0.0 and an2 != 0.0:
            R = rotation_mat(ax2, an2)
            ptx = np.dot(R, ptx.transpose()).transpose()

        if np.sum(c1**2) != 0.0:
            ptx = ptx + np.array(c1)

        return ptx

    def generate_preview_mesh(self, finalize=True):
        def merge_preview_data(data1, data2):
            for k in data2[1]:  # merge l
                data1[1][k] = data2[1][k]
            for k in data2[2]:  # merge s
                data1[2][k] = data2[2][k]
            for k in data2[3]:  # merge v
                data1[3][k] = data2[3][k]
            for k in data2[4]:  # merge vcl
                data1[4][k] = data2[4][k]
            for k in data2[5]:  # merge esize
                data1[5][k] = data2[5][k]

            ptx = np.vstack((data1[6], data2[6],))
            offset = len(data1[6])

            shape = {}
            for k in data1[7]:
                if k in data2[7]:
                    shape[k] = np.vstack((data1[7][k], data2[7][k] + offset))
                else:
                    shape[k] = data1[7][k]
            idx = {}
            for k in data1[8]:
                data = {}
                if k in data2[8]:
                    for kk in data1[8][k]:
                        data[kk] = np.hstack(
                            (data1[8][k][kk], data2[8][k][kk],))
                else:
                    for kk in data1[8][k]:
                        data[kk] = data1[8][k][kk]
                idx[k] = data

            return (data1[0], data1[1], data1[2], data1[3],
                    data1[4], data1[5], ptx, shape, idx)

        if self.queue is not None:
            self.queue.put((False, "Generating preview"))

        if self.isWP == 0:
            return self.generate_preview_mesh0()

        data_wp = self.generate_preview_mesh0()
        if data_wp is not None:
            geom_msh, l, s, v, vcl, esize, ptx, shape, idx = data_wp
            ptx = self.move_wp_points(ptx, *self._last_wp_param)
            data_wp = geom_msh, l, s, v, vcl, esize, ptx, shape, idx

        wp_shape = self.shape
        self.pop_shape_and_topolist()
        data_main = self.generate_preview_mesh0()
        self.set_topolist_group(self.isWP)
        self.shape = wp_shape

        if data_main is not None and data_wp is not None:
            merged_data = merge_preview_data(data_main, data_wp)
            return merged_data
        elif data_wp is not None:
            return data_wp
        elif data_main is not None:
            return data_main
        return None

    def generate_brep(self, filename, trash, finalize=False):
        if finalize and not self.skip_final_frag:
            if self.logfile is not None:
                self.logfile.write("finalize is on \n")
            if self.queue is not None:
                self.queue.put((False, "finalize is on"))
            dprint1('appling fragmentation')
            self.apply_fragments()

        geom_brep = self.make_safe_file(filename, trash, '.brep')

        self.write_brep(geom_brep)

        return geom_brep

    def load_finalized_brep(self, brep_file):
        shape = TopoDS_Shape()
        success = breptools.Read(shape, brep_file, self.builder)

        if not success:
            assert False, "Failed to read brep" + str(brep_file)

        self.shape = shape
        self.prep_topo_list()

        shape = self.select_highest_dim(shape)
        new_objs = self.register_shapes_balk(shape)

    '''
    sequence/preview/brep generator
    '''

    def run_sequence(self, objs, gui_data, start_idx):

        def copy_objs(objs):
            tmp = objs.duplicate()
            org_keys = list(objs)
            for x in org_keys:
                del tmp[x]
            return tmp

        if start_idx < 1:
            self.shape = self.new_compound()
            self.prep_topo_list()
            self.isWP = 0
            self.workplanes = []

        self.org_objs = objs
        if self.isWP == 0:
            self.objs = objs
        else:
            self.objs = copy_objs(objs)

        self.objs = objs

        for gui_name, gui_param, geom_name in self.geom_sequence[start_idx:]:
            if self.logfile is not None:
                self.logfile.write("processing " + gui_name + "\n")
                self.logfile.write(
                    "data " +
                    str(geom_name) +
                    ":" +
                    str(gui_param) +
                    "\n")
            if self.queue is not None:
                self.queue.put((False, "processing " + gui_name))
            dprint1("processing " + gui_name, geom_name)

            if geom_name == "WP_Start":
                #tmp = objs.duplicate()
                #org_keys = list(objs)
                # for x in org_keys:
                #    del tmp[x]
                self.objs = copy_objs(objs)
                self.isWP = self.store_shape_and_topolist()

            elif geom_name == "WP_End_OCC":
                # comes here only when all WP is processed.
                self.isWP = 0

            elif geom_name == "WP_End":
                for x in self.objs:
                    self.org_objs[x] = self.objs[x]
                self.objs = self.org_objs

            else:
                try:
                    method = getattr(self, geom_name + '_build_geom')
                    objkeys, newobjs = method(self.objs, *gui_param)
                    gui_data[gui_name] = (objkeys, newobjs)
                except BaseException:
                    import traceback
                    if self.logfile is not None:
                        self.logfile.write("failed " + traceback.format_exc())
                    raise

        #capcheName = "" if isWP else gui_name
        self.synchronize_topo_list(action='both')

        return gui_data, self.objs

    def inspect_geom(self, inspect_type, args):

        from importlib import reload
        reload(petram.geom.occ_inspect)

        shape_inspector = petram.geom.occ_inspect.shape_inspector
        #print(inspect_type, args)
        if inspect_type in 'property':
            gids = self.get_target2(self.objs, args)
            shapes = [self.get_shape_for_gid(gid) for gid in gids]
        elif inspect_type in 'distance':
            gids = self.get_target2(self.objs, args[1:])
            shapes = [self.get_shape_for_gid(gid) for gid in gids]
            s0 = self.vertices[args[0]]
            shapes = [s0] + [self.get_shape_for_gid(gid) for gid in gids]
        elif inspect_type == 'shortedge':
            shapes = (args, self.edges)
        elif inspect_type == 'smallface':
            shapes = (args, self.faces)
        elif inspect_type == 'findsame':
            gids = [x.strip() for x in args[0].split(',')]
            gids = self.get_target2(self.objs, gids)
            shapes = [self.get_shape_for_gid(gid) for gid in gids]
            shapes = (args[1], shapes, (self.faces, self.edges))
        else:
            assert False, "unknown mode:" + inspect_type
        return shape_inspector(self.shape, inspect_type, shapes)

    def export_shapes(self, selection, filename):
        gids = []
        for i in selection[0]:
            gids.append(VolumeID(i))
        for i in selection[1]:
            gids.append(SurfaceID(i))
        for i in selection[2]:
            gids.append(LineID(i))
        for i in selection[3]:
            gids.append(VertexID(i))

        shapes = [self.get_shape_for_gid(gid) for gid in gids]

        comp = TopoDS_Compound()
        b = self.builder
        b.MakeCompound(comp)
        for s in shapes:
            b.Add(comp, s)

        self.write_brep(filename, shape=comp)

    def export_shapes_step(self, selection, filename):
        '''
        export shapes in STEP and STL format
        '''

        print("entring here")
        stlmode = filename.endswith('.stl')

        if selection is None:
            # create temporary compound to avoid writing duplicate objects
            #
            shape = self.shape
            comp = self.new_compound()
            b = self.builder

            ex1 = TopExp_Explorer(shape, TopAbs_SOLID)
            while ex1.More():
                b.Add(comp, ex1.Current())
                ex1.Next()
            ex1 = TopExp_Explorer(shape, TopAbs_SHELL, TopAbs_SOLID)
            while ex1.More():
                b.Add(comp, ex1.Current())
                ex1.Next()
            ex1 = TopExp_Explorer(shape, TopAbs_FACE, TopAbs_SHELL)
            while ex1.More():
                b.Add(comp, ex1.Current())
                ex1.Next()
            ex1 = TopExp_Explorer(shape, TopAbs_WIRE, TopAbs_FACE)
            while ex1.More():
                b.Add(comp, ex1.Current())
                ex1.Next()
            ex1 = TopExp_Explorer(shape, TopAbs_EDGE, TopAbs_WIRE)
            while ex1.More():
                b.Add(comp, ex1.Current())
                ex1.Next()
            ex1 = TopExp_Explorer(shape, TopAbs_VERTEX, TopAbs_EDGE)
            while ex1.More():
                b.Add(comp, ex1.Current())
                ex1.Next()

        else:
            gids = []
            for i in selection[0]:
                gids.append(VolumeID(i))
            for i in selection[1]:
                gids.append(SurfaceID(i))
            for i in selection[2]:
                gids.append(LineID(i))
            for i in selection[3]:
                gids.append(VertexID(i))

            shapes = [self.get_shape_for_gid(gid) for gid in gids]

            comp = TopoDS_Compound()
            b = self.builder
            b.MakeCompound(comp)
            for s in shapes:
                b.Add(comp, s)

        from OCC.Core.STEPControl import (STEPControl_Writer,
                                          STEPControl_AsIs,
                                          STEPControl_ManifoldSolidBrep,
                                          STEPControl_FacetedBrep,
                                          STEPControl_ShellBasedSurfaceModel,
                                          STEPControl_GeometricCurveSet,)
        from OCC.Core.IFSelect import IFSelect_RetDone, IFSelect_ItemsByEntity

        # On the screen, we use M as unit. x1000 makes it mm
        # which is common in STEP format.

        gt = gp_Trsf()
        pts = gp_Pnt(0., 0., 0.)
        gt.SetScale(pts, 1000.)
        transformer = BRepBuilderAPI_Transform(gt)
        transformer.Perform(comp, True)
        comp1 = transformer.Shape()

        if not stlmode:

            writer = STEPControl_Writer()
            check = Interface_Static.SetIVal("write.step.assembly", 1)

            write_interface_value("write.step.unit", "MM", C=True)

            print("### Exporting Step file", filename)
            read_interface_value("write.precision.mode", I=True)
            read_interface_value("write.precision.val", R=True)
            read_interface_value("write.step.assembly", I=True)
            read_interface_value("write.step.schema", C=True)
            read_interface_value("write.surfacecurve.mode", I=True)
            read_interface_value("write.step.unit", C=True)
            read_interface_value("write.step.vertex.mode", I=True)

            writer.Transfer(comp1, STEPControl_AsIs)
            status = writer.Write(filename)
            if status != IFSelect_RetDone:
                assert False, "failed to write step file"
        else:
            '''
            this is based on
               from OCC.Extend.DataExchange import write_stl_file

               we don't use it as it is, since I suppose we don't need to
               mesh it
            '''
            from OCC.Core.StlAPI import StlAPI_Writer
            stl_exporter = StlAPI_Writer()

            mode = 'ascii'
            if mode == "ascii":
                print("### Exporting STL file (ascii)", filename)
                stl_exporter.SetASCIIMode(True)
            else:  # binary, just set the ASCII flag to False
                print("### Exporting STL file (binary)", filename)
                stl_exporter.SetASCIIMode(False)
            stl_exporter.Write(comp1, filename)

            if not os.path.isfile(filename):
                raise IOError("File not written to disk.")


class OCCGeometryGeneratorBase():
    def __init__(self, q, task_q):
        self.q = q
        self.task_q = task_q
        self.mw = None
        assert hasOCC, "OCC modules are not imported properly"

    def run(self):
        while True:
            time.sleep(0.1)
            try:
                task = self.task_q.get(True)
                self.ready_for_next_task()

            except EOFError:
                self.result_queue.put((-1, None))
                # self.task_queue.task_done()
                continue

            if task[0] == -1:
                # self.task_queue.task_done()
                break

            if task[0] == 2:
                try:
                    ret = self.mw.inspect_geom(*task[1])
                    self.q.put((True, ('success', ret)))
                except BaseException:
                    txt = traceback.format_exc()
                    self.q.put((True, ('fail', txt, None)))

            if task[0] == 3:
                try:
                    #print("exporting", task[1])
                    ret = self.mw.export_shapes(*task[1])
                    self.q.put((True, ('success', ret)))
                except BaseException:
                    txt = traceback.format_exc()
                    self.q.put((True, ('fail', txt)))

            if task[0] == 4:
                try:
                    #print("exporting (STEP/STL)", task[1])
                    ret = self.mw.export_shapes_step(*task[1])
                    self.q.put((True, ('success', ret)))
                except BaseException:
                    txt = traceback.format_exc()
                    self.q.put((True, ('fail', txt)))

            if task[0] == 1:
                try:
                    self.generate_geom(*task[1])
                except BaseException:
                    txt = traceback.format_exc()
                    traceback.print_exc()
                    self.q.put((True, ('fail', txt)))
                    # self.task_queue.task_done()
                    break
        print("exiting prcesss")

    def generate_geom(self, sequence, no_mesh, finalize,
                      filename, start_idx, trash, kwargs):

        kwargs['write_log'] = True
        kwargs['queue'] = self.q
        q = self.q

        if self.mw is None or start_idx == 0:
            from petram.geom.gmsh_geom_model import GeomObjs
            self.mw = Geometry(**kwargs)
            self.objs = GeomObjs()
            self.gui_data = dict()
        else:
            self.mw.process_kwargs(kwargs)

        q.put((self.mw.logfile.name))

        self.mw.geom_sequence = sequence
        self.mw.trash = trash

        self.mw.run_sequence(self.objs, self.gui_data, start_idx)

        if finalize:
            #filename = filename
            brep_file = self.mw.generate_brep(filename, trash,
                                              finalize=True)
        else:
            filename = sequence[-1][0]
            brep_file = self.mw.generate_brep(filename, trash,
                                              finalize=False)

        if no_mesh:
            q.put((True, (self.gui_data, self.objs, brep_file, None, None)))

        else:
            if finalize:
                self.mw.load_finalized_brep(brep_file)
            data = self.mw.generate_preview_mesh()
            # data =  geom_msh, l, s, v,  vcl, esize

            q.put((True, (self.gui_data, self.objs, brep_file, data, None)))


class OCCGeometryGenerator(OCCGeometryGeneratorBase, mp.Process):
    def __init__(self):
        assert hasOCC, "OCC modules are notim ported properly"

        task_q = mp.Queue()  # data to child
        q = mp.Queue()       # data from child
        OCCGeometryGeneratorBase.__init__(self, q, task_q)
        mp.Process.__init__(self)
        dprint1("starting a process for geometry")

    def ready_for_next_task(self):
        pass


class OCCGeometryGeneratorTH(OCCGeometryGeneratorBase, Thread):
    def __init__(self):
        assert hasOCC, "OCC modules are notim ported properly"

        task_q = Queue()  # data to child
        q = Queue()       # data from child
        OCCGeometryGeneratorBase.__init__(self, q, task_q)
        Thread.__init__(self)
        dprint1("starting a thread for geometry")

    def ready_for_next_task(self):
        self.task_q.task_done()
