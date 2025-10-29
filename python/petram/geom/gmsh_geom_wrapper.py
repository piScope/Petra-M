from __future__ import print_function

from petram.geom.geom_id import (GeomIDBase, VertexID, LineID, SurfaceID, VolumeID,
                                 LineLoopID, SurfaceLoopID)
from petram.geom.gmsh_geom_model import get_geom_key
from petram.geom.gmsh_geom_model import GmshPrimitiveBase as GeomPB
from petram.phys.vtable import VtableElement, Vtable
import gmsh
import petram.geom.gmsh_config as gmsh_config

import os
import numpy as np
import time
import tempfile
import traceback
import multiprocessing as mp
from six.moves.queue import Empty as QueueEmpty

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('GmshGeomWrapper')


class Polygon(object):
    def __init__(self, s, ll, lcar):
        self.surface = SurfaceID(s)
        self.line_loop = ll
        self.lcar = lcar


class UniqueCounter(list):
    def add_shape(self, x):
        if x in self:
            return False, self.index(x) + 1
        self.append(x)
        return True, len(self)


def id2dimtag(en):
    if isinstance(en, VertexID):
        return (0, int(en))
    elif isinstance(en, LineID):
        return (1, int(en))
    elif isinstance(en, SurfaceID):
        return (2, int(en))
    elif hasattr(en, 'surface'):
        return (2, int(en))
    elif isinstance(en, VolumeID):
        return (3, int(en))
    else:
        assert False, "Illegal entity" + str(en)


def get_dimtag(entity):
    dimtags = []
    for en in entity:
        dimtags.append(id2dimtag(en))
    return dimtags


def dimtag2id(dimtags):
    out3 = []
    out2 = []
    out1 = []
    out0 = []
    for dim, tag in dimtags:
        if dim == 3 and not tag in out3:
            out3.append(VolumeID(tag))
        elif dim == 2 and not tag in out2:
            out2.append(SurfaceID(tag))
        elif dim == 1 and not tag in out1:
            out1.append(LineID(tag))
        elif dim == 0 and not tag in out1:
            out0.append(VertexID(tag))
    return out3 + out2 + out1 + out0


def get_target1(objs, targets, cls):
    # this is when target type is given
    if cls == 'l':
        cc = LineID
    if cls == 'v':
        cc = VolumeID
    if cls == 'f':
        cc = SurfaceID
    if cls == 'p':
        cc = VertexID

    return [objs[t] if t in objs else cc(t) for t in targets]


def get_target2(objs, targets):
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
    return ret


def find_combined_bbox(model, dimtags):
    xmax = -np.inf
    xmin = np.inf
    ymax = -np.inf
    ymin = np.inf
    zmax = -np.inf
    zmin = np.inf

    def update_maxmin(dim, tag, xmin, ymin, zmin, xmax, ymax, zmax):
        x1, y1, z1, x2, y2, z2 = model.getBoundingBox(dim, tag)
        xmax = np.max([xmax, x2])
        ymax = np.max([ymax, y2])
        zmax = np.max([zmax, z2])
        xmin = np.min([xmin, x1])
        ymin = np.min([ymin, y1])
        zmin = np.min([zmin, z1])
        return xmin, ymin, zmin, xmax, ymax, zmax

    for dim, tag in dimtags:
        xmin, ymin, zmin, xmax, ymax, zmax = update_maxmin(dim, tag,
                                                           xmin, ymin, zmin,
                                                           xmax, ymax, zmax)
    return xmin, ymin, zmin, xmax, ymax, zmax


class Geometry(object):
    def __init__(self, *args, **kwargs):
        self._point_loc = {}

        gmsh.option.setNumber("General.Terminal", 1)
        self.process_kwargs(kwargs)

        modelname = kwargs.pop("modelname", "model1")
        gmsh.clear()
        gmsh.model.add(modelname)

        self.model = gmsh.model
        self.factory = gmsh.model.occ

        #self.p = VertexID(0)
        #self.l = LineID(0)
        #self.ll = LineLoopID(0)
        #self.s = SurfaceID(0)
        #self.sl = SurfaceLoopID(0)
        #self.v = VolumeID(0)
        self.geom_sequence = []

        self._point = {}
        self._point_mask = []
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)

        write_log = kwargs.pop('write_log', False)
        if write_log:
            self.logfile = tempfile.NamedTemporaryFile('w', delete=True)
        else:
            self.logfile = None

        self.queue = kwargs.pop("queue", None)
        self.p = None

    def process_kwargs(self, kwargs):
        self.geom_prev_res = kwargs.pop('PreviewResolution', 30)
        self.geom_prev_algorithm = kwargs.pop('PreviewAlgorithm', 2)
        self.occ_parallel = kwargs.pop('OCCParallel', 0)
        self.maxthreads = kwargs.pop('Maxthreads', 1)
        self.skip_final_frag = kwargs.pop('SkipFrag', False)
        self.use_1d_preview = kwargs.pop('Use1DPreview', False)
        self.use_occ_preview = kwargs.pop('UseOCCPreview', False)
        self.use_curvature = kwargs.pop('UseCurvature', False)
        self.long_edge_thr = kwargs.pop('LongEdgeThr', 0.1)
        self.small_edge_thr = kwargs.pop('SmallEdgeThr', 0.001)
        self.small_edge_seg = kwargs.pop('SmallEdgeSeg', 3)
        self.max_seg = kwargs.pop('MaxSeg', 30)

        gmsh.option.setNumber("Geometry.OCCParallel", self.occ_parallel)

    def set_factory(self, factory_type):
        pass

    def clear(self):
        gmsh.clear()

    def getBoundingBox(self):
        xmax = -np.inf
        xmin = np.inf
        ymax = -np.inf
        ymin = np.inf
        zmax = -np.inf
        zmin = np.inf

        def update_maxmin(dim, tag, xmin, ymin, zmin, xmax, ymax, zmax):
            x1, y1, z1, x2, y2, z2 = self.model.getBoundingBox(dim, tag)
            xmax = np.max([xmax, x2])
            ymax = np.max([ymax, y2])
            zmax = np.max([zmax, z2])
            xmin = np.min([xmin, x1])
            ymin = np.min([ymin, y1])
            zmin = np.min([zmin, z1])
            return xmin, ymin, zmin, xmax, ymax, zmax

        # if (self.model.getEntities(3)) != 0:
        for dim, tag in self.model.getEntities():
            xmin, ymin, zmin, xmax, ymax, zmax = update_maxmin(dim, tag,
                                                               xmin, ymin, zmin,
                                                               xmax, ymax, zmax)
        '''
        elif len(self.model.getEntities(2)) != 0:
           for dim, tag in self.model.getEntities(2):
              xmin, ymin, zmin, xmax, ymax, zmax = update_maxmin(dim, tag,
                                                                 xmin, ymin, zmin,
                                                                 xmax, ymax, zmax)
        elif len(self.model.getEntities(1)) != 0:
           for dim, tag in self.model.getEntities(1):
              xmin, ymin, zmin, xmax, ymax, zmax = update_maxmin(dim, tag,
                                                                 xmin, ymin, zmin,
                                                                 xmax, ymax, zmax)
        else:
           for dim, tag in self.model.getEntities(0):
              xmin, ymin, zmin, xmax, ymax, zmax = update_maxmin(dim, tag,
                                                                 xmin, ymin, zmin,
                                                                 xmax, ymax, zmax)
        '''
        return xmin, ymin, zmin, xmax, ymax, zmax

    def getObjSizes(self):
        size = []
        for dim, tag in self.model.getEntities():
            x1, y1, z1, x2, y2, z2 = self.model.getBoundingBox(dim, tag)
            s = ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)**0.5
            size.append((dim, tag, s))
        return size

    def getVertexCL(self, mincl=0):
        from collections import defaultdict

        lcar = defaultdict(lambda: np.inf)
        esize = {}

        for dim, tag in self.model.getEntities(1):
            x1, y1, z1, x2, y2, z2 = self.model.getBoundingBox(dim, tag)
            s = ((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)**0.5

            esize[tag] = s
            bdimtags = self.model.getBoundary(((dim, tag,),), oriented=False)
            for bdim, btag in bdimtags:
                mm = min((lcar[btag], s))
                if mm > mincl:
                    lcar[btag] = min((lcar[btag], s))

        lcar = dict(lcar)
        for tag in list(lcar):
            if lcar[tag] == np.inf:
                del lcar[tag]
        # for dim, tag in self.model.getEntities(0):
        #    if not tag in lcar:
        #        lcar[tag] = mincl

        return dict(lcar), esize

    def show_only(self, dimtags, recursive=False):
        self.hide_all()
        gmsh.model.setVisibility(dimtags, True, recursive=recursive)

    def hide(self, dimtags, recursive=False):
        gmsh.model.setVisibility(dimtags, False, recursive=recursive)

    def hide_all(self):
        ent = gmsh.model.getEntities()
        gmsh.model.setVisibility(ent, False)

    def show_all(self):
        ent = gmsh.model.getEntities()
        gmsh.model.setVisibility(ent, True)

    def getEntityNumberingInfo(self):
        '''
        Numbering info is collected to understand the change of dimtags
        after brep is saved/loaded. This info may be useful for caching
        the geometry in future?
        '''
        info_digits = 7
        from collections import defaultdict

        # self.factory.removeAllDuplicates()
        self.factory.synchronize()

        points = []
        edges = []
        faces = []
        volumes = []
        for dimtag in self.model.getEntities(3):
            volumes.append(dimtag[1])
            f = self.model.getBoundary(
                [dimtag], combined=False, oriented=False)
            for dim, tag in f:
                if not tag in faces:
                    faces.append(tag)
                e = self.model.getBoundary(
                    [(dim, tag)], combined=False, oriented=False)
                for dim, tag in e:
                    if not tag in edges:
                        edges.append(tag)
                    p = self.model.getBoundary(
                        [(dim, tag)], combined=False, oriented=False)
                    for dim, tag in p:
                        if not tag in points:
                            points.append(tag)
        for dimtag in self.model.getEntities(2):
            dim, tag = dimtag
            if not tag in faces:
                faces.append(tag)
            e = self.model.getBoundary(
                [(dim, tag)], combined=False, oriented=False)
            for dim, tag in e:
                if not tag in edges:
                    edges.append(tag)
                p = self.model.getBoundary(
                    [(dim, tag)], combined=False, oriented=False)
                for dim, tag in p:
                    if not tag in points:
                        points.append(tag)
        for dimtag in self.model.getEntities(1):
            dim, tag = dimtag
            if not tag in edges:
                edges.append(tag)
            p = self.model.getBoundary(
                [(dim, tag)], combined=False, oriented=False)
            for dim, tag in p:
                if not tag in points:
                    points.append(tag)
        for dimtag in self.model.getEntities(0):
            dim, tag = dimtag
            if not tag in points:
                points.append(tag)

        map = points, edges, faces, volumes
        print("entities", points, edges, faces, volumes)
        return map

    def applyEntityNumberingInfo(self, map, objs):
        map1 = self.getEntityNumberingInfo()
        point_map = dict(zip(map[0], map1[0]))
        edge_map = dict(zip(map[1], map1[1]))
        face_map = dict(zip(map[2], map1[2]))
        volume_map = dict(zip(map[3], map1[3]))
        #print("objs", objs)
        for key in objs:
            if isinstance(objs[key], VertexID):
                objs[key] = VertexID(point_map[objs[key]])
            if isinstance(objs[key], LineID):
                objs[key] = LineID(edge_map[objs[key]])
            if isinstance(objs[key], SurfaceID):
                objs[key] = SurfaceID(face_map[objs[key]])
            if isinstance(objs[key], VolumeID):
                objs[key] = VolumeID(volume_map[objs[key]])

    @staticmethod
    def write(filename):
        gmsh.write(filename)

    @staticmethod
    def finalize():
        gmsh.finalize()

    @property
    def dim(self):
        if len(self.model.getEntities(3)) > 0:
            return 3
        if len(self.model.getEntities(2)) > 0:
            return 2
        if len(self.model.getEntities(1)) > 0:
            return 1
        return 0

    def add_point(self, p, lcar=0.0, mask=True):
        p = tuple(p)
        '''
        self.factory.synchronize()
        ptx_tags = gmsh.model.getEntities(0)
        create_new = True
        if len(ptx_tags) > 0:
            points = np.vstack([gmsh.model.getValue(0, x[1], []) for x in ptx_tags])
            dd = np.sum((points-np.array(p))**2, 1)
            dd = np.sqrt(np.sum((points-np.array(p))**2, 1))
            if np.min(dd) == 0:
                idx = np.argmin(dd)
                pp = ptx_tags[idx][1]
                #create_new = False
        #if not p in self._point_loc:
        if create_new:
        '''
        pp = self.factory.addPoint(p[0], p[1], p[2], lcar)
        self._point_loc[p] = VertexID(pp)
        #print("made point ", pp, p)

        p_id = self._point_loc[p]
        self._point[p_id] = np.array(p)
        if mask:
            self._point_mask.append(p_id)
        return p_id

    def add_line(self, p1, p2):
        l = self.factory.addLine(p1, p2)
        return LineID(l)

    def add_circle_arc(self, p2, pc, p3):
        l = self.factory.addCircleArc(p2, pc, p3)
        return LineID(l)

    def add_spline(self, pts, remove_control=True):
        l = self.factory.addSpline(pts)
        if remove_control:
            dimtags = [(0, x) for x in pts[1:-1]]
            self.factory.remove(dimtags)
        return LineID(l)

    def add_plane_surface(self, tags):
        # tags : 1st element exterier, others makes hole
        tags = list(np.atleast_1d(tags))
        s = self.factory.addPlaneSurface(tags)
        return SurfaceID(s)

    def add_surface_filling(self, tags):
        tags = list(np.atleast_1d(tags))
        #print("calling wire", tags)
        self.factory.synchronize()
        # print(gmsh.model.getEntities(1))
        dimtags = [(1, x) for x in tags]

        # reorder tags to make a closed loop
        '''
        this has to be done by coordinates instead of tags !
        corners = [ [yy[1] for yy in self.model.getBoundary((x,), combined=False, oriented=False,)]
                    for x in dimtags]
        order = [0,]
        done_c=[corners[0][0], corners[0][1]]
        while len(order) < len(dimtags):
            #print("order", order)
            for k, c in enumerate(corners):
                if k in order: continue
                if c[0] == done_c[-1] and not c[1] in done_c:
                    done_c.append(c[1])
                    order.append(k)
                    break
                if c[1] == done_c[-1] and not c[0] in done_c:
                    done_c.append(c[0])
                    order.append(k)
                    break
                if ((c[1] == done_c[-1] and c[0] == done_c[0]) or
                    (c[0] == done_c[-1] and c[1] == done_c[0])):
                    done_c.append(c[0])
                    order.append(k)
                    break

            else: # no break here
                assert False, "loop is not closed"
        tags = [tags[x] for x in order]
        '''
        wire = self.factory.addWire(tags)
        self.factory.remove(dimtags, recursive=True)
        self.factory.synchronize()
        # print(wire)
        ent1d = [x[1] for x in gmsh.model.getEntities(1)]
        # print(gmsh.model.getEntities(1))
        s = self.factory.addSurfaceFilling(wire)
        if wire in ent1d:
            self.factory.remove([(1, wire), ], recursive=True)
        self.factory.synchronize()
        #print("boundary", self.model.getBoundary([(2, s)]))
        # print(gmsh.model.getEntities(1))
        return SurfaceID(s)

    def add_line_loop(self, pts, sign=None):
        tags = list(np.atleast_1d(pts))
        if sign is not None:
            for k, v in enumerate(sign):
                if not v:
                    tags[k] = -tags[k]

        # self.factory.synchronize()
        #en1 = self.model.getEntities(1)

        ll = self.factory.addWire(tags, checkClosed=True)

        # self.factory.synchronize()
        #en2 = self.model.getEntities(1)
        # if len(en1) != len(en2):
        #  print("removing", tags[-1])
        #  self.factory.remove(((1, abs(tags[-1])),))

        # (note)
        #   somehow, addWire create a duplicated line sometimes
        #   here I delete input lines to enforce re-numbering.
        #
        dimtags = [(1, x) for x in tags]
        self.factory.remove(dimtags)

        # self.factory.synchronize()
        #en3 = self.model.getEntities(1)
        #print(en1, en2, en3)
        return LineLoopID(ll)

    def add_curve_loop(self, pts, sign=None):
        tags = list(np.atleast_1d(pts))
        if sign is not None:
            for k, v in enumerate(sign):
                if not v:
                    tags[k] = -tags[k]

        ll = self.factory.addCurveLoop(tags)
        return LineLoopID(ll)

    def add_surface_loop(self, sl):
        tags = list(np.atleast_1d(sl))

        sl = self.factory.addSurfaceLoop(tags)
        return SurfaceLoopID(sl)

    def add_sphere(self, x, y, z, radius):
        v = self.factory.addSphere(x, y, z, radius)
        return VolumeID(v)

    def add_cone(self, x, y, z, dx, dy, dz, r1, r2, angle):
        v = self.factory.addCone(x, y, z, dx, dy, dz, r1, r2, angle=angle)
        return VolumeID(v)

    def add_wedge(self, x, y, z, dx, dy, dz, ltx):
        v = self.factory.addWedge(x, y, z, dx, dy, dz, -1, ltx)
        return VolumeID(v)

    def add_cylinder(self, x, y, z, dx, dy, dz, r, angle):
        v = self.factory.addCylinder(x, y, z, dx, dy, dz, r, angle=angle)
        return VolumeID(v)

    def add_torus(self, x, y, z, r1, r2, angle):
        v = self.factory.addTorus(x, y, z, r1, r2, -1, angle)
        return VolumeID(v)

    def add_volume(self, shells):
        tags = list(np.atleast_1d(shells))
        v = self.factory.addVolume(tags)
        return VolumeID(v)

    def add_ellipse_arc(self, startTag, centerTag, endTag):
        a = self._point[startTag] - self._point[centerTag]
        b = self._point[endTag] - self._point[centerTag]
        if np.sum(a * a) > np.sum(b * b):
            l = self.factory.addEllipseArc(startTag, centerTag, endTag)
        else:
            l = self.factory.addEllipseArc(endTag, centerTag, startTag)
        return LineID(l)

    def add_polygon(self, pos, lcar=0.0):
        pts = [self.add_point(p, lcar=lcar) for p in pos]
        lns = [self.add_line(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
        lns.append(self.add_line(pts[-1], pts[0]))
        ll = self.add_line_loop(lns)
        sl = self.add_plane_surface((ll,))
        ret = Polygon(sl, ll, lcar)
        return ret

    def fillet(self, volumes, curves, radii, removeVolume=True):
        volumeTags = list(np.atleast_1d(volumes))
        curveTags = list(np.atleast_1d(curves))
        radii = list(np.atleast_1d(radii))
        outTags = self.factory.fillet(volumeTags,
                                      curveTags,
                                      radii,
                                      removeVolume=removeVolume)
        return [VolumeID(v[1]) for v in outTags]

    def chamfer(self, volumes, curves, surfaces, distances, removeVolume=True):
        volumeTags = list(np.atleast_1d(volumes))
        curveTags = list(np.atleast_1d(curves))
        surfaceTags = list(np.atleast_1d(surfaces))
        distances = list(np.atleast_1d(distances))
        print("surfaceTags", surfaceTags)
        outTags = self.factory.chamfer(volumeTags,
                                       curveTags,
                                       surfaceTags,
                                       distances,
                                       removeVolume=removeVolume)
        return [VolumeID(v[1]) for v in outTags]

    # def import_shapes(self, fileName, highestDimOnly=True, format=""):
    #    out_dimtags = self.factory.importShapes(fileName,
    #                                            highestDimOnly=highestDimOnly,
    #                                            format="")
    #    return dimtag2id(out_dimtags)

    def _boolean_xxx(self, m, input_entity, tool_entity,
                     removeObject=False, removeTool=False,
                     delete=False):

        dimtag1 = get_dimtag(input_entity)
        dimtag2 = get_dimtag(tool_entity)

        if delete:
            removeObject = True
            removeTool = True
        # self.factory.synchronize()
        #print("before", self.model.getEntities(), dimtag1, dimtag2)
        m = getattr(self.factory, m)
        dimtag3, dimtagMap = m(dimtag1, dimtag2,
                               removeObject=removeObject,
                               removeTool=removeTool)
        #print("dimtag3, map", dimtag3, dimtagMap)
        self.factory.synchronize()
        # print(self.model.getEntities())
        return dimtag2id(dimtag3)

    def boolean_intersection(self, input_entity, tool_entity,
                             removeObject=False, removeTool=False, delete=False):
        return self._boolean_xxx('intersect', input_entity, tool_entity,
                                 removeObject=removeObject, removeTool=removeTool,
                                 delete=delete)

    def boolean_union(self, input_entity, tool_entity,
                      removeObject=False, removeTool=False, delete=False):
        return self._boolean_xxx('fuse', input_entity, tool_entity,
                                 removeObject=removeObject, removeTool=removeTool,
                                 delete=delete)

    def boolean_difference(self, input_entity, tool_entity,
                           removeObject=False, removeTool=False, delete=False):
        return self._boolean_xxx('cut', input_entity, tool_entity,
                                 removeObject=removeObject, removeTool=removeTool,
                                 delete=delete)

    def boolean_fragments(self, input_entity, tool_entity,
                          removeObject=False, removeTool=False, delete=False):
        return self._boolean_xxx('fragment', input_entity, tool_entity,
                                 removeObject=removeObject, removeTool=removeTool,
                                 delete=delete)

    def boolean_union2d(self, input_entity, tool_entity,
                        removeObject=False, removeTool=False, delete=False):

        def get_dimtag(entity):
            dimtags = []
            for en in entity:
                dimtags.append(id2dimtag(en))
            return dimtags

        all_entity = input_entity + tool_entity
        out_entity = self._boolean_xxx('fuse', input_entity, tool_entity,
                                       removeObject=removeObject, removeTool=removeTool,
                                       delete=delete)

        self.factory.synchronize()

        out_dimtag = get_dimtag(out_entity)
        xmin, ymin, zmin, xmax, ymax, zmax = find_combined_bbox(
            self.model, out_dimtag)

        dprint1("bounding box", xmin, ymin, zmin, xmax, ymax, zmax)

        dx = xmax - xmin
        dy = ymax - ymin
        bbx = self.factory.addRectangle(xmin - dx / 10., ymin - dy / 10., (zmin + zmax) / 2.,
                                        dx * 1.2, dy * 1.2)
        out_dimtag2, dimtagMap = self.factory.cut(((2, bbx),), out_dimtag)
        # print(out_dimtag2)
        bbx = self.factory.addRectangle(xmin - dx / 10., ymin - dy / 10., (zmin + zmax) / 2.,
                                        dx * 1.2, dy * 1.2)
        out_dimtag3, dimtagMap = self.factory.cut(((2, bbx),), out_dimtag2)
        self.factory.synchronize()
        return dimtag2id(out_dimtag3)

    def apply_fragments(self):
        self.factory.synchronize()
        if self.dim == 0:
            return

        dimtags = self.model.getEntities(self.dim)
        if len(dimtags) > 1:
            if self.logfile is not None:
                self.logfile.write("computing  fragments\n")
            if self.queue is not None:
                self.queue.put((False, "computing  fragments"))

            self.factory.synchronize()
            self.factory.fragment(dimtags[:1], dimtags[1:],
                                  removeObject=True, removeTool=True)

            self.factory.synchronize()
            self.factory.removeAllDuplicates()

            return
        '''
        ## since self.dim returns the highest dim in geometry
        ## we dont need this
        if self.dim > 1:
           dimtags =  self.model.getEntities(self.dim-1)
           print("here2", dimtags)
           if len(dimtags) > 1:
               self.factory.fragment(dimtags[:1], dimtags[1:],
                                  removeObject=True, removeTool=True)
               self.factory.synchronize()
               self.factory.removeAllDuplicates()
               return

        if self.dim > 2:
           dimtags =  self.model.getEntities(self.dim-2)
           if len(dimtags) > 1:
               self.factory.fragment(dimtags[:1], dimtags[1:],
                                  removeObject=True, removeTool=True)
               self.factory.synchronize()
               self.factory.removeAllDuplicates()                                          '''

    def remove(self, entity, recursive=False):
        dimtags = []
        for en in entity:
            dimtags.append(id2dimtag(en))
        self.factory.remove(dimtags, recursive=recursive)
        return []

    def inverse_remove(self, entity):
        dimtags = []
        for en in entity:
            dimtags.append(id2dimtag(en))

        if len(set([x[0] for x in dimtags])) != 1:
            assert False, "Can not choose objects with different dims"
        dim = dimtags[0][0]
        tags = [x[1] for x in dimtags]

        self.factory.synchronize()
        if dim == 3:
            ent3d = [x for x in self.model.getEntities(3) if not x[1] in tags]
            self.factory.remove(ent3d, recursive=True)
        elif dim == 2:
            ent3d = self.model.getEntities(3)
            self.factory.remove(ent3d, recursive=False)
            ent2d = [x for x in self.model.getEntities(2) if not x[1] in tags]
            self.factory.remove(ent2d, recursive=True)
        elif dim == 1:
            ent3d = self.model.getEntities(3)
            self.factory.remove(ent3d, recursive=False)
            ent2d = self.model.getEntities(2)
            self.factory.remove(ent2d, recursive=False)
            ent1d = [x for x in self.model.getEntities(1) if not x[1] in tags]
            self.factory.remove(ent1d, recursive=True)
        else:  # dim == 0:
            ent3d = self.model.getEntities(3)
            self.factory.remove(ent3d, recursive=False)
            ent2d = self.model.getEntities(2)
            self.factory.remove(ent2d, recursive=False)
            ent1d = self.model.getEntities(1)
            self.factory.remove(ent1d, recursive=False)
            ent0d = [x for x in self.model.getEntities(0) if not x[0] in tags]
            self.factory.remove(ent0d, recursive=True)
        self.factory.synchronize()
        return dimtag2id(dimtags)

    def copy(self, entity):
        dimtags = []
        for en in entity:
            dimtags.append(id2dimtag(en))
        dimtags2 = self.factory.copy(dimtags)
        return dimtag2id(dimtags2)

    def rotate(self, entity, x, y, z, ax, ay, az, angle):
        dimtags = []
        for en in entity:
            dimtags.append(id2dimtag(en))
        self.factory.rotate(dimtags, x, y, z, ax, ay, az, angle)
        return []

    def translate(self, entity, dx, dy, dz):
        dimtags = []
        for en in entity:
            dimtags.append(id2dimtag(en))
        self.factory.translate(dimtags, dx, dy, dz)
        return []

    def dilate(self, entity, x, y, z, a, b, c):
        dimtags = []
        for en in entity:
            dimtags.append(id2dimtag(en))
        self.factory.dilate(dimtags, x, y, z, a, b, c)
        return []

    def symmetrize(self, entity, a, b, c, d):
        dimtags = []
        for en in entity:
            dimtags.append(id2dimtag(en))
        self.factory.symmetrize(dimtags, a, b, c, d)
        return []

    def extrude(self, entity, translation_axis=None,
                point_on_axis=None,
                rotation_axis=None,
                angle=0):

        # for en in entity:
        dimtags = [id2dimtag(entity)]
        if translation_axis is not None:
            tax = translation_axis
            dimtags2 = self.factory.extrude(dimtags, tax[0], tax[1], tax[2],)
        else:
            pax = point_on_axis
            rax = rotation_axis
            dimtags2 = self.factory.revolve(dimtags, pax[0], pax[1], pax[2],
                                            rax[0], rax[1], rax[2], angle)
        dprint1("extrude out", dimtags2)
        return dimtag2id(dimtags2)

    def call_synchronize(self):
        self.factory.synchronize()
        #self.p = len(self.model.getEntities(0))

    def recursive_getBdry(self, dimtag, bdr=None):
        if bdr is None:
            bdr = []
        bb = self.model.getBoundary((dimtag,), oriented=False,)
        bdr.extend(bb)

        for x in bb:
            if x[0] > 0:
                bdr = self.recursive_getBdry(x, bdr=bdr)
        return bdr

    def get_unique_entity(self, entity):
        entity = [x for x in entity if isinstance(x, GeomIDBase)
                  and not isinstance(x, LineLoopID)
                  and not isinstance(x, SurfaceLoopID)]
        dimtags = get_dimtag(entity)
        dimtags = [x for x in sorted(dimtags)]

        self.factory.synchronize()

        allent = self.model.getEntities()
        #print('all', allent)
        dimtags = [x for x in dimtags if x in allent]
        outdimtags = dimtags[:]

        #print('input', dimtags)
        for dimtag in reversed(dimtags):
            bdimtags = self.recursive_getBdry(dimtag)
            #print("checking", dimtag, bdimtags)
            for x in bdimtags:
                if x in outdimtags:
                    idx = outdimtags.index(x)
                    del outdimtags[idx]
                if not x in allent:
                    idx = outdimtags.index(x)
                    del outdimtags[idx]
        #print('output', outdimtags)
        return dimtag2id(outdimtags)

    def add_box(self, points):
        p1, p2, p3, p4, p5, p6, p7, p8 = points
        lcar = 0.0

        p1 = self.add_point(p1, lcar)
        p2 = self.add_point(p2, lcar)
        p3 = self.add_point(p3, lcar)
        p4 = self.add_point(p4, lcar)
        p5 = self.add_point(p5, lcar)
        p6 = self.add_point(p6, lcar)
        p7 = self.add_point(p7, lcar)
        p8 = self.add_point(p8, lcar)

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

    '''
    high level interface:
      methods below directroy corresponds to GUI interface
    '''

    def add_sequence(self, gui_name, gui_param, geom_name):
        self.geom_sequence.append((gui_name, gui_param, geom_name))

    def Point_build_geom(self, objs, *args):
        xarr, yarr, zarr = args
        lcar = 0.0
        try:
            pos = np.vstack((xarr, yarr, zarr)).transpose()
        except BaseException:
            print("can not make proper input array")
            return
        PTs = [self.add_point(p, lcar=lcar) for p in pos]
        # apparently I should use this object (poly.surface)...?
        _newobjs = []
        for p in PTs:
            newkey = objs.addobj(p, 'pt')
            _newobjs.append(newkey)

        return list(objs), _newobjs

    def Line_build_geom(self, objs, *args):
        xarr, yarr, zarr, make_spline, periodic = args
        lcar = 0.0
        if len(xarr) < 2:
            return
        try:
            pos = np.vstack((xarr, yarr, zarr)).transpose()
        except BaseException:
            print("can not make proper input array")
            return

        dist = np.sqrt(np.sum((pos[:-1, :] - pos[1:, :])**2, 1))
        print(dist)
        if min(dist) == 0.0:
            assert False, "minimum distance between point is 0.0"
        if max(dist) > min(dist) * 1e4:
            assert False, "some points are too close (d_max > d_min*1e4)"

        pt1 = self.add_point(pos[0], lcar, mask=True)
        pt2 = self.add_point(pos[-1], lcar, mask=True)

        pts = [pt1]
        for ii, p in enumerate(pos[1:-1]):
            pt = self.add_point(p, lcar, mask=False)
            pts.append(pt)
        pts.append(pt2)

        if not make_spline:
            pts1 = pts[:-1]
            pts2 = pts[1:]

            newkeys = []
            for p1, p2 in zip(pts1, pts2):
                ln = self.add_line(p1, p2)
                newkeys.append(objs.addobj(ln, 'ln'))
            _newobjs = newkeys
        else:
            spline = self.add_spline(pts, remove_control=True)
            newobj = objs.addobj(spline, 'sp')
            _newobjs = [newobj]

        newobj1 = objs.addobj(pts[0], 'pt')
        newobj2 = objs.addobj(pts[-1], 'pt')

        _newobjs.append(newobj1)
        _newobjs.append(newobj2)

        return list(objs), _newobjs

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
        poly = self.add_polygon(pos, lcar=lcar)

        # apparently I should use this object (poly.surface)...?
        newkey = objs.addobj(poly.surface, 'pol')
        return list(objs), [newkey]

    def Spline_build_geom(self, objs, *args):
        pts = args
        pts = [x.strip() for x in pts[0].split(',')]

        #pts = [objs[x] for x in pts]
        pts = get_target1(objs, pts, 'p')
        spline = self.add_spline(pts)
        newkey = objs.addobj(spline, 'sp')

        return list(objs), [newkey]

    def CreateLine_build_geom(self, objs, *args):
        pts = args
        pts = [x.strip() for x in pts[0].split(',')]
        pts = get_target1(objs, pts, 'p')
        pts0 = pts[:-1]
        pts1 = pts[1:]

        newkeys = []

        for p0, p1 in zip(pts0, pts1):
            # if not p0 in objs:
            #    assert False, p0 + " does not exist"
            # if not p1 in objs:
            #    assert False, p1 + " does not exist"
            line = self.add_line(p0, p1)
            newkeys.append(objs.addobj(line, 'ln'))

        return list(objs), newkeys

    def LineLoop_build_geom(self, objs, *args):
        pts = args
        pts = [x.strip() for x in pts[0].split(',')]

        ptx = get_target1(objs, pts, 'l')
        #pts = [(objs[x] if not x.startswith('-') else objs[x[1:]]) for x in pts]
        for x in pts:
            if x.startswith('-'):
                if x[1:] in objs:
                    del objs[x[1:]]
            else:
                if x in objs:
                    del objs[x]

        lloop = self.add_line_loop(ptx)
        newkey = objs.addobj(lloop, 'll')

        return list(objs), [newkey]

    def CreateSurface_build_geom(self, objs, *args):
        pts, points, isFilling = args
        pts = [x.strip() for x in pts.split(',')]

        ptx = get_target1(objs, pts, 'l')
        #pts = [(objs[x] if not x.startswith('-') else objs[x[1:]]) for x in pts]
        for x in pts:
            if x.startswith('-'):
                if x[1:] in objs:
                    del objs[x[1:]]
            else:
                if x in objs:
                    del objs[x]

        #objid = [(objs[x] if not x.startswith('-') else objs[x[1:]]) for x in pts]
        #objsign = [not x.startswith('-') for x in pts]
        # for x in pts:
        #   if x.startswith('-'): del objs[x[1:]]
        #   else: del objs[x]

        if isFilling:
            surface = self.add_surface_filling(ptx)
            newobj2 = objs.addobj(surface, 'sf')
#           newobj1 = objs.addobj(line, 'ln')
#           newkeys = [newobj2, newobj1]
#           surface = self.add_surface_filling(ptx)
#
            newkeys = [newobj2, ]
        else:
            ll = self.add_line_loop(ptx)
            #newobj1 = objs.addobj(ll, 'll')
            surface = self.add_plane_surface(ll)
            newobj2 = objs.addobj(surface, 'ps')
            newkeys = [newobj2]

        return list(objs), newkeys

    def SurfaceLoop_build_geom(self, objs, *args):
        pts = args
        pts = [x.strip() for x in pts[0].split(',')]
        #pts = [(objs[x] if not x.startswith('-') else objs[x[1:]]) for x in pts]
        ptx = get_target1(objs, pts, 'f')
        sl = self.add_surface_loop(ptx)
        newobj = objs.addobj(sl, 'sl')

        return list(objs), [newobj]

    def CreateVolume_build_geom(self, objs, *args):
        pts = args
        pts = [x.strip() for x in pts[0].split(',')]

        ptx = get_target1(objs, pts, 'f')
        sl = self.add_surface_loop(ptx)

        self.factory.remove([(2, x) for x in ptx], recursive=True)
        self.factory.synchronize()

        #newobj1 = objs.addobj(sl, 'sl')
        vol = self.add_volume(sl)
        newobj2 = objs.addobj(vol, 'vol')

        return list(objs), [newobj2]

    def Rect_build_geom(self, objs, *args):
        c1, e1, e2 = args
        lcar = 0.0

        c1 = np.array(c1)
        e1 = np.array(e1)
        e2 = np.array(e2)
        p1 = self.add_point(c1, lcar)
        p2 = self.add_point(c1 + e1, lcar)
        p3 = self.add_point(c1 + e1 + e2, lcar)
        p4 = self.add_point(c1 + e2, lcar)
        l1 = self.add_line(p1, p2)
        l2 = self.add_line(p2, p3)
        l3 = self.add_line(p3, p4)
        l4 = self.add_line(p4, p1)
        ll1 = self.add_line_loop([l1, l2, l3, l4])
        rec1 = self.add_plane_surface(ll1)
        newkey = objs.addobj(rec1, 'rec')
        return list(objs), [newkey]

    def Circle_build_geom(self, objs, *args):
        center, ax1, ax2, radius, make_face = args        
        lcar = 0.0
        a1 = np.array(ax1)
        a2 = np.array(ax2)
        a2 = np.cross(np.cross(a1, a2), a1)
        a1 = a1 / np.sqrt(np.sum(a1**2)) * radius
        a2 = a2 / np.sqrt(np.sum(a2**2)) * radius

        c = np.array(center)
        p1 = self.add_point(c + a1, lcar)
        p2 = self.add_point(c + a2, lcar)
        p3 = self.add_point(c - a1, lcar)
        p4 = self.add_point(c - a2, lcar)
        pc = self.add_point(c, lcar)
        ca1 = self.add_circle_arc(p1, pc, p2)
        ca2 = self.add_circle_arc(p2, pc, p3)
        ca3 = self.add_circle_arc(p3, pc, p4)
        ca4 = self.add_circle_arc(p4, pc, p1)
        ll1 = self.add_line_loop([ca1, ca2, ca3, ca4])
        ps1 = self.add_plane_surface(ll1)
        newkey = objs.addobj(ps1, 'ps')
        return list(objs), [newkey]

        #self._objkeys = objs.keys()
        #self._newobjs = [newkey]

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

        '''
        p1 = self.add_point(c1, lcar)
        p2 = self.add_point(c1+e1, lcar)
        p3 = self.add_point(c1+e2, lcar)
        p4 = self.add_point(c1+e3, lcar)
        p5 = self.add_point(c1+e1+e2, lcar)
        p6 = self.add_point(c1+e2+e3, lcar)
        p7 = self.add_point(c1+e3+e1, lcar)
        p8 = self.add_point(c1+e3+e2+e1, lcar)
        '''
        v1 = self.add_box((p1, p2, p3, p4, p5, p6, p7, p8,))

        newkey = objs.addobj(v1, 'bx')

        return list(objs), [newkey]

    def Ball_build_geom(self, objs, *args):
        self.factory.synchronize()

        x0, l1, l2, l3, a1, a2, a3 = args
        lcar = 0.0
        radii = [l1, l2, l3]
        rr = min(radii)

        volumes = []

        v1 = self.factory.addSphere(x0[0], x0[1], x0[2], rr,
                                    angle1=a1 / 180 * np.pi, angle2=a2 / 180 * np.pi, angle3=a3 / 180 * np.pi)
        if (l1 / rr != 1.0 or l2 / rr != 1.0 or l3 / rr != 1.0):
            self.dilate([v1], x0[0], x0[1], x0[2], l1 / rr, l2 / rr, l3 / rr)
        v1 = VolumeID(v1)
        newkey = objs.addobj(v1, 'bl')
        return list(objs), [newkey]

        '''
        v1 = self.factory.addSphere(x0[0], x0[1], x0[2], rr, angle1=-np.pi, angle2 = 0)
        #v2 = self.factory.addSphere(x, y, z, radius, angle1= 0, angle2 = np.pi)
        #v1 = self.add_sphere(x0[0], x0[1], x0[2], rr)

        if (l1/rr != 1.0 or l2/rr != 1.0 or l3/rr != 1.0):
            self.dilate([v1], x0[0], x0[1], x0[2], l1/rr, l2/rr, l3/rr)

        self.factory.synchronize()
        gmsh.write('hoge.brep')
        newkey = objs.addobj(v1, 'bl')
        return  objs.keys(), [newkey]
        '''
        '''
        pc = self.add_point(x0, lcar)
        p1 = self.add_point([x0[0]+rr, x0[1], x0[2]], lcar=lcar)
        p2 = self.add_point([x0[0], x0[1]+rr, x0[2]], lcar=lcar)
        p3 = self.add_point([x0[0]-rr, x0[1], x0[2]], lcar=lcar)
        ca1 = self.add_circle_arc(p1, pc, p2)
        ca2 = self.add_circle_arc(p2, pc, p3)
        ln1 = self.add_line(p3, p1)
        ll1 = self.add_line_loop([ca1, ca2, ln1])
        ps1 = self.add_plane_surface(ll1)
        dst = [id2dimtag(ps1), ]


        for i in range(4):
           ret = self.factory.revolve(dst,
                                   x0[0], x0[1], x0[2],
                                    1, 0, 0, np.pi/2.)
           dst = ret[:1]
           volumes.append(ret[1])

        ret = self.factory.fuse(volumes[:1], volumes[1:])
        v1 = VolumeID(ret[0][0][1])

        if (l1/rr != 1.0 or l2/rr != 1.0 or l3/rr != 1.0):
            self.dilate([v1], x0[0], x0[1], x0[2], l1/rr, l2/rr, l3/rr)
        '''

    def Cone_build_geom(self, objs, *args):
        x0, d0, r1, r2, angle = args

        #an = angle if angle < 180 else angle/2

        v1 = self.add_cone(x0[0], x0[1], x0[2], d0[0], d0[1], d0[2],
                           r1, r2, angle / 180 * np.pi)
        '''
        if angle >=180:
           v2 = self.add_cone(x0[0], x0[1], x0[2], d0[0], d0[1], d0[2],
                              r1, r2,  an/180*np.pi)
           v2 = [id2dimtag(v2), ]
           self.factory.rotate(v2, x0[0], x0[1], x0[2],
                               d0[0], d0[1], d0[2],  an/180*np.pi)
           v1 = [id2dimtag(v1), ]
           ret = self.factory.fuse(v1, v2)
           v1 = VolumeID(ret[0][0][1])
        '''
        v1 = VolumeID(v1)
        newkey = objs.addobj(v1, 'cn')
        return list(objs), [newkey]

    def Cylinder_build_geom(self, objs, *args):
        x0, d0, r1, angle = args
        lcar = 0.0
        d0 = np.array(d0)
        if np.sum(d0 * np.array([1, 0, 0])) > np.sum(d0 * np.array([0, 1, 0])):
            a1 = np.cross(d0, [0, 1, 0])
        else:
            a1 = np.cross(d0, [1, 0, 0])
        a2 = np.cross(d0, a1)

        a1 = a1 / np.sqrt(np.sum(a1**2)) * r1
        a2 = a2 / np.sqrt(np.sum(a2**2)) * r1

        c = np.array(x0)
        p1 = self.add_point(c + a1, lcar)
        p2 = self.add_point(c + a2, lcar)
        p3 = self.add_point(c - a1, lcar)
        p4 = self.add_point(c - a2, lcar)
        pc = self.add_point(c, lcar)
        ca1 = self.add_circle_arc(p1, pc, p2)
        ca2 = self.add_circle_arc(p2, pc, p3)
        ca3 = self.add_circle_arc(p3, pc, p4)
        ca4 = self.add_circle_arc(p4, pc, p1)
        ll1 = self.add_line_loop([ca1, ca2, ca3, ca4])
        ps1 = self.add_plane_surface(ll1)

        ret = self.extrude(ps1, translation_axis=d0,)
        newkey = objs.addobj(ret[0], 'cn')
        return list(objs), [newkey]

    def Wedge_build_geom(self, objs, *args):
        x0, d0, ltx = args
        v1 = self.add_wedge(x0[0], x0[1], x0[2], d0[0], d0[1], d0[2], ltx)

        newkey = objs.addobj(v1, 'wg')
        return list(objs), [newkey]

    def Torus_build_geom(self, objs, *args):
        x0, r1, r2, angle, keep_interior = args

        lcar = 0.0
        a1 = np.array([r2, 0, 0])
        a2 = np.array([0, 0, r2])

        c = np.array(x0) + np.array([r1, 0, 0])
        p1 = self.add_point(c + a1, lcar)
        p2 = self.add_point(c + a2, lcar)
        p3 = self.add_point(c - a1, lcar)
        p4 = self.add_point(c - a2, lcar)
        pc = self.add_point(c, lcar)
        ca1 = self.add_circle_arc(p1, pc, p2)
        ca2 = self.add_circle_arc(p2, pc, p3)
        ca3 = self.add_circle_arc(p3, pc, p4)
        ca4 = self.add_circle_arc(p4, pc, p1)
        ll1 = self.add_line_loop([ca1, ca2, ca3, ca4])
        ps1 = self.add_plane_surface(ll1)

        dst = [id2dimtag(ps1), ]
        volumes = []

        if abs(angle) > 270:
            seg = 4
        elif abs(angle) > 180:
            seg = 3
        elif abs(angle) > 90:
            seg = 2
        else:
            seg = 1

        an = angle / seg

        for i in range(seg):
            ret = self.factory.revolve(dst,
                                       x0[0], x0[1], x0[2],
                                       0, 0, 1, an * np.pi / 180.)
            dst = ret[:1]
            volumes.append(ret[1])

        if keep_interior:
            newkey = []
            for v in volumes:
                v1 = VolumeID(v[1])
                newkey.append(objs.addobj(v1, 'trs'))
        else:
            if seg > 1:
                ret = self.factory.fuse(volumes[:1], volumes[1:])
                v1 = VolumeID(ret[0][0][1])
            else:
                v1 = VolumeID(ret[1][1])

            newkey = [objs.addobj(v1, 'trs')]
        return list(objs), newkey

    def Extrude_build_geom(self, objs, *args):
        targets, tax, lengths = args

        length = lengths[0]
        targets = [x.strip() for x in targets.split(',')]
        targetID = get_target2(objs, targets)

        print("tax", tax)
        if tax[0] == 'normal' or tax[0] == 'normalp':
            assert isinstance(targetID[0], SurfaceID), "target must be surface"
            self.factory.synchronize()
            n1 = np.array(gmsh.model.getNormal(targetID[0], (0, 0)))
            n2 = np.array(gmsh.model.getNormal(targetID[0], (0, 1)))
            n3 = np.array(gmsh.model.getNormal(targetID[0], (1, 0)))
            n1 /= np.sqrt(np.sum(n1**2))
            n2 /= np.sqrt(np.sum(n2**2))
            n3 /= np.sqrt(np.sum(n3**2))

            if np.any(n1 != n2) or np.any(n1 != n3):
                assert False, "surface is not flat"

            if tax[0] == 'normal':
                if tax[1]:
                    tax = -n1 * length
                else:
                    tax = n1 * length
            else:
                destID = get_target1(objs, [tax[1], ], 'p')[0]
                ptx_d = np.array(gmsh.model.getValue(0, int(destID), []))
                dimtags = self.model.getBoundary(((2, targetID[0]),), recursive=True,
                                                 combined=True, oriented=False,)
                print("dimtags", dimtags)
                ptx_s = np.array(gmsh.model.getValue(0, dimtags[0][1], []))
                if tax[2]:
                    tax = -n1 * np.sum((ptx_d - ptx_s) * n1) * length
                else:
                    tax = n1 * np.sum((ptx_d - ptx_s) * n1) * length

        elif tax[0] == 'fromto_points':
            self.factory.synchronize()
            ptx1ID = get_target1(objs, [tax[1], ], 'p')[0]
            ptx1 = np.array(gmsh.model.getValue(0, int(ptx1ID), []))
            ptx2ID = get_target1(objs, [tax[2], ], 'p')[0]
            ptx2 = np.array(gmsh.model.getValue(0, int(ptx2ID), []))
            print('ptx', ptx1, ptx2)
            n1 = ptx2 - ptx1
            if not tax[3]:
                n1 /= np.sqrt(np.sum(n1**2))
                n1 *= length
            if tax[4]:
                n1 *= -1
            tax = n1
        else:
            tax = np.array(tax).flatten()
            tax = tax / np.sqrt(np.sum(np.array(tax)**2)) * length
        newkeys = []
        for t, idd in zip(targets, targetID):
            # if not t in objs:
            #    assert False, t + " does not exist"
            ret = self.extrude(idd,
                               translation_axis=tax,)
            # rotation_axis=rax,
            # point_on_axis=pax
            from petram.geom.gmsh_geom_model import use_gmsh_api

            newkeys.append(objs.addobj(ret[1], t))
            newkeys.append(objs.addobj(ret[0], 'ex'))

        return list(objs), newkeys

    def Revolve_build_geom(self, objs, *args):

        targets, params, angles = args
        
        if params[0] == 'xyz':
            rax = [float(x) for x in params[1]]
            pax = [float(x) for x in params[2]]
        else:
            assert False, "Not supported in GMSH geometry"

        angle = angles[0]
        
        targets = [x.strip() for x in targets.split(',')]
        targetID = get_target2(objs, targets)

        newkeys = []
        for t, idd in zip(targets, targetID):
            # if not t in objs:
            #    assert False, t + " does not exist"
            ret = self.extrude(idd,
                               rotation_axis=rax,
                               point_on_axis=pax,
                               angle=angle * np.pi / 180.)

            from petram.geom.gmsh_geom_model import use_gmsh_api

            newkeys.append(objs.addobj(ret[1], t))
            newkeys.append(objs.addobj(ret[0], 'ex'))

        return list(objs), newkeys

    def Sweep_build_geom(self, objs, *args):
        print("objs", objs)
        targets, lines = args
        targets = [x.strip() for x in targets.split(',')]
        targetID = get_target2(objs, targets)
        lines = [x.strip() for x in lines.split(',')]
        lineID = get_target2(objs, lines)

        newkeys = []
        for t, idd in zip(targets, targetID):
            # if not t in objs:
            #    assert False, t + " does not exist"
            #dimtags = [id2dimtag(wireID)]
            wire = self.factory.addWire(lineID)
            dimtags = [id2dimtag(idd)]
            ret = self.factory.addPipe(dimtags, wire)
            newkeys.append(objs.addobj(ret[0], 'ex'))

        return list(objs), newkeys

    def Move_build_geom(self, objs, *args):
        targets, dx, dy, dz, keep = args
        targets = [x.strip() for x in targets.split(',')]

        newkeys = []
        tt = get_target2(objs, targets)

        if keep:
            tt = self.copy(tt)
        self.translate(tt, dx, dy, dz)
        if keep:
            for t in tt:
                newkeys.append(objs.addobj(t, 'mv'))

        return list(objs), newkeys

    def Rotate_build_geom(self, objs, *args):
        targets, cc, aa, angle, keep = args
        cx, cy, cz = cc
        ax, ay, az = aa
        targets = [x.strip() for x in targets.split(',')]

        newkeys = []
        tt = get_target2(objs, targets)

        if keep:
            tt = self.copy(tt)
        self.rotate(tt, cx, cy, cz, ax, ay, az, np.pi * angle / 180.)
        if keep:
            for t in tt:
                newkeys.append(objs.addobj(t, 'rot'))
        return list(objs), newkeys

    def Scale_build_geom(self, objs, *args):
        targets, cc, ss, keep = args
        cx, cy, cz = cc
        sx, sy, sz = ss
        targets = [x.strip() for x in targets.split(',')]

        newkeys = []
        tt = get_target2(objs, targets)

        if keep:
            tt = self.copy(tt)
        self.dilate(tt, cx, cy, cz, sx, sy, sz)
        if keep:
            for t in tt:
                newkeys.append(objs.addobj(t, 'sc'))

        return list(objs), newkeys

    def Array_build_geom(self, objs, *args):
        targets, count, displacement = args
        dx, dy, dz = displacement
        targets = [x.strip() for x in targets.split(',')]

        newkeys = []
        tt = get_target2(objs, targets)

        for i in range(count):
            tt = self.copy(tt)
            self.translate(tt, dx, dy, dz)
            for t in tt:
                newkeys.append(objs.addobj(t, 'cp'))

        return list(objs), newkeys

    def ArrayRot_build_geom(self, objs, *args):
        targets, count, cc, aa, angle = args
        cx, cy, cz = cc
        ax, ay, az = aa
        targets = [x.strip() for x in targets.split(',')]

        newkeys = []
        tt = get_target2(objs, targets)

        for i in range(count):
            tt = self.copy(tt)
            self.rotate(tt, cx, cy, cz, ax, ay, az, np.pi * angle / 180.)
            for t in tt:
                newkeys.append(objs.addobj(t, 'cp'))

        return list(objs), newkeys

    def Flip_build_geom(self, objs, *args):
        targets, a, b, c, d, keep = args
        targets = [x.strip() for x in targets.split(',')]

        newkeys = []
        tt = get_target2(objs, targets)

        if keep:
            tt = self.copy(tt)
        self.symmetrize(tt, a, b, c, d)
        if keep:
            for t in tt:
                newkeys.append(objs.addobj(t, 'flp'))

        return list(objs), newkeys

    def Fillet_build_geom(self, objs, *args):
        volumes, curves, radius = args
        volumes = [x.strip() for x in volumes.split(',')]
        curves = [x.strip() for x in curves.split(',')]

        radii = [radius]

        volumes = get_target1(objs, volumes, 'v')
        curves = get_target1(objs, curves, 'v')

        ret = self.fillet(volumes, curves, radii, removeVolume=True)
        newkeys = []
        for r in ret:
            newkeys.append(objs.addobj(r, 'vol'))

        return list(objs), newkeys

    def Chamfer_build_geom(self, objs, *args):
        volumes, curves, distances, surfaces = args

        volumes = [x.strip() for x in volumes.split(',')]
        curves = [x.strip() for x in curves.split(',')]
        surfaces = [x.strip() for x in surfaces.split(',')]

        volumes = [objs[t] if t in objs else int(t) for t in volumes]
        curves = [objs[t] if t in objs else int(t) for t in curves]
        surfaces = [objs[t] if t in objs else int(t) for t in surfaces]
        ret = self.chamfer(
            volumes,
            curves,
            surfaces,
            distances,
            removeVolume=True)
        newkeys = []
        for r in ret:
            newkeys.append(objs.addobj(r, 'vol'))

        return list(objs), newkeys

    def Copy_build_geom(self, objs, *args):
        targets = args[0]
        targets = [x.strip() for x in targets.split(',')]

        newkeys = []
        tt = get_target2(objs, targets)
        ret = self.copy(tt)
        for r in ret:
            newkeys.append(objs.addobj(r, 'cp'))

        return list(objs), newkeys

    def do_remove_build_geom(self, objs, targets, recursive, inverse_sel):
        targets = [x.strip() for x in targets.split(',')]

        newkeys = []
        tt = get_target2(objs, targets)

        if inverse_sel:
            ret = self.inverse_remove(tt)
            for t in list(objs):
                del objs[t]
            for rr in ret:
                newkeys.append(objs.addobj(rr, 'kpt'))
        else:
            self.remove(tt, recursive=recursive)

            for t in targets:
                if t in objs:
                    del objs[t]

        return list(objs), newkeys

    def Remove_build_geom(self, objs, *args):
        targets, recursive = args
        return self.do_remove_build_geom(objs, targets, recursive, False)

    def Remove2_build_geom(self, objs, *args):
        targets = args[0]
        return self.do_remove_build_geom(objs, targets, False, True)

    def Difference_build_geom(self, objs, *args):
        tp, tm, delete_input, delete_tool, keep_highest, _do_upgrade = args
        tp = [x.strip() for x in tp.split(',')]
        tm = [x.strip() for x in tm.split(',')]

        input_entity = get_target2(objs, tp)
        tool_entity = get_target2(objs, tm)

        ret = self.boolean_difference(
            input_entity,
            tool_entity,
            removeObject=delete_input,
            removeTool=delete_tool)

        newkeys = []
        for rr in ret:
            if rr.dim == input_entity[0].dim:
                newkeys.append(objs.addobj(rr, 'diff'))
            else:
                if keep_highest:
                    self.remove([rr], recursive=True)
                else:
                    newkeys.append(objs.addobj(rr, get_geom_key(rr)))

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
        tp, tm, delete_input, delete_tool, keep_highest, _do_upgrade = args
        tp = [x.strip() for x in tp.split(',')]
        tm = [x.strip() for x in tm.split(',')]

        input_entity = get_target2(objs, tp)
        tool_entity = get_target2(objs, tm)

        ret = self.boolean_union(
            input_entity,
            tool_entity,
            removeObject=delete_input,
            removeTool=delete_tool)

        newkeys = []
        for rr in ret:
            if rr.dim == input_entity[0].dim:
                newkeys.append(objs.addobj(rr, 'uni'))
            else:
                if keep_highest:
                    self.remove([rr], recursive=True)
                else:
                    newkeys.append(objs.addobj(rr, get_geom_key(rr)))

        if delete_input:
            for x in tp:
                if x in objs:
                    del objs[x]
        if delete_tool:
            for x in tm:
                if x in objs:
                    del objs[x]

        return list(objs), newkeys

    def Union2D_build_geom(self, objs, *args):
        tp, tm, delete_input, delete_tool, keep_highest = args
        tp = [x.strip() for x in tp.split(',')]
        tm = [x.strip() for x in tm.split(',')]

        input_entity = get_target2(objs, tp)
        tool_entity = get_target2(objs, tm)

        ret = self.boolean_union2d(
            input_entity,
            tool_entity,
            removeObject=delete_input,
            removeTool=delete_tool)

        newkeys = []
        for rr in ret:
            if rr.dim == input_entity[0].dim:
                newkeys.append(objs.addobj(rr, 'uni'))
            else:
                if keep_highest:
                    self.remove([rr], recursive=True)
                else:
                    newkeys.append(objs.addobj(rr, get_geom_key(rr)))

        if delete_input:
            for x in tp:
                if x in objs:
                    del objs[x]

        if delete_tool:
            for x in tm:
                if x in objs:
                    del objs[x]

        return list(objs), newkeys

    def Intersection_build_geom(self, objs, *args):
        tp, tm, delete_input, delete_tool, keep_highest, _do_upgrade = args
        tp = [x.strip() for x in tp.split(',')]
        tm = [x.strip() for x in tm.split(',')]

        input_entity = get_target2(objs, tp)
        tool_entity = get_target2(objs, tm)

        ret = self.boolean_intersection(
            input_entity,
            tool_entity,
            removeObject=delete_input,
            removeTool=delete_tool)
        newkeys = []
        for rr in ret:
            if rr.dim == input_entity[0].dim:
                newkeys.append(objs.addobj(rr, 'its'))
            else:
                if keep_highest:
                    self.remove([rr], recursive=True)
                else:
                    newkeys.append(objs.addobj(rr, get_geom_key(rr)))

        if delete_input:
            for x in tp:
                if x in objs:
                    del objs[x]
        if delete_tool:
            for x in tm:
                if x in objs:
                    del objs[x]

        return list(objs), newkeys

    def Fragments_build_geom(self, objs, *args):
        tp, tm, delete_input, delete_tool, keep_highest = args
        tp = [x.strip() for x in tp.split(',')]
        tm = [x.strip() for x in tm.split(',')]

        input_entity = get_target2(objs, tp)
        tool_entity = get_target2(objs, tm)
        ret = self.boolean_fragments(
            input_entity,
            tool_entity,
            removeObject=delete_input,
            removeTool=delete_tool)

        newkeys = []
        newkeys = []
        for rr in ret:
            if rr.dim == input_entity[0].dim:
                newkeys.append(objs.addobj(rr, 'frag'))
            else:
                if keep_highest:
                    self.remove([rr], recursive=True)
                else:
                    newkeys.append(objs.addobj(rr, get_geom_key(rr)))

        if delete_input:
            for x in tp:
                if x in objs:
                    del objs[x]
        if delete_tool:
            for x in tm:
                if x in objs:
                    del objs[x]

        return list(objs), newkeys

    '''
    2D elements
    '''

    def Point2D_build_geom(self, objs, *args):
        xarr, yarr = args
        lcar = 0.0
        xarr = np.atleast_1d(xarr)
        yarr = np.atleast_1d(yarr)
        zarr = xarr * 0.0
        try:
            pos = np.vstack((xarr, yarr, zarr)).transpose()
        except BaseException:
            print("can not make proper input array")
            return
        PTs = [self.add_point(p, lcar=lcar) for p in pos]
        # apparently I should use this object (poly.surface)...?
        _newobjs = []
        for p in PTs:
            newkey = objs.addobj(p, 'pt')
            _newobjs.append(newkey)

        return list(objs), _newobjs

    # Define 2D version the same as 3D
    Line2D_build_geom = Line_build_geom

    def Circle2D_build_geom(self, objs, *args):
        center, ax1, ax2, radius = args
        lcar = 0.0
        a1 = np.array(ax1 + [0])
        a2 = np.array(ax2 + [0])
        a2 = np.cross(np.cross(a1, a2), a1)
        a1 = a1 / np.sqrt(np.sum(a1**2)) * radius
        a2 = a2 / np.sqrt(np.sum(a2**2)) * radius

        c = np.array(center + [0])
        p1 = self.add_point(c + a1, lcar)
        p2 = self.add_point(c + a2, lcar)
        p3 = self.add_point(c - a1, lcar)
        p4 = self.add_point(c - a2, lcar)
        pc = self.add_point(c, lcar)
        ca1 = self.add_circle_arc(p1, pc, p2)
        ca2 = self.add_circle_arc(p2, pc, p3)
        ca3 = self.add_circle_arc(p3, pc, p4)
        ca4 = self.add_circle_arc(p4, pc, p1)
        ll1 = self.add_line_loop([ca1, ca2, ca3, ca4])
        ps1 = self.add_plane_surface(ll1)
        newkey = objs.addobj(ps1, 'ps')

        return list(objs), [newkey]

    def Arc2D_build_geom(self, objs, *args):
        center, ax1, ax2, radius, an1, an2, do_fill = args
        lcar = 0.0
        a1 = np.array(ax1 + [0])
        a2 = np.array(ax2 + [0])
        a2 = np.cross(np.cross(a1, a2), a1)
        a1 = a1 / np.sqrt(np.sum(a1**2)) * radius
        a2 = a2 / np.sqrt(np.sum(a2**2)) * radius
        if an1 > an2:
            tmp = an2
            an2 = an1
            an1 = tmp

        if an2 - an1 >= 360:
            assert False, "angle must be less than 360"

        an3 = (an1 + an2) / 2.0
        pt1 = a1 * np.cos(an1 * np.pi / 180.) + a2 * np.sin(an1 * np.pi / 180.)
        pt2 = a1 * np.cos(an2 * np.pi / 180.) + a2 * np.sin(an2 * np.pi / 180.)
        pt3 = a1 * np.cos(an3 * np.pi / 180.) + a2 * np.sin(an3 * np.pi / 180.)

        c = np.array(center + [0])
        p1 = self.add_point(c + pt1, lcar)
        p2 = self.add_point(c + pt2, lcar)
        p3 = self.add_point(c + pt3, lcar)
        pc = self.add_point(c, lcar)
        ca1 = self.add_circle_arc(p1, pc, p3)
        ca2 = self.add_circle_arc(p3, pc, p2)

        if not do_fill:
            newkey1 = objs.addobj(ca1, 'ln')
            newkey2 = objs.addobj(ca2, 'ln')
            newkeys = [newkey1, newkey2]

        else:
            l1 = self.add_line(pc, p1)
            l2 = self.add_line(p2, pc)
            ll1 = self.add_line_loop([l1, ca1, ca2, l2])
            ps1 = self.add_plane_surface(ll1)
            newkeys = [objs.addobj(ps1, 'ps')]

        return list(objs), newkeys

    def Rect2D_build_geom(self, objs, *args):
        c1, e1, e2 = args
        lcar = 0.0
        c1 = np.array(c1 + [0])
        e1 = np.array(e1 + [0])
        e2 = np.array(e2 + [0])
        p1 = self.add_point(c1, lcar)
        p2 = self.add_point(c1 + e1, lcar)
        p3 = self.add_point(c1 + e1 + e2, lcar)
        p4 = self.add_point(c1 + e2, lcar)
        l1 = self.add_line(p1, p2)
        l2 = self.add_line(p2, p3)
        l3 = self.add_line(p3, p4)
        l4 = self.add_line(p4, p1)
        ll1 = self.add_line_loop([l1, l2, l3, l4])
        rec1 = self.add_plane_surface(ll1)
        newkey = objs.addobj(rec1, 'rec')
        return list(objs), [newkey]

    def Polygon2D_build_geom(self, objs, *args):
        xarr, yarr = args
        zarr = [0] * len(yarr)
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
        poly = self.add_polygon(pos, lcar=lcar)

        # apparently I should use this object (poly.surface)...?
        newkey = objs.addobj(poly.surface, 'pol')

        return list(objs), [newkey]

    def Move2D_build_geom(self, objs, *args):
        targets, dx, dy, keep = args
        dz = 0.0
        targets = [x.strip() for x in targets.split(',')]

        newkeys = []
        tt = get_target2(objs, targets)
        if keep:
            tt = self.copy(tt)
        self.translate(tt, dx, dy, dz)
        if keep:
            for t in tt:
                newkeys.append(objs.addobj(t, 'mv'))

        return list(objs), newkeys

    def Rotate2D_build_geom(self, objs, *args):
        targets, cc, angle, keep = args
        cx, cy = cc
        cz = 0.0
        ax, ay, az = 0.0, 0.0, 1.0
        targets = [x.strip() for x in targets.split(',')]

        newkeys = []
        tt = get_target2(objs, targets)

        if keep:
            tt = self.copy(tt)
        self.rotate(tt, cx, cy, cz, ax, ay, az, np.pi * angle / 180.)
        if keep:
            for t in tt:
                newkeys.append(objs.addobj(t, 'rot'))

        return list(objs), newkeys

    def Flip2D_build_geom(self, objs, *args):
        targets, a, b, d, keep = args
        c = 0.0
        targets = [x.strip() for x in targets.split(',')]

        newkeys = []
        tt = get_target2(objs, targets)
        if keep:
            tt = self.copy(tt)
        self.symmetrize(tt, a, b, c, d)
        if keep:
            for t in tt:
                newkeys.append(objs.addobj(t, 'flp'))

        return list(objs), newkeys

    def Scale2D_build_geom(self, objs, *args):
        targets, cc, ss, keep = args
        cx, cy = cc
        cz = 0.0
        sx, sy = ss
        sz = 1.0
        targets = [x.strip() for x in targets.split(',')]

        newkeys = []
        tt = get_target2(objs, targets)
        if keep:
            tt = self.copy(tt)
        self.dilate(tt, cx, cy, cz, sx, sy, sz)
        if keep:
            for t in tt:
                newkeys.append(objs.addobj(t, 'sc'))

        return list(objs), newkeys

    def Array2D_build_geom(self, objs, *args):
        targets, count, displacement = args
        dx, dy = displacement
        dz = 0.0
        targets = [x.strip() for x in targets.split(',')]

        newkeys = []
        tt = get_target2(objs, targets)
        for i in range(count):
            tt = self.copy(tt)
            self.translate(tt, dx, dy, dz)
            for t in tt:
                newkeys.append(objs.addobj(t, 'cp'))

        return list(objs), newkeys

    def SplitByPlane_build_geom(self, objs, *args):
        print(args)

        def project_ptx_2_plain(normal, cptx, p):
            dp = p - cptx
            dp = dp - np.sum(dp * normal) * normal
            return dp + cptx

        def containing_bbox(normal, cptx, xmin, ymin, zmin, xmax, ymax, zmax):
            corners = (np.array([xmin, ymin, zmin]),
                       np.array([xmin, ymin, zmax]),
                       np.array([xmin, ymax, zmin]),
                       np.array([xmax, ymin, zmin]),
                       np.array([xmax, ymax, zmin]),
                       np.array([xmin, ymax, zmax]),
                       np.array([xmax, ymin, zmax]),
                       np.array([xmax, ymax, zmax]),)
            # projected point
            p = [project_ptx_2_plain(normal, cptx, pp) for pp in corners]

            # distance from plain
            d = [np.sum((pp - cptx) * normal) for pp in corners]
            dist1 = np.max(d)

            # distance on the plain
            d = [np.sqrt(np.sum((pp - cptx)**2)) for pp in p]
            idx = np.argmax(d)
            dist2 = np.max(d)

            size = np.max((dist1, dist2)) * 1.2

            n1 = (p[idx] - cptx)
            n2 = np.cross(normal, n1)

            c1 = cptx - n1 * size - n2 * size
            e1 = 2 * n1 * size
            e2 = 2 * n2 * size
            e3 = normal * size
            box = (c1, c1 + e1, c1 + e2, c1 + e3, c1 + e1 + e2, c1 + e2 + e3, c1 + e3 + e1,
                   c1 + e3 + e2 + e1)

            return box

        targets = [x.strip() for x in args[0].split(',')]
        tt = get_target2(objs, targets)

        dimtags = [id2dimtag(i) for i in tt]
        xmin, ymin, zmin, xmax, ymax, zmax = find_combined_bbox(
            self.model, dimtags)

        if args[1][0] == '3_points':
            # args[1] = ['3_points', '1', '7', '8']

            ptx1ID = get_target1(objs, [args[1][1], ], 'p')[0]
            ptx2ID = get_target1(objs, [args[1][2], ], 'p')[0]
            ptx3ID = get_target1(objs, [args[1][3], ], 'p')[0]
            ptx1 = np.array(gmsh.model.getValue(0, int(ptx1ID), []))
            ptx2 = np.array(gmsh.model.getValue(0, int(ptx2ID), []))
            ptx3 = np.array(gmsh.model.getValue(0, int(ptx3ID), []))

            n = np.cross(ptx1 - ptx2, ptx1 - ptx3)
            if np.sum(n**2) == 0:
                assert False, "three points does not span a surface."
            normal = n / np.sqrt(np.sum(n**2))
            cptx = (ptx1 + ptx2 + ptx3) / 3.0
        elif args[1][0] == 'by_abc':
            data = np.array(args[1][1]).flatten()
            xmin, ymin, zmin, xmax, ymax, zmax = find_combined_bbox(
                self.model, dimtags)
            normal = data[:3]
            xx = np.array(
                [(xmin + xmax) / 2, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0])
            s = data[-1] - np.sum(normal * xx)
            cptx = xx + s * normal
        elif args[1][0] == 'face_parallel':
            faceID = get_target1(objs, [args[1][1], ], 'f')[0]
            ptxID = get_target1(objs, [args[1][2], ], 'p')[0]
            cptx = np.array(gmsh.model.getValue(0, int(ptxID), []))

            n1 = np.array(gmsh.model.getNormal(faceID, (0, 0)))
            n2 = np.array(gmsh.model.getNormal(faceID, (0, 1)))
            n3 = np.array(gmsh.model.getNormal(faceID, (1, 0)))
            n1 /= np.sqrt(np.sum(n1**2))
            n2 /= np.sqrt(np.sum(n2**2))
            n3 /= np.sqrt(np.sum(n3**2))

            if np.any(n1 != n2) or np.any(n1 != n3):
                assert False, "surface is not flat"
            normal = n1

        else:
            assert False, "unknown option:" + args

        points = containing_bbox(
            normal, cptx, xmin, ymin, zmin, xmax, ymax, zmax)
        v = self.add_box(points)

        ret1 = self.boolean_difference(tt, (v,),
                                       removeObject=False,
                                       removeTool=False)
        ret2 = self.boolean_intersection(tt, (v,),
                                         removeObject=True,
                                         removeTool=True)
        newkeys = []
        for rr in ret1 + ret2:
            if rr.dim == tt[0].dim:
                newkeys.append(objs.addobj(rr, 'splt'))
            else:
                if keep_highest:
                    self.remove([rr], recursive=True)
                else:
                    newkeys.append(objs.addobj(rr, get_geom_key(rr)))

        for x in targets:
            if x in objs:
                del objs[x]

        return list(objs), newkeys

        return list(objs), []

    def _WorkPlane_build_geom(self, objs, c1, a1, a2):
        x1 = np.array([1., 0., 0.])

        ax = np.cross(x1, a1)
        an = np.arctan2(np.sqrt(np.sum(ax**2)), np.dot(a1, x1))

        tt = [objs[t] for t in objs]

        #from petram.geom.gmsh_geom_wrapper import VertexID, LineID, SurfaceID

        tt = self.get_unique_entity(tt)

        #print("first rot ???", ax, an, np.sum(ax**2))
        if np.sum(ax**2) == 0.0:
            if an != 0.0:
                # if a1 is [0, 0, -1], rotate 180 deg
                ax = np.array([0, 1, 0])
                an = np.pi
            else:
                ax = x1
                an = 0.0
        if np.sum(ax**2) != 0.0 and an != 0.0:
            #print("first rot", ax, an)
            self.rotate(tt, 0, 0, 0, ax[0], ax[1], ax[2], an)

        from petram.geom.geom_utils import rotation_mat
        R = rotation_mat(ax, an)
        '''
        c = np.cos(an); s = np.sin(an)
        R = np.array(
            [[c + (1-c)*ax[0]**2, ax[0]*ax[1]*(1-c)-ax[2]*s, ax[0]*ax[2]*(1-c)+ax[1]*s],
             [ax[0]*ax[1]*(1-c)+ax[2]*s, c + (1-c)*ax[1]**2,  ax[1]*ax[2]*(1-c)-ax[0]*s],
             [ax[0]*ax[2]*(1-c)-ax[1]*s, ax[1]*ax[2]*(1-c)+ax[0]*s, c + (1-c)*ax[2]**2]]
            )
        '''
        y2 = np.dot(R, np.array([0, 1, 0]))
        ax = a1
        aaa = np.cross(a1, y2)
        an = np.arctan2(np.dot(a2, aaa), np.dot(a2, y2))

        # for t in tt:
        #     if isinstance(t, SurfaceID): continue
        #     print("working on t", t)
        #
        #     geom.rotate([t], 0, 0, 0, ax[0], ax[1], ax[2], an)
        #print("2nd rot ???", ax, an, np.sum(ax**2))
        if np.sum(ax**2) == 0.0 and an != 0.0:
            # rotate 180 deg around a1
            ax = a1
            an = np.pi
        if np.sum(ax**2) != 0.0 and an != 0.0:
            #print("2nd rot", ax, an)
            self.rotate(tt, 0, 0, 0, ax[0], ax[1], ax[2], an)

        if c1[0] != 0.0 or c1[1] != 0.0 or c1[2] != 0.0:
            self.translate(tt, c1[0], c1[1], c1[2])

        #self._newobjs = objs.keys()
        return list(objs), []

    def WorkPlane_build_geom(self, objs, *args):
        c1, a1, a2 = args
        c1 = np.array(c1)
        a1 = np.array(a1)
        a1 = a1 / np.sqrt(np.sum(a1**2))
        a2 = np.array(a2)
        a2 = a2 / np.sqrt(np.sum(a2**2))
        return self._WorkPlane_build_geom(objs, c1, a1, a2)

    def WorkPlaneByPoints_build_geom(self, objs, *args):
        c1, a1, a2, flip1, flip2 = args

        self.factory.synchronize()
        c1 = gmsh.model.getValue(0, int(c1), [])
        a1 = gmsh.model.getValue(0, int(a1), [])
        a2 = gmsh.model.getValue(0, int(a2), [])

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

        return self._WorkPlane_build_geom(objs, c1, d1, d2)

    def get_toplevel_enteties(self):
        non_top = []

        ret_3D = self.model.getEntities(3)
        tmp = ret_3D
        if len(tmp) != 0:
            non_top = list(
                self.model.getBoundary(
                    tmp,
                    combined=False,
                    oriented=False))

        ret_2D = [x for x in self.model.getEntities(2) if x not in non_top]

        tmp = non_top + ret_2D
        if len(tmp) != 0:
            non_top = self.model.getBoundary(
                tmp, combined=False, oriented=False)

        ret_1D = [x for x in self.model.getEntities(1) if x not in non_top]

        tmp = non_top + ret_1D
        if len(tmp) != 0:
            non_top = self.model.getBoundary(
                tmp, combined=False, oriented=False)

        ret_0D = [x for x in self.model.getEntities(0) if x not in non_top]

        ret = ret_3D + ret_2D + ret_1D + ret_0D
        return ret

    def healShapes(self, dimtags, fix_tol, fixDegenerated=False,
                   fixSmallEdges=False,
                   fixSmallFaces=False,
                   sewFaces=False):

        self.factory.synchronize()
        top_level = self.get_toplevel_enteties()
        if dimtags is None:
            dimtags = top_level

        ret = []
        removed = []

        for dimtag in dimtags:
            if not dimtag in top_level:
                print(
                    "skipping " +
                    str(dimtag) +
                    " since it is not top level entitiy")
                continue
            outdimtags = self.factory.healShapes(dimTags=[dimtag],
                                                 tolerance=fix_tol,
                                                 fixDegenerated=fixDegenerated,
                                                 fixSmallEdges=fixSmallEdges,
                                                 fixSmallFaces=fixSmallFaces,
                                                 sewFaces=sewFaces)
            #print("heal outdimtags", outdimtags)
            self.factory.synchronize()
            self.factory.remove([dimtag], recursive=True)
            ret.append(outdimtags[0])
            removed.append(dimtag)

        self.factory.synchronize()
        return ret, removed

    def healCAD_build_geom(self, objs, *args):
        targets, use_fix_param, use_fix_tol = args

        self.factory.synchronize()

        targets = [x.strip()
                   for x in targets.split(',') if len(x.strip()) != 0]
        if len(targets) == 0:
            dimtags = None
        else:
            targetID = get_target2(objs, targets)
            dimtags = get_dimtag(targetID)

        ret, removed = self.healShapes(dimtags, use_fix_tol, fixDegenerated=use_fix_param[0],
                                       fixSmallEdges=use_fix_param[1],
                                       fixSmallFaces=use_fix_param[2],
                                       sewFaces=use_fix_param[3])

        for k in list(objs):
            if objs[k].to_dimtag() in removed:
                del objs[k]

        newkeys = []
        ret = dimtag2id(ret)
        for rr in ret:
            newkeys.append(objs.addobj(rr, 'hld'))

        return list(objs), newkeys

    def BrepImport_build_geom(self, objs, *args):
        cad_file, use_fix, use_fix_param, use_fix_tol, use_fix_rescale, highestDimOnly = args
        print(type(use_fix_rescale))
        assert float(use_fix_rescale) == 1.0, "rescale is not supported in GMSH geometry"

        PTs = self.factory.importShapes(cad_file,
                                        highestDimOnly=highestDimOnly)
        # debug to load one element...
        '''
        for dim, tag in PTs:
            if dim == 3:
               self.factory.remove(((dim, tag),), recursive=False)
            if dim == 2:
                if tag != 165:
                   self.factory.remove(((dim, tag),), recursive=True)
        PTs = ((2, 165),)
        '''
        # apparently I should use this object (poly.surface)...?

        newkeys = []

        if use_fix:
            PTs, void = self.healShapes(PTs, use_fix_tol, fixDegenerated=use_fix_param[0],
                                        fixSmallEdges=use_fix_param[1],
                                        fixSmallFaces=use_fix_param[2],
                                        sewFaces=use_fix_param[3])

        dim = max([p[0] for p in PTs])
        for p in PTs:
            if p[0] == dim:
                pp = dimtag2id([p])
                newkeys.append(objs.addobj(pp[0], 'impt'))

        return list(objs), newkeys

    def CADImport_build_geom(self, objs, *args):
        unit = args[-1]
        gmsh.option.setString("Geometry.OCCTargetUnit", unit)
        args = args[:-1]
        ret = self.BrepImport_build_geom(objs, *args)
        gmsh.option.setString("Geometry.OCCTargetUnit", "")
        return ret

    '''
    sequence/preview/brep generator
    '''

    def run_sequence(self, objs, gui_data, start_idx):
        isWP = False

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

            if geom_name == "WP_Start":
                tmp = objs.duplicate()
                org_keys = list(objs)

                for x in org_keys:
                    del tmp[x]

                org_objs = objs
                objs = tmp
                isWP = True

            elif geom_name == "WP_End":
                for x in objs:
                    org_objs[x] = objs[x]
                objs = org_objs
                isWP = False

            else:
                try:
                    method = getattr(self, geom_name + '_build_geom')
                    objkeys, newobjs = method(objs, *gui_param)
                    gui_data[gui_name] = (objkeys, newobjs)
                except BaseException:
                    import traceback
                    if self.logfile is not None:
                        self.logfile.write("failed " + traceback.format_exc())
                    assert False, traceback.format_exc()

        #capcheName = "" if isWP else gui_name
        return gui_data, objs

    def find_tinyloop(self, esize, thr=1e-5):
        # magic number to define too small
        el_max = max(list(esize.values()))
        edges = np.array([x for x in esize if esize[x] < el_max * thr])

        dimtags = self.model.getEntities(1)
        loops = []
        for dim, tag in dimtags:
            if len(self.model.getBoundary([(dim, tag)], oriented=False)) == 0:
                loops.append(tag)
        edges = [e for e in edges if e in loops]
        print("tiny loop edges ", edges)

        return edges

    def mesh_edge_algorithm1(self):
        '''
        use characteristic length
        '''
        vcl, esize = self.getVertexCL()
        xmin, ymin, zmin, xmax, ymax, zmax = self.getBoundingBox()
        modelsize = ((xmax - xmin)**2 + (ymax - ymin)
                     ** 2 + (zmax - zmin)**2)**0.5

        for tag in vcl:
            gmsh.model.mesh.setSize(((0, tag),), vcl[tag] / 2.5)

        gmsh.option.setNumber(
            "Mesh.CharacteristicLengthMax",
            modelsize / self.geom_prev_res)
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 1)
        gmsh.model.mesh.generate(1)

        return vcl, esize, modelsize

    def mesh_edge_algorithm2(self):
        vcl, esize = self.getVertexCL()
        for tag in vcl:
            gmsh.model.mesh.setSize(((0, tag),), vcl[tag] / 2.5)

        xmin, ymin, zmin, xmax, ymax, zmax = self.getBoundingBox()
        modelsize = ((xmax - xmin)**2 + (ymax - ymin)
                     ** 2 + (zmax - zmin)**2)**0.5

        emax = max(list(esize.values()))
        if len(esize) == 0:
            return vcl, esize, modelsize

        too_small = []

        if self.logfile is not None:
            self.logfile.write("Running Edge Mesh Alg.2 \n")
        for l in esize:
            if esize[l] > emax / 10.:
                seg = 5
            elif esize[l] > emax / 100.:
                seg = 5
            elif esize[l] > emax / 1000.:
                seg = 5
            elif esize[l] > emax / 10000.:
                seg = 4
            elif esize[l] > emax / 100000.:
                seg = 3
            else:
                too_small.append(l)
                seg = 3

                if self.logfile is not None:
                    self.logfile.write(
                        "Edge too small" + str(l) + "/" + str(esize[l] / emax) + "\n")

            gmsh.model.mesh.setTransfiniteCurve(
                l, seg, meshType="Bump", coef=1.3)

        for tag in vcl:
            gmsh.model.mesh.setSize(((0, tag),), emax / 1000)

        gmsh.model.mesh.generate(1)
        return vcl, esize, modelsize

    def mesh_edge_algorithm3(self):
        '''
        use characteristic length
        '''

        xmin, ymin, zmin, xmax, ymax, zmax = self.getBoundingBox()
        modelsize = ((xmax - xmin)**2 + (ymax - ymin)
                     ** 2 + (zmax - zmin)**2)**0.5

        too_large = self.long_edge_thr
        too_small = self.small_edge_thr
        vcl, esize = self.getVertexCL(modelsize * too_small)

        if len(esize) == 0:
            return vcl, esize, modelsize

        emax = max(list(esize.values()))

        print("Max/Min CL size", modelsize * too_large, modelsize * too_small)

        gmsh.option.setNumber(
            "Mesh.CharacteristicLengthMax",
            modelsize * too_large)
        gmsh.option.setNumber(
            "Mesh.CharacteristicLengthMin",
            modelsize * too_small)
        gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshOnlyVisible", 1)

        tiny_loops = self.find_tinyloop(esize, thr=1e-5)
        dont_do = [(1, tag) for tag in tiny_loops]

        ### if curse mesh large edge first ###
        do_first = []
        too_small2 = 5e-2

        for tag in list(vcl):
            s = np.max([vcl[tag] / self.geom_prev_res, modelsize * too_small])
            if s > modelsize * too_large:
                s = modelsize * too_large
            vcl[tag] = s
        # print(vcl)

        for dim, tag in self.model.getEntities(1):
            bdimtags = self.model.getBoundary(((dim, tag,),), oriented=False)
            ll = [vcl[vtag] for dim, vtag in bdimtags if vtag in vcl]
            if len(ll) == 0:
                continue
            if esize[tag] > np.max(ll) * self.max_seg:
                do_first.append((1, tag))
                for dim, vtag in bdimtags:
                    gmsh.model.mesh.setSize(
                        ((0, tag),), esize[tag] / self.max_seg)
            if esize[tag] < modelsize * too_small:
                do_first.append((1, tag))
                gmsh.model.mesh.setTransfiniteCurve(
                    tag, self.small_edge_seg, meshType="Bump", coef=1)

        self.show_only(do_first)
        self.hide(dont_do)
        gmsh.model.mesh.generate(1)

        self.show_all()
        self.hide(do_first)
        self.hide(dont_do)

        for tag in vcl:
            gmsh.model.mesh.setSize(
                ((0, tag),), modelsize / self.geom_prev_res)
            if np.isfinite(vcl[tag]):
                gmsh.model.mesh.setSize(((0, tag),), vcl[tag])

        print(self.use_curvature)
        if self.use_curvature:
            gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
        else:
            gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
        gmsh.model.mesh.generate(1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
        return vcl, esize, modelsize

    def mesh_face_algorithm1(self, esize):
        '''
        avoid face with strange short length loop
        make big surface with cuvertuer
        '''
        edges = self.find_tinyloop(esize)
        s = {}
        dimtags = self.model.getEntities(2)
        for dim, tag in dimtags:
            s[tag] = [y for x, y in self.model.getBoundary(
                [(dim, tag)], oriented=False)]
        from petram.mesh.mesh_utils import line2surf
        l2s = line2surf(s)
        faces = sum([l2s[edge] for edge in edges if len(l2s[edge]) == 1], [])
        print("loop faces with tiny single edge", faces)

        #dont_do = [(2, tag) for tag in faces]

        do_first = []
        size_thr = 0.01

        max_edge = np.max(list(esize.values()))
        for dim, tag in self.model.getEntities(2):
            bdimtags = self.model.getBoundary(((dim, tag,),), oriented=False)
            ll = [esize[etag] for dim, etag in bdimtags]
            if len(ll) == 0:
                continue
            if np.max(ll) > max_edge * size_thr:
                do_first.append((2, tag))

        self.show_only(do_first)
        # self.hide(dont_do)

        if self.use_curvature:
            gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
        else:
            gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)

        gmsh.model.mesh.generate(2)

        self.show_all()
        # self.hide(dont_do)
        self.hide(do_first)

        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)
        gmsh.model.mesh.generate(2)

    def make_safe_file(self, filename, trash, ext):
        #map = self.getEntityNumberingInfo()
        # make filename safe
        filename = '_'.join(filename.split("/"))
        filename = '_'.join(filename.split(":"))
        filename = '_'.join(filename.split("\\"))

        if trash == '':  # when finalizing
            return os.path.join(os.getcwd(), filename + ext)
        else:
            return os.path.join(trash, filename + ext)

    def generate_preview_mesh_gmsh(self, filename, trash):
        if self.queue is not None:
            self.queue.put((False, "generating preview"))

        ss = self.getObjSizes()

        dim2_size = min([s[2] for s in ss if s[0] == 2] + [3e20])
        dim1_size = min([s[2] for s in ss if s[0] == 1] + [3e20])

        gmsh.option.setNumber("Mesh.MaxNumThreads1D", self.maxthreads)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", self.maxthreads)

        #ent = gmsh.model.getEntities()
        #gmsh.model.setVisibility(ent, False, recursive=True)
        #gmsh.model.setVisibility(((2, 165),), True, recursive=True)

        # make 1D mesh
        vcl, esize, modelsize = self.mesh_edge_algorithm3()

        if not self.use_1d_preview and len(esize) > 0:
            gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1e22)
            #gmsh.option.setNumber("Mesh.CharacteristicLengthMax", modelsize/10)
            gmsh.option.setNumber(
                "Mesh.CharacteristicLengthExtendFromBoundary", 1)
            gmsh.option.setNumber(
                "Mesh.CharacteristicLengthMin", max(
                    list(
                        esize.values())) / 100.)
            gmsh.option.setNumber("Mesh.Optimize", 0)
            gmsh.option.setNumber("Mesh.IgnorePeriodicity", 1)
            gmsh.option.setNumber("Mesh.RefineSteps", 1)
            gmsh.option.setNumber("Mesh.Algorithm", self.geom_prev_algorithm)

            if self.use_curvature:
                gmsh.option.setNumber(
                    "Mesh.CharacteristicLengthFromCurvature", 1)
            else:
                gmsh.option.setNumber(
                    "Mesh.CharacteristicLengthFromCurvature", 0)

            self.mesh_face_algorithm1(esize)

        import os

        geom_msh = self.make_safe_file(filename, trash, '.msh')
        # filename = '_'.join(filename.split('/')) ### this avoids a problem when filename is  STEP/IGES (having / in it)
        #geom_msh = os.path.join(os.getcwd(), filename+'.msh')
        gmsh.write(geom_msh)

        return vcl, esize, geom_msh

    def generate_preview_mesh_occ(self, filename, trash):
        if self.queue is not None:
            self.queue.put((False, "generating preview"))

        gmsh.option.setNumber("Mesh.MeshOnlyVisible", 1)

        xmin, ymin, zmin, xmax, ymax, zmax = self.getBoundingBox()
        modelsize = ((xmax - xmin)**2 + (ymax - ymin)
                     ** 2 + (zmax - zmin)**2)**0.5

        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 1e22)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", modelsize / 30)

        too_large = self.long_edge_thr
        too_small = self.small_edge_thr
        vcl, esize = self.getVertexCL(modelsize * too_small)
        return vcl, esize, ''

    def generate_preview_mesh(self, filename, trash):
        from petram.geom.read_gmsh import read_pts_groups, read_loops, read_loops_do_meshloop

        if self.use_occ_preview:
            vcl, esize, geom_msh = self.generate_preview_mesh_occ(
                filename, trash)
            l, s, v = read_loops_do_meshloop(gmsh)
        else:
            vcl, esize, geom_msh = self.generate_preview_mesh_gmsh(
                filename, trash)
            l, s, v = read_loops(gmsh)
        return geom_msh, l, s, v, vcl, esize

    def create_entity_mapping(self):
        '''
        create a mapping between entitiy number and the numbering read from
        brep
        '''
        self.factory.synchronize()

        uvols = UniqueCounter()
        ufaces = UniqueCounter()
        uedges = UniqueCounter()
        uverts = UniqueCounter()

        vmap = {}
        fmap = {}
        emap = {}
        pmap = {}

        for dim1, tag1 in gmsh.model.getEntities(3):
            flag, ivol = uvols.add_shape(tag1)
            vmap[ivol] = tag1
            print(self.model.getBoundary(((dim1, tag1,),), oriented=False))
            for dim2, tag2 in self.model.getBoundary(
                    ((dim1, tag1,),), oriented=False):
                flag, iface = ufaces.add_shape(tag2)
                fmap[iface] = tag2
                for dim3, tag3 in self.model.getBoundary(
                        ((dim2, tag2,),), oriented=False):
                    flag, iedge = uedges.add_shape(tag3)
                    emap[iedge] = tag3
                    for dim4, tag4 in self.model.getBoundary(
                            ((dim3, tag3,),), oriented=False):
                        flag, ivert = uverts.add_shape(tag4)
                        pmap[ivert] = tag4

        for dim2, tag2 in gmsh.model.getEntities(2):
            flag, iface = ufaces.add_shape(tag2)
            if not flag:
                continue
            fmap[iface] = tag2
            for dim3, tag3 in self.model.getBoundary(
                    ((dim2, tag2,),), oriented=False):
                flag, iedge = uedges.add_shape(tag3)
                emap[iedge] = tag3
                for dim4, tag4 in self.model.getBoundary(
                        ((dim3, tag3,),), oriented=False):
                    flag, ivert = uverts.add_shape(tag4)
                    pmap[ivert] = tag4

        for dim3, tag3 in gmsh.model.getEntities(1):
            flag, iedge = uedges.add_shape(tag3)
            if not flag:
                continue
            emap[iedge] = tag3
            for dim4, tag4 in self.model.getBoundary(
                    ((dim3, tag3,),), oriented=False):
                flag, ivert = uverts.add_shape(tag4)
                pmap[ivert] = tag4

        for dim4, tag4 in gmsh.model.getEntities(1):
            flag, ivert = uverts.add_shape(tag4)
            pmap[ivert] = tag4

        print("mapping", vmap, fmap, emap, pmap)
        return vmap, fmap, emap, pmap

    def generate_brep(self, objs, filename='', trash='', finalize=False):

        if finalize and not self.skip_final_frag:
            if self.logfile is not None:
                self.logfile.write("finalize is on \n")
            if self.queue is not None:
                self.queue.put((False, "finalize is on"))

            self.apply_fragments()

        self.factory.synchronize()

        '''
        Save BREP for meshing.
        '''
        import os

        geom_brep = self.make_safe_file(filename, trash, '.brep')
        gmsh.write(geom_brep)

        do_map_always = False
        if finalize or do_map_always:
            '''
            We need to reload it here so that indexing is consistent
            in meshing.
            '''
            gmsh.clear()

            # We keep highestDimOnly = False, sinse low dim elemtns could be
            # used for embeding (cl control)
            gmsh.model.occ.importShapes(geom_brep, highestDimOnly=False)
            gmsh.model.occ.synchronize()
            #self.applyEntityNumberingInfo(map, objs)

        mappings = self.create_entity_mapping()

        return geom_brep, mappings


class GMSHGeometryGeneratorBase():
    def __init__(self, q, task_q):
        self.q = q
        self.task_q = task_q
        self.mw = None

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
            if task[0] == 1:
                try:
                    self.generator(*task[1])
                except BaseException:
                    txt = traceback.format_exc()
                    traceback.print_exc()
                    self.q.put((True, ('fail', txt)))
                    # self.task_queue.task_done()
                    break
        print("exiting prcesss")

    def generator(self, sequence, no_mesh, finalize,
                  filename, start_idx, trash, kwargs):

        kwargs['write_log'] = True
        kwargs['queue'] = self.q
        q = self.q

        if self.mw is None:
            from petram.geom.gmsh_geom_model import GeomObjs
            self.mw = Geometry(**kwargs)
            self.objs = GeomObjs()
            self.gui_data = dict()
        else:
            self.mw.process_kwargs(kwargs)

        q.put((self.mw.logfile.name))

        self.mw.geom_sequence = sequence
        self.mw.run_sequence(self.objs, self.gui_data, start_idx)

        if finalize:
            brep_file, mappings = self.mw.generate_brep(
                self.objs, filename=filename, trash=trash, finalize=True)
        else:
            filename = sequence[-1][0]
            brep_file, mappings = self.mw.generate_brep(
                self.objs, filename=filename, trash=trash, finalize=False)
            #brep_file = ''

        if no_mesh:
            q.put((True, (self.gui_data, self.objs, brep_file, None, None)))

        else:
            data = self.mw.generate_preview_mesh(filename, trash)
            # data =  geom_msh, l, s, v,  vcl, esize

            q.put((True, (self.gui_data, self.objs, brep_file, data, mappings)))

class GMSHGeometryGenerator(GMSHGeometryGeneratorBase, mp.Process):
    def __init__(self):
        task_q = mp.Queue()  # data to child
        q = mp.Queue()       # data from child
        GMSHGeometryGeneratorBase.__init__(self, q, task_q)
        mp.Process.__init__(self)
        dprint1("starting a process for geometry")
        
    def ready_for_next_task(self):
        pass
        
from threading import Thread
from queue import Queue
        
class GMSHGeometryGeneratorTH(GMSHGeometryGeneratorBase, Thread):
    def __init__(self):
        task_q = Queue()  # data to child
        q = Queue()       # data from child
        GMSHGeometryGeneratorBase.__init__(self, q, task_q)        
        Thread.__init__(self)
        dprint1("starting a thread for geometry")
        
    def ready_for_next_task(self):
        self.task_q.task_done()
