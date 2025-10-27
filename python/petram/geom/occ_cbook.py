

from petram.geom.geom_id import (GeomIDBase, VertexID, LineID, SurfaceID, VolumeID,
                                 LineLoopID, SurfaceLoopID)
import numpy as np


class bcolors:

    WARNING = '\033[93m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


hasOCC = False

try:
    import OCC
    import OCC.Core.Geom
    from OCC.Core.GeomAPI import (GeomAPI_Interpolate,
                                  GeomAPI_ProjectPointOnSurf,
                                  GeomAPI_ProjectPointOnCurve)
    from OCC.Core.Geom import Geom_Plane
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.TopLoc import TopLoc_Location
    from OCC.Core.TopExp import (TopExp_Explorer,
                                 topexp_MapShapes,
                                 topexp_MapShapesAndAncestors)
    from OCC.Core.BRep import BRep_Builder, BRep_Tool
    from OCC.Core.BRepTools import (breptools_Write,
                                    breptools_Read,
                                    breptools_Clean,
                                    BRepTools_WireExplorer)

    from OCC.Core.TopTools import (TopTools_IndexedMapOfShape,
                                   TopTools_IndexedDataMapOfShapeListOfShape,
                                   TopTools_ListIteratorOfListOfShape,
                                   TopTools_ListOfShape)
    from OCC.Core.ShapeFix import (ShapeFix_Shape,
                                   ShapeFix_Solid,
                                   ShapeFix_Shell,
                                   ShapeFix_Face,
                                   ShapeFix_Wire,
                                   ShapeFix_Wireframe,
                                   ShapeFix_FixSmallFace)
    from OCC.Core.TopoDS import (TopoDS_CompSolid,
                                 TopoDS_Compound,
                                 TopoDS_Shape,
                                 TopoDS_Solid,
                                 TopoDS_Shell,
                                 TopoDS_Face,
                                 TopoDS_Wire,
                                 TopoDS_Edge,
                                 TopoDS_Vertex,
                                 topods_Compound,
                                 topods_CompSolid,
                                 topods_Solid,
                                 topods_Shell,
                                 topods_Face,
                                 topods_Wire,
                                 topods_Edge,
                                 topods_Vertex)
    from OCC.Core.TopAbs import (TopAbs_COMPSOLID,
                                 TopAbs_COMPOUND,
                                 TopAbs_SOLID,
                                 TopAbs_SHELL,
                                 TopAbs_FACE,
                                 TopAbs_WIRE,
                                 TopAbs_EDGE,
                                 TopAbs_VERTEX)
    from OCC.Core.BRepPrimAPI import (BRepPrimAPI_MakePrism,
                                      BRepPrimAPI_MakeRevol,
                                      BRepPrimAPI_MakeCone,
                                      BRepPrimAPI_MakeWedge,
                                      BRepPrimAPI_MakeSphere,
                                      BRepPrimAPI_MakeTorus,
                                      BRepPrimAPI_MakeCylinder,)
    from OCC.Core.BRepFilletAPI import (BRepFilletAPI_MakeFillet,
                                        BRepFilletAPI_MakeChamfer,
                                        BRepFilletAPI_MakeFillet2d,)
    from OCC.Core.BRepOffsetAPI import (BRepOffsetAPI_MakePipe,
                                        BRepOffsetAPI_MakeOffsetShape,
                                        BRepOffsetAPI_MakeOffset,
                                        BRepOffsetAPI_MakeFilling,
                                        BRepOffsetAPI_ThruSections,
                                        BRepOffsetAPI_NormalProjection,
                                        BRepOffsetAPI_MakeThickSolid)

    from OCC.Core.BRepBuilderAPI import (BRepBuilderAPI_Sewing,
                                         BRepBuilderAPI_Copy,
                                         BRepBuilderAPI_Transform,
                                         BRepBuilderAPI_GTransform,
                                         BRepBuilderAPI_MakeSolid,
                                         BRepBuilderAPI_MakeShell,
                                         BRepBuilderAPI_MakeFace,
                                         BRepBuilderAPI_MakeWire,
                                         BRepBuilderAPI_MakeEdge,
                                         BRepBuilderAPI_MakeVertex)

    from OCC.Core.BRepAlgo import BRepAlgo_NormalProjection

    from OCC.Core.BRepAlgoAPI import (BRepAlgoAPI_Fuse,
                                      BRepAlgoAPI_Cut,
                                      BRepAlgoAPI_Common,
                                      BRepAlgoAPI_BuilderAlgo,
                                      BRepAlgoAPI_Defeaturing)
    from OCC.Core.gp import (gp_Ax1, gp_Ax2, gp_Pnt, gp_Circ,
                             gp_Dir, gp_Pnt2d, gp_Trsf, gp_Vec2d,
                             gp_Vec, gp_XYZ, gp_GTrsf, gp_Mat,
                             gp_Lin2d, gp_Dir2d)
    from OCC.Core.GC import (GC_MakeArcOfCircle,
                             GC_MakeSegment,
                             GC_MakeCircle)
    from OCC.Core.BOPTools import BOPTools_AlgoTools3D
    from OCC.Core.BOPAlgo import BOPAlgo_Splitter

    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.BRepGProp import (brepgprop_LinearProperties,
                                    brepgprop_SurfaceProperties)
    from OCC.Core.TColgp import TColgp_HArray1OfPnt

    from OCC.Core.ShapeBuild import ShapeBuild_ReShape
    from OCC.Core.ShapeExtend import (ShapeExtend_OK,
                                      ShapeExtend_DONE1,
                                      ShapeExtend_DONE2,
                                      ShapeExtend_DONE3,
                                      ShapeExtend_DONE4,
                                      ShapeExtend_DONE5,
                                      ShapeExtend_DONE6,
                                      ShapeExtend_DONE7,
                                      ShapeExtend_DONE8,
                                      ShapeExtend_FAIL1,
                                      ShapeExtend_FAIL2,
                                      ShapeExtend_FAIL3)

    from OCC.Core.IMeshTools import IMeshTools_Parameters

    from OCC.Core.BRepCheck import BRepCheck_Analyzer
    from OCC.Core.BRepLib import breplib_OrientClosedSolid

    from OCC.Core.ShapeUpgrade import (ShapeUpgrade_UnifySameDomain,
                                       ShapeUpgrade_RemoveInternalWires)

    from OCC.Core.GeomPlate import (GeomPlate_BuildPlateSurface,
                                    GeomPlate_PointConstraint,
                                    GeomPlate_MakeApprox)

    #from OCC.Core.BRepAdaptor import BRepAdaptor_HCurve
    from OCC.Core.BRepFill import BRepFill_CurveConstraint
    from OCC.Core.GeomAbs import GeomAbs_C0

    __ex1 = TopExp_Explorer()
    __ex2 = TopExp_Explorer()
    _system = GProp_GProps()
    _bt = BRep_Tool()
    __expparam = {'compound': (TopAbs_COMPOUND, topods_Compound, ''),
                  'compsolid': (TopAbs_COMPSOLID, topods_CompSolid, ''),
                  'solid': (TopAbs_SOLID, topods_Solid, ''),
                  'shell': (TopAbs_SHELL, topods_Shell, 'solid'),
                  'face': (TopAbs_FACE, topods_Face, 'shell'),
                  'wire': (TopAbs_WIRE, topods_Wire, 'face'),
                  'edge': (TopAbs_EDGE, topods_Edge, 'wire'),
                  'vertex': (TopAbs_VERTEX, topods_Vertex, 'edge')}
    __topo_names = ('solid', 'shell', 'face', 'wire', 'edge', 'vertex')
    hasOCC = True

    from packaging import version
    OCC_after_7_7_0 = version.parse(OCC.VERSION) >= version.parse("7.7.0")
    OCC_after_7_6_0 = version.parse(OCC.VERSION) >= version.parse("7.6.0")
    OCC_after_7_5_0 = version.parse(OCC.VERSION) >= version.parse("7.5.0")

except ImportError:
    import traceback
    traceback.print_exc()

    tfail = "\033[91m"
    tendc = "\033[0m"

    import warnings
    warnings.warn(
        tfail + "python-occ is not found and geometry construction feature is not avaialble"
        "consider intalling python-occ (https://github.com/tpaviot/pythonocc-core)" + tendc,
        UserWarning,
        stacklevel=2  # Points the warning to the caller's location
    )


def iter_shape(shape, shape_type='shell', exclude_parent=False, use_ex2=False):
    '''
    iterate over shape. this allows to write

    for subshape in iter_shape(shape, 'shell'):
        ...
    '''
    args = [__expparam[shape_type][0], ]
    cast = __expparam[shape_type][1]

    if exclude_parent:
        args.append(__expparam[shape_type][2])

    ex = __ex2 if use_ex2 else __ex1
    ex.Init(shape, *args)
    while ex.More():
        sub_shape = cast(ex.Current())
        yield sub_shape
        ex.Next()


def iter_shape_once(shape, shape_type='shell', exclude_parent=False, use_ex2=False):
    mapper = get_mapper(shape, shape_type)
    seen = topo_seen(mapping=mapper)

    for s in iter_shape(shape, shape_type=shape_type,
                        exclude_parent=exclude_parent,
                        use_ex2=use_ex2):
        if seen.check_shape(s) == 0:
            yield s


def shape_dim(shape):
    if isinstance(shape, TopoDS_Solid):
        return 3
    if isinstance(shape, TopoDS_Face):
        return 2
    if isinstance(shape, TopoDS_Edge):
        return 1
    if isinstance(shape, TopoDS_Vertex):
        return 0
    if isinstance(shape, TopoDS_Shell):
        return -2
    if isinstance(shape, TopoDS_Wire):
        return -2
    return -3


def shape_name(shape):
    if isinstance(shape, TopoDS_Solid):
        return 'solid'
    elif isinstance(shape, TopoDS_Face):
        return 'face'
    elif isinstance(shape, TopoDS_Edge):
        return 'edge'
    elif isinstance(shape, TopoDS_Vertex):
        return 'vertex'
    elif isinstance(shape, TopoDS_Shell):
        return 'shell'
    elif isinstance(shape, TopoDS_Wire):
        return 'wire'
    elif isinstance(shape, TopoDS_CompSolid):
        return 'compsolid'
    elif isinstance(shape, TopoDS_Compound):
        return 'compound'
    else:
        assert False, "unknown topoDS:" + str(type(shape))


def get_mapper(shape, shape_type):
    mapper = TopTools_IndexedMapOfShape()
    topo_abs = __expparam[shape_type][0]
    topexp_MapShapes(shape, topo_abs, mapper)
    return mapper


def iterdouble_shape(shape_in, inner_type='shell'):
    outer_type = __expparam[inner_type][2]

    args1 = [__expparam[outer_type][0], ]
    args2 = [__expparam[inner_type][0], ]
    cast1 = __expparam[outer_type][1]
    cast2 = __expparam[inner_type][1]

    __ex1.Init(shape_in, *args1)

    while __ex1.More():
        p_shape = cast1(__ex1.Current())
        __ex2.Init(p_shape, *args2)

        while __ex2.More():
            shape = cast2(__ex2.Current())
            yield shape, p_shape
            __ex2.Next()
        __ex1.Next()


def do_rotate(shape, ax, an, txt=''):
    ax = [float(x) for x in ax]
    trans = gp_Trsf()
    axis_revolution = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(ax[0], ax[1], ax[2]))
    trans.SetRotation(axis_revolution, an)
    transformer = BRepBuilderAPI_Transform(trans)
    transformer.Perform(shape)
    if not transformer.IsDone():
        assert False, "can not rotate (WP "+txt+")"
    return transformer.ModifiedShape(shape)


def do_translate(shape, delta):
    delta = [float(x) for x in delta]
    trans = gp_Trsf()
    trans.SetTranslation(gp_Vec(delta[0], delta[1], delta[2]))
    transformer = BRepBuilderAPI_Transform(trans)
    transformer.Perform(shape)
    if not transformer.IsDone():
        assert False, "can not translate (WP) "
    return transformer.ModifiedShape(shape)


def get_topo_list():
    lists = {}
    for x in __topo_names:
        cls = 'topo_list_'+x
        lists[x] = globals()[cls]()
    return lists


def calc_wp_projection(c1, a1, a2):
    x1 = np.array([1., 0., 0.])

    ax = np.cross(x1, a1)
    an = np.arctan2(np.sqrt(np.sum(ax**2)), np.dot(a1, x1))

    if np.sum(ax**2) == 0.0:
        if an != 0.0:
            # if a1 is [0, 0, -1], rotate 180 deg
            ax = np.array([0, 1, 0])
            an = np.pi
        else:
            ax = x1
            an = 0.0
    if np.sum(ax**2) != 0.0 and an != 0.0:
        ax1 = ax
        an1 = an
    else:
        ax1 = x1
        an1 = 0.0

    from petram.geom.geom_utils import rotation_mat
    R = rotation_mat(ax1, an1)
    y2 = np.dot(R, np.array([0, 1, 0]))
    ax = a1
    aaa = np.cross(a1, y2)
    an = np.arctan2(np.dot(a2, aaa), np.dot(a2, y2))

    if np.sum(ax**2) == 0.0 and an != 0.0:
        # rotate 180 deg around a1
        ax2 = a1
        an2 = np.pi
    else:
        ax2 = ax
        an2 = an

    if c1[0] != 0.0 or c1[1] != 0.0 or c1[2] != 0.0:
        cxyz = c1
    else:
        cxyz = None

    return ax1, an1, ax2, an2, cxyz


def prep_maps(shape, return_all=True, return_compound=False):

    solidMap = get_mapper(shape, 'solid')
    faceMap = get_mapper(shape, 'face')
    edgeMap = get_mapper(shape, 'edge')
    vertMap = get_mapper(shape, 'vertex')

    maps = {'solid': solidMap, 'face': faceMap, 'edge': edgeMap,
            'vertex': vertMap}

    if not return_all:
        return maps

    shellMap = get_mapper(shape, 'shell')
    wireMap = get_mapper(shape, 'wire')

    maps['shell'] = shellMap
    maps['wire'] = wireMap

    if not return_compound:
        return maps

    compoundMap = get_mapper(shape, 'compound')
    compsolidMap = get_mapper(shape, 'compsolid')

    maps['compound'] = compoundMap
    maps['compsolid'] = compsolidMap

    return maps


def register_shape(shape, topolists):
    maps = prep_maps(shape)
    seens = {x: topo_seen(mapping=maps[x]) for x in maps}

    new_objs = []
    # registor solid
    for solid in iter_shape(shape, 'solid'):
        if seens['solid'].check_shape(solid) == 0:
            solid_id = topolists['solid'].add(solid)
            new_objs.append(solid_id)

    def register_topo(shape, shape_type, add_newobj=False):
        seen = seens[shape_type]
        topo_ll = topolists[shape_type]

        for sub_shape, sub_shape_p in iterdouble_shape(shape, shape_type):
            if seen.check_shape(sub_shape) == 0:
                topo_id = topo_ll.add(sub_shape)
        for sub_shape in iter_shape(shape, shape_type, exclude_parent=True):
            if seen.check_shape(sub_shape) == 0:
                topo_id = topo_ll.add(sub_shape)
                if add_newobj:
                    new_objs.append(topo_id)

    register_topo(shape, 'shell')
    register_topo(shape, 'face', True)
    register_topo(shape, 'wire')
    register_topo(shape, 'edge', True)
    register_topo(shape, 'vertex', True)

    '''
    ex1 = TopExp_Explorer(shape, TopAbs_SOLID)
    while ex1.More():
        solid = topods_Solid(ex1.Current())
        if usolids.check_shape(solid) == 0:
             solid_id = self.solids.add(solid)
             new_objs.append(solid_id)
        ex1.Next()

    def register_topo(shape, ucounter, topabs, topabs_p, topods, topods_p,
                      topo_list, dim=-1):
        ex1 = TopExp_Explorer(shape, topabs_p)
        while ex1.More():
            topo_p = topods_p(ex1.Current())
            ex2 = TopExp_Explorer(topo_p, topabs)
            while ex2.More():
                topo = topods(ex2.Current())
                if ucounter.check_shape(topo) == 0:
                    topo_id = topo_list.add(topo)
                ex2.Next()
            ex1.Next()
        ex1.Init(shape, topabs, topabs_p)
        while ex1.More():
            topo = topods(ex1.Current())
            if ucounter.check_shape(topo) == 0:
                topo_id = topo_list.add(topo)
                if dim != -1:
                    new_objs.append(topo_id)
            ex1.Next()
    
    register_topo(shape, ushells, TopAbs_SHELL, TopAbs_SOLID,
                  topods_Shell, topods_Solid, self.shells)
    register_topo(shape, ufaces, TopAbs_FACE, TopAbs_SHELL,
                  topods_Face, topods_Shell, self.faces, dim=2)
    register_topo(shape, uwires, TopAbs_WIRE, TopAbs_FACE,
                  topods_Wire, topods_Face, self.wires)
    register_topo(shape, uedges, TopAbs_EDGE, TopAbs_WIRE,
                  topods_Edge, topods_Wire, self.edges, dim=1)
    register_topo(shape, uvertices, TopAbs_VERTEX, TopAbs_EDGE,
                  topods_Vertex, topods_Edge,self.vertices, dim=0)
    '''


def read_interface_value(name, R=False, I=False, C=False, verbose=True):
    from OCC.Core.Interface import (Interface_Static_CVal,
                                    Interface_Static_RVal,
                                    Interface_Static_IVal)

    if R:
        rp = Interface_Static_RVal(name)
    if I:
        rp = Interface_Static_IVal(name)
    if C:
        rp = Interface_Static_CVal(name)

    if verbose:
        print(name, rp)

    return rp


def write_interface_value(name, rp, R=False, I=False, C=False, verbose=True):
    from OCC.Core.Interface import (Interface_Static_SetCVal,
                                    Interface_Static_SetRVal,
                                    Interface_Static_SetIVal)

    if R:
        Interface_Static_SetRVal(name, rp)
    if I:
        Interface_Static_SetIVal(name, rp)
    if C:
        Interface_Static_SetCVal(name, rp)

    if verbose:
        print("setting", name, rp)

    return rp


def display_shape(shape):
    # show shape on python-occ display for debug.
    from OCC.Display.SimpleGui import init_display

    display, start_display, add_menu, add_function_to_menu = init_display()
    display.DisplayShape(shape)
    display.FitAll()
    start_display()


def project_ptx_2_plain(normal, cptx, p):
    dp = p - cptx
    dp = dp - np.sum(dp * normal) * normal
    return dp + cptx


def rect_by_bbox_projection(normal, cptx, xmin, ymin, zmin,
                            xmax, ymax, zmax, scale=1.5):
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

    c1 = np.mean(np.vstack(p), 0)
    # distance on the plane
    d = [np.sqrt(np.sum((pp - c1)**2)) for pp in p]
    idx = np.argmax(d)
    dist2 = np.max(d)

    n1 = (p[idx] - c1)
    n1 = n1/np.sqrt(np.sum(n1**2))
    n2 = np.cross(normal, n1)

    #c1 = (cptx + p[idx])/2.0
    e1 = n1 * dist2*scale
    e2 = n2 * dist2*scale

    return [c1+e1, c1+e2, c1-e1, c1-e2]


def box_containing_bbox(normal, cptx, xmin, ymin, zmin,
                        xmax, ymax, zmax, scale=1.2):
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

    # distance from plane
    d = [np.sum((pp - cptx) * normal) for pp in corners]
    dist1 = np.max(d)

    print("distance from plane", dist1)
    # distance on the plane
    d = [np.sqrt(np.sum((pp - cptx)**2)) for pp in p]
    idx = np.argmax(d)
    dist2 = np.max(d)

    size = np.max((dist1, dist2)) * scale

    n1 = (p[idx] - cptx)
    n1 = n1/np.sqrt(np.sum(n1**2))
    n2 = np.cross(normal, n1)

    c1 = cptx - n1 * size - n2 * size
    e1 = 2 * n1 * size
    e2 = 2 * n2 * size
    e3 = normal * size
    box = (c1, c1 + e1, c1 + e2, c1 + e3, c1 + e1 + e2,
           c1 + e2 + e3, c1 + e3 + e1, c1 + e3 + e2 + e1)

    return box


def measure_edge_length(edge):
    brepgprop_LinearProperties(edge, _system)
    return _system.Mass()


def measure_face_area(face):
    brepgprop_SurfaceProperties(face, _system)
    return _system.Mass()


def find_point_on_curve(edge, lengths, tol=1e-4, flip=False):

    length = measure_edge_length(edge)

    curve, first, last = _bt.Curve(edge)

    # Number of points
    segs = 100

    while segs < 1e5:

        if flip:
            u = np.linspace(last, first, segs)
        else:
            u = np.linspace(first, last, segs)

        pnts = [curve.Value(x) for x in u]
        ptx = np.array([(p.X(), p.Y(), p.Z()) for p in pnts])
        l = np.cumsum(np.sqrt(np.sum((ptx[:-1] - ptx[1:])**2, -1)))
        rtol = abs(l[-1]-length)/length
        if rtol < tol:
            break
        # this array converges somewhat different value from
        # what brepgprop returns. We check convergence with the
        # last step value

        length = l[-1]
        segs = segs * 3

    assert rtol < tol, ("(fint_point_on_curve) cannot " +
                        "achive request tol. : "+str(tol))

    l = np.hstack(([0], l))/l[-1]*length
    flag = (lengths <= length)

    ufit = np.interp(lengths[flag], l, u)
    return ufit


def check_shape_area(shape, thr, return_area=False):
    surfacecount = []
    faces = []
    for face in iter_shape_once(shape, 'face'):
        a = measure_face_area(face)
        surfacecount.append(a)
        faces.append(face)

    if len(faces) == 0:
        smax = 0.0
        idx = []
        faces = []
    else:
        smax = np.max(surfacecount)
        idx = np.where(surfacecount < smax*thr)[0]
        faces = [faces[i] for i in idx]

    if return_area:
        areas = [surfacecount[i] for i in idx]
        return len(idx), smax, faces, areas
    return len(idx), smax, faces


def check_shape_length(shape, thr, return_area=False):
    lcount = []
    edges = []
    for edge in iter_shape_once(shape, 'edge'):
        l = measure_edge_length(edge)
        lcount.append(l)
        edges.append(edge)

    lmax = np.max(lcount)
    idx = np.where(lcount < lmax*thr)[0]
    edges = [edges[i] for i in idx]

    if return_area:
        ll = [lcount[i] for i in idx]
        return len(idx), lmax, edges, ll
    return len(idx), lmax, edges


c_kinds = ('Geom_BezierCurve',
           'Geom_BSplineCurve',
           'Geom_TrimmedCurve',
           'Geom_Circle',
           'Geom_Ellipse',
           'Geom_Hyperbola',
           'Geom_Parabola',
           'Geom_Line',
           'Geom_OffsetCurve',
           'ShapeExtend_ComplexCurve',
           'Geom_BoundedCurve',
           'Geom_Conic',)

s_kinds = ('Geom_BezierSurface',
           'Geom_BSplineSurface',
           'Geom_RectangularTrimmedSurface',
           'Geom_ConicalSurface',
           'Geom_CylindricalSurface',
           'Geom_Plane',
           'Geom_SphericalSurface',
           'Geom_ToroidalSurface',
           'Geom_SurfaceOfLinearExtrusion',
           'Geom_SurfaceOfRevolution',
           'Geom_PlateSurface',
           'Geom_OffsetSurface',
           'Geom_BoundedSurface',
           'ShapeExtended_CompositeSurface',
           'Geom_ElementarySurface',
           'Geom_Surface',)


def downcast_curve(curve):
    kind = 'unknown'
    for k in c_kinds:
        if curve.IsKind(k):
            kind = k
            break
    handle = OCC.Core.Geom.__dict__[kind]
    curve = handle.DownCast(curve)

    return curve, kind.split('_')[-1]


def downcast_surface(surf):
    kind = 'unknown'
    for k in s_kinds:
        if surf.IsKind(k):
            kind = k
            break
    handle = OCC.Core.Geom.__dict__[kind]
    surf = handle.DownCast(surf)

    return surf, kind.split('_')[-1]


class topo_seen(list):
    def __init__(self, mapping):
        self.mapping = mapping
        self.check = np.array([0] * mapping.Size())

    def check_shape(self, x):
        i = self.mapping.FindIndex(x) - 1
        ret = self.check[i]
        self.check[i] += 1
        return ret

    def index(self, x):
        i = self.mapping.FindIndex(x) - 1
        return i

    def find_from_index(self, i):
        return self.mapping.FindFromIndex(i)

    def seen(self, x):
        return self.check_shape(x) != 0

    def not_seen(self, x):
        return self.check_shape(x) == 0


class topo_list():
    name = 'base'
    myclass = type(None)
    gidclass = type(None)

    def __init__(self):
        self.gg = {0: {}, }
        self.d = self.gg[0]
        self.next_id = 0

    def add(self, shape):
        if not isinstance(shape, self.myclass):
            assert False, ("invalid object type" + self.myclass.__name__ +
                           ':' + shape.__class__.__name__)
        self.next_id += 1
        self.d[self.next_id] = shape
        return self.next_id

    def new_group(self):
        group = max(list(self.gg.keys()))+1
        self.set_group(group)
        return group

    def set_group(self, group):
        if not group in self.gg:
            self.gg[group] = {}
        self.d = self.gg[group]

    def current_group(self):
        for g in self.gg:
            if self.gg[g] is self.d:
                return g

    def __iter__(self):
        return self.d.__iter__()

    def __len__(self):
        return len(self.d)

    def __getitem__(self, val):
        return self.d[int(val)]

    def __setitem__(self, val, value):
        self.d[int(val)] = value

    def __contains__(self, val):
        return val in self.d

    def find_gid(self, shape1):
        mapper = self.get_mapper(shape1)

        for k in self.d:
            if mapper.Contains(self.d[k]):
                idx = k
                break
        else:
            assert False, "Can not find a shape number to replace"
        return self.gidclass(idx)

    def get_item_from_group(self, val, group=0):
        return self.gg[group][val]

    def is_toplevel(self, *args):
        del args  # not used
        assert False, "subclass need to add this"

    def synchronize(self, mapper, action='remove', verbose=False):
        if verbose:
            print("Synchronize:", self.name, mapper.Size())

        if action in ('remove', 'both'):
            removal = []
            found_idx = []
            for k in self.d:
                shape = self.d[k]
                if not mapper.Contains(shape):
                    removal.append(k)
            for k in removal:
                del self.d[k]
            if verbose:
                print("removed gid", removal, list(self.d))

        if action in ('add', 'both'):
            found_idx = []
            new_gids = []
            for k in self.d:
                shape = self.d[k]
                if mapper.Contains(shape):
                    idx = mapper.FindIndex(shape)
                    found_idx.append(idx)
            tmp = np.arange(1, mapper.Size() + 1)
            new_shape_idx = tmp[np.in1d(tmp, np.array(found_idx), invert=True)]

            for idx in new_shape_idx:
                shape = mapper(int(idx))
                new_gids.append(self.add(shape))

            if verbose:
                print("added gid", new_gids)

    def get_ancestors(self, val, kind):
        shape = self[val]
        return iter_shape(shape, kind)


class topo_list_vertex(topo_list):
    name = 'vertex'
    myclass = TopoDS_Vertex
    gidclass = VertexID

    def child_generator(self, val):
        del val  # unused
        return []

    def is_toplevel(self, val, compound):
        mapper = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp_MapShapesAndAncestors(
            compound, TopAbs_VERTEX, TopAbs_EDGE, mapper)
        shape = self[val]
        if mapper.FindFromKey(shape).Size() == 0:
            return True
        return False

    def get_mapper(self, shape):
        mapper = TopTools_IndexedMapOfShape()
        topexp_MapShapes(shape, TopAbs_VERTEX, mapper)
        return mapper

    def get_child_mapper(self, args):
        del args  # unused
        return None

    def add(self, shape):
        ret = topo_list.add(self, shape)
        return VertexID(ret)

    def keys(self):
        return [VertexID(x) for x in self.d]


class topo_list_edge(topo_list):
    name = 'edge'
    myclass = TopoDS_Edge
    gidclass = LineID

    def get_children(self, val):
        shape = self[val]
        return iter_shape(shape, 'vertex')
        '''
        ex1 = TopExp_Explorer(shape, TopAbs_VERTEX)
        while ex1.More():
            vertex = topods_Vertex(ex1.Current())
            yield vertex
            ex1.Next()
        '''

    def is_toplevel(self, val, compound):
        mapper = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp_MapShapesAndAncestors(
            compound, TopAbs_EDGE, TopAbs_FACE, mapper)
        shape = self[val]
        if mapper.FindFromKey(shape).Size() == 0:
            return True
        return False

    def get_mapper(self, shape):
        mapper = TopTools_IndexedMapOfShape()
        topexp_MapShapes(shape, TopAbs_EDGE, mapper)
        return mapper

    def get_chilld_mapper(self, shape):
        mapper = TopTools_IndexedMapOfShape()
        topexp_MapShapes(shape, TopAbs_VERTEX, mapper)
        return mapper

    def add(self, shape):
        ret = topo_list.add(self, shape)
        return LineID(ret)

    def keys(self):
        return [LineID(x) for x in self.d]


class topo_list_wire(topo_list):
    name = 'wire'
    myclass = TopoDS_Wire
    gidclass = LineLoopID

    def get_children(self, val):
        shape = self[val]
        return iter_shape(shape, 'edge')

    def get_mapper(self, shape):
        mapper = TopTools_IndexedMapOfShape()
        topexp_MapShapes(shape, TopAbs_WIRE, mapper)
        return mapper

    def get_chilld_mapper(self, shape):
        mapper = TopTools_IndexedMapOfShape()
        topexp_MapShapes(shape, TopAbs_EDGE, mapper)
        return mapper

    def add(self, shape):
        ret = topo_list.add(self, shape)
        return LineLoopID(ret)

    def keys(self):
        return [LineLoopID(x) for x in self.d]


class topo_list_face(topo_list):
    name = 'face'
    myclass = TopoDS_Face
    gidclass = SurfaceID

    def get_children(self, val):
        shape = self[val]
        return iter_shape(shape, 'wire')

    def is_toplevel(self, val, compound):
        mapper = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp_MapShapesAndAncestors(
            compound, TopAbs_FACE, TopAbs_SOLID, mapper)
        shape = self[val]
        if mapper.FindFromKey(shape).Size() == 0:
            return True
        return False

    def get_mapper(self, shape):
        mapper = TopTools_IndexedMapOfShape()
        topexp_MapShapes(shape, TopAbs_FACE, mapper)
        return mapper

    def get_chilld_mapper(self, shape):
        mapper = TopTools_IndexedMapOfShape()
        topexp_MapShapes(shape, TopAbs_WIRE, mapper)
        return mapper

    def add(self, shape):
        ret = topo_list.add(self, shape)
        return SurfaceID(ret)

    def keys(self):
        return [SurfaceID(x) for x in self.d]


class topo_list_shell(topo_list):
    name = 'shell'
    myclass = TopoDS_Shell
    gidclass = SurfaceLoopID

    def get_children(self, val):
        shape = self[val]
        return iter_shape(shape, 'face')

    def get_mapper(self, shape):
        mapper = TopTools_IndexedMapOfShape()
        topexp_MapShapes(shape, TopAbs_SHELL, mapper)
        return mapper

    def get_chilld_mapper(self, shape):
        mapper = TopTools_IndexedMapOfShape()
        topexp_MapShapes(shape, TopAbs_FACE, mapper)
        return mapper

    def add(self, shape):
        ret = topo_list.add(self, shape)
        return SurfaceLoopID(ret)

    def keys(self):
        return [SurfaceLoopID(x) for x in self.d]


class topo_list_solid(topo_list):
    name = 'solid'
    myclass = TopoDS_Solid
    gidclass = VolumeID

    def get_children(self, val):
        shape = self[val]
        return iter_shape(shape, 'shell')

    def is_toplevel(self, val, compound):
        mapper = TopTools_IndexedDataMapOfShapeListOfShape()
        topexp_MapShapesAndAncestors(
            compound, TopAbs_SOLID, TopAbs_COMPSOLID, mapper)
        shape = self[val]
        if mapper.FindFromKey(shape).Size() == 0:
            return True
        return False

    def get_mapper(self, shape):
        mapper = TopTools_IndexedMapOfShape()
        topexp_MapShapes(shape, TopAbs_SOLID, mapper)
        return mapper

    def get_chilld_mapper(self, shape):
        mapper = TopTools_IndexedMapOfShape()
        topexp_MapShapes(shape, TopAbs_SHELL, mapper)
        return mapper

    def add(self, shape):
        ret = topo_list.add(self, shape)
        return VolumeID(ret)

    def keys(self):
        return [VolumeID(x) for x in self.d]
