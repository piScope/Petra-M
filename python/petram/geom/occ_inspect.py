import numpy as np

from petram.geom.occ_cbook import *


def xyz2txt(c):
    return ", ".join([str(x) for x in (c.X(), c.Y(), c.Z())])


def shape_property_txt(bt, shape):
    if isinstance(shape, TopoDS_Vertex):
        pnt = bt.Pnt(shape)
        c_txt = xyz2txt(pnt)
        txt = ['Point:',
               '  Coords:\t' + c_txt]

    if isinstance(shape, TopoDS_Edge):
        loc = TopLoc_Location()
        curve, first, last = bt.Curve(shape, loc)
        is_closed = curve.IsClosed()
        is_periodic = curve.IsPeriodic()
        curve, kind = downcast_curve(curve)
        length = measure_edge_length(shape)
        txt = ['Curve:',
               '  Kind:\t' + kind,
               '  Length:\t' + str(length),
               '  Parameter:\t' + str([first, last]),
               '  Closed:\t' + str(is_closed),
               '  Periodic:\t' + str(is_periodic)]

        if curve.IsKind('Geom_Conic'):
            txt_c = xyz2txt(curve.Location())
            txt_n = xyz2txt(curve.Axis().Direction())
            txt.extend(['  Center:\t' + txt_c,
                        '  Normal:\t' + txt_n])
        if curve.IsKind('Geom_Circle'):
            r = curve.Radius()
            txt.extend(['  Radius:\t' + str(r)])
        if curve.IsKind('Geom_Ellipse'):
            r1 = curve.MajorRadius()
            r2 = curve.MinorRadius()
            txt_f1 = xyz2txt(curve.Focus1())
            txt_f2 = xyz2txt(curve.Focus2())
            txt.extend(['  Radius1:\t' + str(r1),
                        '  Radius2:\t' + str(r2),
                        '  Focus1:\t' + txt_f1,
                        '  Focus2:\t' + txt_f2])
        if curve.IsKind('Geom_Hypabola'):
            r1 = curve.MajorRadius()
            r2 = curve.MinorRadius()
            txt_f1 = xyz2txt(curve.Focus1())
            txt_f2 = xyz2txt(curve.Focus2())
            txt.extend(['  Radius1\t:' + str(r1),
                        '  Radius2:\t' + str(r2),
                        '  Focus1:\t' + txt_f1,
                        '  Focus2:\t' + txt_f2])
        if curve.IsKind('Geom_Parabola'):
            txt_f = xyz2txt(curve.Focus())
            txt.extend(['  Focus:\t' + txt_f])
        if curve.IsKind('Geom_BSplineCurve'):
            txt.extend(['  Start:\t' + xyz2txt(curve.StartPoint()),
                        '  End:\t' + xyz2txt(curve.EndPoint()),
                        '  #Knots:\t' + str(curve.NbKnots()),
                        '  #Poles:\t' + str(curve.NbPoles())])
        if curve.IsKind('Geom_BezierCurve'):
            txt.extend(['  Start:\t' + xyz2txt(curve.StartPoint()),
                        '  End:\t' + xyz2txt(curve.EndPoint()),
                        '  #Poles:\t' + str(curve.NbPoles())])
        if curve.IsKind('Geom_TrimmedCurve'):
            txt.extend(['  Start:\t' + xyz2txt(curve.StartPoint()),
                        '  End:\t' + xyz2txt(curve.EndPoint()), ])

    if isinstance(shape, TopoDS_Face):
        surf = bt.Surface(shape)
        u1, u2, v1, v2 = surf.Bounds()
        is_uperiodic = surf.IsUPeriodic()
        is_vperiodic = surf.IsVPeriodic()

        system = GProp_GProps()
        brepgprop_SurfaceProperties(shape, system)
        surfacecount = system.Mass()

        surf, kind = downcast_surface(surf)

        txt = ['Surface:',
               ' Kind:\t' + kind,
               ' Area:\t' + str(surfacecount),
               ' U-Parameter:\t' + str([u1, u2]),
               ' V-Parameter:\t' + str([v1, v2]),
               ' Periodic (U,V):\t' + str([is_uperiodic, is_vperiodic]), ]

        if surf.IsKind('Geom_Plane'):
            a, b, c, d = surf.Coefficients()
            txt2 = ', '.join([str(x) for x in (a, b, c, d)])
            txt.extend(['  Coefficient:\t' + txt2])

    if isinstance(shape, TopoDS_Solid):
        txt = ['', ]

    return '\n'.join(txt)


def find_sameface(bt, shape, faces, tol):
    '''
    find faces which has same kind,
    same area, edges with same length
    '''
    system = GProp_GProps()

    def count_edges(face):
        k = 0
        lens = []
        for edge in iter_shape_once(face, 'edge', use_ex2=True):
            brepgprop_LinearProperties(edge, system)
            lens.append(system.Mass())
            k = k + 1
        return k, np.sort(lens)

    dataset = []
    for f1 in faces:
        surf = bt.Surface(f1)
        surf, kind = downcast_surface(surf)
        brepgprop_SurfaceProperties(f1, system)
        area = system.Mass()
        nedges, lengths = count_edges(f1)
        dataset.append((kind, area, nedges, lengths))

    samefaces = []
    for f2 in iter_shape_once(shape, 'face'):
        surf2 = bt.Surface(f2)
        surf2, kind2 = downcast_surface(surf2)

        subset = [x for x in dataset if x[0] == kind2]
        if len(subset) == 0:
            continue

        brepgprop_SurfaceProperties(f2, system)
        area2 = system.Mass()

        subset = [x for x in subset
                  if abs(x[1] - area2)/(x[1] + area2) < tol]
        if len(subset) == 0:
            continue

        nedges2, lengths2 = count_edges(f2)
        subset = [x for x in subset if x[2] == nedges2]
        if len(subset) == 0:
            continue

        subset = [x for x in subset
                  if np.all([abs(l1 - l2)/(l1 + l2) < tol for l1, l2 in zip(x[3], lengths2)])]

        if len(subset) == 0:
            continue
        samefaces.append(f2)

    return samefaces


def find_sameedge(bt, shape, edges, tol):
    '''
    find edges which has same kind and length
    '''
    system = GProp_GProps()

    def measure_len(edge):
        brepgprop_LinearProperties(edge, system)
        return system.Mass()

    dataset = []
    for edge in edges:
        l1 = measure_len(edge)
        params = bt.Curve(edge)
        if len(params) == 2:
            # null handle case
            continue
        curve = params[0]
        curve, kind = downcast_curve(curve)
        dataset.append((kind, l1))

    sameedges = []

    for e2 in iter_shape_once(shape, 'edge'):
        params = bt.Curve(e2)
        if len(params) == 2:
            # null handle case
            continue
        curve2 = params[0]
        curve2, kind2 = downcast_curve(curve2)

        subset = [x for x in dataset if x[0] == kind2]
        if len(subset) == 0:
            continue

        l2 = measure_len(e2)
        subset = [x for x in subset
                  if abs(x[1] - l2)/(x[1] + l2) < tol]
        if len(subset) == 0:
            continue

        sameedges.append(e2)
    return sameedges


def find_min_distance_in_face(bt, shape):
    '''
    shape is face
    check all distances between vertices in face
    '''
    from scipy.spatial import distance_matrix

    vertices = [p for p in iter_shape_once(shape, 'vertex')]

    ptx = []
    for v in vertices:
        pnt = bt.Pnt(v)
        p = np.array((pnt.X(), pnt.Y(), pnt.Z(),))
        ptx.append(p)

    ptx = np.vstack(ptx)
    md = distance_matrix(ptx, ptx, p=2)

    # diagnal is zero. needs to inflate it to find minimum
    for i in range(len(vertices)):
        md[i, i] = np.infty

    return np.min(md.flatten())


def shape_inspector(shape, inspect_type, shapes):

    # print("inspection ", shape, inspect_type, shapes)
    bt = BRep_Tool()

    ret = ''
    data = None

    if inspect_type == 'property':
        prop = [shape_property_txt(bt, s) for s in shapes]
        return ' \n\n'.join(prop), ''

    elif inspect_type == 'smallface':
        args, topolist = shapes
        thr = args[0]
        nsmall, smax, faces, areas = check_shape_area(shape, thr,
                                                      return_area=True)
        gids = [topolist.find_gid(f) for f in faces]
        kinds = [downcast_surface(bt.Surface(f))[1] for f in faces]
        min_ds = [find_min_distance_in_face(bt, f) for f in faces]
        txt = '\n'.join([str(int(gid)) + "\t(area = "+str(a) +
                         ",\tmin D= " + str(min_d) + ")\t:" + k
                         for gid, a, k, min_d in zip(gids, areas, kinds, min_ds)])

        txt = txt + '\n smax = ' + str(smax)

        if nsmall != 0:
            data = gids
        return txt, data

    elif inspect_type == 'shortedge':
        args, topolist = shapes
        thr = args[0]
        nsmall, lmax, edges, ll = check_shape_length(shape, thr,
                                                     return_area=True)
        gids = [topolist.find_gid(e) for e in edges]
        kinds = []
        for e in edges:
            params = bt.Curve(e)
            if len(params) == 2:
                # null handle case
                kinds.append("Unknown")
            else:
                curve = params[0]
                curve, kind = downcast_curve(curve)
                kinds.append(kind)

        txt = ',\n'.join([str(int(gid)) + " (L = "+str(l) + "): "+k
                          for gid, l, k in zip(gids, ll, kinds)])

        txt = txt + '\n smax = ' + str(lmax)

        if nsmall != 0:
            data = gids
        return txt, data

    elif inspect_type == 'distance':
        if shape_dim(shapes[0]) > shape_dim(shapes[1]):
            shapes = (shapes[1], shapes[0])

        if (isinstance(shapes[0], TopoDS_Vertex) and
                isinstance(shapes[1], TopoDS_Face)):
            # distance between point and surface
            pnt = bt.Pnt(shapes[0])
            p1 = np.array((pnt.X(), pnt.Y(), pnt.Z(),))
            surf = bt.Surface(shapes[1])

            pj = GeomAPI_ProjectPointOnSurf(pnt, surf)
            # print("number of solution ", pj.NbPoints())

            pnt = pj.NearestPoint()
            p2 = np.array((pnt.X(), pnt.Y(), pnt.Z(),))

            dist = np.sqrt(np.sum((p1 - p2)**2))
            ret = dist
            txt = "\n".join(["Number of projection point: " +
                             str(pj.NbPoints()),
                             "Nearest point: " + str(p2),
                             "Distance: " + str(dist)])
            return txt, data

        elif (isinstance(shapes[0], TopoDS_Vertex) and
              isinstance(shapes[1], TopoDS_Edge)):
            # distance between point and edge
            pnt = bt.Pnt(shapes[0])
            p1 = np.array((pnt.X(), pnt.Y(), pnt.Z(),))
            curve, _first, _last = bt.Curve(shapes[1])

            pj = GeomAPI_ProjectPointOnCurve(pnt, curve)
            # print("number of solution ", pj.NbPoints())

            pnt = pj.NearestPoint()
            p2 = np.array((pnt.X(), pnt.Y(), pnt.Z(),))

            dist = np.sqrt(np.sum((p1 - p2)**2))
            ret = dist

            txt = "\n".join(["Number of projection point: " +
                             str(pj.NbPoints()),
                             "Nearest point: " + str(p2),
                             "Distance: " + str(dist)])
            return txt, data

        elif (isinstance(shapes[0], TopoDS_Vertex) and
              isinstance(shapes[1], TopoDS_Vertex)):
            # distance between point and point
            pnt = bt.Pnt(shapes[0])
            p1 = np.array((pnt.X(), pnt.Y(), pnt.Z(),))
            pnt = bt.Pnt(shapes[1])
            p2 = np.array((pnt.X(), pnt.Y(), pnt.Z(),))
            dist = np.sqrt(np.sum((p1 - p2)**2))

            txt = "Distance: " + str(dist)
            return txt, data

        elif (isinstance(shapes[0], TopoDS_Edge) and
              isinstance(shapes[1], TopoDS_Face)):
            # distance between edge and face
            assert False, "not implemented"

        elif (isinstance(shapes[0], TopoDS_Edge) and
              isinstance(shapes[1], TopoDS_Edge)):
            # distance between edge and edge
            assert False, "not implemented"

        elif (isinstance(shapes[0], TopoDS_Face) and
              isinstance(shapes[1], TopoDS_Face)):
            # distance between face and face
            assert False, "not implemented"
        else:
            assert False, "not implemented"

    elif inspect_type == 'findsame':
        tol, shapes, topolists = shapes
        facelist = topolists[0]
        edgelist = topolists[1]
        gids = []
        nface = 0
        nedge = 0
        for s in shapes:
            if isinstance(s, TopoDS_Face):
                nface = nface + 1
            if isinstance(s, TopoDS_Edge):
                nedge = nedge + 1

        assert nface > 0 or nedge > 0, "Specify either faces or edges"
        assert nface == 0 or nedge == 0, "Specify either faces or edges"

        faces = [s for s in shapes if isinstance(s, TopoDS_Face)]
        edges = [s for s in shapes if isinstance(s, TopoDS_Edge)]

        if len(faces) > 0:
            samefaces = find_sameface(bt, shape, faces, tol)
            gidsf = [facelist.find_gid(f) for f in samefaces]
            gids.extend(gidsf)
        if len(edges) > 0:
            sameedges = find_sameedge(bt, shape, edges, tol)
            gidse = [edgelist.find_gid(e) for e in sameedges]
            gids.extend(gidse)

        gids_int = np.unique([int(x) for x in gids])
        if nface > 0:
            gids = [SurfaceID(int(x)) for x in gids_int]
        if nedge > 0:
            gids = [LineID(int(x)) for x in gids_int]
        txt = ',\n'.join([str(int(x)) for x in gids])

        return txt, gids
    else:
        assert False, "unknown mode" + inspect_type
    return ret, data
