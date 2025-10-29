'''
 OCC heal shape extra
'''
import petram.geom.occ_cbook
from petram.geom.occ_cbook import *
from scipy.spatial import distance_matrix


def _make_wire(edges):
    wireMaker = BRepBuilderAPI_MakeWire()
    for e in edges:
        wireMaker.Add(e)
    wireMaker.Build()

    if not wireMaker.IsDone():
        assert False, "Failed to make wire"

    return wireMaker.Wire()


def _create_plane_filling(edges1):
    # then, create group of edges becomes wire
    wire = _make_wire(edges1)

    faceMaker = BRepBuilderAPI_MakeFace(wire)
    faceMaker.Build()

    if not faceMaker.IsDone():
        assert False, "can not create face"

    face = faceMaker.Face()
    fixer = ShapeFix_Face(face)
    fixer.Perform()
    face = fixer.Face()
    return face


def _create_surface_filling(edges, occ_geom_tolerance):
    from OCC.Core.GeomAbs import GeomAbs_C0
    from OCC.Core.BRepTools import BRepTools_WireExplorer

    bt = BRep_Tool()
    f = BRepOffsetAPI_MakeFilling()

    # make wire first
    wire = _make_wire(edges)

    # make wire constraints
    ex1 = BRepTools_WireExplorer(wire)
    while ex1.More():
        edge = topods_Edge(ex1.Current())
        f.Add(edge, GeomAbs_C0)
        ex1.Next()

    f.Build()

    if not f.IsDone():
        assert False, "Cannot make filling"

    face = f.Shape()
    s = bt.Surface(face)

    faceMaker = BRepBuilderAPI_MakeFace(s, wire)

    result = faceMaker.Face()

    fix = ShapeFix_Face(result)
    fix.SetPrecision(occ_geom_tolerance)
    fix.Perform()
    fix.FixOrientation()
    return fix.Face()


def split_hairlineface(face, limit=0.1):
    '''
    x------------------x-----------------x
    |                                    |
    x------------------x-----------------x
    When a face is very skinny, split the face using the two vertices
    which are not on the same edge, and has a smallest distance.

    at moment, the face needs to be planner. once the face is split,
    a user can use CADfix collapse the short edge.
    '''
    bt = BRep_Tool()

    # vertices
    vertices = [p for p in iter_shape_once(face, 'vertex')]

    # create group of edges
    edge_connection = {}
    edge_shapes = {}
    edges = [p for p in iter_shape_once(face, 'edge')]

    for e in edges:
        mapper = get_mapper(e, 'vertex')
        idx = np.where([mapper.Contains(v) for v in vertices])[0]

        if len(idx) != 2:
            assert False, "self-looping edge was found"
        if idx[0] not in edge_connection:
            edge_connection[idx[0]] = idx[1]
        else:
            edge_connection[idx[1]] = idx[0]
        edge_shapes[tuple(idx)] = e

    ptx = []
    for v in vertices:
        pnt = bt.Pnt(v)
        p = np.array((pnt.X(), pnt.Y(), pnt.Z(),))
        ptx.append(p)

    ptx = np.vstack(ptx)
    md = distance_matrix(ptx, ptx, p=2)

    print("number of vertex in face: " + str(len(vertices)))
    short_pair = []
    for i in range(len(ptx)):
        for j in range(len(ptx)):
            if j > i:
                break
            if md[i, j] < limit:
                if np.abs(i-j) == 0:
                    continue
                if (i, j) in edge_shapes:
                    continue
                if (j, i) in edge_shapes:
                    continue
                short_pair.append((i, j))
    print("short distance pair", short_pair)
    if len(short_pair) > 1:
        dd = np.sort(md.flatten())
        dd = dd[dd > 0]
        print("distances: " + ", ".join([str(x) for x in dd]))
        print("more than two short pairs are found (try smaller limit?)")
    if len(short_pair) == 0:
        print("short internal edge is not found")
        print(md)
        return None
    p1 = min(short_pair[0])
    p2 = max(short_pair[0])

    # creat a new edge
    edgeMaker = BRepBuilderAPI_MakeEdge(vertices[p1], vertices[p2])
    edgeMaker.Build()
    if not edgeMaker.IsDone():
        assert False, "Can not make line"
    new_edge = edgeMaker.Edge()

    # make loops for new faces
    loop1 = [p1]
    while True:
        pp = edge_connection[loop1[-1]]
        loop1.append(pp)
        if pp == p2:
            # loop1.append(p1)
            break
    loop2 = [p2]
    while True:
        pp = edge_connection[loop2[-1]]
        loop2.append(pp)
        if pp == p1:
            # loop2.append(p2)
            break

    # make a list of edges to create surface
    edges1 = [edge_shapes[(loop1[i], loop1[i+1])]
              if (loop1[i], loop1[i+1]) in edge_shapes else
              edge_shapes[(loop1[i+1], loop1[i])]
              for i in range(len(loop1)-1)]
    edges1.append(new_edge)
    edges2 = [edge_shapes[(loop2[i], loop2[i+1])]
              if (loop2[i], loop2[i+1]) in edge_shapes else
              edge_shapes[(loop2[i+1], loop2[i])]
              for i in range(len(loop2)-1)]
    edges2.append(new_edge)

    #face1 = make_filling(wire1)
    #face2 = make_filling(wire2)
    face1 = _create_plane_filling(edges1)
    face2 = _create_plane_filling(edges2)

    new_faces = (face1, face2)
    return new_faces


def create_cap_face(volume, faces, use_filling, occ_geom_tolerance):
    '''
    creat a capping face (surface filling using the outer edge
    of faces
    '''
    all_edges = []
    mapper = get_mapper(volume, 'edge')

    for f in faces:
        for e in iter_shape_once(f, 'edge'):
            all_edges.append(e)

    eidx = []
    arr = sorted([mapper.FindIndex(e) for e in all_edges])

    if arr[0] != arr[1]:
        eidx.append(arr[0])
    if arr[-2] != arr[-1]:
        eidx.append(arr[-1])
    for i in range(1, len(arr) - 1):
        if (arr[i] != arr[i + 1] and
                arr[i] != arr[i - 1]):
            eidx.append(arr[i])

    outer_edges = []
    for e in all_edges:
        if mapper.FindIndex(e) in eidx:
            outer_edges.append(e)

    if use_filling:
        face = _create_surface_filling(outer_edges, occ_geom_tolerance)
    else:
        face = _create_plane_filling(outer_edges)

    return face

def replace_faces(volume, old_faces, new_faces):
    '''
    relace old_faces by new_faces of volume
    '''
    mapper = get_mapper(volume, 'face')
    ifaces = [mapper.FindIndex(f) for f in old_faces]

    allfaces = [p for p in iter_shape_once(volume, 'face')
                if mapper.FindIndex(p) not in ifaces]

    allfaces.extend(new_faces)

    def make_surface_loop(faces):
        # first sew the surfaces.
        try:
            sewingMaker = BRepBuilderAPI_Sewing()
            for face in faces:
                sewingMaker.Add(face)
            sewingMaker.Perform()
            result = sewingMaker.SewedShape()
        except BaseException:
            assert False, "Failed to sew faces"

        fixer = ShapeFix_Shell(result)
        fixer.Perform()
        shell = fixer.Shell()

        return shell

    shell = make_surface_loop(allfaces)

    solidMaker = BRepBuilderAPI_MakeSolid()
    solidMaker.Add(shell)
    result = solidMaker.Solid()

    if not solidMaker.IsDone():
        assert False, "Failed to make solid"

    fixer = ShapeFix_Solid(result)
    fixer.Perform()
    solid = topods_Solid(fixer.Solid())

    return solid
