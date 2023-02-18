'''
  nodal_refinement

  generate geometrically refined dataset using nodal values.
  note that the data inside the element is simply interplated
  from nodal values and H1 weight. 

'''
import numpy as np
import weakref
import six
from petram.mfem_config import use_parallel
if use_parallel:
    import mfem.par as mfem
    from mfem.par import GlobGeometryRefiner as GR    
else:
    import mfem.ser as mfem
    from mfem.ser import GlobGeometryRefiner as GR
    
Geom = mfem.Geometry()

weight = {}

def surface_weight(refine, gtype):
    if (refine, gtype) in globals()['weight']:
        return globals()['weight'][(refine, gtype)]
    
    quad_v = [[0, 0], [1, 0], [1, 1], [0, 3]]
    quad_e = [[0, 1, 2, 3]]
    tri_v =   [[0, 0], [1, 0], [0, 1]]
    tri_e = [[0, 1, 2,]]
    seg_v =   [[0,], [1, ],]
    seg_e = [[0, 1,]]

    if gtype == mfem.Geometry.TRIANGLE:
        mesh = mfem.Mesh(2, 3, 1, 0, 2)
        for j in range(3):
            mesh.AddVertex(tri_v[j])
        for j in range(1):
            mesh.AddTri(tri_e[j], 11)
        mesh.FinalizeTriMesh(1,1, True)

    elif gtype == mfem.Geometry.SQUARE:
        mesh = mfem.Mesh(2, 4, 1, 0, 2)
        for j in range(4):
            mesh.AddVertex(quad_v[j])
        for j in range(1):
            mesh.AddQuad(quad_e[j], 11)
        mesh.FinalizeQuadMesh(1,1, True)
    elif gtype == mfem.Geometry.SEGMENT:
        mesh = mfem.Mesh(1, 2, 1, 0, 1)
        for j in range(2):
            mesh.AddVertex(seg_v[j])
        for j in range(1):
            seg = mfem.Segment(seg_e[j], j+1)
            mesh.AddElement(seg)
            seg.thisown = False        
            mesh.FinalizeTopology()         
            mesh.Finalize(False, False)

    fec_type = mfem.H1_FECollection
    fe_coll = fec_type(1, 2)
    fes = mfem.FiniteElementSpace(mesh, fe_coll, 2)
    el = fes.GetFE(0)
    npt = Geom.GetVertices(gtype).GetNPoints()
    RefG = GR.Refine(gtype, refine)

    shape = mfem.Vector(el.GetDof())              
    ir = RefG.RefPts

    shapes =[]
    for i in range(ir.GetNPoints()):
        el.CalcShape(ir.IntPoint(i), shape)
        shapes.append(shape.GetDataArray().copy())
        
    w = np.vstack(shapes)
    globals()['weight'][(refine, gtype)] = w

    return w
    

def refine_surface_data(mesh, ibele, val, idx, refine):

    if mesh.Dimension() == 3:
        getarray = mesh.GetBdrArray
        gettrans = mesh.GetBdrElementTransformation
        getelement = mesh.GetBdrElement
        getbasegeom = mesh.GetBdrElementBaseGeometry
        getvertices = mesh.GetBdrElementVertices
    elif mesh.Dimension() == 2:
        getarray = mesh.GetDomainArray
        gettrans = mesh.GetElementTransformation        
        getelement = mesh.GetElement
        getbasegeom = mesh.GetElementBaseGeometry
        getvertices = mesh.GetElementVertices
    else:
        assert False, "BdrNodal Evaluator is not supported for this dimension"


    ptx = []
    data = []
    ridx = []

    nele = 0
    gtype_st = -1    
    sdim = mesh.SpaceDimension()

    for k, i in enumerate(ibele):
        gtype = getbasegeom(i)
        verts = getvertices(i)

        if gtype != gtype_st:
            RefG = GR.Refine(gtype, 3)
            ir = RefG.RefPts
            npt = ir.GetNPoints()
            ele = np.array(RefG.RefGeoms.ToList()).reshape(-1, len(verts))
            w = surface_weight(refine, gtype)
            gtype_st = gtype

        T = gettrans(i)
        pt = np.vstack([T.Transform(ir.IntPoint(j)) for j in range(npt)])
        ptx.append(pt)
        d = val[idx[k]]
        data.append(np.dot(w, d))
        ridx.append(ele + nele)
        nele = nele + ir.GetNPoints()
        
    ptx = np.vstack(ptx)
    data = np.hstack(data)
    ridx = np.vstack(ridx)
    return ptx, data, ridx

def refine_edge_data(mesh, ibele, val, idx, refine):

    if mesh.Dimension() == 2:
        getarray = mesh.GetBdrArray
        gettrans = mesh.GetBdrElementTransformation
        getelement = mesh.GetBdrElement
        getbasegeom = mesh.GetBdrElementBaseGeometry
        getvertices = mesh.GetBdrElementVertices
    elif mesh.Dimension() == 1:
        getarray = mesh.GetDomainArray
        gettrans = mesh.GetElementTransformation        
        getelement = mesh.GetElement
        getbasegeom = mesh.GetElementBaseGeometry
        getvertices = mesh.GetElementVertices
    else:
        assert False, "BdrEdge Evaluator is not supported for this dimension"


    ptx = []
    data = []
    ridx = []

    nele = 0
    gtype_st = -1    
    sdim = mesh.SpaceDimension()

    for k, i in enumerate(ibele):
        gtype = getbasegeom(i)
        verts = getvertices(i)

        if gtype != gtype_st:
            RefG = GR.Refine(gtype, refine)
            ir = RefG.RefPts
            npt = ir.GetNPoints()
            ele = np.array(RefG.RefGeoms.ToList()).reshape(-1, len(verts))
            w = surface_weight(refine, gtype)
            gtype_st = gtype

        T = gettrans(i)
        pt = np.vstack([T.Transform(ir.IntPoint(j)) for j in range(npt)])
        ptx.append(pt)
        d = val[idx[k]]
        data.append(np.dot(w, d))
        ridx.append(ele + nele)
        nele = nele + ir.GetNPoints()
        
    ptx = np.vstack(ptx)
    data = np.hstack(data)
    ridx = np.vstack(ridx)
    return ptx, data, ridx

