'''
   eval_shape:

   this routine demonstrates how to evaluate shape function 
   with finer resolution

   (example)
    dof, ptx, shape_l2 = petram.helper.eval_shape.eval_shape(order=1, refine=10, fec='RT')
    petram.helper.eval_shape.plot_shape(dof, ptx, shape_l2)


'''
import numpy as np

from petram.mfem_config import use_parallel
if use_parallel:
   from petram.helper.mpi_recipes import *
   import mfem.par as mfem   
else:
   import mfem.ser as mfem

def eval_shape(order = 1, refine = 5, elem_type = 0,
               fec = 'ND'):

    if fec == 'ND':
        fec_type = mfem.ND_FECollection
        if order < 1: assert False, "ND order is 1 and above"        
    elif fec == 'RT':
        fec_type = mfem.RT_FECollection
    elif fec == 'H1':
        fec_type = mfem.H1_FECollection
        if order < 1: assert False, "H1 order is 1 and above"
    elif fec == 'L2':
        fec_type = mfem.L2_FECollection
    else:
        assert False, "unknown basis"
        
    if elem_type == 0:
        Nvert = 3; Nelem = 1; spaceDim = 2
    elif elem_type == 1:    
        Nvert = 4; Nelem = 1; spaceDim = 2

    mesh = mfem.Mesh(2, Nvert, Nelem, 0, spaceDim)

    if elem_type == 0:
        tri_v = [[1.,  0.3 ], [0.,  1.,], [0, 0]]
        tri_e = [[0, 1, 2], ]
        for j in range(Nvert):
            mesh.AddVertex(tri_v[j])
        for j in range(Nelem):
            mesh.AddTriangle(tri_e[j], j+1)
        mesh.FinalizeTriMesh(1,1, True)
    else:
        quad_v = [[-1, -1.3, ], [+1, -1, ], [+1, +1, ], [-1, +1,]]
        quad_e = [[0, 1, 2, 3]]
        for j in range(Nvert):
            mesh.AddVertex(quad_v[j])
        for j in range(Nelem):
            mesh.AddQuad(quad_e[j], j+1)
        mesh.FinalizeQuadMesh(1,1, True)

    #mesh.PrintToFile('plot_basis.mesh', 8)

    fe_coll = fec_type(order, spaceDim)
    fespace = mfem.FiniteElementSpace(mesh, fe_coll)

    x = mfem.GridFunction(fespace)
    x.Assign(0.0)
    x[0] = 1.0

    idx = 0
    geom = mesh.GetElementBaseGeometry(idx)
    T = mesh.GetElementTransformation(idx)
    fe = fespace.GetFE(idx)
    fe_nd = fe.GetDof()

    ir = fe.GetNodes()
    npt = ir.GetNPoints()
    dof = np.vstack([T.Transform(ir.IntPoint(i)) for i in range(npt)])
    
    RefG = mfem.GlobGeometryRefiner.Refine(geom, refine, 1);
    ir = RefG.RefPts

    npt = ir.GetNPoints()
    ptx = np.vstack([T.Transform(ir.IntPoint(i)) for i in range(npt)])

    shape = []
    if fec == 'ND' or fec == 'RT':
        mat = mfem.DenseMatrix(fe_nd, spaceDim)
        shape_func = fe.CalcVShape
        for i in range(npt):
            ip = ir.IntPoint(i)
            T.SetIntPoint(ip)
            fe.CalcVShape(T, mat)
            shape.append(mat.GetDataArray().copy())
    else:
        vec = mfem.Vector(fe_nd)
        for i in range(npt):
            ip = ir.IntPoint(i)            
            fe.CalcShape(ip, vec)
            shape.append(vec.GetDataArray().copy())

    return dof, ptx, np.stack(shape)

def plot_shape(dof, ptx,  shape, viewer = None):
    try:
        viewer.Validate()
    except:
        from ifigure.interactive import figure
        viewer = figure()

    ll = shape.shape[1]
    n = int(np.ceil(np.sqrt(ll)))
    
    viewer.cls()
    viewer.nsec(n, int(np.ceil(float(ll)/float(n))))

    for i in range(ll):
        x = ptx[:, 0]; y= ptx[:, 1]
        viewer.isec(i)
        if len(shape.shape) == 3:
            viewer.tripcolor(ptx[:,0], ptx[:,1], ptx[:,1]*0)            
            u=shape[:, i, 0]; v = shape[:, i, 1]            
            viewer.quiver(x, y, u, v)
        else:
            viewer.tripcolor(ptx[:,0], ptx[:,1], shape[:,i])
            idx = np.where(np.abs(shape[:,i]) < 1e-10)[0]
            if len(idx)>0: viewer.plot(ptx[idx,0], ptx[idx,1], 'gs')
        viewer.plot([dof[i][0]], [dof[i][1]], 'or')
    
if __name__ == '__main__':
    ans = eval_shape(fec = 'RT')
    for k in ans:
        print(k.shape)
