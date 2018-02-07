import os
import numpy as np
import mfem.ser as mfem
from petram.sol.solsets import read_solsets

def generate_surface(mesh = None, filename = '', precision=8,
                     obj=None, i = 157):

    '''
    mesh must be 
    if sdim == 3:
       a domain of   2D mesh
       a boundary of 3D mesh
    if sdim == 2:
       a domain  in 2D mesh

    filename : an option to save the file 
    return new surface mesh

    # MFEM Geometry Types (see mesh/geom.hpp):
    #
    # POINT       = 0
    # SEGMENT     = 1
    # TRIANGLE    = 2
    # SQUARE      = 3
    # TETRAHEDRON = 4
    # CUBE        = 5
    '''
    

    if obj is not None:
        path = obj.owndir()
    else:
        path = os.getcwd()
    if mesh is None:
        solsets = read_solsets(path)
        mesh = solsets.meshes[0]

    sdim = mesh.SpaceDimension()
    dim = mesh.Dimension()
    Nodal = mesh.GetNodalFESpace()
    hasNodal = (Nodal is not None)    

    if sdim == 3:
       if dim == 3:
           GetXAttributeArray  = mesh.GetBdrAttributeArray
           GetXElementVertices = mesh.GetBdrElementVertices
           GetXBaseGeometry    = mesh.GetBdrElementBaseGeometry
       elif dim == 2:
           GetXAttributeArray  = mesh.GetAttributeArray
           GetXElementVertices = mesh.GetElementVertices
           GetXBaseGeometry    = mesh.GetElementBaseGeometry
       else:
            assert False, "not supprint sdim==3, dim==1"
    elif sdim == 2:
       GetXAttributeArray  = mesh.GetAttributeArray
       GetXElementVertices = mesh.GetElementVertices
       GetXBaseGeometry    = mesh.GetElementBaseGeometry
       if hasNodal:
    else:
        assert False, "not supprint sdim==1"
    attrs = GetXAttributeArray()
    idx = np.where(attrs == i)[0]

    u, indices = np.unique([GetXElementVertices(i) for i in idx],
                            return_inverse = True)
    base = [GetXBaseGeometry(i) for i in idx]
    vtx = np.vstack([mesh.GetVertexArray(i) for i in u])

    Nvert = len(u)
    Nelem = len(idx)
    spaceDim = sdim

    if hasNodal:



    omesh = mfem.Mesh(2, Nvert, Nelem, 0, spaceDim)

    k = 0
    for i in range(Nelem):
        iv = GetXElementVertices(i)
        iv = indices[k:k+len(iv)]
        k = k + len(iv)
        if base[i] == 2:  # triangle
            omesh.AddTri(list(iv), 1)
        elif base[i] == 3: # quad
            omesh.AddQuad(list(iv), 1)
        else:
            assert False, "unsupported base geometry: " + str(base[i])

    for i in range(Nvert):
         omesh.AddVertex(list(vtx[i]))

    if hasNodal:
        fec = Nodal.FEColl()
        dNodal = mfem.FiniteElementSpace(mesh, fec, spaceDim)
        omesh.SetNodalFESpace(dNodal)
        omesh._nodal= dNodal

        if sdim == 3:
           if dim == 3:
               GetXDofs        =  Nodal.GetBdrElementDofs
               dGetXDofs       = dNodal.GetBdrElementDofs
           elif dim == 2:
               GetXDofs        =  Nodal.GetElementDofs
               dGetXDofs       = dNodal.GetElementDofs
        elif sdim == 2:
           GetXDofs         =  Nodal.GetElementDofs
           dGetXDofs        = dNodal.GetElementDofs
        DofToVDof        =  Nodal.DofToVDof
        dDofToVDof       = dNodal.DofToVDof

        nodes = mesh.GetNodes()
        node_ptx1 = nodes.GetDataArray()

        onodes = mesh.GetNodes()
        node_ptx2 = onodes.GetDataArray()

        for j in idx: 
            dofs1 = GetXDofs(j)
            dof1_idx = np.array([[DofToVDof(i, d)
                                 for d in range(sdim)] for i in dofs1])
            dofs2 = dGetXDofs(j)
            dof2_idx = np.array([[dDofToVDof(i, d)
                                 for d in range(sdim)] for i in dofs2])
            for i1, i2 in zip(dof1_idx, dof2_idx):
                node_ptx2[i2] = node_ptx1[i1]

    omesh.FinalizeTopology()
    omesh.Finalize(refine=False, fix_orientation=True)
    #mesh.FinalizeTriMesh(1,1, True)
    if filename != '':
        omesh.PrintToFile(filename, precision)

    return omesh


