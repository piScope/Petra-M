import numpy as np

import petram.debug
from petram.mfem_config import use_parallel

if use_parallel:
    import mfem.par as mfem
    from mpi4py import MPI
    num_proc = MPI.COMM_WORLD.size
    myid = MPI.COMM_WORLD.rank
    from petram.helper.mpi_recipes import *
else:
    import mfem.ser as mfem

dprint1, dprint2, dprint3 = petram.debug.init_dprints('BoundaryRefinement')



def apply_boundary_refinement(mesh, sels, nlayers=6):
    '''
    refinement near the boundary element sels.
    '''
    dprint1("applying boundary refinement: " + str(sels) + ", nlayers=" + str(nlayers))
    dim = mesh.Dimension()

    bdrs = mesh.GetBdrAttributeArray()
    ibdrs = np.where(np.in1d(bdrs, sels))[0]

    ifaces = [mesh.GetBdrFace(i) for i in ibdrs]
    iels = [mesh.GetFaceElementTransformations(i).Elem1No for i in ifaces]

    layers = [iels]
    iels = layers[-1][:]

    if dim == 2:
        for i in range(nlayers):
            new_layer = []
            for i in layers[-1]:
                edges = mesh.GetElementEdges(i)[0]
                for edge in edges:
                    trs = mesh.GetFaceElementTransformations(edge)
                    e1 = trs.Elem1No
                    e2 = trs.Elem2No
                    if e1 not in iels and e1 != -1:
                        if e1 not in new_layer:
                            new_layer.append(e1)
                    if e2 not in iels and e2 != -1:
                        if e2 not in new_layer:
                            new_layer.append(e2)

            iels.extend(new_layer)
            layers.append(new_layer)

    elif dim == 3:
        for i in range(nlayers):
            new_layer = []
            for i in layers[-1]:
                faces = mesh.GetElementFaces(i)[0]
                for face in faces:
                    trs = mesh.GetFaceElementTransformations(face)
                    e1 = trs.Elem1No
                    e2 = trs.Elem2No
                    if e1 not in iels and e1 != -1:
                        if e1 not in new_layer:
                            new_layer.append(e1)
                    if e2 not in iels and e2 != -1:
                        if e2 not in new_layer:
                            new_layer.append(e2)

            iels.extend(new_layer)
            layers.append(new_layer)

        pass

    else:
        pass

    iii = mfem.intArray(iels)
    mesh.GeneralRefinement(iii)
    return mesh
