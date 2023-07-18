import numpy as np
import mfem.ser as mfem


def apply_boundary_refinement(mesh, sels, nlayers=6):
    '''
    refinement near the boundary element sels.
    '''
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
