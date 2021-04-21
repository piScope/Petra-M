'''
   extract_refined_mesh_data3

   extract refined  mesh data from 2D (=ndim)  mesh
'''
import numpy as np
from collections import defaultdict

from petram.mesh.find_edges import find_edges
from petram.mesh.find_vertex import find_vertex
from mfem.ser import GlobGeometryRefiner as GR

def extract_refined_mesh_data1(mesh, refine=None):
    ndim = mesh.Dimension()
    sdim = mesh.SpaceDimension()

    ivert0 = [mesh.GetElement(i).GetVerticesArray()
              for i in range(mesh.GetNE())]
    idx2 = np.arange(mesh.GetNE())
    iv2 = [np.arange(mesh.GetNV())]

    table = np.arange(mesh.GetNV())

    battrs = mesh.GetAttributeArray()
    gt = mesh.GetElementTransformation

    cells = {}
    cell_data = {}

    ptx = [mesh.GetVertexArray() for i in range(mesh.GetNV())]

    base = 1

    from petram.mesh.refined_mfem_geom import get_geom

    if len(idx2) > 0:
        attr2, ptx2, ivx2, ivxe2, attrx2 = get_geom(idx2, 2, base, gt, battrs,
                                                    sdim, refine,)
        ptx.append(ptx2)
        cells['line_x'] = ivx2
        cell_data['line_x'] = {}
        cell_data['line_x']['physical'] = attrx2
        cell_data['X_refined_edge']=ptx2

    X = np.vstack(ptx)

    from petram.mesh.mesh_utils import populate_plotdata

    l_s_loop = populate_plotdata(mesh, table, cells, cell_data)

    iedge2bb = None # is it used?

    #print "X", X.shape
    #for k in cells['vertex']:print X[k]
    return X, cells, cell_data, l_s_loop, iedge2bb


