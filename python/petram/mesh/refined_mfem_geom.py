import numpy as np
from collections import defaultdict

from mfem.ser import GlobGeometryRefiner as GR
from mfem.ser import DenseMatrix
default_refine = 5

def get_geom(idx, size, base, get_transformation, attrs, sdim, refine = None):
    # base 
    # SEGMENT     = 1
    # TRIANGLE    = 2
    # SQUARE      =p 3
    # TETRAHEDRON = 4
    # CUBE        = 5
    if refine is None: refine = default_refine

    RefG = GR.Refine(base, refine)
    T = get_transformation(idx[0])
    ir = RefG.RefPts
    npt = ir.GetNPoints()
    ref_idx = np.array(RefG.RefGeoms.ToList()).reshape(-1, size)
    refe_idx = np.array(RefG.RefEdges.ToList()).reshape(-1, 2)
    ptx_x = np.zeros((npt*len(idx), sdim), dtype=np.float32)

    m = DenseMatrix()

    ptx_data = [None]*len(idx)
    for k, i in enumerate(idx):
        T = get_transformation(i)
        T.Transform(ir, m)
        ptx_data[k] = m.GetDataArray().copy()
    ptx_x = np.hstack(ptx_data).transpose().astype(np.float32, copy=False)

    '''
    for k, i in enumerate(idx):
        T = get_transformation(i)
        ptx = np.vstack([T.Transform(ir.IntPoint(j)) for j in range(npt)])
        ptx_x[k*npt:(k+1)*npt, :] = ptx.astype(np.float32, copy = False)

    '''
    ivert_x = (np.tile(ref_idx, (len(idx), 1)) +
             np.repeat(np.arange(len(idx))*npt,
                       len(ref_idx)).reshape(-1, 1))
    iverte_x = (np.tile(refe_idx, (len(idx), 1)) +
             np.repeat(np.arange(len(idx))*npt,
                       len(refe_idx)).reshape(-1, 1))

    if isinstance(attrs, dict):  # for edge
        a = [attrs[i] for i in idx]
    else:
        a = attrs[idx]
    attrs_x = np.repeat(a, npt)

    ivert_x = ivert_x.flatten().reshape(-1,size)
    attrs = a


    return attrs, ptx_x, ivert_x, iverte_x, attrs_x
