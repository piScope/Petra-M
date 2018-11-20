import numpy as np

from petram.mfem_config import use_parallel

if use_parallel:
   import mfem.par as mfem
   from mpi4py import MPI
   num_proc = MPI.COMM_WORLD.size
   myid     = MPI.COMM_WORLD.rank
   from petram.helper.mpi_recipes import *   
else:
   import mfem.ser as mfem
   
import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('Mesh1Dl')

def straight_line_mesh(lengths, nsegs, filename='',
                       refine=False, fix_orientation=False,
                       sdim = 3, x0=0.0):

    Nvert = np.sum(nsegs)+1
    Nelem = np.sum(nsegs)
    Nbdrelem = len(lengths)+1
    mesh = mfem.Mesh(1, Nvert, Nelem,  Nbdrelem, sdim)

    ivert = {}
    L = np.hstack(([0], np.cumsum(lengths))).astype(float)
    P = np.hstack(([0], np.cumsum(nsegs))).astype(int)
    X = [np.linspace(L[i], L[i+1], n+1)[1:] for i, n in enumerate(nsegs)]
    X = np.hstack(([0], np.hstack(X)))
    A = np.hstack([[i+1]*n for i, n in enumerate(nsegs)])
    for k, i in enumerate(P):
        ptx = mfem.Point(i)
        ptx.SetAttribute(k+1)
        mesh.AddBdrElement(ptx)
        ptx.thisown = False
        
    for i in range(X.shape[0]-1):
        seg = mfem.Segment((i, i+1), A[i])
        mesh.AddElement(seg)
        seg.thisown = False        
    for i in range(X.shape[0]):
         pt = [0]*sdim
         pt[0] = X[i] + x0
         mesh.AddVertex(pt)
         
    mesh.FinalizeTopology()         
    mesh.Finalize(refine, fix_orientation)

    if filename != '':
        mesh.PrintToFile(filename, 8)
    return mesh


