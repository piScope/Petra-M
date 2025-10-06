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

def quad_rectangle_mesh(xlengths, xnsegs, ylengths, ynsegs,
                        filename='', refine=False, fix_orientation=False,
                        sdim=3, x0=None):

    x0 = (0.0, 0.0) if x0 is None else x0
    
    Nvert = (np.sum(xnsegs)+1)*(np.sum(ynsegs)+1)
    Nelem = np.sum(xnsegs)*np.sum(ynsegs)
    Nbdrelem = (np.sum(xnsegs)*(len(ylengths)+1) + np.sum(ynsegs)*(len(xlengths)+1))
    mesh = mfem.Mesh(2, Nvert, Nelem,  Nbdrelem, sdim)
    
    Lx = np.hstack(([0], np.cumsum(xlengths)))
    x = [np.linspace(Lx[i], Lx[i+1], n+1)[:-1].astype(float) for i, n in enumerate(xnsegs)]+[np.sum(xlengths)]
    x = np.hstack(x)
    kx = np.hstack(([0], np.cumsum(xnsegs)))
    kkx = sum([[i]*n for i,n in enumerate(xnsegs)],[])
    
    Ly = np.hstack(([0], np.cumsum(ylengths)))
    y = [np.linspace(Ly[i], Ly[i+1], n+1)[:-1].astype(float) for i, n in enumerate(ynsegs)]+[np.sum(ylengths)]
    y = np.hstack(y)
    ky = np.hstack(([0], np.cumsum(ynsegs)))
    kky = sum([[i]*n for i,n in enumerate(ynsegs)],[])

    X, Y = np.meshgrid(x, y)

    X = X + x0[0]
    Y = Y + x0[1]
    
    IDX = np.arange(len(x)*len(y)).reshape(len(y), len(x))

    ax = np.hstack([[i]*n for i, n in enumerate(xnsegs)])
    ay = np.hstack([[j]*n for j, n in enumerate(ynsegs)])
    AX, AY = np.meshgrid(ax, ay)
    KDOM = AY*len(xnsegs) + AX + 1

    e_count = 1
    for i in range(len(y[:-1])):
        for j in range(len(x[:-1])):
            elem = [IDX[i, j], IDX[i, j+1], IDX[i+1, j+1], IDX[i+1, j]]
            mesh.AddQuad(elem, KDOM[i, j])
            e_count = e_count + 1

    kbdr = 1
    b_count = 1

    for k, j in enumerate(kx):
       for i in range(len(y[:-1])):
            elem = [IDX[i, j], IDX[i+1, j]]
            mesh.AddBdrSegment(elem, kbdr+kky[i])
            b_count = b_count + 1
       kbdr = kbdr + len(ynsegs)
        
    for k, i in enumerate(ky):
       for j in range(len(x[:-1])):
            elem = [IDX[i, j], IDX[i, j+1]]
            mesh.AddBdrSegment(elem, kbdr+kkx[j])
            b_count = b_count + 1            
       kbdr = kbdr + len(xnsegs)
        
    v_count = 1
    for xx, yy in zip(X.flatten(), Y.flatten()):
        ptx = [xx, yy, 0.0]
        mesh.AddVertex(ptx)
        v_count = v_count + 1
        
    #print(e_count, b_count, v_count)
    #print(Nelem,  Nbdrelem, Nvert)

    mesh.FinalizeTopology()         
    mesh.Finalize(refine, fix_orientation)

    if filename != '':
        mesh.PrintToFile(filename, 8)
    return mesh
 
def hex_box_mesh(xlengths, xnsegs, ylengths, ynsegs, zlengths, znsegs,
                 filename='', refine=False, fix_orientation=False,
                 sdim=3, x0=None):
   
    x0 = (0.0, 0.0, 0.0) if x0 is None else x0

    Nvert = (np.sum(xnsegs)+1)*(np.sum(ynsegs)+1)*(np.sum(znsegs)+1)
    Nelem = np.sum(xnsegs)*np.sum(ynsegs)*np.sum(znsegs)
    Nbdrelem = (np.sum(xnsegs)*np.sum(ynsegs)*(len(zlengths)+1) + 
                np.sum(xnsegs)*np.sum(znsegs)*(len(ylengths)+1) +
                np.sum(znsegs)*np.sum(ynsegs)*(len(xlengths)+1))

    mesh = mfem.Mesh(3, Nvert, Nelem,  Nbdrelem, sdim)
    
    Lx = np.hstack(([0], np.cumsum(xlengths)))
    x = [np.linspace(Lx[i], Lx[i+1], n+1)[:-1].astype(float) for i, n in enumerate(xnsegs)]+[np.sum(xlengths)]
    x = np.hstack(x)
    kx = np.hstack(([0], np.cumsum(xnsegs)))
    kkx = sum([[i]*n for i,n in enumerate(xnsegs)],[])
    
    Ly = np.hstack(([0], np.cumsum(ylengths)))
    y = [np.linspace(Ly[i], Ly[i+1], n+1)[:-1].astype(float) for i, n in enumerate(ynsegs)]+[np.sum(ylengths)]
    y = np.hstack(y)
    ky = np.hstack(([0], np.cumsum(ynsegs)))
    kky = sum([[i]*n for i,n in enumerate(ynsegs)],[])

    Lz = np.hstack(([0], np.cumsum(zlengths)))
    z = [np.linspace(Lz[i], Lz[i+1], n+1)[:-1].astype(float) for i, n in enumerate(znsegs)]+[np.sum(zlengths)]
    z = np.hstack(z)
    kz = np.hstack(([0], np.cumsum(znsegs)))
    kkz = sum([[i]*n for i,n in enumerate(znsegs)],[])
    
    X, Y, Z = np.meshgrid(x, y, z)   # y-idx, x-idx, z-idx

    X = X + x0[0]
    Y = Y + x0[1]
    Z = Z + x0[2]    
    
    IDX = np.arange(len(x)*len(y)*len(z)).reshape(len(y), len(x), len(z))

    ax = np.hstack([[i]*n for i, n in enumerate(xnsegs)])
    ay = np.hstack([[j]*n for j, n in enumerate(ynsegs)])
    az = np.hstack([[j]*n for j, n in enumerate(znsegs)])
    
    AX, AY, AZ = np.meshgrid(ax, ay, az)
    KDOM = AZ*len(xnsegs)*len(ynsegs) + AY*len(xnsegs) + AX + 1

    e_count = 1
    
    for i in range(len(y[:-1])):
        for j in range(len(x[:-1])):
            for k in range(len(z[:-1])):
               elem = [IDX[i, j, k ], IDX[i, j+1, k  ], IDX[i+1, j+1, k  ], IDX[i+1, j, k  ],
                       IDX[i, j,k+1], IDX[i, j+1, k+1], IDX[i+1, j+1, k+1], IDX[i+1, j, k+1]]
               mesh.AddHex(elem, KDOM[i, j, k])
               e_count = e_count + 1

    kbdr = 1
    b_count = 1

    for j in kx:
        for i in range(len(y[:-1])):
            for k in range(len(z[:-1])):
                elem = [IDX[i, j, k], IDX[i+1, j, k], IDX[i+1, j, k+1], IDX[i, j, k+1]] 
                mesh.AddBdrQuad(elem, kbdr+kky[i]*len(znsegs) + kkz[k])
                b_count = b_count + 1                
        kbdr = kbdr + len(ynsegs)*len(znsegs)


    for i in ky:
        for k in range(len(z[:-1])):                 
            for j in range(len(x[:-1])):
                elem = [IDX[i, j, k], IDX[i, j+1, k], IDX[i, j+1, k+1], IDX[i, j, k+1]]
                mesh.AddBdrQuad(elem, kbdr+kkz[k]*len(xnsegs) + kkx[j])
                b_count = b_count + 1                                                
        kbdr = kbdr + len(xnsegs)*len(znsegs)

    for k in kz:
        for j in range(len(x[:-1])):       
            for i in range(len(y[:-1])):
                elem = [IDX[i, j, k], IDX[i+1, j, k], IDX[i+1, j+1, k], IDX[i, j+1, k]]               
                mesh.AddBdrQuad(elem, kbdr+kkx[j]*len(ynsegs) + kky[i])
                b_count = b_count + 1                                
        kbdr = kbdr + len(xnsegs)*len(ynsegs)
       
    v_count = 1
    for xx, yy, zz in zip(X.flatten(), Y.flatten(), Z.flatten()):
        ptx = [xx, yy, zz]
        mesh.AddVertex(ptx)
        v_count = v_count + 1
    #print(e_count, b_count, v_count)
    #print(Nelem,  Nbdrelem, Nvert)

    mesh.FinalizeTopology()         
    mesh.Finalize(refine, fix_orientation)

    if filename != '':
        mesh.PrintToFile(filename, 8)
    return mesh


