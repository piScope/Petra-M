'''
  export_noralvalues.py

  a helper routine to export nordalvalues from  
  a pair of solution and mesh file.

  perform 3D interpolation of using GridData

  sample usage:

     #  go to Petra-M solution directory

     >>>  import petram.helper.export_nodalvalues as exporter
     >>>  solset = exporter.load_sol(fesvar = 'E')
     >>>  values = exporter.get_nodalvalues(solset)

     >>>  R =   np.arange(0.1, 1.5, 100)
     >>>  Z =   np.arange(-1.,  1., 100)
     >>>  Phi = np.arange(0, 2*np.pi., 100)


  Author: S. Shiraiwa

'''
import os
import six
import numpy as np

import mfem.ser as mfem
    
def load_sol(fesvar = '', path = '.', refine = 0, ):
    '''
    read sol files in directory path. 
    it reads only a file for the certain FES variable given by fesvar
    '''
    from petram.sol.solsets import find_solfiles, MeshDict
    solfiles = find_solfiles(path)

    def fname2idx(t):    
           i = int(os.path.basename(t).split('.')[0].split('_')[-1])
           return i
    solfiles = solfiles.set
    solset = []

    for meshes, solf, in solfiles:
        s = {}
        for key in six.iterkeys(solf):
            name = key.split('_')[0]
            if name != fesvar: continue
            idx_to_read = int(key.split('_')[1])
            break

    for meshes, solf, in solfiles:
        idx = [fname2idx(x) for x in meshes]
        meshes = {i:  mfem.Mesh(str(x), 1, refine) for i, x in zip(idx, meshes) if i == idx_to_read}
        meshes=MeshDict(meshes) # to make dict weakref-able
        ### what is this refine = 0 !?
        for i in meshes.keys():
            meshes[i].ReorientTetMesh()
            meshes[i]._emesh_idx = i

        s = {}
        for key in six.iterkeys(solf):
            name = key.split('_')[0]
            if name != fesvar: continue

            fr, fi =  solf[key]
            i = fname2idx(fr)
            m = meshes[i]
            solr = (mfem.GridFunction(m, str(fr)) if fr is not None else None)
            soli = (mfem.GridFunction(m, str(fi)) if fi is not None else None)
            if solr is not None: solr._emesh_idx = i
            if soli is not None: soli._emesh_idx = i
            s[name] = (solr, soli)
        solset.append((meshes, s))
        
    return solset

def get_nodalvalues(solset,
                    curl = False,
                    grad = False,
                    div  = False):

    if grad:
        assert False, "evaluating Grad is not implemented"
    if div:
        assert False, "evaluating Div is not implemented"

    import petram.helper.eval_deriv as eval_deriv

    from collections import defaultdict
    nodalvalues =defaultdict(list)

    for meshes, s in solset:   # this is MPI rank loop
        for name in s:
            gfr, gfi = s[name]
            m = gfr.FESpace().GetMesh()
            size = m.GetNV()

            ptx = np.vstack([m.GetVertexArray(i) for i in range(size)])
            
            if curl:
                gfr, gfi, extra = eval_deriv.eval_curl(gfr, gfi)            
            dim = gfr.VectorDim()
            
            ret = np.zeros((size, dim), dtype = float)            
            for comp in range(dim):
                values = mfem.Vector()
                gfr.GetNodalValues(values, comp+1)
                ret[:, comp] = values.GetDataArray()
                values.StealData()

            if gfi is None:
                nodalvalues[name].append((ptx, ret, gfr))
                continue

            ret2 = np.zeros((size, dim), dtype = float)                        
            for comp in range(dim):
                values = mfem.Vector()
                gfi.GetNodalValues(values, comp+1)

                if ret2 is None:
                    ret2 = np.zeros((values.Size(), dim), dtype = float)

                ret2[:, comp] = values.GetDataArray()
                values.StealData()

            ret = ret + 1j*ret2
            nodalvalues[name].append((ptx, ret, gfr))

            
    nodalvalues.default_factory = None
    return nodalvalues

def make_mask(values, X, Y, Z):
    '''
    mask for interpolation
    '''
    size = len(X.flatten())    
    mask= np.zeros(len(X.flatten()), dtype=int) - 1

    for kk, data in enumerate(values):
       ptx, ret, gfr = data
       mesh = gfr.FESpace().GetMesh()

       m = mfem.DenseMatrix(3, size)
       ptx = np.vstack([X.flatten(), Y.flatten(), Z.flatten()])
       m.Assign(ptx)
       ips = mfem.IntegrationPointArray()
       elem_ids = mfem.intArray()
       
       pts_found = mesh.FindPoints(m, elem_ids, ips)
       
       mask[np.array(elem_ids.ToList()) !=-1 ] = kk
            
    return mask


def interp3D(values, X, Y, Z, mask, vdim=1, complex = False):
    from scipy.interpolate import griddata
    
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()        
    
    size = len(X.flatten())
    if complex:
        res = np.zeros((vdim, size), dtype=np.complex)
    else:
        res = np.zeros((vdim, size), dtype=np.float)
    
    for kk, data in enumerate(values):
        idx = mask == kk
        ptx, ret, gfr = data

        for ii in range(vdim):
           print(ptx.shape, ret.shape)
           res[ii, idx] = griddata(ptx, ret[:, ii].flatten(),
                                   (X[idx], Y[idx], Z[idx]))

    if vdim == 1: res = res.flatten()       
    return res
       


        



              
