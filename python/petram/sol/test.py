from __future__ import print_function

import numpy as np
import weakref
import scipy
import six

from weakref import WeakKeyDictionary as WKD
from weakref import WeakValueDictionary as WVD

from petram.mfem_config import use_parallel
if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

from petram.sol.bdr_nodal_evaluator import process_iverts2nodals, eval_at_nodals

def pg(self, attrs, plane = None):                
    '''

    '''
    self.vertices = None
    self.iverts = None
    
    self.knowns = WKD()
    mesh = self.mesh()
    self.iverts = []
    self.attrs = attrs
    self.plane = plane

    attrs = self.attrs
    axyz = self.plane[:3]
    c    = self.plane[-1]

    attr = mesh.GetAttributeArray()
    x = [np.where(attr == a)[0] for a in attrs]
    if np.sum([len(xx) for xx in x]) == 0: return

    ialleles = np.hstack(x).astype(int).flatten()
    ieles = []
    num_tri = 0
    tri_iverts = []
    f_values = []
    for iel in ialleles:
        verts = np.vstack([mesh.GetVertexArray(i)
                           for i in mesh.GetElement(iel).GetVerticesArray()])
        f = np.sum(verts*axyz, -1) + c
        ips = np.where(f > 0)[0]
        ims = np.where(f < 0)[0]
        izs = np.where(f == 0)[0]           
        if (len(ips) == 0 or len(ims) == 0) and len(izs) != 3:
            continue # outside the plane
        ieles.append(iel)
        f_values.append(f)
        
        if f.prod() > 0: # cross section is quad.
           num_tri += 2
        else:
           num_tri += 1

    # then get unique set of elements relating to the verts.
    if num_tri == 0:
        print("not found")
        return 
    vert2el = mesh.GetVertexToElementTable()
           
    iverts = np.array([mesh.GetElement(iel).GetVerticesArray()
                          for iel in ieles])
    self.iverts = iverts
    
    data = process_iverts2nodals(mesh, iverts)
    for k in six.iterkeys(data):
        setattr(self, k, data[k])

    iverts_f = self.iverts_f
    
    # how to interpolate nodal values...
    vertices = np.zeros((num_tri, 3, 3))
    mat      = scipy.sparse.lil_matrix((num_tri*3, len(iverts_f)))

    def fill_vert_mat(vertics, mat, ivv, itri, k, i1, i2, verts, f):
        v1 = verts[i1]; v2 = verts[i2]
        f1 = np.abs(f[i1]); f2 = np.abs(f[i2])
        v = (v1*f2 + v2*f1)/(f1 + f2)
        vertices[itri, k, :] = v
        mat[itri*3+k, ivv[i1]] = np.abs(f1)
        mat[itri*3+k, ivv[i2]] = np.abs(f2)
    
    itri = 0
    for iel, f in zip(ieles, f_values):
        iv = mesh.GetElement(iel).GetVerticesArray()
        verts = np.vstack([mesh.GetVertexArray(i) for i in iv])
        ivv = [np.where(iverts_f == i)[0] for i in iv]
        ips = np.where(f > 0)[0]
        ims = np.where(f < 0)[0]           
        if (len(ips) == 2 and len(ims) == 2):
            fill_vert_mat(vertices, mat, ivv, itri, 0, ips[0], ims[0], verts, f)
            fill_vert_mat(vertices, mat, ivv, itri, 1, ips[0], ims[1], verts, f)
            fill_vert_mat(vertices, mat, ivv, itri, 2, ips[1], ims[0], verts, f)            
            itri += 1

            fill_vert_mat(vertices, mat, ivv, itri, 0, ips[1], ims[0], verts, f)
            fill_vert_mat(vertices, mat, ivv, itri, 1, ips[1], ims[1], verts, f)
            fill_vert_mat(vertices, mat, ivv, itri, 2, ips[0], ims[1], verts, f)            
            itri += 1
        else:

            if len(ips) == 1:
                ims = np.where(f <= 0)[0]                                       
                i =  ips[0]
                ii = ims
            else:
                ips = np.where(f >= 0)[0]   
                i = ims[0]
                ii = ips

            fill_vert_mat(vertices, mat, ivv, itri, 0, i, ii[0], verts, f)
            fill_vert_mat(vertices, mat, ivv, itri, 1, i, ii[1], verts, f)
            fill_vert_mat(vertices, mat, ivv, itri, 2, i, ii[2], verts, f)            
            itri += 1

    self.ibeles = None # can not use boundary variable in this evaulator        
    self.vertices = vertices
    self.interp_mat = mat
    
def eval(self, expr, solvars, phys):
    if self.vertices is None: return None, None
    
    val = eval_at_nodals(self, expr, solvars, phys)

    if val is None: return None, None
    value = self.interp_mat.dot(val)
    value = value.reshape(len(self.vertices), -1)
    return self.vertices,  value

def edge_detect(index):
    print("edge_detect", index.shape)
    store = []
    def check_pair(store, a, b):
        a1 = min(a,b)
        b1 = max(a,b)
        p = (a1, b1)
        if p in store: store.remove(p)
        else: store.append(p)
        return store
        
    for iv in index:
        store = check_pair(store, iv[0], iv[1])
        store = check_pair(store, iv[0], iv[2])
        store = check_pair(store, iv[1], iv[2])        
    ret = np.vstack(store)
    return  ret 
    
        
    
