'''
 DoF mapper
    this routine returns M.
     [y_dest] = M [y_src]

     fes1 : source
     fes2 : dest (size of fes2 can be smaller)

     For periodic BC, DoF corresponts to y_dest will be eliminated
     from the final linear system.

        Since, mapping is 
           [1,  0][y_src ]   [y_src]
           [     ][      ] = []
           [M,  0][void  ]   [y_dest]

        A linear system below
             [y_src ]   [b1]
           A [      ] = [  ]
             [y_dest]   [b2]
        becomes
           Pt A P [y_src] = [b1 + Mt b2]

     For H1, L2 element M^-1 = M^t,
     For ND, and RT, M^-1 needs inversion. 
'''

import numpy as np
import scipy
from scipy.sparse import lil_matrix
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('find_dof_map3_h1')

from petram.helper.matrix_file import write_matrix, write_vector

from petram.mfem_config import use_parallel
if use_parallel:
   from mpi4py import MPI
   num_proc = MPI.COMM_WORLD.size
   myid     = MPI.COMM_WORLD.rank
   from petram.helper.mpi_recipes import *
   import mfem.par as mfem
else:
   import mfem.ser as mfem
   num_proc = 1
   myid = 0
   def allgather(x): return [x]

mapper_debug = False
gf_debug = False
element_data_debug = False

methods = {}
methods['Dom'] = {'N': 'GetNE',
                  'AttributeArray': 'GetAttributeArray',
                  'Vertices':'GetElementVertices',
                  'Transformation':'GetElementTransformation',
                  'Element': 'GetFE',
                  'VDofs': 'GetElementVDofs'}
methods['Bdr'] = {'N': 'GetNBE',
                  'AttributeArray': 'GetBdrAttributeArray',
                  'Vertices':'GetBdrElementVertices',
                  'Transformation':'GetBdrElementTransformation',
                  'Element': 'GetBE',
                  'VDofs': 'GetBdrElementVDofs'}
def notrans(xyz):
    return np.array(xyz, copy=False)

def get_surface_mode(dim, sdim):
    mode1 = ''
    if sdim == 3:
       if dim == 3: mode1 = 'Bdr'
       elif dim == 2:mode1 = 'Dom'
    elif sdim == 2:
       if dim == 2: mode1 = 'Dom'
 
    if mode1 == '':
        assert False, "not supprint dim/sdim "+str(dim)+'/'+str(sdim)
    return mode1

def find_element(fes, attr, mode = 'Bdr'):
    mesh = fes.GetMesh()
    m = getattr(mesh, methods[mode]['AttributeArray'])
    arr = m()
    flag = np.in1d(arr, attr)
    return np.arange(len(arr))[flag]

def find_el_center(fes, ibdr1, trans1, mode = 'Bdr'):
    
    if len(ibdr1) == 0: return np.empty(shape=(0,2))
    mesh = fes.GetMesh()
    m = getattr(mesh, methods[mode]['Vertices'])
    print methods[mode]['Vertices']
    pts = np.vstack([np.mean([trans1(mesh.GetVertexArray(kk))
                              for kk in m(k)],0) for k in ibdr1])
    return pts

def get_element_data(fes, idx, trans, mode='Bdr'):
    mesh = fes.GetMesh()  

    GetTrans = getattr(fes, methods[mode]['Transformation'])
    GetElement = getattr(fes, methods[mode]['Element'])
    GetVDofs = getattr(fes, methods[mode]['VDofs'])

    ret = [None]*len(idx)
    for iii, k1 in enumerate(idx):
       tr1 = GetTrans(k1)
       nodes1 = GetElement(k1).GetNodes()
       vdof1 = GetVDofs(k1)

       pt1 = np.vstack([trans(tr1.Transform(nodes1.IntPoint(kk)))
                        for kk in range(len(vdof1))])
       pt1o = np.vstack([tr1.Transform(nodes1.IntPoint(kk))
                         for kpk in range(len(vdof1))])

       subvdof1 = [x if x>= 0 else -1-x for x in vdof1]
       if use_parallel:
            subvdof2= [fes.GetLocalTDofNumber(i) for i in subvdof1]
            flag = False
            for k, x in enumerate(subvdof2):
                if x >=0:
                    subvdof2[k] = fes.GetMyTDofOffset()+ x
                else: flag = True
            if element_data_debug and flag:
               dprint1(subvdof1, vdof1, subvdof2)
            ## note subdof2 = -1 if it is not owned by the node
       else:
           subvdof2 = subvdof1

       newk1 = np.vstack([(k, xx[0], xx[1])
                           for k, xx in enumerate(zip(vdof1, subvdof2))])
       pt1 =  np.vstack([pt1[kk] for kk, v, s in newk1])
       pt1o = np.vstack([pt1o[kk] for kk, v, s in newk1])

       ret[iii] = (newk1, pt1, pt1o)
    return ret

def get_h1_shape(fes, ibdr , mode='Bdr'):
    mesh = fes.GetMesh()  

    GetTrans = getattr(fes, methods[mode]['Transformation'])
    GetElement = getattr(fes, methods[mode]['Element'])
    GetVDofs = getattr(fes, methods[mode]['VDofs'])

    ret = [None]*len(ibdr)
    for iii, k1 in enumerate(ibdr):
        tr1 = GetTrans(k1)
        el = GetElement(k1)
        nodes1 = el.GetNodes()
        v = mfem.Vector(nodes1.GetNPoints())
        shape = [None]*nodes1.GetNPoints()
        for idx in range(len(shape)):
            el.CalcShape(nodes1.IntPoint(idx), v)
            shape[idx] = v.GetDataArray()[idx]
        ret[iii]  = shape
    return ret

def resolve_nonowned_dof(pt1all, pt2all, k1all, k2all, map_1_2):
    '''
    resolves shadowed DoF
    this is done based on integration point distance.
    It searches a closeest true (non-shadow) DoF point
    '''
    for k in range(len(pt1all)):
        subvdof1 = k1all[k][:,2]
        k2 = map_1_2[k]
        subvdof2 = k2all[k2][:,2]
        pt2 = pt2all[k2]
        check = False
        if -1 in subvdof2:
                check = True
                dprint1('before resolving dof', subvdof2)
        for kk, x in enumerate(subvdof2):
             if x == -1:
                dist = pt2all-pt2[kk]
                dist = np.sqrt(np.sum((dist)**2, -1))
                fdist= dist.flatten()
                isort = np.argsort(fdist)
                minidx =  np.where(dist.flatten() == np.min(dist.flatten()))[0]
                #dprint1("distances", np.min(dist),fdist[isort[:25]])
                while all(k2all[:,:,2].flatten()[minidx] == -1):
                    dprint1("distances (non -1 exists?)", fdist[minidx],
                            k2all[:,:,2].flatten()[minidx])
                    minidx = np.hstack((minidx, isort[len(minidx)]))
                dprint1("distances", np.min(dist),fdist[minidx],
                        fdist[isort[:len(minidx)+1]])                    
                dprint1("minidx",  minidx, k2all[:,:,2].flatten()[minidx])
                for i in minidx:
                    if k2all[:,:,2].flatten()[i] != -1:
                       subvdof2[kk] = k2all[:,:,2].flatten()[i]

        if check:
             dprint1('resolved dof', k2all[k2][:,2])
             if -1 in subvdof2: 
                 assert False, "failed to resolve shadow DoF"

def map_dof_h1(map, fes1, fes2, pt1all, pt2all, pto1all, pto2all, 
               k1all, k2all, sh1all, sh2all, map_1_2,
               trans1, trans2, tol, tdof, rstart):

    pt = []
    subvdofs1 = []
    subvdofs2 = []

    num_entry = 0
    num_pts = 0

    decimals = int(np.abs(np.log10(tol)))       
    for k0 in range(len(pt1all)):
        k2 = map_1_2[k0]
        pt1 = pt1all[k0]
        pto1 = pto1all[k0]
        newk1 = k1all[k0] #(i local DoF, global DoF)
        sh1 = sh1all[k0]           
        pt2 = pt2all[k2]
        pto2 = pto2all[k2]
        newk2 = k2all[k2]
        sh2 = sh2all[k2]
        for k, p in enumerate(pt1):
            num_pts = num_pts + 1
            if newk1[k,2] == -1: continue # not owned by the node
            if newk1[k,2] in tdof: continue
            if newk1[k,2] in subvdofs1: continue               

            dist = np.sum((pt2-p)**2, 1)
            d = np.where(dist == np.min(dist))[0]
            #dprint1('min_dist', np.min(dist))
            if len(d) == 1:
               d = d[0]
               s1 = sh1[newk1[k, 0]]
               s2 = sh2[newk2[d, 0]]
               #dprint1("case1 ", s1, s2) this looks all 1
               if s1/s2 < 0: dprint1("not positive")

               map[newk1[d][2]+rstart, 
                   newk2[k][2]] = np.around(s1/s2, decimals)
               num_entry = num_entry + 1
            else:
                raise AssertionError("more than two dofs at same plase is not asupported. ")
        subvdofs1.extend([s for k, v, s in newk1])
        subvdofs2.extend([s for k, v, s in newk2])
        #print len(subvdofs1), len(subvdofs2)

    if use_parallel:
        total_entry = sum(allgather(num_entry))
        total_pts = sum(allgather(num_pts))
    else:
        total_entry = num_entry
        total_pts = num_pts
       
    dprint1("map size", map.shape)       
    dprint1("local pts/entry", num_pts, " " , num_entry)
    dprint1("total pts/entry", total_pts, " " , total_entry)       
    return map

def map_surface_h1(idx1, idx2, fes1, fes2=None, trans1=None,
                trans2=None, tdof=None, tol=1e-4):
    '''
    map DoF on surface to surface

      fes1: source finite element space
      fes2: destination finite element space

      idx1: surface attribute (Bdr for 3D/3D, Domain for 2D/3D or 2D/2D)

    '''
    if fes2 is None: fes2 = fes1
    if trans1 is None: trans1=notrans
    if trans2 is None: trans2=notrans
    if tdof is None: tdof=[]

    mesh1= fes1.GetMesh()  
    mesh2= fes2.GetMesh()  
    mode1 = get_surface_mode(mesh1.Dimension(), mesh1.SpaceDimension())
    mode2 = get_surface_mode(mesh2.Dimension(), mesh2.SpaceDimension())

    # collect data
    ibdr1 = find_element(fes1, idx1, mode = mode1)
    ibdr2 = find_element(fes2, idx2, mode = mode2)
    ct1 = find_el_center(fes1, ibdr1, trans1, mode=mode1)
    ct2 = find_el_center(fes2, ibdr2, trans2, mode=mode2)
    arr1 = get_element_data(fes1, ibdr1, trans1, mode=mode1)
    arr2 = get_element_data(fes2, ibdr2, trans1, mode=mode2)
    sh1all = get_h1_shape(fes1, ibdr1, mode=mode1)
    sh2all = get_h1_shape(fes2, ibdr2, mode=mode2)

    # pt is on (u, v), pto is (x, y, z)
    k1all, pt1all, pto1all = zip(*arr1)
    k2all, pt2all, pto2all = zip(*arr2)

    if use_parallel:
       # share ibr2 (destination information among nodes...)
       ct2 =  allgather_vector(ct2, MPI.DOUBLE)
       fesize1 = fes1.GetTrueVSize()
       fesize2 = fes2.GlobalTrueVSize()
       rstart = fes1.GetMyTDofOffset()
    else:
       fesize1 = fes1.GetNDofs()
       fesize2 = fes2.GetNDofs()
       rstart = 0
    
    ctr_dist = np.array([np.min(np.sum((ct2-c)**2, 1)) for c in ct1])
    if ctr_dist.size > 0 and np.max(ctr_dist) > 1e-15:
       print('Center Dist may be too large (check mesh): ' + 
            str(np.max(ctr_dist)))
    # mapping between elements
    map_1_2= [np.argmin(np.sum((ct2-c)**2, 1)) for c in ct1]

    if use_parallel:
       pt2all =  allgather_vector(pt2all, MPI.DOUBLE)
       pto2all = allgather_vector(pto2all, MPI.DOUBLE)
       k2all =  allgather_vector(k2all, MPI.INT)
       sh2all =  allgather_vector(sh2all, MPI.DOUBLE)
       resolve_nonowned_dof(pt1all, pt2all, k1all, k2all, map_1_2)

    # map is fill as transposed shape (row = fes1)
    map = lil_matrix((fesize1, fesize2), dtype=float)
    map_dof_h1(map, fes1, fes2, pt1all, pt2all, pto1all, pto2all, 
              k1all, k2all, sh1all, sh2all, map_1_2,
              trans1, trans2, tol, tdof, rstart)

    return map

