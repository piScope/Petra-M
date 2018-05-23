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
debug.debug_default_level = 1
dprint1, dprint2, dprint3 = debug.init_dprints('dof_map')


from petram.helper.matrix_file import write_matrix, write_vector

from petram.mfem_config import use_parallel
if use_parallel:
   from mpi4py import MPI
   comm = MPI.COMM_WORLD
   num_proc = MPI.COMM_WORLD.size
   myid     = MPI.COMM_WORLD.rank
   from petram.helper.mpi_recipes import *
   import mfem.par as mfem
   from mfem.common.mpi_debug import nicePrint
   
else:
   import mfem.ser as mfem
   num_proc = 1
   myid = 0
   def allgather(x): return [x]
   nicePrint = dprint1

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

def get_edge_mode(dim, sdim):
    mode1 = ''
    if sdim == 3:
       if dim == 2: mode1 = 'Bdr'
    elif sdim == 2:
       if dim == 2: mode1 = 'Bdr'
       if dim == 1: mode1 = 'Dom'       
    if mode1 == '':
        assert False, "not supprint dim/sdim "+str(dim)+'/'+str(sdim)
    return mode1
   
def get_volume_mode(dim, sdim):
    if sdim == 3 and dim == 3:
        return 'Dom'
    else:
        assert False, "volume mapping must be 3d mesh"
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
               dprint2(subvdof1, vdof1, subvdof2)
            ## note subdof2 = -1 if it is not owned by the node
       else:
           subvdof2 = subvdof1

       newk1 = np.vstack([(k, xx[0], xx[1])
                           for k, xx in enumerate(zip(vdof1, subvdof2))])
       pt1 =  np.vstack([pt1[kk] for kk, v, s in newk1])
       pt1o = np.vstack([pt1o[kk] for kk, v, s in newk1])

       ret[iii] = (newk1, pt1, pt1o)
    return ret

def get_shape(fes, ibdr , mode='Bdr'):
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
 
def get_vshape(fes, ibdr , mode='Bdr'):
    mesh = fes.GetMesh()  

    GetTrans = getattr(fes, methods[mode]['Transformation'])
    GetElement = getattr(fes, methods[mode]['Element'])
    GetVDofs = getattr(fes, methods[mode]['VDofs'])

    ret = [None]*len(ibdr)
    for iii, k1 in enumerate(ibdr):
        tr1 = GetTrans(k1)
        el = GetElement(k1)
        nodes1 = el.GetNodes()
        m = mfem.DenseMatrix(nodes1.GetNPoints(),
                             tr1.GetSpaceDim())        
        shape = [None]*nodes1.GetNPoints()
        for idx in range(len(shape)):
            tr1.SetIntPoint(nodes1.IntPoint(idx))
            el.CalcVShape(tr1,m)
            shape[idx] = m.GetDataArray()[idx,:].copy()
        ret[iii]  = shape
    return ret

def resolve_nonowned_dof(pt1all, pt2all, k1all, k2all, map_1_2):
    '''
    resolves shadowed DoF
    this is done based on integration point distance.
    It searches a closeest true (non-shadow) DoF point
    '''
    k2all = np.stack(k2all)
    for k in range(len(pt1all)):
        subvdof1 = k1all[k][:,2]
        k2 = map_1_2[k]
        subvdof2 = k2all[k2][:,2]
        pt2 = pt2all[k2]
        check = False
        if -1 in subvdof2:
                check = True
                dprint2('before resolving dof', subvdof2)
        for kk, x in enumerate(subvdof2):
             if x == -1:
                dist = pt2all-pt2[kk]
                dist = np.sqrt(np.sum((dist)**2, -1))
                fdist= dist.flatten()
                isort = np.argsort(fdist)
                minidx =  np.where(dist.flatten() == np.min(dist.flatten()))[0]
                #dprint1("distances", np.min(dist),fdist[isort[:25]])
                while all(k2all[:,:,2].flatten()[minidx] == -1):
                    dprint2("distances (non -1 exists?)", fdist[minidx],
                            k2all[:,:,2].flatten()[minidx])
                    minidx = np.hstack((minidx, isort[len(minidx)]))
                dprint2("distances", np.min(dist),fdist[minidx],
                        fdist[isort[:len(minidx)+1]])                    
                dprint2("minidx",  minidx, k2all[:,:,2].flatten()[minidx])
                for i in minidx:
                    if k2all[:,:,2].flatten()[i] != -1:
                       subvdof2[kk] = k2all[:,:,2].flatten()[i]

        if check:
             dprint2('resolved dof', k2all[k2][:,2])
             if -1 in subvdof2: 
                 assert False, "failed to resolve shadow DoF"
    return k2all

def map_dof_scalar(map, fes1, fes2, pt1all, pt2all, pto1all, pto2all, 
               k1all, k2all, sh1all, sh2all, map_1_2,
               trans1, trans2, tol, tdof, rstart):

    pt = []
    subvdofs1 = []
    subvdofs2 = []

    num_entry = 0
    num_pts = 0

    decimals = int(np.abs(np.log10(tol)))

    if use_parallel:
        P = fes1.Dof_TrueDof_Matrix()
        from mfem.common.parcsr_extra import ToScipyCoo
        P = ToScipyCoo(P).tocsr()
        VDoFtoGTDoF = P.indices  #this is global TrueDoF (offset is not subtracted)
        external_entry = []
        gtdof_check = []
        
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
            if newk1[k,2] in tdof: continue
            if newk1[k,2] in subvdofs1: continue               

            dist = np.sum((pt2-p)**2, 1)
            d = np.where(dist == np.min(dist))[0]
            if myid == 1: dprint2('min_dist', np.min(dist))

            if len(d) == 1:
               d = d[0]
               s1 = sh1[newk1[k, 0]]
               s2 = sh2[newk2[d, 0]]
               #dprint1("case1 ", s1, s2) this looks all 1
               if s1/s2 < 0: dprint2("not positive")
               #if myid == 1: print(newk1[d][2]-rstart, newk2[k][2])
               value = np.around(s1/s2, decimals)               
               if newk1[k,2] != -1: 
                   map[newk1[k][2]-rstart, 
                       newk2[d][2]] = value
                   num_entry = num_entry + 1
                   subvdofs1.append(newk1[k][2])
               else:
                   # for scalar, this is perhaps not needed
                   # rr = newk1[k][1]] if newk1[k][1]] >= 0 else -1-newk1[k][1]]
                   # gtdof = VDoFtoGTDoF[rr]
                   assert newk1[k][1]>=0, "Negative index found"
                   gtdof = VDoFtoGTDoF[newk1[k][1]]
                   if not gtdof in gtdof_check:
                       external_entry.append((gtdof, newk2[d][2], value))
                       gtdof_check.append(gtdof)
                   

            else:
                raise AssertionError("more than two dofs at same plase is not asupported. ")
        #subvdofs1.extend([s for k, v, s in newk1])
        subvdofs2.extend([s for k, v, s in newk2])
        #print len(subvdofs1), len(subvdofs2)

    if use_parallel:
        dprint1("total entry (before)",sum(allgather(num_entry)))
        #nicePrint(len(subvdofs1), subvdofs1)
        external_entry =  sum(comm.allgather(external_entry),[])
        for r, c, d in external_entry:
           h = map.shape[0]
           if (r - rstart >= 0 and r - rstart < h and
               not r  in subvdofs1):
               num_entry = num_entry + 1                                 
               print("adding",myid, r,  c, d )
               map[r-rstart, c] = d
               subvdofs1.append(r)
        total_entry = sum(allgather(num_entry))
        total_pts = sum(allgather(num_pts))
        if sum(allgather(map.nnz)) != total_entry:
           assert False, "total_entry does not match with nnz"
    else:
        total_entry = num_entry
        total_pts = num_pts
        
    #dprint1("map size", map.shape)       
    dprint1("local pts/entry", num_pts, " " , num_entry)
    dprint1("total pts/entry", total_pts, " " , total_entry)       
    return map

def map_dof_vector(map, fes1, fes2, pt1all, pt2all, pto1all, pto2all, 
                   k1all, k2all, sh1all, sh2all, map_1_2,
                   trans1, trans2, tol, tdof, rstart):

    pt = []
    subvdofs1 = []
    subvdofs2 = []

    num1 = 0
    num2 = 0    
    num_pts = 0

    decimals = int(np.abs(np.log10(tol)))

    if use_parallel:
        P = fes1.Dof_TrueDof_Matrix()
        from mfem.common.parcsr_extra import ToScipyCoo
        P = ToScipyCoo(P).tocsr()
        VDoFtoGTDoF = P.indices  #this is global TrueDoF (offset is not subtracted)
        external_entry = []
        gtdof_check = []
        
    def make_entry(r, c, value, num_entry):
        value = np.around(value, decimals)
        if value == 0: return num_entry
        if r[1] != -1: 
            map[r[1]-rstart, c] = value 
            num_entry = num_entry + 1
            subvdofs1.append(r[1])
        else:
            rr = r[0] if r[0] >= 0 else -1-r[0]
            gtdof = VDoFtoGTDoF[rr]
            if not gtdof in gtdof_check:
               external_entry.append((gtdof, c, value))
               gtdof_check.append(gtdof)
        return num_entry      
        
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
        #if myid == 1: print newk1[:,2], newk1[:,1], rstart
        #if myid == 1:
        #    x = [r if r >= 0 else -1-r for r in newk1[:,1]]
        #    print [VDoFtoGTDoF[r] for r in x]
        for k, p in enumerate(pt1):
            num_pts = num_pts + 1
            if newk1[k,2] in tdof: continue
            if newk1[k,2] in subvdofs1: continue               

            dist = np.sum((pt2-p)**2, 1)
            d = np.where(dist == np.min(dist))[0]
            #if myid == 1: dprint1('min_dist', np.min(dist))
            if len(d) == 1:            
               '''
               this factor is not always 1
               '''

               d = d[0]               
               s = np.sign(newk1[k,1] +0.5)*np.sign(newk2[d,1] + 0.5)

               p1 = pto1[k]; p2 = pto2[d]
               delta = np.sum(np.std(p1, 0))/np.sum(np.std(sh1, 0))/10.

               v1 = trans1(p1) - trans1(p1 + delta*sh1[newk1[k, 0]])
               v2 = trans2(p2) - trans2(p2 + delta*sh2[newk2[d, 0]])

               fac = np.sum(v1*v2)/np.sum(v1*v1)*s

               num1 = make_entry(newk1[k, [1,2]], newk2[d, 2], fac, num1)
                             
            elif len(d) == 2:
               dd = np.argsort(np.sum((pt1 - p)**2, 1))
                   
               p1 = pto1[dd[0]]; p3 = pto2[d[0]]
               p2 = pto1[dd[1]]; p4 = pto2[d[1]]
               delta = np.sum(np.std(p1, 0))/np.sum(np.std(sh1, 0))/10.
                   
               v1 = trans1(p1) - trans1(p1 + delta*sh1[newk1[dd[0], 0]])
               v2 = trans1(p2) - trans1(p2 + delta*sh1[newk1[dd[1], 0]])
               v3 = trans2(p3) - trans2(p3 + delta*sh2[newk2[d[0], 0]])
               v4 = trans2(p4) - trans2(p4 + delta*sh2[newk2[d[1], 0]])
               v1 = v1*np.sign(newk1[dd[0], 1] +0.5)
               v2 = v2*np.sign(newk1[dd[1], 1] +0.5)
               v3 = v3*np.sign(newk2[d[0], 1] +0.5)
               v4 = v4*np.sign(newk2[d[1], 1] +0.5)
               s = np.sign(newk1[k,1] +0.5)*np.sign(newk2[d,1] + 0.5)
               def vnorm(v):
                   return v/np.sqrt(np.sum(v**2))
               v1n = vnorm(v1) ; v2n = vnorm(v2)
               v3n = vnorm(v3) ; v4n = vnorm(v4)                   

               if (np.abs(np.abs(np.sum(v1n*v3n))-1) < tol and
                   np.abs(np.abs(np.sum(v2n*v4n))-1) < tol):
                   fac1 = np.sum(v1*v3)/np.sum(v1*v1)
                   fac2 = np.sum(v2*v4)/np.sum(v2*v2)
                   num2 = make_entry(newk1[dd[0],[1,2]], newk2[d[0],2], fac1, num2)
                   num2 = make_entry(newk1[dd[1],[1,2]], newk2[d[1],2], fac2, num2)

               elif (np.abs(np.abs(np.sum(v2n*v3n))-1) < tol and
                     np.abs(np.abs(np.sum(v1n*v4n))-1) < tol):
                   fac1 = np.sum(v1*v4)/np.sum(v1*v1)
                   fac2 = np.sum(v2*v3)/np.sum(v2*v2)
                   num2 = make_entry(newk1[dd[0],[1,2]], newk2[d[1],2], fac1, num2)
                   num2 = make_entry(newk1[dd[1],[1,2]], newk2[d[0],2], fac2, num2)
               else:
                   def proj2d(v, e1, e2):
                       return np.array([np.sum(v*e1), np.sum(v*e2)])
                   if len(v1) == 3: # if vector is 3D, needs to prjoect on surface
                      e3 = np.cross(v1n, v2n)
                      e1 = v1n
                      e2 = np.cross(e1, e3)
                      v1 = proj2d(v1, e1, e2)
                      v2 = proj2d(v2, e1, e2)
                      v3 = proj2d(v3, e1, e2)
                      v4 = proj2d(v4, e1, e2)
                   m1 = np.transpose(np.vstack((v1, v2)))
                   m2 = np.transpose(np.vstack((v3, v4)))
                   m = np.dot(np.linalg.inv(m1), m2)
                   m = np.around(np.linalg.inv(m), decimals = decimals)
                   num2 = make_entry(newk1[dd[0],[1,2]], newk2[d[0],2], m[0,0], num2)
                   num2 = make_entry(newk1[dd[0],[1,2]], newk2[d[1],2], m[1,0], num2)
                   num2 = make_entry(newk1[dd[1],[1,2]], newk2[d[0],2], m[0,1], num2)
                   num2 = make_entry(newk1[dd[1],[1,2]], newk2[d[1],2], m[1,1], num2)
                   
            elif len(d) == 3:
                dd = np.argsort(np.sum((pt1 - p)**2, 1))
                   
                p1 = [pto1[dd[i]] for i in [0, 1, 2]]
                p2 = [pto2[d[i]]  for i in [0, 1, 2]]                       

                delta = np.sum(np.std(p1[0], 0))/np.sum(np.std(sh1, 0))/10.
                 
                v1 = [trans1(p1[i]) - trans1(p1[i] + delta*sh1[newk1[dd[i], 0]])
                       for i in [0, 1, 2]]
                v2 = [trans2(p2[i]) - trans2(p2[i] + delta*sh2[newk2[d[i], 0]])
                       for i in [0, 1, 2]]
                       
                v1 = [v1[i]*np.sign(newk1[dd[i], 1] +0.5) for i in [0, 1, 2]]
                v2 = [v2[i]*np.sign(newk2[d[i], 1]  +0.5) for i in [0, 1, 2]]

                s = np.sign(newk1[k,1] +0.5)*np.sign(newk2[d,1] + 0.5)
                def vnorm(v):
                    return v/np.sqrt(np.sum(v**2))
                v1n = [vnorm(v) for v in v1]
                v2n = [vnorm(v) for v in v2]
                
                m1 = np.transpose(np.vstack(v1))
                m2 = np.transpose(np.vstack(v2))
                m = np.dot(np.linalg.inv(m1), m2)
                m = np.around(np.linalg.inv(m), decimals = decimals)
                num2 = make_entry(newk1[dd[0],[1,2]], newk2[d[0],2], m[0,0], num2)
                num2 = make_entry(newk1[dd[0],[1,2]], newk2[d[1],2], m[1,0], num2)
                num2 = make_entry(newk1[dd[0],[1,2]], newk2[d[2],2], m[2,0], num2)
                                  
                num2 = make_entry(newk1[dd[1],[1,2]], newk2[d[0],2], m[0,1], num2)
                num2 = make_entry(newk1[dd[1],[1,2]], newk2[d[1],2], m[1,1], num2)
                num2 = make_entry(newk1[dd[1],[1,2]], newk2[d[2],2], m[2,1], num2)
                                  
                num2 = make_entry(newk1[dd[2],[1,2]], newk2[d[0],2], m[0,2], num2)
                num2 = make_entry(newk1[dd[2],[1,2]], newk2[d[1],2], m[1,2], num2)
                num2 = make_entry(newk1[dd[2],[1,2]], newk2[d[2],2], m[2,2], num2)
                                  
            else:
                print pt1, pt2
                '''
                 newk1 = k1all[k0] #(i local DoF, global DoF)
                 sh1 = sh1all[k0]           
                 pto2 = pto2all[k2]
                 newk2 = k2all[k2]
                 sh2 = sh2all[k2]
                '''
                # to do support three vectors
                raise AssertionError("more than three dofs at same place")
        subvdofs2.extend([s for k, v, s in newk2])

    num_entry = num1 + num2
    
    if use_parallel:
        dprint1("total entry (before)",sum(allgather(num_entry)))
        #nicePrint(len(subvdofs1), subvdofs1)
        external_entry =  sum(comm.allgather(external_entry),[])
        #nicePrint(external_entry)        
        for r, c, d in external_entry:
           h = map.shape[0]
           if (r - rstart >= 0 and r - rstart < h and
               not r  in subvdofs1):
               num_entry = num_entry + 1                                 
               print("adding",myid, r,  c, d )
               map[r-rstart, c] = d
               subvdofs1.append(r)
        total_entry = sum(allgather(num_entry))
        total_pts = sum(allgather(num_pts))
        if sum(allgather(map.nnz)) != total_entry:
           assert False, "total_entry does not match with nnz"
    else:
        total_entry = num_entry
        total_pts = num_pts
        
    #dprint1("map size", map.shape)       
    dprint1("local pts/entry", num_pts, " " , num_entry)
    dprint1("total pts/entry", total_pts, " " , total_entry)
                             
    return map

def gather_dataset(idx1, idx2, fes1, fes2, trans1,
                               trans2, tol, shape_type = 'scalar',
                               mode = 'surface'):

    if fes2 is None: fes2 = fes1
    if trans1 is None: trans1=notrans
    if trans2 is None: trans2=notrans

    mesh1= fes1.GetMesh()  
    mesh2= fes2.GetMesh()
    
    if mode == 'volume':
        mode1 = get_volume_mode(mesh1.Dimension(), mesh1.SpaceDimension())
        mode2 = get_volume_mode(mesh2.Dimension(), mesh2.SpaceDimension())
    elif mode == 'surface':
        mode1 = get_surface_mode(mesh1.Dimension(), mesh1.SpaceDimension())
        mode2 = get_surface_mode(mesh2.Dimension(), mesh2.SpaceDimension())
    elif mode == 'edge':
        mode1 = get_edge_mode(mesh1.Dimension(), mesh1.SpaceDimension())
        mode2 = get_edge_mode(mesh2.Dimension(), mesh2.SpaceDimension())

    # collect data
    ibdr1 = find_element(fes1, idx1, mode = mode1)
    ibdr2 = find_element(fes2, idx2, mode = mode2)
    ct1 = find_el_center(fes1, ibdr1, trans1, mode=mode1)
    ct2 = find_el_center(fes2, ibdr2, trans2, mode=mode2)
    arr1 = get_element_data(fes1, ibdr1, trans1, mode=mode1)
    arr2 = get_element_data(fes2, ibdr2, trans1, mode=mode2)

    if shape_type == 'scalar':
        sh1all = get_shape(fes1, ibdr1, mode=mode1)
        sh2all = get_shape(fes2, ibdr2, mode=mode2)
    elif shape_type == 'vector':
        sh1all = get_vshape(fes1, ibdr1, mode=mode1)
        sh2all = get_vshape(fes2, ibdr2, mode=mode2)
    else:
        assert False, "Unknown shape type"

    # pt is on (u, v), pto is (x, y, z)
    try:
       k1all, pt1all, pto1all = zip(*arr1)
    except:
       k1all, pt1all, pto1all = (), (), ()
    try:
       k2all, pt2all, pto2all = zip(*arr2)
    except:
       k2all, pt2all, pto2all = (), (), ()

    if use_parallel:
       # share ibr2 (destination information among nodes...)
       ct2 = np.atleast_1d(ct2).reshape(-1, ct1.shape[1])
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
       pt2all =  sum(comm.allgather(pt2all),())
       pto2all = sum(comm.allgather(pto2all),())
       k2all =  sum(comm.allgather(k2all),())
       sh2all =  sum(comm.allgather(sh2all),[])
       k2all = resolve_nonowned_dof(pt1all, pt2all, k1all, k2all, map_1_2)

    # map is fill as transposed shape (row = fes1)
    
    map = lil_matrix((fesize1, fesize2), dtype=float)
    data = pt1all, pt2all, pto1all, pto2all, k1all, k2all, sh1all, sh2all,

    return  map, data, map_1_2, rstart    

def map_xxx_h1(xxx, idx1, idx2, fes1, fes2=None, trans1=None,
                   trans2=None, tdof1=None, tdof2=None, tol=1e-4):
    '''
    map DoF on surface to surface

      fes1: source finite element space
      fes2: destination finite element space

      idx1: surface attribute (Bdr for 3D/3D, Domain for 2D/3D or 2D/2D)

    '''
                             
    if fes2 is None: fes2 = fes1
    if trans1 is None: trans1=notrans
    if trans2 is None: trans2=trans1
    if tdof1 is None: tdof1=[]
    if tdof2 is None: tdof2=[]    

    tdof = tdof1 # ToDo support tdof2    
    map, data, elmap, rstart = gather_dataset(idx1, idx2, fes1, fes2, trans1,
                                              trans2, tol, shape_type = 'scalar',
                                              mode=xxx)


    pt1all, pt2all, pto1all, pto2all, k1all, k2all, sh1all, sh2all  = data

    map_dof_scalar(map, fes1, fes2, pt1all, pt2all, pto1all, pto2all, 
                   k1all, k2all, sh1all, sh2all, elmap,
                   trans1, trans2, tol, tdof1, rstart)

    return map
def map_volume_h1(*args, **kwargs):
    return map_xxx_h1('volume', *args, **kwargs)
def map_surface_h1(*args, **kwargs):
    return map_xxx_h1('surface', *args, **kwargs)
def map_edge_h1(*args, **kwargs):
    return map_xxx_h1('edge', *args, **kwargs)

''' 
def map_surface_h1(idx1, idx2, fes1, fes2=None, trans1=None,
                   trans2=None, tdof1=None, tdof2=None, tol=1e-4):
    if fes2 is None: fes2 = fes1
    if trans1 is None: trans1=notrans
    if trans2 is None: trans2=trans1
    if tdof1 is None: tdof1=[]
    if tdof2 is None: tdof2=[]    

    tdof = tdof1 # ToDo support tdof2    
    map, data, elmap, rstart = gather_dataset(idx1, idx2, fes1, fes2, trans1,
                                              trans2, tol, shape_type = 'scalar',
                                              mode='surface')


    pt1all, pt2all, pto1all, pto2all, k1all, k2all, sh1all, sh2all  = data

    map_dof_scalar(map, fes1, fes2, pt1all, pt2all, pto1all, pto2all, 
                   k1all, k2all, sh1all, sh2all, elmap,
                   trans1, trans2, tol, tdof1, rstart)

    return map

def map_edge_h1(idx1, idx2, fes1, fes2=None, trans1=None,
                   trans2=None, tdof1=None, tdof2=None, tol=1e-4):
    if fes2 is None: fes2 = fes1
    if trans1 is None: trans1=notrans
    if trans2 is None: trans2=trans1
    if tdof1 is None: tdof1=[]
    if tdof2 is None: tdof2=[]    

    tdof = tdof1 # ToDo support tdof2

    map, data, elmap, rstart = gather_dataset(idx1, idx2, fes1, fes2, trans1,
                                              trans2, tol, shape_type = 'scalar',
                                              mode="edge")

    
    pt1all, pt2all, pto1all, pto2all, k1all, k2all, sh1all, sh2all  = data
    map_dof_scalar(map, fes1, fes2, pt1all, pt2all, pto1all, pto2all, 
                   k1all, k2all, sh1all, sh2all, elmap,
                   trans1, trans2, tol, tdof1, rstart)

    return map
''' 
def map_xxx_nd(xxx, idx1, idx2, fes1, fes2=None, trans1=None,
                   trans2=None, tdof1=None, tdof2=None, tol=1e-4):
 
    '''
    map DoF on surface to surface

      fes1: source finite element space
      fes2: destination finite element space

      idx1: surface attribute (Bdr for 3D/3D, Domain for 2D/3D or 2D/2D)

    '''
                             
    if fes2 is None: fes2 = fes1
    if trans1 is None: trans1=notrans
    if trans2 is None: trans2=trans1
    if tdof1 is None: tdof1=[]
    if tdof2 is None: tdof2=[]    

    tdof = tdof1 # ToDo support tdof2    
    map, data, elmap, rstart = gather_dataset(idx1, idx2, fes1, fes2, trans1,
                                              trans2, tol, shape_type = 'vector',
                                              mode=xxx)
    pt1all, pt2all, pto1all, pto2all, k1all, k2all, sh1all, sh2all  = data
    
    map_dof_vector(map, fes1, fes2, pt1all, pt2all, pto1all, pto2all, 
                   k1all, k2all, sh1all, sh2all, elmap,
                   trans1, trans2, tol, tdof1, rstart)

    return map
def map_volume_nd(*args, **kwargs):
    return map_xxx_nd('volume', *args, **kwargs)
def map_surface_nd(*args, **kwargs):
    return map_xxx_nd('surface', *args, **kwargs)
def map_edge_nd(*args, **kwargs):
    return map_xxx_nd('edge', *args, **kwargs)
''' 
def map_volume_nd(idx1, idx2, fes1, fes2=None, trans1=None,
                   trans2=None, tdof1=None, tdof2=None, tol=1e-4):
 
    if fes2 is None: fes2 = fes1
    if trans1 is None: trans1=notrans
    if trans2 is None: trans2=trans1
    if tdof1 is None: tdof1=[]
    if tdof2 is None: tdof2=[]    

    tdof = tdof1 # ToDo support tdof2    
    map, data, elmap, rstart = gather_dataset(idx1, idx2, fes1, fes2, trans1,
                                              trans2, tol, shape_type = 'vector',
                                              mode="volume")
    pt1all, pt2all, pto1all, pto2all, k1all, k2all, sh1all, sh2all  = data
    
    map_dof_vector(map, fes1, fes2, pt1all, pt2all, pto1all, pto2all, 
                   k1all, k2all, sh1all, sh2all, elmap,
                   trans1, trans2, tol, tdof1, rstart)

    return map

def map_surface_nd(idx1, idx2, fes1, fes2=None, trans1=None,
                   trans2=None, tdof1=None, tdof2=None, tol=1e-4):
 
                             
    if fes2 is None: fes2 = fes1
    if trans1 is None: trans1=notrans
    if trans2 is None: trans2=trans1
    if tdof1 is None: tdof1=[]
    if tdof2 is None: tdof2=[]    

    tdof = tdof1 # ToDo support tdof2    
    map, data, elmap, rstart = gather_dataset(idx1, idx2, fes1, fes2, trans1,
                               trans2, tol, shape_type = 'vector')
    
    pt1all, pt2all, pto1all, pto2all, k1all, k2all, sh1all, sh2all  = data
    
    map_dof_vector(map, fes1, fes2, pt1all, pt2all, pto1all, pto2all, 
                   k1all, k2all, sh1all, sh2all, elmap,
                   trans1, trans2, tol, tdof1, rstart)

    return map
'''
 
# ToDO test these
# map_surface_rt = map_surface_nd
# map_surface_l2 = map_surface_h1

def projection_matrix(idx1,  idx2,  fes, tdof1, fes2=None, tdof2=None,
                      trans1=None, trans2 = None, dphase=0.0, weight = None,
                      tol = 1e-7, mode = 'surface', filldiag=True):
    '''
     map: destinatiom mapping 
     smap: source mapping
    '''
    fec_name = fes.FEColl().Name()

    if fec_name.startswith('ND') and mode == 'volume':
        mapper = map_volume_nd
    elif fec_name.startswith('ND') and mode == 'surface':
        mapper = map_surface_nd
    elif fec_name.startswith('ND') and mode == 'edge':
        mapper = map_edge_nd
    elif fec_name.startswith('H1') and mode == 'volume':
        mapper = map_volume_h1
    elif fec_name.startswith('H1') and mode == 'surface':
        mapper = map_surface_h1
    elif fec_name.startswith('H1') and mode == 'edge':
        mapper = map_edge_h1
    else:
        raise NotImplementedError("mapping :" + fec_name + ", mode: " + mode)

    map = mapper(idx2, idx1, fes, fes2=fes2, trans1=trans1, trans2=trans2, tdof1=tdof1,
                 tdof2=tdof2, tol=tol)


    if weight is None:
        iscomplex = False       
        if (dphase == 0.):
            pass
        elif (dphase == 180.):
            map = -map
        else:
            iscomplex = True
            map = map.astype(complex)        
            map *= np.exp(-1j*np.pi/180*dphase)
    else:
        iscomplex = np.iscomplexobj(weight)
        if iscomplex:
            map = map.astype(complex)        
        map *= -weight
      
    m_coo = map.tocoo()
    row = m_coo.row
    col = m_coo.col
    col = np.unique(col)

    
    if use_parallel:
        start_row = fes.GetMyTDofOffset()
        end_row = fes.GetMyTDofOffset() + fes.GetTrueVSize()
        col =  np.unique(allgather_vector(col))
        row = row + start_row
    else:
        start_row = 0
        end_row = map.shape[0]

    if filldiag:
        for i in range(min(map.shape[0], map.shape[1])):
            r = start_row+i
            if not r in col: map[i, r] = 1.0
        
    from scipy.sparse import coo_matrix, csr_matrix
    if use_parallel:
        if iscomplex:
            m1 = csr_matrix(map.real, dtype=float)
            m2 = csr_matrix(map.imag, dtype=float) 
        else:
            m1 = csr_matrix(map.real, dtype=float)
            m2 = None
        from mfem.common.chypre import CHypreMat
        start_col = fes2.GetMyTDofOffset()
        end_col = fes2.GetMyTDofOffset() + fes2.GetTrueVSize()
        col_starts = [start_col, end_col, map.shape[1]]
        M = CHypreMat(m1, m2, col_starts=col_starts)
    else:
        from petram.helper.block_matrix import convert_to_ScipyCoo

        M = convert_to_ScipyCoo(coo_matrix(map, dtype=map.dtype))

    return M, row, col
