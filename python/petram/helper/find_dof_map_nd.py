from __future__ import print_function

import numpy as np
import scipy

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('find_dof_map_nd')

#from petram.solver.mumps.hypre_to_mumps import get_HypreParMatrixRow

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
def find_dof_map_nd(idx1, idx2, transu, transv, transu2, transv2, 
                    fes, engine, tdof, tol=1e-4):
   '''
   find_dof_map

   projection coupling between DoFs. so far, it is tested
   only for triangle Nedelec boundary elemets.

   it maps element nodes of on bdr1 (attribute = idx1) to 
   bdr2(attribute =idx2)

   this version accept two mapping function which mappps
   (x, y, z) to (u, v) plane

   In this version, returned matrix is interpolation constraints P.
   P will be used as follows.
    P^t A P y = P^t f, x = Py
       
   For example..
      if  x2 = x1 and   [A B][x3]  = [x5]
                        [C D][x4]    [x6]
      P should be filled as follows. It starts from identity 
      sparse  matrix and changes the following matrix elements.

      [ 1            ][y1]    [x1]
      [ 1 0          ][y2]    [x2]
      [     1        ][y3]  = [x3] 
      [        1     ][y4]    [x4]
      [     A  B 0   ][y5]    [x5]
      [     C  D   0 ][y6]    [x6]

    row and col of  P^t A P corresponding to x2, x5, x6 will
    be all zero.

    This module actually assemble the transpose of P.
    Diagonal elements should be handled in caller. See em3d_floquet

   '''
   def get_vshape(fes, k1, idx):
       tr1 = fes.GetBdrElementTransformation(k1)
       el = fes.GetBE(k1)
       nodes1 = el.GetNodes()
       m = mfem.DenseMatrix(nodes1.GetNPoints(), tr1.GetSpaceDim())
       tr1.SetIntPoint(nodes1.IntPoint(idx))
       el.CalcVShape(tr1,m)
       return m.GetDataArray()[idx, :]
    
   def get_vshape_all(fes, k1):
       tr1 = fes.GetBdrElementTransformation(k1)
       el = fes.GetBE(k1)
       nodes1 = el.GetNodes()
       m = mfem.DenseMatrix(nodes1.GetNPoints(), tr1.GetSpaceDim())
       shape = []
       for idx in range(nodes1.GetNPoints()):
          tr1.SetIntPoint(nodes1.IntPoint(idx))
          el.CalcVShape(tr1,m)
          shape.append(m.GetDataArray()[idx,:].copy())
       return np.stack(shape)

   def get_all_intpoint2(fes, pt1all, pt2all, pto1all, pto2all, k1all,
                         k2all, sh1all, sh2all, map_1_2,
                         tru, trv, tru2, trv2, 
                         tol, tdof, fesize):
       pt = []
       subvdofs1 = []
       subvdofs2 = []

       map = scipy.sparse.lil_matrix((fesize, fesize), dtype=float)

       num_entry = 0
       num_entry2 = 0       
       num_pts = 0       

       decimals = int(np.abs(np.log10(tol)))
       for k0 in range(len(pt1all)):
           k2 = map_1_2[k0]
           pt1 = pt1all[k0]
           pto1 = pto1all[k0]
           newk1 = k1all[k0]
           sh1 = sh1all[k0]           
           pt2 = pt2all[k2]
           pto2 = pto2all[k2]
           newk2 = k2all[k2]
           sh2 = sh2all[k2]
           for k, p in enumerate(pt1):
               num_pts = num_pts + 1              
               if newk1[k,2] == -1: continue
               if newk1[k,2] in tdof: continue
               if newk1[k,2] in subvdofs1: continue               
               dist = np.sum((pt2-p)**2, 1)
               d = np.where(dist == np.min(dist))[0]
               #dprint1(dist-np.min(dist), len(d))
               if len(d) == 1:
                   '''
                   this factor is not always 1
                   '''
                   num_entry = num_entry + 1                   
                   s = np.sign(newk1[k,1] +0.5)*np.sign(newk2[d,1] + 0.5)
                   d = d[0]
                   
                   p1 = pto1[k]; p2 = pto2[d]
                   delta = np.sum(np.std(p1, 0))/np.sum(np.std(sh1, 0))/10.

                   v1 = np.array([tru(p1) - tru(p1 + delta*sh1[newk1[k, 0]]),
                                  trv(p1) - trv(p1 + delta*sh1[newk1[k, 0]])])
                   v2 = np.array([tru2(p2) - tru2(p2 + delta*sh2[newk2[d, 0]]),
                                  trv2(p2) - trv2(p2 + delta*sh2[newk2[d, 0]])])
                   fac_2 = np.sum(v1*v2)/np.sum(v1*v1)*s 

                   map[newk2[d,2], newk1[k,2]] = np.around(fac_2, decimals)

               elif len(d) == 2:
                   num_entry2 = num_entry2 + 1                  
                   dd = np.argsort(np.sum((pt1 - p)**2, 1))
                   
                   p1 = pto1[dd[0]]; p3 = pto2[d[0]]
                   p2 = pto1[dd[1]]; p4 = pto2[d[1]]
                   delta = np.sum(np.std(p1, 0))/np.sum(np.std(sh1, 0))/10.
                   
                   v1 = np.array([tru(p1) - tru(p1 + delta*sh1[newk1[dd[0], 0]]),
                                  trv(p1) - trv(p1 + delta*sh1[newk1[dd[0], 0]])])
                   v2 = np.array([tru(p2) - tru(p2 + delta*sh1[newk1[dd[1], 0]]),
                                  trv(p2) - trv(p2 + delta*sh1[newk1[dd[1], 0]])])
                   v3 = np.array([tru2(p3) - tru2(p3 + delta*sh2[newk2[d[0], 0]]),
                                  trv2(p3) - trv2(p3 + delta*sh2[newk2[d[0], 0]])])
                   v4 = np.array([tru2(p4) - tru2(p4 + delta*sh2[newk2[d[1], 0]]),
                                  trv2(p4) - trv2(p4 + delta*sh2[newk2[d[1], 0]])])

                   v1 = v1*np.sign(newk1[dd[0], 1] +0.5)
                   v2 = v2*np.sign(newk1[dd[1], 1] +0.5)
                   v3 = v3*np.sign(newk2[d[0], 1] +0.5)
                   v4 = v4*np.sign(newk2[d[1], 1] +0.5)
                   s = np.sign(newk1[k,1] +0.5)*np.sign(newk2[d,1] + 0.5) 

                   def vnorm(v):
                       return v/np.sqrt(np.sum(v**2))
                   v1n = vnorm(v1) ; v2n = vnorm(v2)
                   v3n = vnorm(v3) ; v4n = vnorm(v4)                   

#                   print v1, v2, v3, v4                   
#                   if np.abs((np.abs(np.sum(v1*v3)/np.sum(v1*v1)) - 1) < tol
#                       and np.abs(np.abs(np.sum(v2*v4)/np.sum(v2*v2)) -  1) < tol):
#                   print 'check', np.abs(np.abs(np.sum(v1n*v3n))-1),   np.abs(np.abs(np.sum(v2n*v4n))-1)

                   if (np.abs(np.abs(np.sum(v1n*v3n))-1) < tol and
                       np.abs(np.abs(np.sum(v2n*v4n))-1) < tol):
                       fac1 = np.sum(v1*v3)/np.sum(v1*v1)
                       fac2 = np.sum(v2*v4)/np.sum(v2*v2)
                       #print 'first case', fac1, fac2                  
                       map[newk2[d[0],2], newk1[dd[0],2]] = np.around(fac1, decimals)
                       map[newk2[d[1],2], newk1[dd[1],2]] = np.around(fac2, decimals)
                   elif (np.abs(np.abs(np.sum(v2n*v3n))-1) < tol and
                         np.abs(np.abs(np.sum(v1n*v4n))-1) < tol):
                       fac1 = np.sum(v1*v4)/np.sum(v1*v1)
                       fac2 = np.sum(v2*v3)/np.sum(v2*v2)
                       #print 'second case', fac1, fac2, decimals               
                       map[newk2[d[1],2], newk1[dd[0],2]] = np.around(fac1, decimals)
                       map[newk2[d[0],2], newk1[dd[1],2]] = np.around(fac2, decimals)
                   else:
                       #print 'two shape vector couples!', v1, v2, v3, v4
                       #m2 = np.transpose(np.vstack((v1, v2)))
                       #m1 = np.transpose(np.vstack((v3, v4)))
                       #m = np.dot(np.linalg.inv(m1), m2)
                       m1 = np.transpose(np.vstack((v1, v2)))
                       m2 = np.transpose(np.vstack((v3, v4)))
                       m = np.dot(np.linalg.inv(m1), m2)
                       m = np.around(np.linalg.inv(m), decimals = decimals)
                       #print m
                       #print pt1[dd[0]], pt1[dd[1]], pt2[d[0]], pt2[d[1]]
                       #print v1, v2, v3, v4, newk1[dd[0]][2], newk1[dd[1]][2],newk2[d[0]][2], newk2[d[1]][2]
                       if m[0,0] != 0.: map[newk2[d[0],2], newk1[dd[0],2]] = m[0,0]
                       if m[0,1] != 0.: map[newk2[d[1],2], newk1[dd[0],2]] = m[0,1]
                       if m[1,0] != 0.: map[newk2[d[0],2], newk1[dd[1],2]] = m[1,0]
                       if m[1,1] != 0.: map[newk2[d[1],2], newk1[dd[1],2]] = m[1,1]
               else:
                   raise AssertionError("three dof coupling is not supported. ")
                   pass
           subvdofs1.extend([s for k, v, s in newk1])
           subvdofs2.extend([s for k, v, s in newk2])
       #print len(subvdofs1), len(subvdofs2)
       total_entry2 = sum(allgather(num_entry2))
       total_entry = sum(allgather(num_entry))
       total_pts = sum(allgather(num_pts))
       
       dprint2("map size", map.shape)
       dprint2("local pts/edge entry/mid-face entry", num_pts, " " , num_entry, " ",
               num_entry2)
       dprint1("total pts/edge entry/mid-face entry", total_pts, " " , total_entry, " ",
               total_entry2)
       
       return map


   def find_el_center(fes, ibdr1, tru, trv):
       if len(ibdr1) == 0: return np.empty(shape=(0,2))
       mesh = fes.GetMesh()
       pts = np.vstack([np.mean([(tru(mesh.GetVertexArray(kk)),
                                  trv(mesh.GetVertexArray(kk)))
                                  for kk in mesh.GetBdrElementVertices(k)],0)
                        for k in ibdr1])
       return pts

   def get_vshape_all2(fes, ibdr1):
       if len(ibdr1) == 0:
           tmp = np.stack([get_vshape_all(fes, k) for k in [0]])
           res = np.empty(shape=(0, tmp.shape[1], tmp.shape[2]))
       else:
           res =  np.stack([get_vshape_all(fes, k) for k in ibdr1])
       dprint2('get_vshape_all2', res.shape)
       return res
        
   def get_element_data(k1, debug, tru, trv):
       tr1 = fes.GetBdrElementTransformation(k1)
       nodes1 = fes.GetBE(k1).GetNodes()
       vdof1 = fes.GetBdrElementVDofs(k1)
       pt1 = np.vstack([np.array([tru(tr1.Transform(nodes1.IntPoint(kk))),
                                  trv(tr1.Transform(nodes1.IntPoint(kk)))])
              for kk in range(len(vdof1))])
       pt1o = np.vstack([np.array(tr1.Transform(nodes1.IntPoint(kk)))
                       for kk in range(len(vdof1))])


       #print vdof1
       subvdof1 = [x if x>= 0 else -1-x for x in vdof1]

       if use_parallel:
           subvdof2= [fes.GetLocalTDofNumber(i)
                      for i in subvdof1]


           flag = False
           for k, x in enumerate(subvdof2):
               if x >=0:
                  #subvdof2[k] = get_HypreParMatrixRow(engine.matvec[2][0], x)
                  subvdof2[k] = fes.GetMyTDofOffset()+ x                  
               else:
                  if debug:
                     print('not own')
                     flag = True
           if flag: print(subvdof1, vdof1, subvdof2)

             ## note subdof1 = -1 if it is not owned by the node
       else:
           subvdof2 = subvdof1
       newk1 = np.vstack([(k, xx[0], xx[1])
                for k, xx in enumerate(zip(vdof1, subvdof2))])
       pt1 =  np.vstack([pt1[kk] for kk, v, s in newk1])
       pt1o = np.vstack([pt1o[kk] for kk, v, s in newk1])

       return pt1, newk1, pt1o

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
                dprint3("distances", np.min(dist),fdist[isort[:5]])
                #if fdist[isort[1]] <  np.mean(fdist[isort[2:4]])*1e-6:
                #     minidx = list(isort[:2])
                #else:
                #     minidx = [isort[0]]
                minidx =  np.where(dist.flatten() == np.min(dist.flatten()))[0]
                while all(k2all[:,:,2].flatten()[minidx] == -1):
                    dprint3("distances (non -1 exists?)", fdist[minidx],
                            k2all[:,:,2].flatten()[minidx])
                    minidx = np.hstack((minidx, isort[len(minidx)]))
                dprint3("distances", np.min(dist),fdist[minidx], fdist[isort[:len(minidx)+1]])
                for i in minidx:
                    if k2all[:,:,2].flatten()[i] != -1:
                       subvdof2[kk] = k2all[:,:,2].flatten()[i]
          if check:
              dprint1('resolved dof', k2all[k2][:,2])
              if -1 in subvdof2: raise AssertionError("failed to resolve shadow DoF")
   
   nbe = fes.GetNBE()
   ibdr1 = [i for i in range(nbe) if fes.GetBdrAttribute(i) in idx1]
   ibdr2 = [i for i in range(nbe) if fes.GetBdrAttribute(i) in idx2]
   ct1 = find_el_center(fes, ibdr1, transu, transv)
   ct2 = find_el_center(fes, ibdr2, transu2, transv2)

   arr1 = [get_element_data(k, False, transu, transv) for k in ibdr1]
   arr2 = [get_element_data(k, False, transu2, transv2) for k in ibdr2]
   
   sh1all = get_vshape_all2(fes, ibdr1)
   sh2all = get_vshape_all2(fes, ibdr2)

   # prepare default shape...
   ttt = [get_element_data(k, False, transu, transv) for k in [0]]
   ptzero = np.stack([x for x, y, z in ttt])
   ptozero = np.stack([z for x, y, z in ttt])
   kzero = np.stack([y for x, y, z in ttt])
   
   # pt is on (u, v), pto is (x, y, z)
   pt1all = (np.stack([x for x, y, z in arr1]) if len(arr1)!=0 else
             np.empty(shape=(0,ptzero.shape[1],ptzero.shape[2])))
   pt2all = (np.stack([x for x, y, z in arr2]) if len(arr2)!=0 else
             np.empty(shape=(0,ptzero.shape[1],ptzero.shape[2])))            
   pto1all = (np.stack([z for x, y, z in arr1]) if len(arr1)!=0 else
             np.empty(shape=(0,ptozero.shape[1],ptozero.shape[2])))
   pto2all = (np.stack([z for x, y, z in arr2]) if len(arr2)!=0 else
             np.empty(shape=(0,ptozero.shape[1],ptozero.shape[2])))            

   if use_parallel:
      if MPI.INT.size == 4:
          dtype = np.int32
      else:
          dtype = np.int64
   else:
      dtype = np.int32
   k1all = (np.stack([y for x, y, z in arr1]).astype(dtype) if len(arr1)!=0
            else np.empty(shape=(0,kzero.shape[1],kzero.shape[2])).astype(dtype))
   k2all = (np.stack([y for x, y, z in arr2]).astype(dtype) if len(arr2)!=0
            else np.empty(shape=(0,kzero.shape[1],kzero.shape[2])).astype(dtype))

   if mapper_debug:         
   #   for i in range(num_proc):
      for i in range(1):
          MPI.COMM_WORLD.Barrier()      
          if ( myid == 1): 
             dprint3("checking ptall ", myid)
             for k, x in enumerate(pt1all):
                print(x, k1all[k], sh1all[k])
      for i in range(num_proc):
          MPI.COMM_WORLD.Barrier()      
          if ( myid == i): 
             dprint3("checking ptall2 ", myid)
             for k, x in enumerate(pt2all):
                 print(x, k2all[k], sh2all[k])             

   #print 'k1all', k1all.shape
   #print 'sh1all', sh1all.shape
   fesize = fes.GetNDofs()
   if use_parallel:
       # share ibr2 (destination information among nodes...)
       ct2 =  allgather_vector(ct2, MPI.DOUBLE)
       pt2all =  allgather_vector(pt2all, MPI.DOUBLE)
       pto2all =  allgather_vector(pto2all, MPI.DOUBLE)
       k2all =  allgather_vector(k2all, MPI.INT)
       sh2all =  allgather_vector(sh2all, MPI.DOUBLE)
       fesize = fes.GlobalTrueVSize()
       dprint1(fesize, fes.GetTrueVSize())

   if mapper_debug:         
      MPI.COMM_WORLD.Barrier()                   
      if ( myid == 1 ):
         dprint3("checking pt2all ", myid)
         for k, x in enumerate(pt2all):
              print(x, k2all[k], sh2all[k])
      MPI.COMM_WORLD.Barrier()                   
   #print 'k2all', k2all.shape
   #print 'sh2all', sh2all.shape
   ctr_dist =  np.array([np.min(np.sum((ct2-c)**2, 1)) for c in ct1])
   if ctr_dist.size > 0 and np.max(ctr_dist) > 1e-15:
       print('Center Dist may be too large (check mesh): ' + str(np.max(ctr_dist)))
   map_1_2= [np.argmin(np.sum((ct2-c)**2, 1)) for c in ct1]
   
   if use_parallel:
       resolve_nonowned_dof(pt1all, pt2all, k1all, k2all, map_1_2)

   map =  get_all_intpoint2(fes, pt1all, pt2all, pto1all, pto2all, k1all,
                            k2all, sh1all, sh2all, map_1_2,
                            transu, transv,
                            transu2, transv2,
                            tol, tdof, fesize)

   return map
