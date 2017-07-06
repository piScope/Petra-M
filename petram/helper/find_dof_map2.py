import numpy as np
import scipy

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('find_dof_map2')

from petram.solver.mumps.hypre_to_mumps import get_HypreParMatrixRow

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
   from mpi4py import MPI
   num_proc = MPI.COMM_WORLD.size
   myid     = MPI.COMM_WORLD.rank
else:
   import mfem.ser as mfem
   num_proc = 1
   myid = 0

mapper_debug = False
def find_dof_map2(idx1, idx2, transu, transv, tdof=None,
                  engine=None, tor=1e-4, use_complex = False):
   '''
   find_dof_map

   projection coupling between DoFs. so far, it is tested
   only for triangle Nedelec boundary elemets.

   it maps element nodes of on bdr1 (attribute = idx1) to 
   bdr2(attribute =idx2)

   this version accept two mapping function which mappps
   (x, y, z) to (u, v) plane

   Need to test that this approach works even when two 
   surface is not parallel...

   '''
   fes = engine.fespace
   
#   if trans is None:
#       trans = np.array([[0, 1, 0], [0, 0, 1]]) # projection to y, z         
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

   def get_all_intpoint(fes, ibdr1, ibdr2, transu, transv, tor, tdo):

       pt = []
       subvdofs1 = []
       subvdofs2 = []
       map = scipy.sparse.lil_matrix((fes.GetNDofs(), fes.GetNDofs()),
                                     dtype=float)

       for k1, k2 in zip(ibdr1, ibdr2):
           tr1 = fes.GetBdrElementTransformation(k1)
           nodes1 = fes.GetBE(k1).GetNodes()
           vdof1 = fes.GetBdrElementVDofs(k1)
           pt1 = [np.array([transu(tr1.Transform(nodes1.IntPoint(kk))),
                            transv(tr1.Transform(nodes1.IntPoint(kk)))])
                   for kk in range(len(vdof1))]
           subvdof1 = [x if x>= 0 else -1-x for x in vdof1]
           vdof2 = fes.GetBdrElementVDofs(k2)
           nodes2 = fes.GetBE(k2).GetNodes()
           tr2 = fes.GetBdrElementTransformation(k2)
           pt2 = [np.array([transu(tr2.Transform(nodes2.IntPoint(kk))),
                            transv(tr2.Transform(nodes2.IntPoint(kk)))]) 
                   for kk in range(len(vdof2))]
           subvdof2 = [x if x>= 0 else -1-x for x in vdof2]

           newk1 = [(k, xx[0], xx[1])
                    for k, xx in enumerate(zip(vdof1, subvdof1)) 
                    if not xx[1] in subvdofs1]
           newk2 = [(k, xx[0], xx[1])
                    for k, xx in enumerate(zip(vdof2, subvdof2)) 
                    if not xx[1] in subvdofs2]
           pt1 = [pt1[kk] for kk, v, s in newk1]
           pt2 = [pt2[kk] for kk, v, s in newk2]
           if len(pt1) != len(pt2):
              dprint1("something wrong in DoF mappling")
              break
           #print subvdof1, newk1
           #print subvdof2, newk2
           for k, p in enumerate(pt1):
               if newk1[k][2] in tdof: continue
               dist = np.sum((pt2-p)**2, 1)
               d = np.where(dist == np.min(dist))[0]
               if len(d) == 1:
                   '''
                   this factor is not always 1
                   '''
                   s = np.sign(newk1[k][1] +0.5)*np.sign(newk2[d][1] + 0.5)
                   v1 = np.array([transu(get_vshape(fes, k1, newk1[k][0])),
                                  transv(get_vshape(fes, k1, newk1[k][0]))])
                   v2 = np.array([transu(get_vshape(fes, k2, newk2[d][0])),
                                  transv(get_vshape(fes, k2, newk2[d][0]))])
                   fac = np.sum(v1*v2)/np.sum(v1*v1)*s 
                   #print fac, v1, v2, newk1[k][1], newk2[d][1]
                   print [newk2[d][2], newk1[k][2]] , fac
                   map[newk2[d][2], newk1[k][2]] = fac
               elif len(d) == 2:
                   dd = np.where(np.sum((pt1 - p)**2, 1) < tor)[0]
                   v1 = np.array((transu(get_vshape(fes, k1, newk1[dd[0]][0])),
                                  transv(get_vshape(fes, k1, newk1[dd[0]][0]))))
                   v2 = np.array((transu(get_vshape(fes, k1, newk1[dd[1]][0])),
                                  transv(get_vshape(fes, k1, newk1[dd[1]][0]))))
                   v3 = np.array((transu(get_vshape(fes, k2, newk2[d[0]][0])),
                                  transv(get_vshape(fes, k2, newk2[d[0]][0]))))
                   v4 = np.array((transu(get_vshape(fes, k2, newk2[d[1]][0])),
                                  transv(get_vshape(fes, k2, newk2[d[1]][0]))))
                   v1 = v1*np.sign(newk1[dd[0]][1] +0.5)
                   v2 = v2*np.sign(newk1[dd[1]][1] +0.5)
                   v3 = v3*np.sign(newk2[d[0]][1] +0.5)
                   v4 = v4*np.sign(newk2[d[1]][1] +0.5)
                   if (np.abs(np.sum(v1*v3)/np.sum(v1*v1)) - 1 < tor
                       and np.abs(np.sum(v2*v4)/np.sum(v2*v2)) -  1 < tor):
                       fac1 = np.sum(v1*v3)/np.sum(v1*v1)
                       fac2 = np.sum(v2*v4)/np.sum(v2*v2)

                       print [newk2[d[0]][2], newk1[dd[0]][2]], fac1
                       print [newk2[d[1]][2], newk1[dd[1]][2]], fac2
                       map[newk2[d[0]][2], newk1[dd[0]][2]] = fac1
                       map[newk2[d[1]][2], newk1[dd[1]][2]] = fac2
                   elif (np.abs(np.sum(v1*v4)/np.sum(v1*v1)) - 1 < tor and
                         np.abs(np.sum(v2*v3)/np.sum(v2*v2)) - 1 < tor):
                       fac1 = np.sum(v1*v4)/np.sum(v1*v1)
                       fac2 = np.sum(v2*v3)/np.sum(v2*v2)
                       print [newk2[d[1]][2], newk1[dd[0]][2]], fac1
                       print [newk2[d[0]][2], newk1[dd[1]][2]], fac2
                       map[newk2[d[1]][2], newk1[dd[0]][2]] = fac1
                       map[newk2[d[0]][2], newk1[dd[1]][2]] = fac2
                   else:
                       print 'two shape vector couples!'
                       m2 = np.transpose(np.vstack((v1, v2)))
                       m1 = np.transpose(np.vstack((v3, v4)))
                       m = np.dot(np.linalg.inv(m1), m2)
                       #print v1, v2, v3, v4, newk1[dd[0]][2], newk1[dd[1]][2],newk2[d[0]][2], newk2[d[1]][2]
                       map[newk2[d[0]][2], newk1[dd[0]][2]] = m[0,0]
                       map[newk2[d[1]][2], newk1[dd[0]][2]] = m[1,0]
                       map[newk2[d[0]][2], newk1[dd[1]][2]] = m[0,1]
                       map[newk2[d[1]][2], newk1[dd[1]][2]] = m[1,1]
               else:
                   print 'error'
           subvdofs1.extend([s for k, v, s in newk1])
           subvdofs2.extend([s for k, v, s in newk2])
       return map
    
   def get_all_intpoint2(fes, pt1all, pt2all, k1all,
                         k2all, sh1all, sh2all, map_1_2,
                         transu, transv, tor, tdof, fesize,
                         use_complex=False):
       pt = []
       subvdofs1 = []
       subvdofs2 = []
       if use_complex:
           map = scipy.sparse.lil_matrix((fesize, fesize), dtype=complex)
       else:
           map = scipy.sparse.lil_matrix((fesize, fesize), dtype=float)

       for k0 in range(len(pt1all)):
           k2 = map_1_2[k0]
           pt1 = pt1all[k0]
           newk1 = k1all[k0]
           sh1 = sh1all[k0]           
           pt2 = pt2all[k2]
           newk2 = k2all[k2]
           sh2 = sh2all[k2]
           #if myid == 1: print newk1
           for k, p in enumerate(pt1):
               if newk1[k,2] == -1: continue
               if newk1[k,2] in tdof: continue
               if newk1[k,2] in subvdofs1: continue               
               dist = np.sum((pt2-p)**2, 1)
               d = np.where(dist == np.min(dist))[0]
               #dprint1('min_dist', np.min(dist))
               if len(d) == 1:
                   '''
                   this factor is not always 1
                   '''
                   s = np.sign(newk1[k,1] +0.5)*np.sign(newk2[d,1] + 0.5)
                   d = d[0]
                   v1 = np.array([transu(sh1[newk1[k, 0]]),
                                  transv(sh1[newk1[k, 0]])])
                   v2 = np.array([transu(sh2[newk2[d, 0]]),
                                  transv(sh2[newk2[d, 0]])])
                   fac = np.sum(v1*v2)/np.sum(v1*v1)*s 
                   #print fac, v1, v2, newk1[k][1], newk2[d][1]
                   #print v1, v2, [newk2[d][2], newk1[k][2]] , fac
                   map[newk2[d][2], newk1[k][2]] = fac
               elif len(d) == 2:
#                   print 'I am here', tor
                   dd = np.argsort(np.sum((pt1 - p)**2, 1))
                   #dd = np.where(np.sum((pt1 - p)**2, 1) < tor)[0]
                   #print dd
                   #print (sh1[newk1[dd[0], 0]],sh1[newk1[dd[1], 0]], sh2[newk2[d[0], 0]], sh2[newk2[d[1], 0]])
                   #print pt1[dd[0]], pt1[dd[1]], pt2[d[0]], pt2[d[1]]
                   v1 = np.array((transu(sh1[newk1[dd[0], 0]]),
                                  transv(sh1[newk1[dd[0], 0]])))
                   v2 = np.array((transu(sh1[newk1[dd[1], 0]]),
                                  transv(sh1[newk1[dd[1], 0]])))
                   v3 = np.array((transu(sh2[newk2[d[0], 0]]),
                                  transv(sh2[newk2[d[0], 0]])))
                   v4 = np.array((transu(sh2[newk2[d[1], 0]]),
                                  transv(sh2[newk2[d[1], 0]])))
#                   print newk1[dd[0], 1], newk1[dd[1], 1], newk2[d[0], 1], newk2[d[1], 1] 
                   
                   v1 = v1*np.sign(newk1[dd[0], 1] +0.5)
                   v2 = v2*np.sign(newk1[dd[1], 1] +0.5)
                   v3 = v3*np.sign(newk2[d[0], 1] +0.5)
                   v4 = v4*np.sign(newk2[d[1], 1] +0.5)
                   #print np.abs(np.sum(v1*v3)/np.sum(v1*v1)),  np.abs(np.sum(v2*v4)/np.sum(v2*v2))
                   #print np.abs(np.sum(v1*v4)/np.sum(v1*v1)),  np.abs(np.sum(v2*v3)/np.sum(v2*v2))
                   def vnorm(v):
                       return v/np.sqrt(np.sum(v**2))
                   v1n = vnorm(v1) ; v2n = vnorm(v2)
                   v3n = vnorm(v3) ; v4n = vnorm(v4)                   
#                   print v1, v2, v3, v4                   
#                   if np.abs((np.abs(np.sum(v1*v3)/np.sum(v1*v1)) - 1) < tor
#                       and np.abs(np.abs(np.sum(v2*v4)/np.sum(v2*v2)) -  1) < tor):
#                   print 'check', np.abs(np.abs(np.sum(v1n*v3n))-1),   np.abs(np.abs(np.sum(v2n*v4n))-1)
                   if (np.abs(np.abs(np.sum(v1n*v3n))-1) < tor and
                       np.abs(np.abs(np.sum(v2n*v4n))-1) < tor):
                       fac1 = np.sum(v1*v3)/np.sum(v1*v1)
                       fac2 = np.sum(v2*v4)/np.sum(v2*v2)
                       #print 'first case'                       
                       #print  np.sum(v1*v3), np.sum(v1*v1)
                       #print [newk2[d[0]][2], newk1[dd[0]][2]], fac1
                       #print [newk2[d[1]][2], newk1[dd[1]][2]], fac2
                       #print fac1, fac2
                       map[newk2[d[0],2], newk1[dd[0],2]] = fac1
                       map[newk2[d[1],2], newk1[dd[1],2]] = fac2
                   elif (np.abs(np.abs(np.sum(v2n*v3n))-1) < tor and
                         np.abs(np.abs(np.sum(v1n*v4n))-1) < tor):
#                   elif np.abs((np.abs(np.sum(v1*v4)/np.sum(v1*v1)) - 1) < tor and
#                         np.abs(np.abs(np.sum(v2*v3)/np.sum(v2*v2)) - 1) < tor):
                       fac1 = np.sum(v1*v4)/np.sum(v1*v1)
                       fac2 = np.sum(v2*v3)/np.sum(v2*v2)
                       #print 'second case'
                       #print [newk2[d[1]][2], newk1[dd[0]][2]], fac1
                       #print [newk2[d[0]][2], newk1[dd[1]][2]], fac2
                       #print fac1, fac2                       
                       map[newk2[d[1],2], newk1[dd[0],2]] = fac1
                       map[newk2[d[0],2], newk1[dd[1],2]] = fac2
                   else:
                       #print 'two shape vector couples!', v1, v2, v3, v4
                       #m2 = np.transpose(np.vstack((v1, v2)))
                       #m1 = np.transpose(np.vstack((v3, v4)))
                       #m = np.dot(np.linalg.inv(m1), m2)
                       m1 = np.transpose(np.vstack((v1, v2)))
                       m2 = np.transpose(np.vstack((v3, v4)))
                       m = np.dot(np.linalg.inv(m1), m2)
                       
                       #print m
                       #print pt1[dd[0]], pt1[dd[1]], pt2[d[0]], pt2[d[1]]
                       #print v1, v2, v3, v4, newk1[dd[0]][2], newk1[dd[1]][2],newk2[d[0]][2], newk2[d[1]][2]
                       map[newk2[d[0],2], newk1[dd[0],2]] = m[0,0]
                       map[newk2[d[1],2], newk1[dd[0],2]] = m[1,0]
                       map[newk2[d[0],2], newk1[dd[1],2]] = m[0,1]
                       map[newk2[d[1],2], newk1[dd[1],2]] = m[1,1]
               else:
                   print 'here??'
                   pass
           subvdofs1.extend([s for k, v, s in newk1])
           subvdofs2.extend([s for k, v, s in newk2])
       #print len(subvdofs1), len(subvdofs2)
       return map


   def find_el_center(fes, ibdr1, transu, transv):
       if len(ibdr1) == 0: return np.empty(shape=(0,2))
       mesh = fes.GetMesh()
       pts = np.vstack([np.mean([(transu(mesh.GetVertexArray(kk)),
                                  transv(mesh.GetVertexArray(kk)))
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
        
   def get_element_data(k1, debug = False):
       tr1 = fes.GetBdrElementTransformation(k1)
       nodes1 = fes.GetBE(k1).GetNodes()
       vdof1 = fes.GetBdrElementVDofs(k1)
       pt1 = np.vstack([np.array([transu(tr1.Transform(nodes1.IntPoint(kk))),
                        transv(tr1.Transform(nodes1.IntPoint(kk)))])
              for kk in range(len(vdof1))])
       #print vdof1
       subvdof1 = [x if x>= 0 else -1-x for x in vdof1]

       if use_parallel:
           subvdof2= [engine.fespace.GetLocalTDofNumber(i)
                      for i in subvdof1]


#           print  subvdof2
           flag = False
           for k, x in enumerate(subvdof2):
               if x >=0:
                  subvdof2[k] = get_HypreParMatrixRow(engine.r_A, x)
               else:
                  if debug:
                     print 'not own'
                     flag = True
           if flag: print subvdof1, vdof1, subvdof2

             ## note subdof1 = -1 if it is not owned by the node
       else:
           subvdof2 = subvdof1
       newk1 = np.vstack([(k, xx[0], xx[1])
                for k, xx in enumerate(zip(vdof1, subvdof2))])
       pt1 = np.vstack([pt1[kk] for kk, v, s in newk1])
       return pt1, newk1

   def resolve_nonowned_dof(pt1all, pt2all, k1all, k2all, map_1_2):
       for k in range(len(pt1all)):
          subvdof1 = k1all[k][:,2]
          k2 = map_1_2[k]
          subvdof2 = k2all[k2][:,2]
          pt2 = pt2all[k2]
          check = False
          for kk, x in enumerate(subvdof2):
             if x == -1:
                dist = pt2all-pt2[kk]
                dist = np.sum((dist)**2, -1)
                minidx =  np.where(dist.flatten() == np.min(dist.flatten()))[0]
                for i in minidx:
                    if k2all[:,:,2].flatten()[i] != -1:
                       subvdof2[kk] = k2all[:,:,2].flatten()[i]
                check = True
          if check:  dprint2('resolved dof', k2all[k2][:,2])
   
   nbe = fes.GetNBE()
   ibdr1 = [i for i in range(nbe) if fes.GetBdrAttribute(i) in idx1]
   ibdr2 = [i for i in range(nbe) if fes.GetBdrAttribute(i) in idx2]
   ct1 = find_el_center(fes, ibdr1, transu, transv)
   ct2 = find_el_center(fes, ibdr2, transu, transv)

   arr1 = [get_element_data(k, False) for k in ibdr1]
   arr2 = [get_element_data(k, False) for k in ibdr2]
   
   sh1all = get_vshape_all2(fes, ibdr1)
   sh2all = get_vshape_all2(fes, ibdr2)

   # prepare default shape...
   ttt = [get_element_data(k, False) for k in [0]]
   ptzero = np.stack([x for x, y in ttt])
   kzero = np.stack([y for x, y in ttt])
   
   pt1all = (np.stack([x for x, y in arr1]) if len(arr1)!=0 else
             np.empty(shape=(0,ptzero.shape[1],ptzero.shape[2])))
   pt2all = (np.stack([x for x, y in arr2]) if len(arr2)!=0 else
             np.empty(shape=(0,ptzero.shape[1],ptzero.shape[2])))            

   if use_parallel:
      if MPI.INT.size == 4:
          dtype = np.int32
      else:
          dtype = np.int64
   else:
      dtype = np.int32
   k1all = (np.stack([y for x, y in arr1]).astype(dtype) if len(arr1)!=0
            else np.empty(shape=(0,kzero.shape[1],kzero.shape[2])).astype(dtype))
   k2all = (np.stack([y for x, y in arr2]).astype(dtype) if len(arr2)!=0
            else np.empty(shape=(0,kzero.shape[1],kzero.shape[2])).astype(dtype))

   if mapper_debug:         
   #   for i in range(num_proc):
      for i in range(1):
          MPI.COMM_WORLD.Barrier()      
          if ( myid == 1): 
             dprint3("checking ptall ", myid)
             for k, x in enumerate(pt1all):
                print x, k1all[k], sh1all[k]
      for i in range(num_proc):
          MPI.COMM_WORLD.Barrier()      
          if ( myid == i): 
             dprint3("checking ptall2 ", myid)
             for k, x in enumerate(pt2all):
                 print x, k2all[k], sh2all[k]             


   #print 'k1all', k1all.shape
   #print 'sh1all', sh1all.shape
   fesize = engine.fespace.GetNDofs()
   if use_parallel:
       # share ibr2 (destination information among nodes...)
       ct2 =  engine.allgather_vector(ct2, MPI.DOUBLE)
       pt2all =  engine.allgather_vector(pt2all, MPI.DOUBLE)
       k2all =  engine.allgather_vector(k2all, MPI.INT)
       sh2all =  engine.allgather_vector(sh2all, MPI.DOUBLE)
       fesize = engine.fespace.GlobalTrueVSize()
       
   if mapper_debug:         
      MPI.COMM_WORLD.Barrier()                   
      if ( myid == 1 ):
         dprint3("checking pt2all ", myid)
         for k, x in enumerate(pt2all):
              print x, k2all[k], sh2all[k]
      MPI.COMM_WORLD.Barrier()                   
   #print 'k2all', k2all.shape
   #print 'sh2all', sh2all.shape
       
   map_1_2= [np.argmin(np.sum((ct2-c)**2, 1)) for c in ct1]
   if use_parallel:
       resolve_nonowned_dof(pt1all, pt2all, k1all, k2all, map_1_2)

   map =  get_all_intpoint2(fes, pt1all, pt2all, k1all,
                            k2all, sh1all, sh2all, map_1_2,
                            transu, transv, tor, tdof, fesize,
                            use_complex)

   
   if debug.debug_default_level > 1:
      if use_parallel: 
          for i in range(num_proc):
             MPI.COMM_WORLD.Barrier()      
             if ( myid == i ): 
                dprint3("checking map on node ", myid)
                dprint3(map)
      else:
          pass
          #print map
   return map
