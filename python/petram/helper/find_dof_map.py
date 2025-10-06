import numpy as np
def find_dof_map(fes, idx1, idx2, trans = None):
   '''
   find_dof_map

   projection coupling between DoFs. so far, it is tested
   only for triangle Nedelec boundary elemets.

   it maps element nodes of on bdr1 (attribute = idx1) to 
   bdr2(attribute =idx2)

   boundary elements are projected to a 2D surface 
   and matched on the surface (using trans in keyword). 
   By default, it projects DoF points on y-z plan. 

   Need to test that this approach works even when two 
   surface is not parallel...

   '''
   if trans is None:
       trans = np.array([[0, 1, 0], [0, 0, 1]]) # projection to y, z         
   def get_vshape(fes, k1, idx):
       tr1 = fes.GetBdrElementTransformation(k1)
       el = fes.GetBE(k1)
       nodes1 = el.GetNodes()
       m = mfem.DenseMatrix(nodes1.GetNPoints(), tr1.GetSpaceDim())
       tr1.SetIntPoint(nodes1.IntPoint(idx))
       el.CalcVShape(tr1,m)
       return m.GetDataArray()[idx, :]
   def get_all_intpoint(fes, ibdr1, ibdr2, trans):
       pt = []
       subvdofs1 = []
       subvdofs2 = []
       map = scipy.sparse.lil_matrix((fes.GetNDofs(), fes.GetNDofs()), dtype=float)

       for k1, k2 in zip(ibdr1, ibdr2):
           tr1 = fes.GetBdrElementTransformation(k1)
           nodes1 = fes.GetBE(k1).GetNodes()
           vdof1 = fes.GetBdrElementVDofs(k1)
           pt1 = [np.dot(trans, tr1.Transform(nodes1.IntPoint(kk))) 
                 for kk in range(len(vdof1))]
           subvdof1 = [x if x>= 0 else -1-x for x in vdof1]
           vdof2 = fes.GetBdrElementVDofs(k2)
           nodes2 = fes.GetBE(k2).GetNodes()
           tr2 = fes.GetBdrElementTransformation(k2)
           pt2 = [np.dot(trans, tr2.Transform(nodes2.IntPoint(kk))) 
                 for kk in range(len(vdof2))]
           subvdof2 = [x if x>= 0 else -1-x for x in vdof2]

           newk1 = [(k, xx[0], xx[1]) for k, xx in enumerate(zip(vdof1, subvdof1)) 
                   if not xx[1] in subvdofs1]
           newk2 = [(k, xx[0], xx[1]) for k, xx in enumerate(zip(vdof2, subvdof2)) 
                   if not xx[1] in subvdofs2]
           pt1 = [pt1[kk] for kk, v, s in newk1]
           pt2 = [pt2[kk] for kk, v, s in newk2]

           #print subvdof1, newk1
           #print subvdof2, newk2
           for k, p in enumerate(pt1):
               dist = np.sum((pt2-p)**2, 1)
               d = np.where(dist == np.min(dist))[0]
               if len(d) == 1:
                   '''
                   this factor is not always 1
                   '''
                   s = np.sign(newk1[k][1] +0.5)*np.sign(newk2[d][1] + 0.5)
                   v1 = np.dot(trans, get_vshape(fes, k1, newk1[k][0])) 
                   v2 = np.dot(trans, get_vshape(fes, k2, newk2[d][0]))
                   fac = np.sum(v1*v2)/np.sum(v1*v1)*s 
                   #print fac, v1, v2, newk1[k][1], newk2[d][1]
                   #print [newk2[d][2], newk1[k][2]] , fac
                   map[newk2[d][2], newk1[k][2]] = fac
               elif len(d) == 2:
                   dd = np.where(np.sum((pt1 - p)**2, 1) == 0.0)[0]
                   v1 = np.dot(trans, get_vshape(fes, k1, newk1[dd[0]][0])) 
                   v2 = np.dot(trans, get_vshape(fes, k1, newk1[dd[1]][0]))
                   v3 = np.dot(trans, get_vshape(fes, k2, newk2[d[0]][0])) 
                   v4 = np.dot(trans, get_vshape(fes, k2, newk2[d[1]][0]))
                   v1 = v1*np.sign(newk1[dd[0]][1] +0.5)
                   v2 = v2*np.sign(newk1[dd[1]][1] +0.5)
                   v3 = v3*np.sign(newk2[d[0]][1] +0.5)
                   v4 = v4*np.sign(newk2[d[1]][1] +0.5)
                   if (np.abs(np.sum(v1*v3)/np.sum(v1*v1)) == 1
                       and np.abs(np.sum(v2*v4)/np.sum(v2*v2)) == 1):
                       fac1 = np.sum(v1*v3)/np.sum(v1*v1)
                       fac2 = np.sum(v2*v4)/np.sum(v2*v2)

                       #print [newk2[d[0]][2], newk1[dd[0]][2]]
                       #print [newk2[d[1]][2], newk1[dd[1]][2]]
                       map[newk2[d[0]][2], newk1[dd[0]][2]] = fac1
                       map[newk2[d[1]][2], newk1[dd[1]][2]] = fac2
                   elif (np.abs(np.sum(v1*v4)/np.sum(v1*v1)) == 1 and
                         np.abs(np.sum(v2*v3)/np.sum(v2*v2)) == 1):
                       fac1 = np.sum(v1*v4)/np.sum(v1*v1)
                       fac2 = np.sum(v2*v3)/np.sum(v2*v2)
                       #print [newk2[d[1]][2], newk1[dd[0]][2]], fac1
                       #print [newk2[d[0]][2], newk1[dd[1]][2]], fac2
                       map[newk2[d[1]][2], newk1[dd[0]][2]] = fac1
                       map[newk2[d[0]][2], newk1[dd[1]][2]] = fac2
                   else:
                       #print 'two shape vector couples!'
                       m2 = np.transpose(np.vstack((v1, v2)))
                       m1 = np.transpose(np.vstack((v3, v4)))
                       m = np.dot(np.linalg.inv(m1), m2)
                       #print v1, v2, v3, v4, newk1[dd[0]][2], newk1[dd[1]][2],newk2[d[0]][2], newk2[d[1]][2]

                       map[newk2[d[0]][2], newk1[dd[0]][2]] = m[0,0]
                       map[newk2[d[1]][2], newk1[dd[0]][2]] = m[1,0]
                       map[newk2[d[0]][2], newk1[dd[1]][2]] = m[0,1]
                       map[newk2[d[1]][2], newk1[dd[1]][2]] = m[1,1]
               else:
                   pass
           subvdofs1.extend([s for k, v, s in newk1])
           subvdofs2.extend([s for k, v, s in newk2])
       return map

   def find_el_center(fes, ibdr1, trans):
       mesh = fes.GetMesh()
       pts = np.vstack([np.mean([np.dot(trans, mesh.GetVertexArray(kk))
                                 for kk in mesh.GetBdrElementVertices(k)],0)
                        for k in ibdr1])
       return pts
   nbe = fes.GetNBE()
   ibdr1 = [i for i in range(nbe) if fes.GetBdrAttribute(i) == idx1]
   ibdr2 = [i for i in range(nbe) if fes.GetBdrAttribute(i) == idx2]

   ct1 = find_el_center(fes, ibdr1, trans)
   ct2 = find_el_center(fes, ibdr2, trans)

   map_1_2= [np.argmin(np.sum((ct2-c)**2, 1)) for c in ct1]
   ibdr2 = [ibdr2[x] for x in map_1_2]
   ct2 = ct2[map_1_2]

   map = get_all_intpoint(fes, ibdr1, ibdr2, trans)

   return map
