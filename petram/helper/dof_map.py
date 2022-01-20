from __future__ import print_function
from petram.mfem_config import use_parallel
from petram.helper.matrix_file import write_matrix, write_vector
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
import bisect
import warnings
import traceback
from scipy.sparse import lil_matrix
import petram.debug as debug
debug.debug_default_level = 1
dprint1, dprint2, dprint3 = debug.init_dprints('dof_map')

# this is for testing. need to add three lines below to suppress
# numpy related warning (copied from numpy.__init__.py)
warnings.filterwarnings('error', category=RuntimeWarning)
# Filter out Cython harmless warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")


if use_parallel:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    num_proc = MPI.COMM_WORLD.size
    myid = MPI.COMM_WORLD.rank
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
                  'Vertices': 'GetElementVertices',
                  'Transformation': 'GetElementTransformation',
                  'Element': 'GetFE',
                  'VDofs': 'GetElementVDofs',
                  'VDofTransformation': 'GetElementVDofTransformation'}
methods['Bdr'] = {'N': 'GetNBE',
                  'AttributeArray': 'GetBdrAttributeArray',
                  'Vertices': 'GetBdrElementVertices',
                  'Transformation': 'GetBdrElementTransformation',
                  'Element': 'GetBE',
                  'VDofs': 'GetBdrElementVDofs',
                  'VDofTransformation': 'GetBdrElementVDofTransformation'}


def notrans(xyz):
    return np.array(xyz, copy=False)


def get_surface_mode(dim, sdim):
    mode1 = ''
    if sdim == 3:
        if dim == 3:
            mode1 = 'Bdr'
        elif dim == 2:
            mode1 = 'Dom'
    elif sdim == 2:
        if dim == 2:
            mode1 = 'Dom'

    if mode1 == '':
        assert False, "not supprint dim/sdim "+str(dim)+'/'+str(sdim)
    return mode1


def get_edge_mode(dim, sdim):
    mode1 = ''
    if sdim == 3:
        if dim == 2:
            mode1 = 'Bdr'
        if dim == 1:
            mode1 = 'Dom'
    elif sdim == 2:
        if dim == 2:
            mode1 = 'Bdr'
        if dim == 1:
            mode1 = 'Dom'
    if mode1 == '':
        assert False, "not supported dim/sdim "+str(dim)+'/'+str(sdim)
    return mode1


def get_volume_mode(dim, sdim):
    if sdim == 3 and dim == 3:
        return 'Dom'
    else:
        assert False, "volume mapping must be 3d mesh"
    return mode1


def find_element(fes, attr, mode='Bdr'):
    mesh = fes.GetMesh()
    m = getattr(mesh, methods[mode]['AttributeArray'])
    arr = m()
    flag = np.in1d(arr, attr)
    return np.arange(len(arr))[flag]


def find_el_center(fes, ibdr1, trans1, mode='Bdr'):

    if len(ibdr1) == 0:
        return np.empty(shape=(0, 2))
    mesh = fes.GetMesh()
    m = getattr(mesh, methods[mode]['Vertices'])
    pts = np.mean(np.array([[trans1(mesh.GetVertexArray(i)) for i in m(k)]
                            for k in ibdr1]), 1)
    # pts = np.vstack([np.mean([trans1(mesh.GetVertexArray(kk))
    #                          for kk in m(k)],0) for k in ibdr1])
    return pts


def get_element_data(fes, idx, trans, mode='Bdr', use_global=False):
    mesh = fes.GetMesh()

    GetTrans = getattr(fes, methods[mode]['Transformation'])
    GetElement = getattr(fes, methods[mode]['Element'])
    GetVDofs = getattr(fes, methods[mode]['VDofs'])

    ret = [None]*len(idx)
    for iii, k1 in enumerate(idx):
        tr1 = GetTrans(k1)
        nodes1 = GetElement(k1).GetNodes()
        vdof1 = GetVDofs(k1)
        # pt1 = np.vstack([trans(tr1.Transform(nodes1.IntPoint(kk)))
        #                 for kk in range(len(vdof1))])
        pt1o = np.vstack([tr1.Transform(nodes1.IntPoint(kk))
                          for kk in range(len(vdof1))])
        pt1 = np.vstack([trans(pt) for pt in pt1o])

        subvdof1 = [x if x >= 0 else -1-x for x in vdof1]
        if use_parallel:
            if use_global:
                subvdof2 = [fes.GetGlobalTDofNumber(i) for i in subvdof1]
            else:
                subvdof2 = [fes.GetLocalTDofNumber(i) for i in subvdof1]
                flag = False
                myoffset = fes.GetMyTDofOffset()
                for k, x in enumerate(subvdof2):
                    if x >= 0:
                        subvdof2[k] = myoffset + x
                    else:
                        flag = True
                if element_data_debug and flag:
                    dprint2(subvdof1, vdof1, subvdof2)
            # note subdof2 = -1 if it is not owned by the node
        else:
            subvdof2 = subvdof1

        newk1 = np.vstack([(k, xx[0], xx[1])
                           for k, xx in enumerate(zip(vdof1, subvdof2))])

        # this doesn't do anything since kk is [0, 1,2,3,...]??
        #pt1 =  np.vstack([pt1[kk] for kk, v, s in newk1])
        #pt1o = np.vstack([pt1o[kk] for kk, v, s in newk1])

        ret[iii] = (newk1, pt1, pt1o)
    return ret


def get_shape(fes, ibdr, mode='Bdr'):
    mesh = fes.GetMesh()

    use_weight = False
    fec_name = fes.FEColl().Name()
    if (fec_name.startswith('RT') and
            mode == 'Bdr'):
        use_weight = True
    if (fec_name.startswith('ND')):
        use_weight = False
        # mode == 'Bdr'):
        use_weight = True
        #use_weight = False

    GetTrans = getattr(fes, methods[mode]['Transformation'])
    GetElement = getattr(fes, methods[mode]['Element'])
    GetVDofs = getattr(fes, methods[mode]['VDofs'])
    GetVDofTrans = getattr(fes, methods[mode]['VDofTransformation'])

    ret = [None]*len(ibdr)
    if len(ibdr) == 0:
        return ret

    # this is to make sure that IntegraitonPoint in Eltrans is
    # set once...
    tr1 = GetTrans(0)
    el = GetElement(0)
    nodes1 = el.GetNodes()
    doftrans = GetVDofTrans(0)
    tr1.SetIntPoint(nodes1.IntPoint(0))

    for iii, k1 in enumerate(ibdr):
        tr1 = GetTrans(k1)
        el = GetElement(k1)
        nodes1 = el.GetNodes()

        weight = tr1.Weight() if use_weight else 1

        v = mfem.Vector(nodes1.GetNPoints())
        shape = [None]*nodes1.GetNPoints()
        for idx in range(len(shape)):
            el.CalcShape(nodes1.IntPoint(idx), v)
            shape[idx] = v.GetDataArray()[idx]*weight
        ret[iii] = np.array(shape)

    return ret


def get_vshape(fes, ibdr, mode='Bdr'):
    mesh = fes.GetMesh()

    GetTrans = getattr(fes, methods[mode]['Transformation'])
    GetElement = getattr(fes, methods[mode]['Element'])
    GetVDofs = getattr(fes, methods[mode]['VDofs'])
    GetVDofTrans = getattr(fes, methods[mode]['VDofTransformation'])

    ret = [None]*len(ibdr)
    if len(ibdr) == 0:
        return ret

    # this is to make sure that IntegraitonPoint in Eltrans is
    # set once...
    tr1 = GetTrans(0)
    el = GetElement(0)
    nodes1 = el.GetNodes()
    doftrans = GetVDofTrans(0)
    tr1.SetIntPoint(nodes1.IntPoint(0))

    v0 = mfem.Vector(tr1.GetSpaceDim())

    use_weight = True
    for iii, k1 in enumerate(ibdr):
        tr1 = GetTrans(k1)
        el = GetElement(k1)
        nodes1 = el.GetNodes()
        doftrans = GetVDofTrans(k1)

        m = mfem.DenseMatrix(nodes1.GetNPoints(),
                             tr1.GetSpaceDim())
        shape = [None]*nodes1.GetNPoints()

        vv = mfem.Vector(nodes1.GetNPoints())

        for idx in range(len(shape)):
            tr1.SetIntPoint(nodes1.IntPoint(idx))
            el.CalcVShape(tr1, m)

            if doftrans is not None:
                vv.Assign(0.0)
                vv[idx] = 1
                doftrans.InvTransformPrimal(vv)
                m.MultTranspose(vv, v0)
                shape[idx] = v0.GetDataArray().copy()
            else:
                shape[idx] = m.GetDataArray()[idx, :].copy()
        ret[iii] = shape
    return ret


def redistribute_pt2_k2(pt2all,  pto2all, k2all, sh2all, map_1_2):
    # map matrix is filled where fes1 elements are owned
    # we deliver pt2 data to nodes where map filling takes place.

    # sort map_1_2 to make bisect work
    sort_idx = np.argsort(map_1_2)
    map_1_2 = np.sort(map_1_2)
    isort_idx = np.arange(len(sort_idx))[np.argsort(sort_idx)]

    k2offset = np.cumsum(comm.allgather(len(k2all)))

    segs = [0]+[bisect.bisect_left(map_1_2, i) for i in k2offset]

    data = [map_1_2[segs[i]:segs[i+1]] for i in range(num_proc)]

    destinations = comm.alltoall(data)
    # nicePrint(destinations)
    k2offset = np.hstack(([0], k2offset))
    myoffset2 = k2offset[myid]

    data1 = []
    data2 = []
    data3 = []
    data4 = []
    for mm in destinations:
        idx = [x-myoffset2 for x in mm]
        data1.append([pt2all[x] for x in idx])
        data2.append([pto2all[x] for x in idx])
        data3.append([k2all[x] for x in idx])
        data4.append([sh2all[x] for x in idx])

    pt2all = sum(comm.alltoall(data1), [])
    pto2all = sum(comm.alltoall(data2), [])
    k2all = sum(comm.alltoall(data3), [])
    sh2all = sum(comm.alltoall(data4), [])

    pt2all = [pt2all[x] for x in isort_idx]
    pto2all = [pto2all[x] for x in isort_idx]
    k2all = [k2all[x] for x in isort_idx]
    sh2all = [sh2all[x] for x in isort_idx]

    map_1_2 = np.arange(len(pt2all))
    # nicePrint(k2all)
    return pt2all,  pto2all, k2all, sh2all, map_1_2


def redistribute_external_entry(external_entry, rstart):
    #  redistribute external entry. first regroup
    #  entries to the destination. then all-to-all

    rstarts = comm.allgather(rstart)
    dests = [bisect.bisect_right(rstarts, r)for r, c, d in external_entry]

    data = [[] for x in range(num_proc)]
    for i, d in zip(dests, external_entry):
        data[i].append(d)

    external_entry = sum(comm.alltoall(data), [])
    return np.array(external_entry)


def map_dof_scalar(map, fes1, fes2, pt1all, pt2all, pto1all, pto2all,
                   k1all, k2all, sh1all, sh2all, map_1_2,
                   trans1, trans2, tol, tdof, rstart):

    dprint1("map_dof_scalar1", debug.format_memory_usage())

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
        # this is global TrueDoF (offset is not subtracted)
        VDoFtoGTDoF = P.indices
        external_entry = []
        gtdof_check = []

    for k0 in range(len(pt1all)):
        k2 = map_1_2[k0]
        pt1 = pt1all[k0]
        pto1 = pto1all[k0]
        newk1 = k1all[k0]  # (i local DoF, global DoF)
        sh1 = sh1all[k0]
        pt2 = pt2all[k2]
        pto2 = pto2all[k2]
        newk2 = k2all[k2]
        sh2 = sh2all[k2]
        for k, p in enumerate(pt1):
            num_pts = num_pts + 1

            # if newk1[k,2] in tdof: continue
            iii = bisect.bisect_left(tdof, newk1[k, 2])
            if iii != len(tdof) and tdof[iii] == newk1[k, 2]:
                continue

            # if newk1[k,2] in subvdofs1: continue
            iii = bisect.bisect_left(subvdofs1, newk1[k, 2])
            if iii != len(subvdofs1) and subvdofs1[iii] == newk1[k, 2]:
                continue
            #print("maping points", pt1, pt2)
            dist = np.sum((pt2-p)**2, 1)
            d = np.where(dist == np.min(dist))[0]
            if myid == 1:
                dprint2('min_dist', np.min(dist))

            if len(d) == 1:
                d = d[0]
                s1 = sh1[newk1[k, 0]]
                s2 = sh2[newk2[d, 0]]
                # dprint1("case1 ", s1, s2) #this looks all 1
                if s1/s2 < 0:
                    dprint2("not positive")
                #if myid == 1: print(newk1[d][2]-rstart, newk2[k][2])
                value = np.around(s1/s2, decimals)
                #print("looking into this...", newk1[k], newk2[d])
                if newk1[k, 2] != -1:
                    # flip the sign to match the orientation
                    # need this for mapping ND on edge
                    # we may need to come back here to handle the case
                    # where the edge segment is randomly oriented....
                    if (newk2[d][1]+0.5)*(newk1[k][1]+0.5) < 0:
                        value *= -1
                    map[newk1[k][2]-rstart,
                        newk2[d][2]] = value
                    num_entry = num_entry + 1
                    bisect.insort_left(subvdofs1, newk1[k][2])
                    # subvdofs1.append(newk1[k][2])
                else:
                    # for scalar, this is perhaps not needed
                    # rr = newk1[k][1]] if newk1[k][1]] >= 0 else -1-newk1[k][1]]
                    # gtdof = VDoFtoGTDoF[rr]
                    assert newk1[k][1] >= 0, "Negative index found"
                    gtdof = VDoFtoGTDoF[newk1[k][1]]
                    if not gtdof in gtdof_check:
                        external_entry.append((gtdof, newk2[d][2], value))
                        gtdof_check.append(gtdof)
            else:
                print("failed to map points", pt1, pt2)
                raise AssertionError(
                    "more than two dofs at same places is not supported. ")
        #subvdofs1.extend([s for k, v, s in newk1])
        subvdofs2.extend([s for k, v, s in newk2])

    dprint1("map_dof_scalar2", debug.format_memory_usage())

    if use_parallel:
        dprint1("total entry (before)", sum(allgather(num_entry)))
        external_entry = redistribute_external_entry(
            external_entry, rstart+map.shape[0])

        if len(external_entry.shape) == 2:
            idx1 = np.in1d(external_entry[:, 0], subvdofs1, invert=True)
            val, idx2 = np.unique(external_entry[idx1, 0], return_index=True)
            external_entry = external_entry[idx1][idx2]

            for r, c, d in external_entry:
                # if not r in subvdofs1:
                num_entry = num_entry + 1
                map[r-rstart, c] = d

                #print("adding",myid, r,  c, d )
                # subvdofs1.append(r)

        dprint1("map_dof_scalar3", debug.format_memory_usage())
        '''
        external_entry =  sum(comm.allgather(external_entry),[])           
        for r, c, d in external_entry:
           h = map.shape[0]
           if (r - rstart >= 0 and r - rstart < h and
               not r  in subvdofs1):
               num_entry = num_entry + 1                                 
               print("adding",myid, r,  c, d )
               map[r-rstart, c] = d
               subvdofs1.append(r)
        '''
        total_entry = sum(allgather(num_entry))
        total_pts = sum(allgather(num_pts))
        if sum(allgather(map.nnz)) != total_entry:
            assert False, "total_entry does not match with nnz"
    else:
        total_entry = num_entry
        total_pts = num_pts

    #dprint1("map size", map.shape)
    dprint1("local pts/entry", num_pts, " ", num_entry)
    dprint1("total pts/entry", total_pts, " ", total_entry)
    return map


def map_dof_vector(map, fes1, fes2, pt1all, pt2all, pto1all, pto2all,
                   k1all, k2all, sh1all, sh2all, map_1_2,
                   trans1, trans2, tol, tdof, rstart,
                   old_mapping=True):

    dprint1("map_dof_vector1", debug.format_memory_usage())

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
        P1mat = ToScipyCoo(P).tocsr()
        # this is global TrueDoF (offset is not subtracted)
        external_entry = []
        gtdof_check = []

    def make_entry(r, c, value, num_entry):
        value = np.around(value, decimals)
        if value == 0:
            return num_entry
        if r[1] != -1:
            map[r[1]-rstart, c] = value
            num_entry = num_entry + 1
            bisect.insort_left(subvdofs1, r[1])
            # subvdofs1.append(r[1])
        else:
            rr = r[0] if r[0] >= 0 else -1-r[0]
            gtdofs = P1mat.indices[P1mat.indptr[rr]:P1mat.indptr[rr+1]]
            weights = P1mat.data[P1mat.indptr[rr]:P1mat.indptr[rr+1]]
            for gtdof, w in zip(gtdofs, weights):
                if not gtdof in gtdof_check:
                    external_entry.append((gtdof, c, value*w))
                    gtdof_check.append(gtdof)

        return num_entry

    tdof = sorted(tdof)

    for k0 in range(len(pt1all)):
        k2 = map_1_2[k0]
        pt1 = pt1all[k0]
        pto1 = pto1all[k0]
        newk1 = k1all[k0]  # (i local DoF, global DoF)
        sh1 = sh1all[k0]
        pt2 = pt2all[k2]
        pto2 = pto2all[k2]
        newk2 = k2all[k2]
        sh2 = sh2all[k2]

        #dprint1(len(np.unique(newk1[:,2])) == len(newk1[:,2]))
        for k, p in enumerate(pt1):
            # if idx[k]: continue
            num_pts = num_pts + 1

            # if newk1[k,2] in tdof: continue
            iii = bisect.bisect_left(tdof, newk1[k, 2])
            if iii != len(tdof) and tdof[iii] == newk1[k, 2]:
                continue

            # if newk1[k,2] in subvdofs1: continue
            iii = bisect.bisect_left(subvdofs1, newk1[k, 2])
            if iii != len(subvdofs1) and subvdofs1[iii] == newk1[k, 2]:
                continue

            dist = np.sum((pt2-p)**2, 1)
            d = np.where(dist == np.min(dist))[0]

            #if myid == 1: dprint1('min_dist', np.min(dist))
            if len(d) == 1:
                '''
                this factor is not always 1
                '''

                d = d[0]
                s = np.sign(newk1[k, 1] + 0.5)*np.sign(newk2[d, 1] + 0.5)
                p1 = pto1[k]
                p2 = pto2[d]

                if np.sum(np.std(sh1, 0)) != 0:
                    delta = np.sum(np.std(pto1, 0))/np.sum(np.std(sh1, 0))/10.
                else:
                    delta = 1  # this happens on edge....

                v1 = trans1(p1) - trans1(p1 + delta*sh1[newk1[k, 0]])
                v2 = trans2(p2) - trans2(p2 + delta*sh2[newk2[d, 0]])

                fac = np.sum(v1*v2)/np.sum(v1*v1)*s
                # except RuntimeWarning:
                #   print(pto1, pto1.shape, p1,p2, s, delta, sh1[newk1[k, 0]], sh2[newk2[d, 0]])
                #   assert False, "Got Here"

                num1 = make_entry(newk1[k, [1, 2]], newk2[d, 2], fac, num1)

            elif len(d) == 2:
                dd = np.argsort(np.sum((pt1 - p)**2, 1))

                p1 = pto1[dd[0]]
                p3 = pto2[d[0]]
                p2 = pto1[dd[1]]
                p4 = pto2[d[1]]
                delta = np.sum(np.std(pto1, 0))/np.sum(np.std(sh1, 0))/10.

                v1 = trans1(p1) - trans1(p1 + delta*sh1[newk1[dd[0], 0]])
                v2 = trans1(p2) - trans1(p2 + delta*sh1[newk1[dd[1], 0]])
                v3 = trans2(p3) - trans2(p3 + delta*sh2[newk2[d[0], 0]])
                v4 = trans2(p4) - trans2(p4 + delta*sh2[newk2[d[1], 0]])
                v1 = v1*np.sign(newk1[dd[0], 1] + 0.5)
                v2 = v2*np.sign(newk1[dd[1], 1] + 0.5)
                v3 = v3*np.sign(newk2[d[0], 1] + 0.5)
                v4 = v4*np.sign(newk2[d[1], 1] + 0.5)
                s = np.sign(newk1[k, 1] + 0.5)*np.sign(newk2[d, 1] + 0.5)

                def vnorm(v):
                    return v/np.sqrt(np.sum(v**2))
                v1n = vnorm(v1)
                v2n = vnorm(v2)
                v3n = vnorm(v3)
                v4n = vnorm(v4)

                if (np.abs(np.abs(np.sum(v1n*v3n))-1) < tol and
                        np.abs(np.abs(np.sum(v2n*v4n))-1) < tol):
                    fac1 = np.sum(v1*v3)/np.sum(v1*v1)
                    fac2 = np.sum(v2*v4)/np.sum(v2*v2)

                    num2 = make_entry(
                        newk1[dd[0], [1, 2]], newk2[d[0], 2], fac1, num2)
                    num2 = make_entry(
                        newk1[dd[1], [1, 2]], newk2[d[1], 2], fac2, num2)

                elif (np.abs(np.abs(np.sum(v2n*v3n))-1) < tol and
                      np.abs(np.abs(np.sum(v1n*v4n))-1) < tol):
                    fac1 = np.sum(v1*v4)/np.sum(v1*v1)
                    fac2 = np.sum(v2*v3)/np.sum(v2*v2)

                    num2 = make_entry(
                        newk1[dd[0], [1, 2]], newk2[d[1], 2], fac1, num2)
                    num2 = make_entry(
                        newk1[dd[1], [1, 2]], newk2[d[0], 2], fac2, num2)
                else:
                    def proj2d(v, e1, e2):
                        return np.array([np.sum(v*e1), np.sum(v*e2)])
                    if len(v1) == 3:  # if vector is 3D, needs to prjoect on surface
                        e3 = np.cross(v1n, v2n)
                        e1 = v1n
                        e2 = np.cross(e1, e3)
                        v1 = proj2d(v1, e1, e2)
                        v2 = proj2d(v2, e1, e2)
                        v3 = proj2d(v3, e1, e2)
                        v4 = proj2d(v4, e1, e2)

                    if old_mapping:
                        m1 = np.transpose(np.vstack((v1, v2)))
                        m2 = np.transpose(np.vstack((v3, v4)))
                        m = np.dot(np.linalg.inv(m1), m2)
                        m = np.around(np.linalg.inv(m), decimals=decimals)
                        m = np.transpose(m)
                    else:
                        m1 = np.vstack((v1, v2))
                        m2 = np.vstack((v3, v4))
                        m = np.dot(m2, np.linalg.inv(m1))
                        m = np.around(m, decimals=decimals)
                        m = np.transpose(m)
                    '''
                   #this was wrong. keeping it for now for record (2020.03)
                   '''
                    num2 = make_entry(
                        newk1[dd[0], [1, 2]], newk2[d[0], 2], m[0, 0], num2)
                    num2 = make_entry(
                        newk1[dd[0], [1, 2]], newk2[d[1], 2], m[0, 1], num2)
                    num2 = make_entry(
                        newk1[dd[1], [1, 2]], newk2[d[0], 2], m[1, 0], num2)
                    num2 = make_entry(
                        newk1[dd[1], [1, 2]], newk2[d[1], 2], m[1, 1], num2)

            elif len(d) == 3:
                dd = np.argsort(np.sum((pt1 - p)**2, 1))

                p1 = [pto1[dd[i]] for i in [0, 1, 2]]
                p2 = [pto2[d[i]] for i in [0, 1, 2]]

                delta = np.sum(np.std(pto1, 0))/np.sum(np.std(sh1, 0))/10.

                v1 = [trans1(p1[i]) - trans1(p1[i] + delta*sh1[newk1[dd[i], 0]])
                      for i in [0, 1, 2]]
                v2 = [trans2(p2[i]) - trans2(p2[i] + delta*sh2[newk2[d[i], 0]])
                      for i in [0, 1, 2]]

                v1 = [v1[i]*np.sign(newk1[dd[i], 1] + 0.5) for i in [0, 1, 2]]
                v2 = [v2[i]*np.sign(newk2[d[i], 1] + 0.5) for i in [0, 1, 2]]

                s = np.sign(newk1[k, 1] + 0.5)*np.sign(newk2[d, 1] + 0.5)

                def vnorm(v):
                    return v/np.sqrt(np.sum(v**2))
                v1n = [vnorm(v) for v in v1]
                v2n = [vnorm(v) for v in v2]

                '''
                note: see old_mapping flag note below

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
                '''
                if old_mapping:
                    m1 = np.transpose(np.vstack(v1))
                    m2 = np.transpose(np.vstack(v2))
                    m = np.dot(np.linalg.inv(m1), m2)
                    m = np.around(np.linalg.inv(m), decimals=decimals)
                    m = np.transpose(m)
                else:
                    m1 = np.vstack(v1)
                    m2 = np.vstack(v2)
                    m = np.dot(m2, np.linalg.inv(m1))
                    m = np.around(m, decimals=decimals)
                    m = np.transpose(m)

                num2 = make_entry(newk1[dd[0], [1, 2]],
                                  newk2[d[0], 2], m[0, 0], num2)
                num2 = make_entry(newk1[dd[0], [1, 2]],
                                  newk2[d[1], 2], m[0, 1], num2)
                num2 = make_entry(newk1[dd[0], [1, 2]],
                                  newk2[d[2], 2], m[0, 2], num2)

                num2 = make_entry(newk1[dd[1], [1, 2]],
                                  newk2[d[0], 2], m[1, 0], num2)
                num2 = make_entry(newk1[dd[1], [1, 2]],
                                  newk2[d[1], 2], m[1, 1], num2)
                num2 = make_entry(newk1[dd[1], [1, 2]],
                                  newk2[d[2], 2], m[1, 2], num2)

                num2 = make_entry(newk1[dd[2], [1, 2]],
                                  newk2[d[0], 2], m[2, 0], num2)
                num2 = make_entry(newk1[dd[2], [1, 2]],
                                  newk2[d[1], 2], m[2, 1], num2)
                num2 = make_entry(newk1[dd[2], [1, 2]],
                                  newk2[d[2], 2], m[2, 2], num2)

            else:
                print(pt1, pt2)
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

    dprint1("map_dof_vector2", debug.format_memory_usage())
    num_entry = num1 + num2

    if use_parallel:

        dprint1("total entry (before)", sum(allgather(num_entry)))

        #nicePrint("data to exchange", len(external_entry))
        external_entry = redistribute_external_entry(
            external_entry, rstart+map.shape[0])

        if len(external_entry.shape) == 2:
            idx1 = np.in1d(external_entry[:, 0], subvdofs1, invert=True)
            val, idx2 = np.unique(external_entry[idx1, 0], return_index=True)
            external_entry = external_entry[idx1][idx2]

            for r, c, d in external_entry:
                # if not r in subvdofs1:
                num_entry = num_entry + 1
                #print("adding",myid, r,  c, d )
                map[r-rstart, c] = d
                # subvdofs1.append(r)

        dprint1("map_dof_vector3", debug.format_memory_usage())
        '''        
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
        '''
        total_entry = sum(allgather(num_entry))
        total_pts = sum(allgather(num_pts))
        if sum(allgather(map.nnz)) != total_entry:
            assert False, "total_entry does not match with nnz"
    else:
        total_entry = num_entry
        total_pts = num_pts

    #dprint1("map size", map.shape)
    dprint1("local pts/entry", num_pts, " ", num_entry)
    dprint1("total pts/entry", total_pts, " ", total_entry)

    return map


def gather_dataset(idx1, idx2, fes1, fes2, trans1,
                   trans2, tol, shape_type='scalar',
                   mode='surface'):

    #dprint1("gather_dataset1", debug.format_memory_usage())

    if fes2 is None:
        fes2 = fes1
    if trans1 is None:
        trans1 = notrans
    if trans2 is None:
        trans2 = notrans

    mesh1 = fes1.GetMesh()
    mesh2 = fes2.GetMesh()

    if mode == 'volume':
        mode1 = get_volume_mode(mesh1.Dimension(), mesh1.SpaceDimension())
        mode2 = get_volume_mode(mesh2.Dimension(), mesh2.SpaceDimension())
    elif mode == 'surface':
        mode1 = get_surface_mode(mesh1.Dimension(), mesh1.SpaceDimension())
        mode2 = get_surface_mode(mesh2.Dimension(), mesh2.SpaceDimension())
    elif mode == 'edge':
        mode1 = get_edge_mode(mesh1.Dimension(), mesh1.SpaceDimension())
        mode2 = get_edge_mode(mesh2.Dimension(), mesh2.SpaceDimension())
    elif mode == 'edge':
        mode1 = get_point_mode(mesh1.Dimension(), mesh1.SpaceDimension())
        mode2 = get_point_mode(mesh2.Dimension(), mesh2.SpaceDimension())

    # collect data
    ibdr1 = find_element(fes1, idx1, mode=mode1)
    ibdr2 = find_element(fes2, idx2, mode=mode2)

    ct1 = find_el_center(fes1, ibdr1, trans1, mode=mode1)
    ct2 = find_el_center(fes2, ibdr2, trans2, mode=mode2)
    arr1 = get_element_data(fes1, ibdr1, trans1, mode=mode1)
    arr2 = get_element_data(fes2, ibdr2, trans1, mode=mode2, use_global=True)

    if shape_type == 'scalar':
        sh1all = get_shape(fes1, ibdr1, mode=mode1)
        sh2all = get_shape(fes2, ibdr2, mode=mode2)
    elif shape_type == 'vector':
        sh1all = get_vshape(fes1, ibdr1, mode=mode1)
        sh2all = get_vshape(fes2, ibdr2, mode=mode2)
    else:
        assert False, "Unknown shape type"

    #dprint1("gather_dataset2", debug.format_memory_usage())

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
        ct1dim = ct1.shape[1] if ct1.size > 0 else 0
        ct1dim = comm.allgather(ct1dim)
        ct1 = np.atleast_2d(ct1).reshape(-1, max(ct1dim))
        ct2 = np.atleast_2d(ct2).reshape(-1, max(ct1dim))
        ct2 = allgather_vector(ct2, MPI.DOUBLE)

    #dprint1("gather_dataset3", debug.format_memory_usage())

    # mapping between elements
    from scipy.spatial import cKDTree
    tree = cKDTree(ct2)
    ctr_dist, map_1_2 = tree.query(ct1)

    #dprint1("gather_dataset4", debug.format_memory_usage())

    if ctr_dist.size > 0 and np.max(ctr_dist) > 1e-15:
        print('Center Dist may be too large (check mesh): ' +
              str(np.max(ctr_dist)))

    if use_parallel:

        pt2all, pto2all, k2all, sh2all, map_1_2 = redistribute_pt2_k2(pt2all,
                                                                      pto2all, k2all, sh2all,
                                                                      map_1_2)

        #dprint1("gather_dataset5", debug.format_memory_usage())

    # map is fill as transposed shape (row = fes1)

    data = pt1all, pt2all, pto1all, pto2all, k1all, k2all, sh1all, sh2all,

    return data, map_1_2


def get_empty_map(fes1, fes2, is_complex=False):
    '''
    empty matrix (fes1: test, fes2: 
    '''
    if use_parallel:
        fesize1 = fes1.GetTrueVSize()
        fesize2 = fes2.GlobalTrueVSize()
        rstart = fes1.GetMyTDofOffset()
    else:
        fesize1 = fes1.GetNDofs()
        fesize2 = fes2.GetNDofs()
        rstart = 0

    dtype = complex if is_complex else float
    map = lil_matrix((fesize1, fesize2), dtype=dtype)
    return map, rstart


def map_xxx_h1(xxx, idx1, idx2, fes1, fes2=None, trans1=None,
               trans2=None, tdof1=None, tdof2=None, tol=1e-4,
               old_mapping=True):
    '''
    map DoF on surface to surface

      fes1: source finite element space
      fes2: destination finite element space

      idx1: surface attribute (Bdr for 3D/3D, Domain for 2D/3D or 2D/2D)

    '''

    if fes2 is None:
        fes2 = fes1
    if trans1 is None:
        trans1 = notrans
    if trans2 is None:
        trans2 = trans1
    if tdof1 is None:
        tdof1 = []
    if tdof2 is None:
        tdof2 = []

    tdof = tdof1  # ToDo support tdof2
    map, rstart = get_empty_map(fes1, fes2)
    data, elmap = gather_dataset(idx1, idx2, fes1, fes2, trans1,
                                 trans2, tol, shape_type='scalar',
                                 mode=xxx)

    pt1all, pt2all, pto1all, pto2all, k1all, k2all, sh1all, sh2all = data

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


def map_point_h1(idx1, idx2, fes1, fes2=None, trans1=None,
                 trans2=None, tdof1=None, tdof2=None, tol=1e-4,
                 old_mapping=True):
    '''
    map DoF from vertex to vertex

      fes1: source finite element space
      fes2: destination finite element space

      idx1: surface attribute (Bdr for 3D/3D, Domain for 2D/3D or 2D/2D)

    '''

    if fes2 is None:
        fes2 = fes1
    if trans1 is None:
        trans1 = notrans
    if trans2 is None:
        trans2 = trans1
    if tdof1 is None:
        tdof1 = []
    if tdof2 is None:
        tdof2 = []

    tdof = tdof1  # ToDo support tdof2
    map, rstart = get_empty_map(fes1, fes2)

    R = fes1.GetConformingRestriction()
    if R is not None:
        assert False, "Non-conforming mesh not supported"
    '''
    If R is not none. we have to do something like this...
        VDof2TDof = np.zeros(fes.GetNDofs(), dtype=int)
        for i, j in enumerate(R.GetJArray()):
            VDof2TDof[j] = i
        TDof2Vdof = R.GetJArray().copy()            
    '''
    mesh = fes1.GetMesh()
    battrs = {battr: k for k, battr in enumerate(mesh.GetBdrAttributeArray())}

    vdofs1 = [fes1.GetBdrElementVDofs(
        battrs[i])[0] if i in battrs else -1 for i in idx1]
    vdofs2 = [fes2.GetBdrElementVDofs(
        battrs[i])[0] if i in battrs else -1 for i in idx2]

    if use_parallel:
        P = fes1.Dof_TrueDof_Matrix()
        from mfem.common.parcsr_extra import ToScipyCoo
        P = ToScipyCoo(P).tocsr()

        # this is global TrueDoF (offset is not subtracted # this is okay for scalar)
        VDoFtoGTDoF = P.indices
        gtdofs1 = [VDoFtoGTDoF[x] if x >= 0 else -1 for x in vdofs1]
        gtdofs2 = [VDoFtoGTDoF[x] if x >= 0 else -1 for x in vdofs2]

        # collecting non-negative tdofs2
        gtdofs2 = allgather_vector(np.array(gtdofs2, dtype=int))
        tmp = gtdofs2.reshape(-1, len(idx2))
        gtdofs2 = [np.max(tmp[:, i]) for i in range(len(idx2))]

    else:
        gtdofs1 = vdofs1
        gtdofs2 = vdofs2

    for v1, v2 in zip(gtdofs1, gtdofs2):
        if v1 <= 0:
            continue
        map[v1-rstart, v2] = 1
    return map


def map_xxx_nd(xxx, idx1, idx2, fes1, fes2=None, trans1=None,
               trans2=None, tdof1=None, tdof2=None, tol=1e-4,
               old_mapping=True):
    '''
    map DoF on surface to surface

      fes1: source finite element space
      fes2: destination finite element space

      idx1: surface attribute (Bdr for 3D/3D, Domain for 2D/3D or 2D/2D)

    '''

    if fes2 is None:
        fes2 = fes1
    if trans1 is None:
        trans1 = notrans
    if trans2 is None:
        trans2 = trans1
    if tdof1 is None:
        tdof1 = []
    if tdof2 is None:
        tdof2 = []

    tdof = tdof1  # ToDo support tdof2
    map, rstart = get_empty_map(fes1, fes2)
    data, elmap = gather_dataset(idx1, idx2, fes1, fes2, trans1,
                                 trans2, tol, shape_type='vector',
                                 mode=xxx)

    pt1all, pt2all, pto1all, pto2all, k1all, k2all, sh1all, sh2all = data

    map_dof_vector(map, fes1, fes2, pt1all, pt2all, pto1all, pto2all,
                   k1all, k2all, sh1all, sh2all, elmap,
                   trans1, trans2, tol, tdof1, rstart, old_mapping=old_mapping)

    return map


def map_volume_nd(*args, **kwargs):
    return map_xxx_nd('volume', *args, **kwargs)


def map_surface_nd(*args, **kwargs):
    return map_xxx_nd('surface', *args, **kwargs)


def map_edge_nd(*args, **kwargs):
    return map_xxx_nd('edge', *args, **kwargs)


def map_xxx_rt(xxx, idx1, idx2, fes1, fes2=None, trans1=None,
               trans2=None, tdof1=None, tdof2=None, tol=1e-4,
               old_mapping=True):
    '''
    map DoF on surface to surface

      fes1: source finite element space (row)
      fes2: destination finite element space (col)

      idx1: surface attribute (Bdr for 3D/3D, Domain for 2D/3D or 2D/2D)

    '''

    if fes2 is None:
        fes2 = fes1
    if trans1 is None:
        trans1 = notrans
    if trans2 is None:
        trans2 = trans1
    if tdof1 is None:
        tdof1 = []
    if tdof2 is None:
        tdof2 = []

    map, rstart = get_empty_map(fes1, fes2)
    if xxx == 'volume':
        tdof = tdof1  # ToDo support tdof2
        data, elmap = gather_dataset(idx1, idx2, fes1, fes2, trans1,
                                     trans2, tol, shape_type='vector',
                                     mode=xxx)
        pt1all, pt2all, pto1all, pto2all, k1all, k2all, sh1all, sh2all = data
        map_dof_vector(map, fes1, fes2, pt1all, pt2all, pto1all, pto2all,
                       k1all, k2all, sh1all, sh2all, elmap,
                       trans1, trans2, tol, tdof1, rstart,
                       old_mapping=old_mapping)
    else:
        tdof = tdof1  # ToDo support tdof2
        data, elmap = gather_dataset(idx1, idx2, fes1, fes2, trans1,
                                     trans2, tol, shape_type='scalar',
                                     mode=xxx)

        pt1all, pt2all, pto1all, pto2all, k1all, k2all, sh1all, sh2all = data

        map_dof_scalar(map, fes1, fes2, pt1all, pt2all, pto1all, pto2all,
                       k1all, k2all, sh1all, sh2all, elmap,
                       trans1, trans2, tol, tdof1, rstart)

    return map


def map_volume_rt(*args, **kwargs):
    return map_xxx_rt('volume', *args, **kwargs)


def map_surface_rt(*args, **kwargs):
    return map_xxx_rt('surface', *args, **kwargs)


def projection_matrix(idx1,  idx2,  fes, tdof1, fes2=None, tdof2=None,
                      trans1=None, trans2=None, dphase=0.0, weight=None,
                      tol=1e-7, mode='surface', filldiag=True,
                      old_mapping=True):
    '''
     map: destinatiom mapping 
     smap: source mapping

     old_mapping: True : periodic boundary conditions are implemented this way
                         x = M*y
                  False: projection operator should use this flag.
                         y = M*x
        the difference is only when mapping two/three DoFs sitting at the
        same location. therefore, it only matters for ND and RT cases

    '''
    fec_name = fes.FEColl().Name()
    dprint1("constructing mapping to fec_name", fec_name, mode)
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
    elif fec_name.startswith('H1') and mode == 'point':
        mapper = map_point_h1
    elif fec_name.startswith('L2') and mode == 'volume':
        mapper = map_volume_h1
    elif fec_name.startswith('L2') and mode == 'surface':
        mapper = map_surface_h1
    elif fec_name.startswith('L2') and mode == 'edge':
        mapper = map_edge_h1
    elif fec_name.startswith('RT') and mode == 'volume':
        mapper = map_volume_rt
    elif fec_name.startswith('RT') and mode == 'surface':
        mapper = map_surface_rt
    else:
        raise NotImplementedError("mapping :" + fec_name + ", mode: " + mode)

    map = mapper(idx2, idx1, fes, fes2=fes2, trans1=trans1, trans2=trans2, tdof1=tdof1,
                 tdof2=tdof2, tol=tol, old_mapping=old_mapping)

    if weight is None:
        iscomplex = False
        if (dphase == 0.):
            pass
        elif (dphase == 180.):
            map = map.tocsr()
            map *= -1
            map = map.tolil()

        else:
            iscomplex = True
            map = map.astype(complex)
            # need to this to make efficient....
            map = map.tocsr()
            map *= np.exp(-1j*np.pi/180*dphase)
            map = map.tolil()

    else:
        iscomplex = np.iscomplexobj(weight)
        if iscomplex:
            map = map.astype(complex)
        if map.nnz > 0:
            map = map.tocsr()
            map *= -weight
            map = map.tolil()

    m_coo = map.tocoo()
    row = m_coo.row
    col = m_coo.col
    col = np.unique(col)

    if use_parallel:
        start_row = fes.GetMyTDofOffset()
        end_row = fes.GetMyTDofOffset() + fes.GetTrueVSize()
        col = np.unique(allgather_vector(col))
        row = row + start_row
    else:
        start_row = 0
        end_row = map.shape[0]

    if filldiag:
        for i in range(min(map.shape[0], map.shape[1])):
            r = start_row+i
            if not r in col:
                map[i, r] = 1.0

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
