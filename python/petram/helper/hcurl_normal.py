from petram.mfem_config import use_parallel
import numpy as np
import itertools
from scipy.sparse import lil_matrix
import itertools
from collections import defaultdict, OrderedDict
from mfem.common.mpi_debug import nicePrint
from mfem.common.parcsr_extra import ToScipyCoo

from petram.helper.dof_map import get_empty_map

import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('hcurln')

if use_parallel:
    USE_PARALLEL = True
    import mfem.par as mfem
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    myid = comm.rank
    nprc = comm.size
    from mfem.common.mpi_debug import nicePrint, niceCall
    from petram.helper.mpi_recipes import *
else:
    USE_PARALLEL = False
    import mfem.ser as mfem
    myid = 0
    nprc = 1
    mfem.ParFiniteElementSpace = type(None)
    mfem.ParGridfunction = type(None)
    mfem.ParMesh = type(None)
    mfem.ParMixedBilinearForm = type(None)

    def nicePrint(*x):
        print(*x)

rules = {}


def eval_element_center(fe, trans):
    nodes = fe.GetNodes()
    ptx = [trans.Transform(ip) for ip in nodes]

    return np.mean(np.vstack(ptx), 0)


def get_rule(fe1o, fe2, trans, orderinc=1, verbose=True):
    fe2o = fe2.GetOrder()

    order = fe1o + fe2o + trans.OrderW() + orderinc
    if (fe2.Space() == mfem.FunctionSpace.rQk):
        assert False, "not supported"

    if not (fe2.GetGeomType(), order) in rules:
        ir = mfem.IntRules.Get(fe2.GetGeomType(), order)
        rules[(fe2.GetGeomType(), order)] = ir

    ir = rules[(fe2.GetGeomType(), order)]
    if verbose:
        dprint1("Order/N Points: ", order, "/", ir.GetNPoints())
    return ir


ref_shapes = {}
ref_shapes["tri"] = [np.array(x, dtype=float)
                     for x in [[0, 0], [1, 0], [0, 1]]]
ref_shapes["quad"] = [np.array(x, dtype=float)
                      for x in [[0, 0], [1, 0], [1, 1], [0, 1]]]
ref_shapes["tet"] = [np.array(x, dtype=float) for x in
                     [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]]
ref_shapes["hex"] = [np.array(x, dtype=float) for x in
                     [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                      [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]]


def eval_coeff(coeff, T, ip, MV):
    if isinstance(coeff[0], mfem.MatrixCoefficient):
        coeff[0].Eval(MV[1], T, ip)
        if coeff[1] is None:
            return MV[1].GetDataArray()
        else:
            d = MV[1].GetDataArray().copy()
            coeff[1].Eval(MV[1], T, ip)
            d = d + 1j*MV[1].GetDataArray().copy()
            return d
    elif isinstance(coeff[0], mfem.VectorCoefficient):
        coeff[0].Eval(MV[0], T, ip)
        if coeff[1] is None:
            return MV[0].GetDataArray()
        else:
            d = MV[0].GetDataArray().copy()
            coeff[1].Eval(MV[0], T, ip)
            d = d + 1j*MV[0].GetDataArray().copy()
            return d
    elif isinstance(coeff[0], mfem.Coefficient):
        value = coeff[0].Eval(T, ip)
        if coeff[1] is None:
            return np.array(value)
        else:
            value = value + 1j*coeff[1].Eval(T, ip)
            return np.array(value)
    else:
        assert False, "unsupported type of coefficient: " + str(type(coeff[0]))


def get_inv_doftrans(doftrans, sign1):
    Msign = np.diag(sign1.flatten())
    if doftrans is not None:
        Mdoftrans = np.zeros((doftrans.Size(), doftrans.Size()))
        vv = mfem.Vector(doftrans.Size())
        for i in range(doftrans.Size()):
            vv.Assign(0)
            vv[i] = 1
            doftrans.InvTransformPrimal(vv)
            Mdoftrans[:, i] = vv.GetDataArray()

        #if myid == 1: print(Mdoftrans.astype(int))
        return Mdoftrans.dot(Msign)
    else:
        return Msign


def check_inv_doftrans(doftrans):
    Mdoftrans = np.zeros((doftrans.Size(), doftrans.Size()))
    vv = mfem.Vector(doftrans.Size())
    for i in range(doftrans.Size()):
        vv.Assign(0)
        vv[i] = 1
        doftrans.InvTransformPrimal(vv)
        Mdoftrans[:, i] = vv.GetDataArray()
    print(Mdoftrans)
    Mdoftrans = np.zeros((doftrans.Size(), doftrans.Size()))
    for i in range(doftrans.Size()):
        vv.Assign(0)
        vv[i] = 1
        doftrans.TransformPrimal(vv)
        Mdoftrans[:, i] = vv.GetDataArray()
    print(Mdoftrans)
    Mdoftrans = np.zeros((doftrans.Size(), doftrans.Size()))
    for i in range(doftrans.Size()):
        vv.Assign(0)
        vv[i] = 1
        doftrans.InvTransformDual(vv)
        Mdoftrans[:, i] = vv.GetDataArray()
    print(Mdoftrans)

    Mdoftrans = np.zeros((doftrans.Size(), doftrans.Size()))
    for i in range(doftrans.Size()):
        vv.Assign(0)
        vv[i] = 1
        doftrans.TransformDual(vv)
        Mdoftrans[:, i] = vv.GetDataArray()
    print(Mdoftrans)


def map_ir(fe1, eltrans, coeff1,
           shape1, dim2, sdim2, locnors2,
           sign1, Mdoftrans, MV, th=1e-7):

    g1 = fe1.GetGeomType()
    ip = mfem.IntegrationPoint()

    res = []

    def get_options(g1, *pp):
        if g1 == 2 or g1 == 3:
            s = ref_shapes['tri'] if g1 == 2 else ref_shapes['quad']
            options = [pp[0]*s[i] + (1-pp[0])*s[j] for i, j
                       in itertools.permutations(range(len(s)), 2)]
        elif g1 == 4 or g1 == 5:
            s = ref_shapes['tet'] if g1 == 4 else ref_shapes['hex']
            options = [pp[0]*s[i] + pp[1]*s[j] + (1-pp[0]-pp[1])*s[k] for i, j, k
                       in itertools.permutations(range(len(s)), 3)]
        else:
            print(pp)
            assert False, "geometry type: " + str(g1) + " is not supported."
        return options

    d_misalginment = []

    for data in locnors2:
        pp = data[:dim2]
        p = data[dim2:dim2 + sdim2]
        nor = data[dim2 + sdim2:]

        options = get_options(g1, *pp)

        dd = []

        pxs = []
        for o in options:
            if g1 == 2 or g1 == 3:
                ip.Set2(*o)
            elif g1 == 4 or g1 == 5:
                ip.Set3(*o)
            else:
                assert False, "geometry type: " + \
                    str(g1) + " is not supported."
            px = eltrans.Transform(ip)
            pxs.append(px)
            dd.append(np.sum((p - px)**2))

        assert np.min(dd) < np.max(dd) * \
            th, "point not found: " + str(np.min(dd))

        kk = np.argmin(dd)
        if g1 == 2 or g1 == 3:
            ip.Set2(*options[kk])
        elif g1 == 4 or g1 == 5:
            ip.Set3(*options[kk])
        else:
            assert False, "geometry type: " + str(g1) + " is not supported."
        #if myid == 1: print("location", pxs[kk])
        eltrans.SetIntPoint(ip)
        fe1.CalcVShape(eltrans, shape1)

        # if doftrans is not None:
        s1 = shape1.GetDataArray().transpose().dot(Mdoftrans)
        # else:
        #    s1 = shape1.GetDataArray().transpose().dot(Msign)

        cc = eval_coeff(coeff1, eltrans, ip, MV)

        val = nor.dot(cc.dot(s1))  # *sign1
        # shape1.GetDataArray().transpose())) * sign1   #/ww

        res.append(val)
        d_misalginment.append(np.min(dd))

    return np.vstack(res), np.array(d_misalginment)


def hcurln(fes1, fes2, coeff,
           is_complex=False, bdr='all', orderinc=1, verbose=False):

    mat, rstart = get_empty_map(fes2, fes1, is_complex=is_complex)
    mat2, rstart = get_empty_map(fes2, fes1, is_complex=is_complex)

    from petram.helper.element_map import map_element

    name_fes1 = fes1.FEColl().Name()[:2]
    name_fes2 = fes2.FEColl().Name()[:2]

    if verbose:
        if myid == 0:
            dprint1("fes", name_fes1, name_fes2)

    mesh1 = fes1.GetMesh()
    mesh2 = fes2.GetMesh()

    mesh2.Print("/home/shiraiwa/part.mesh")

    if verbose:
        if myid == 0:
            dprint1("NE", mesh1.GetNE(), mesh2.GetNE())
    elmap, elmap_r = map_element(mesh1, mesh2, bdr, map_bdr=True)

    sdim1 = mesh1.SpaceDimension()
    sdim2 = mesh1.SpaceDimension()
    dim1 = mesh1.Dimension()
    dim2 = mesh2.Dimension()

    shape1 = mfem.DenseMatrix()
    shape2 = mfem.Vector()
    ip = mfem.IntegrationPoint()
    nor = mfem.Vector(sdim1)

    if USE_PARALLEL:
        # this is global TrueDoF (offset is not subtracted)
        P = fes1.Dof_TrueDof_Matrix()
        P1mat = ToScipyCoo(P).tocsr()
        #VDoFtoGTDoF1 = P.indices
        #P = fes2.Dof_TrueDof_Matrix()
        #P = ToScipyCoo(P).tocsr()
        #VDoFtoGTDoF2 = P.indices
        #P2mat = P

    vdofs1_senddata = []

    shared_data = []

    el2_2_node = {}
    el2_2_el1 = {}

    for d in elmap_r:
        for x in list(elmap_r[d]):
            el2_2_node[x] = d
        for x in list(elmap_r[d]):
            el2_2_el1[x] = elmap_r[d][x]

    # working for fes2
    # find boundary element on mesh1 using mesh2 boundary
    el2_arr = [list() for x in range(nprc)]
    el1_arr = [list() for x in range(nprc)]
    fe2o_arr = [list() for x in range(nprc)]
    for i_el in range(fes2.GetNE()):
        attr = fes2.GetAttribute(i_el)
        if bdr != 'all' and not attr in bdr:
            continue
        el1_arr[el2_2_node[i_el]].append(el2_2_el1[i_el])
        el2_arr[el2_2_node[i_el]].append(i_el)
        fe2 = fes2.GetFE(i_el)
        fe2o_arr[el2_2_node[i_el]].append(fe2.GetOrder())

    if USE_PARALLEL:
        el1_arr = alltoall_vector(el1_arr, int)  # transfer to mesh1 owners

    # working for fes1
    # find elemet order on mesh1
    fe1o_arr = [list() for x in range(nprc)]
    i_fe1_arr = [list() for x in range(nprc)]
    rank = 0
    for rank, i_bdrs in enumerate(el1_arr):
        for i_bdr in i_bdrs:
            iface = mesh1.GetBdrElementEdgeIndex(i_bdr)
            transs = mesh1.GetFaceElementTransformations(iface)
            i_el1 = transs.Elem1No
            assert transs.Elem2No == -1, "boundary must be exterior for this operator"
            fe1 = fes1.GetFE(i_el1)
            fe1o_arr[rank].append(fe1.GetOrder())
            i_fe1_arr[rank].append(i_el1)
        rank = rank + 1

    if USE_PARALLEL:
        fe1o_arr = alltoall_vector(fe1o_arr, int)  # transfer to mesh2

    # working for fes2
    locnor_arr = [list() for x in range(nprc)]
    data2_arr = [list() for x in range(nprc)]
    verbose1 = verbose
    for rank, i_el2s in enumerate(el2_arr):
        for i_el2, fe1o in zip(i_el2s, fe1o_arr[rank]):
            eltrans = fes2.GetElementTransformation(i_el2)
            fe2 = fes2.GetFE(i_el2)
            nd2 = fe2.GetDof()
            ir = get_rule(
                fe1o,
                fe2,
                eltrans,
                orderinc=orderinc,
                verbose=verbose1)
            verbose1 = False
            shape2.SetSize(nd2)
            data2 = []
            locnors = []
            for jj in range(ir.GetNPoints()):
                ip1 = ir.IntPoint(jj)
                eltrans.SetIntPoint(ip1)
                w = eltrans.Weight() * ip1.weight
                mfem.CalcOrtho(eltrans.Jacobian(), nor)
                nor2 = nor.GetDataArray() / np.linalg.norm(nor.GetDataArray())
                fe2.CalcShape(ip1, shape2)

                if dim2 == 1:
                    d = np.array([ip1.x] + list(eltrans.Transform(ip1))
                                         + list(nor2))
                    locnors.append(d)
                elif dim2 == 2:
                    d = np.array([ip1.x, ip1.y] + list(eltrans.Transform(ip1))
                                                + list(nor2))

                    locnors.append(d)
                else:
                    assert False, "boundary mesh must be dim=1 or 2"
                data2.append(w * shape2.GetDataArray().copy())

            #   np.vstack(locnors).shape = (#Npoints, dim2+sdim2*2)
            #   np.vstack(data2).shape = (#Npoints, #NDoF2)
            #print("size here", np.vstack(locnors).shape, np.vstack(data2).shape)

            locnor_arr[rank].append(np.vstack(locnors))
            data2_arr[rank].append(np.vstack(data2))

    if USE_PARALLEL:
        locnor_arr = alltoall_vectorv(locnor_arr, float)  # transfer to mesh1

    ll = dim2 + 2 * sdim2

    vdofs1_arr = [list() for x in range(nprc)]
    data1_arr = [list() for x in range(nprc)]

    # space to compute the coefficient
    MV = [mfem.Vector(sdim1),
          mfem.DenseMatrix(sdim1, sdim1)]

    max_misalignment = -np.inf
    for rank, i_fe1s in enumerate(i_fe1_arr):
        locnorss = locnor_arr[rank]

        sign_dict = {}

        for k, i_fe1 in enumerate(i_fe1s):
            fe1 = fes1.GetFE(i_fe1)
            nd1 = fe1.GetDof()
            eltrans = fes1.GetElementTransformation(i_fe1)
            doftrans = fes1.GetElementDofTransformation(i_fe1)
            #ctr = eval_element_center(fe1, eltrans)

            locnors2 = locnorss[k]
            shape1.SetSize(nd1, sdim1)
            vdofs1 = fes1.GetElementVDofs(i_fe1)

            dof_sign1 = np.array([[1 if vv >= 0 else -1
                                   for vv in vdofs1], ])
            vdofs1 = [-1 - x if x < 0 else x for x in vdofs1]

            mat_doftrans = get_inv_doftrans(doftrans, dof_sign1)

            if USE_PARALLEL:
                #  After DofTransformation is introduced we can not use GetGlobalTDofNumber, because
                #  element local DoF could be linked with two TrueDoFs in neighber processes
                #  We construct submatrix of Prolongation to construct element matrix
                #  in TrueDof space

                vv1 = [P1mat.indices[P1mat.indptr[ii]:P1mat.indptr[ii+1]]
                       for ii in vdofs1]
                vv3 = [P1mat.data[P1mat.indptr[ii]:P1mat.indptr[ii+1]]
                       for ii in vdofs1]
                ngtof = np.sum([len(x) for x in vv3])
                sub_p = np.zeros((nd1, ngtof))
                k1 = 0
                k2 = 0
                for gtofs, weights in zip(vv1, vv3):
                    for g, w in zip(gtofs, weights):
                        sub_p[k1, k2] = w
                        k2 = k2 + 1
                    k1 = k1 + 1

                vdofs1 = np.hstack(vv1).flatten()
                mat_doftrans = mat_doftrans.dot(sub_p)

            res, misalignment = map_ir(fe1, eltrans, coeff,
                                       shape1, dim2, sdim2,
                                       locnors2, dof_sign1,
                                       mat_doftrans, MV)

            vdofs1_arr[rank].append(np.array(vdofs1))
            data1_arr[rank].append(res)

            max_misalignment = np.max([max_misalignment, np.max(misalignment)])
            # res.shape = (#Npoints, #DoF1)

    if USE_PARALLEL:
        vdofs1_arr = alltoall_vectorv(vdofs1_arr, int)  # transfer to mesh2
        if is_complex:
            data1_arr = alltoall_vectorv(
                data1_arr, complex)  # transfer to mesh2
        else:
            data1_arr = alltoall_vectorv(data1_arr, float)  # transfer to mesh2
        max_misalignment = np.max(
            MPI.COMM_WORLD.gather(max_misalignment, root=0))
    dprint1("Max misalignment: ", max_misalignment)

    shared_data = []

    for rank, i_el2s in enumerate(el2_arr):
        for k, i_el2 in enumerate(i_el2s):
            vdofs1 = vdofs1_arr[rank][k]

            fe2 = fes2.GetFE(i_el2)
            eltrans2 = fes2.GetElementTransformation(i_el2)
            vdofs2 = fes2.GetElementVDofs(i_el2)
            vdofs2 = [-1 - x if x < 0 else x for x in vdofs2]

            d1 = data1_arr[rank][k]
            d2 = data2_arr[rank][k]

            mm = d2.transpose().dot(d1)

            if USE_PARALLEL:
                # prepare data for not-owned DoFs, which will be shared later
                vdofs22 = [fes2.GetLocalTDofNumber(ii) for ii in vdofs2]
                vdofs22g = [fes2.GetGlobalTDofNumber(ii) for ii in vdofs2]

                kkk = 0
                for v2, v2g in zip(vdofs22, vdofs22g):
                    if v2 < 0:
                        shared_data.append([v2g, mm[kkk, :], vdofs1])
                    kkk = kkk + 1
            else:
                vdofs22 = vdofs2

            for i, ltdof2 in enumerate(vdofs22):
                if ltdof2 < 0:
                    continue
                for j, gtdof1 in enumerate(vdofs1):
                    mat[ltdof2, gtdof1] = mat[ltdof2, gtdof1] + mm[i, j]

    if USE_PARALLEL:
        #nicePrint("shared data", shared_data)
        for source_id in range(nprc):
            data = comm.bcast(shared_data, root=source_id)
            myoffset = fes2.GetMyTDofOffset()
            for v2g, elmat, vdofs1 in data:
                if v2g >= myoffset and v2g < myoffset + mat.shape[0]:
                    i = v2g - myoffset
                    for j, gtdof1 in enumerate(vdofs1):
                        mat[i, gtdof1] = mat[i, gtdof1] + elmat[j]
                    #mat[i, vdofs1] = mat[i, vdofs1] + elmat

    from scipy.sparse import coo_matrix, csr_matrix

    if USE_PARALLEL:
        if is_complex:
            m1 = csr_matrix(mat.real, dtype=float)
            m2 = csr_matrix(mat.imag, dtype=float)
        else:
            m1 = csr_matrix(mat.real, dtype=float)
            m2 = None
        from mfem.common.chypre import CHypreMat

        start_col = fes1.GetMyTDofOffset()
        end_col = fes1.GetMyTDofOffset() + fes1.GetTrueVSize()
        col_starts = [start_col, end_col, mat.shape[1]]
        M = CHypreMat(m1, m2, col_starts=col_starts)
    else:
        from petram.helper.block_matrix import convert_to_ScipyCoo
        M = convert_to_ScipyCoo(coo_matrix(mat, dtype=mat.dtype))

    return M
