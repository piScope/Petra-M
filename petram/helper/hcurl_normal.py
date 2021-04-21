import numpy as np
from scipy.sparse import lil_matrix
import itertools
from collections import defaultdict, OrderedDict
from mfem.common.mpi_debug import nicePrint
from mfem.common.parcsr_extra import ToScipyCoo

from petram.helper.dof_map import get_empty_map

import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('convolve1d')

from petram.mfem_config import use_parallel
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
    def alltoall_vector(x):
        return np.array(x)
    
rules = {}

def get_rule(fe1o, fe2, trans, orderinc=1, verbose=False):
    fe2o = fe2.GetOrder()
    
    order = fe1o + fe2o + trans.OrderW() + orderinc
    if (fe2.Space() == mfem.FunctionSpace.rQk):
        assert False, "not supported"

    if not (fe2.GetGeomType(), order) in rules:
        ir = mfem.IntRules.Get(fe2.GetGeomType(), order)
        rules[(fe2.GetGeomType(), order)] = ir
        
    ir = rules[(fe2.GetGeomType(), order)]
    if verbose:
        dprint1("Order, N Points", order, ir.GetNPoints())
    return ir

def map_ir(fe1, bfe1, trans, ptxs, th = 1e-7):
    g1 = fe1.GetGeomType()
    bg1 = bfe1.GetGeomType()

    ip = mfem.IntegrationPoint()

    res = []
    dim = -1
    if g1==2 and bg1==1:   #triangle <- line
        dim = 2
        for pp, p, w, nor in ptxs:
            print("target", p)
            options = ((pp.x, 0),
                       (1 - pp.x, 0),
                       (0, pp.x),
                       (0, 1 - pp.x),
                       (pp.x, 1 - pp.x),
                       (1- pp.x, pp.x),)
            dd = []
            for o in options:
                ip.Set2(*o)
                dd.append(np.sum((p - trans.Transform(ip))**2))
            res.append((options[np.argmin(dd)], w, nor))
            print(np.min(dd))
            assert np.min(dd) < np.max(dd)*th, "point not found"
    return dim, res
        
def hcurln(fes1, fes2, coeff,
           is_complex=False, bdr='all',
           verbose=False):
    
    mat, rstart = get_empty_map(fes2, fes1, is_complex=is_complex)

    from petram.helper.element_map import map_element
    
    name_fes1 = fes1.FEColl().Name()[:2]
    name_fes2 = fes2.FEColl().Name()[:2]

    if verbose:
        print("fes", name_fes1, name_fes2)

    mesh1 = fes1.GetMesh()
    mesh2 = fes2.GetMesh()

    print("NE", mesh1.GetNE(),mesh2.GetNE())
    elmap, elmap_r = map_element(mesh1, mesh2, bdr, map_bdr=True)

    nicePrint(elmap, elmap_r)
    
    sdim1 = mesh1.SpaceDimension()
    dim1 = mesh1.Dimension()
    dim2 = mesh2.Dimension()    
    
    shape1 = mfem.DenseMatrix()
    shape2 = mfem.Vector()
    ip = mfem.IntegrationPoint()
    nor = mfem.Vector(sdim1)
    
    if USE_PARALLEL:
        #this is global TrueDoF (offset is not subtracted)        
        P = fes1.Dof_TrueDof_Matrix()
        P = ToScipyCoo(P).tocsr()
        VDoFtoGTDoF1 = P.indices  
        P = fes2.Dof_TrueDof_Matrix()
        P = ToScipyCoo(P).tocsr()
        VDoFtoGTDoF2 = P.indices
        
    vdofs1_senddata = []

    shared_data = []

    el2_2_node = {}
    el2_2_el1 = {} 
    for d in elmap_r:
       for x in list(elmap_r[d]): el2_2_nodes[x] = d
       for x in list(elmap_r[d]): el2_2_el1[x] = elmap_r[d][x]

    ### working for fes2           
    ## find boundary element on mesh1 using mesh2 boundary
    el2_arr = [list() for x in range(nproc)]    
    el1_arr = [list() for x in range(nproc)]
    fe2o_arr = [list() for x in range(nproc)]    
    for i_el in range(fes2.GetNE()):
        attr = fes2.GetAttribute(i_el)
        if bdr != 'all' and not attr in bdr:
            continue
        el1_arr[el2_2_node[i_el]].append(el2_2_el1[i_el])
        el2_arr[el2_2_node[i_el]].append(i_el)
        fe2 = fes2.GetFE(i_el)        
        fe2o_arr[el2_2_node[i_el]].append(fe2.GetOrder())
        
    el1_arr = alltoall_vector(el1_arr) # transfer to mesh1 owners

    ### working for fes1    
    ## find elemet order on mesh1    
    fe1o_arr = [list() for x in range(nproc)]
    i_fe1_arr = [list() for x in range(nproc)]
    rank = 0
    for rank, i_bdrs in enumerate(el1_arr):
        for i_bdr in i_bdrs:
            iface = mesh1.GetBdrElementEdgeIndex(i_bdr)
            transs = mesh1.GetFaceElementTransformations(iface)
            i_el1 = transs.Elem1No
            i_el2 = transs.Elem2No
            assert i_el2 == -1, "boundary must be exterior for this operator"            
            fe1 = fes1.GetFE(i_el1)
            fe1o_arr[rank].append(fe1.GetOrder())
            i_fe1_arr[rank].append(i_el1)
        rank = rank+1
    fe1o_arr =  alltoall_vector(fe1o_arr) # transfer to mesh2

    ### working for fes2
    loc_arr = [list() for x in range(nproc)]
    nir_arr = [list() for x in range(nproc)]
    for rank, i_el2s in enumerate(el2_arr):
        for i_el2, fe1o in i_el2s, fe1o_arr[rank]:
            eltrans = fes2.GetElementTransformation(i_el2)            
            fe2 = fes2.GetFE(i_el2)
            ir = get_rule(fe1o, fe2, eltrans, verbose=verbose)

            ptxs = []
            for jj in range(ir.GetNPoints()):
                ip1 = ir.IntPoint(jj)
                eltrans.SetIntPoint(ip1)
                w = eltrans.Weight() * ip1.weight
                mfem.CalcOrtho(eltrans.Jacobian(), nor)
                fe2.CalcShape(ip1, shape2)
                if dim2 == 1:
                    loc_arr[rank].extend([ip1.x]+list(eltrans.Transform(ip1)))
                elif dim2 == 2:
                    loc_arr[rank].extend([ip1.x, ip1.y]+list(eltrans.Transform(ip1)))
                else:
                    assert False, "boundary mesh must be dim=1 or 2"
                ptxs[rank].append((np.atleast_2d(w*shape2.GetDataArray().copy()).transpose(),
                             nor.GetDataArray().copy()))
            nir[rank].append(ir.GetNPoints())
            
    loc_arr =  alltoall_vector(loc_arr) # transfer to mesh1

    L = 4 if dim2 == 2 else 3
    
    for rank, i_fe1s in enumerate(i_fe1_arr):
        locs = loc_arr[rank]
        locs = np.array(locs).reshape(-1 ,L)
        
        

        
    fe1o_arr = alltoall_vector(fe1o_arr) # transfer to mesh2 owners

    i = 0
    for i_els, fe1os in zip(el2_arr, fe1o_arr):
        for i_el, fe1o in zip(i_els, fe1os):
            fe2o = fes2.GetFE(i_el)
            eltrans = fes1.GetElementTransformation(i_el)        

            
def get_rule(fe1o, fe2o, trans, orderinc=1, verbose=False):    
    
        fe1os.append(fe1o)
        if USE_PARALLEL:            
            data = comm.scatter(fe1os, root=i)
        else:
            data = fe1os[0]
        fe1o_arr[i] = data
            
        fe2 = fes2.GetFE(i_bdr)
        nd1 = fe1.GetDof()
        nd2 = fe2.GetDof()
            
    if USE_PARALLEL:
            
        fe2 = fes2.GetFE(i_el)
        el2_order 
        eltrans = fes1.GetBdrElementTransformation(i_bdr)
        ir = get_rule(fe1, fe2, eltrans, verbose=verbose)

        transs = mesh1.GetFaceElementTransformations(iface)
        el1 = transs.Elem1No
        el2 = transs.Elem2No
        assert el2 == -1, "boundary must be exterior for this operator"
        print(iface, el1)

        fe1 = fes1.GetFE(el1)
        
        fe2 = fes2.GetFE(i_bdr)
        nd1 = fe1.GetDof()
        nd2 = fe2.GetDof()
        
        eltrans = fes1.GetBdrElementTransformation(i_bdr)
        ir = get_rule(fe1, fe2, eltrans, verbose=verbose)

        shape2.SetSize(nd2)
        
        ptxs = []
        for jj in range(ir.GetNPoints()):
             ip1 = ir.IntPoint(jj)
             eltrans.SetIntPoint(ip1)
             w = eltrans.Weight() * ip1.weight
             mfem.CalcOrtho(eltrans.Jacobian(), nor)
             fe2.CalcShape(ip1, shape2)
             ptxs.append((ip1,
                          eltrans.Transform(ip1),
                          np.atleast_2d(w*shape2.GetDataArray().copy()).transpose(),
                          nor.GetDataArray().copy()))

        eltrans = fes1.GetElementTransformation(el1)
        dim, irptx = map_ir(fe1, fe2, eltrans, ptxs)
                         
        shape1.SetSize(nd1, sdim1)

        print('nd', nd1, nd2)
        vdofs1 = fes1.GetElementVDofs(el1)                         
        vdofs2 = fes2.GetElementVDofs(i_bdr)
        dof_sign1 = np.array([[1 if vv >= 0 else -1
                                   for vv in vdofs1],])
        vdofs1 = [-1-x if x < 0 else x for x in vdofs1]
        print("before", vdofs1)
        if USE_PARALLEL:
            vdofs1 = [VDoFtoGTDoF1[i] for i in vdofs1]
            print("after", vdofs1)
        vdofs2 = [-1-x if x < 0 else x for x in vdofs2]

        mm = None
        for ptx, w, nor2 in irptx:
            if dim == 2:
                ip.Set2(*ptx)
            elif dim == 3:
                ip.Set3(*ptx)
            eltrans.SetIntPoint(ip)
            fe1.CalcVShape(eltrans, shape1)
            ww = eltrans.Weight()
            #print(w.shape, coeff.shape, shape1.GetDataArray().shape, nor2.shape)

            tmp = nor2.dot(coeff.dot(shape1.GetDataArray().transpose()))
            print(tmp.shape, dof_sign1.shape, ww)
            tmp = np.atleast_2d(tmp*dof_sign1)
            
            if mm is None:
                mm = w.dot(tmp)#/ww
            else:
                mm += w.dot(tmp)#/ww
            
                         
        if USE_PARALLEL:
             vdofs22 = [fes2.GetLocalTDofNumber(ii) for ii in vdofs2]
             vdofs22g = [VDoFtoGTDoF2[ii] for ii in vdofs2]
             kkk = 0
             #for v2, v2g in zip(vdofs22, vdofs22g):
             for v2, v2g in zip(vdofs2, vdofs22g):
                 if v2 < 0:
                     shared_data.append([v2g, mm[kkk, :], vdofs1])
                 kkk = kkk + 1

        for k, vv in enumerate(vdofs1):
             if USE_PARALLEL:            
                 mmm = mm[np.where(np.array(vdofs22) >= 0)[0], :]                            
                 vdofs222 = [x for x in vdofs22 if x >= 0]
             else:
                 vdofs222 = vdofs2
                 mmm = mm
             print(vdofs222, vv)
             print("dest", mat[vdofs222, vv])
             print("data", mmm[:, [k]])
             tmp = mat[vdofs222, vv] + mmm[:, [k]]             
             mat[vdofs222, vv] = tmp.flatten()

    if USE_PARALLEL:
        for source_id in range(nprc):
            data = comm.bcast(shared_data, root=source_id)
            myoffset = fes2.GetMyTDofOffset()
            for v2g, elmat, vdofs1 in data:
                if v2g >= myoffset and v2g < myoffset + mat.shape[0]:
                    i = v2g - myoffset
                    mat[i, vdofs1] = mat[i, vdofs1] + elmat
             

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

