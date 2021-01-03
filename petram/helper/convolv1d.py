import numpy as np
from scipy.sparse import lil_matrix
import itertools
from collections import defaultdict, OrderedDict
from mfem.common.mpi_debug import nicePrint
from mfem.common.parcsr_extra import ToScipyCoo

from petram.helper.dof_map import get_empty_map

from petram.mfem_config import use_parallel
if use_parallel:
    USE_PARALLEL = True
    import mfem.par as mfem
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    myid = comm.rank
    nprc = comm.size
else:
    USE_PARALLEL = False    
    import mfem.ser as mfem
    myid = 0
    nprc = 1
    mfem.ParFiniteElementSpace = type(None)
    mfem.ParGridfunction = type(None)
    mfem.ParMesh = type(None)
    mfem.ParMixedBilinearForm = type(None)

def delta(x, w=None):
    '''
    delta function like coefficient.
    has to return 1/w instead of 1.
    (this is for debugging)
    '''
    #return 1.0
    if np.abs(x) < 1e-5:
        return 1/w
    return 0.0

def get_rule(fe1, fe2, trans, orderinc):
    order = fe1.GetOrder() + fe2.GetOrder() + trans.OrderW() + orderinc
    if (fe1.Space() == mfem.FunctionSpace.rQk):
        assert False, "not supported"
    ir = mfem.IntRules.Get(fe1.GetGeomType(), order)
    print("Order, N Points", order, ir.GetNPoints())
    return ir

def convolve1d(fes1, fes2, func=delta, orderinc=5, is_complex=False):
    '''
    fill linear operator for convolution
    \int phi_test(x) func(x-x') phi_trial(x') dx
    '''
    #smesh = mfem.Mesh(3, 1.0)
    #if USE_PARALLEL:
    #    mesh = mfem.ParMesh(comm, smesh)
    #else:
    #    mesh = smesh
    #sdim = mesh.SpaceDimension()
    #fes1 = new_fespace(mesh, fec1, vdim=1)
    #fes2 = new_fespace(mesh, fec2, vdim=1)

    mat, rstart = get_empty_map(fes2, fes1, is_complex=is_complex)

    eltrans1 = fes1.GetElementTransformation(0)
    ir = get_rule(fes1.GetFE(0), fes2.GetFE(0), eltrans1, orderinc)

    shape1 = mfem.Vector()
    shape2 = mfem.Vector()
    tmp_int = mfem.Vector()

    #nicePrint("shape", mat.shape, fes2.GetNE(), fes1.GetNE())

    # communication strategy
    #   (1) x2 (ir points on test space) is collected in each nodes
    #   (2) x2 is send to other nodes
    #   (3) each nodes compute \int f(x2-x1) phi(x1)
    #   (4) non-zero results of (3) and global index should be send back

    # Step (1, 2)
    x2_arr = []

    ptx = mfem.DenseMatrix(ir.GetNPoints(), 1)

    for i in range(fes2.GetNE()): # scan test space
        eltrans = fes2.GetElementTransformation(i)
        eltrans.Transform(ir, ptx)
        x2_arr.append(ptx.GetDataArray().copy())
    ptx_x2 = np.vstack(x2_arr)

    #nicePrint("x2 shape", ptx_x2.shape)
    if USE_PARALLEL:
        ## note: we could implement more advanced alg. to reduce
        ## the amount of data exchange..
        x2_all = comm.allgather(ptx_x2)
    else:
        x2_all = [ptx_x2]
    #nicePrint("x2_all shape", x2_all.shape)

    if USE_PARALLEL:
        P = fes1.Dof_TrueDof_Matrix()
        P = ToScipyCoo(P).tocsr()
        VDoFtoGTDoF = P.indices  #this is global TrueDoF (offset is not subtracted)

    # Step 3
    for knode1, x2_onenode in enumerate(x2_all):
        elmats_all = []
        vdofs1_all = []

        # collect vdofs
        for j in range(fes1.GetNE()):
            local_vdofs = fes1.GetElementVDofs(j)
            if USE_PARALLEL:
                subvdofs2 = [VDoFtoGTDoF[i] for i in local_vdofs]
                vdofs1_all.append(subvdofs2)
            else:
                vdofs1_all.append(local_vdofs)

        for x2s in x2_onenode: # loop over fes2
            nd2 = len(x2s)
            #nicePrint(x2s)
            elmats = []
            for j in range(fes1.GetNE()):
                # collect integration
                fe1 = fes1.GetFE(j)
                nd1 = fe1.GetDof()
                shape1.SetSize(nd1)
                eltrans = fes1.GetElementTransformation(j)

                tmp_int.SetSize(shape1.Size())
                elmat = np.zeros((nd2, nd1), dtype=float)

                #if myid == 0: print("fes1 idx", j)
                for kkk, x2 in enumerate(x2s):
                    tmp_int.Assign(0.0)
                    for jj in range(ir.GetNPoints()):
                        ip1 = ir.IntPoint(jj)
                        eltrans.SetIntPoint(ip1)
                        x1 = eltrans.Transform(ip1)[0]
                        #if myid == 0: print("check here", x1, x2)


                        fe1.CalcShape(ip1, shape1)
                        #print("shape1", shape1.GetDataArray())

                        w = eltrans.Weight() * ip1.weight

                        #val = delta(x2-x1, w)
                        val = func(x2-x1, w=w)

                        shape1 *= w*val
                        tmp_int += shape1
                    elmat[kkk, :] = tmp_int.GetDataArray()
                elmats.append(elmat)
                #print(elmats)

            elmats_all.append(elmats)

        # send this information to knodes;
        if USE_PARALLEL:
            if myid == knode1:
                vdofs1_data = comm.gather(vdofs1_all, root=knode1)
                elmats_data = comm.gather(elmats_all, root=knode1)
            else:
                _ = comm.gather(vdofs1_all, root=knode1)
                _ = comm.gather(elmats_all, root=knode1)
        else:
            vdofs1_data = [vdofs1_all,]
            elmats_data = [elmats_all,]

    # Step 4
    shared_data = []
    for vdofs1, elmats_all in zip(vdofs1_data, elmats_data): # loop over MPI nodes
        #nicePrint("len elmats", len(elmats_all))
        for i, elmats in enumerate(elmats_all):  # corresponds to loop over fes2
            #nicePrint(len(vdofs1), len(elmats))
            vdofs2 = fes2.GetElementVDofs(i)
            fe2 = fes2.GetFE(i)
            nd2 = fe2.GetDof()
            shape2.SetSize(nd2)

            eltrans = fes2.GetElementTransformation(i)
            for j, elmat in enumerate(elmats):
                #print(vdofs1[j], elmat.shape)
                mm = np.zeros((len(vdofs2), len(vdofs1[j])), dtype=float)

                for ii in range(ir.GetNPoints()):
                    ip2 = ir.IntPoint(ii)
                    eltrans.SetIntPoint(ip2)
                    ww = eltrans.Weight() * ip2.weight
                    fe2.CalcShape(ip2, shape2)
                    shape2 *= ww

                    tmp_int = elmat[ii, :]
                    tmp = np.dot(np.atleast_2d(shape2.GetDataArray()).transpose(),
                                 np.atleast_2d(tmp_int))
                    mm = mm + tmp
                    #print("check here", myid, mm.shape, tmp.shape)

                # merge contribution to final mat
                if USE_PARALLEL:
                    vdofs22 = [fes2.GetLocalTDofNumber(i) for i in vdofs2]
                    vdofs22g = [VDoFtoGTDoF[i] for i in vdofs2]
                    kkk = 0
                    for v2, v2g in zip(vdofs22, vdofs22g):
                        if v2 < 0:
                            shared_data.append([v2g, mm[kkk, :], vdofs1[j]])
                        kkk = kkk + 1

                for k, vv in enumerate(vdofs1[j]):
                    try:
                        if USE_PARALLEL:
                            mmm = mm[np.where(np.array(vdofs22) >= 0)[0], :]                            
                            vdofs222 = [x for x in vdofs22 if x >= 0]
                        #if myid == 1:
                        #    print("check here", vdofs2, vdofs22, vdofs222)
                        #print(mmm[:, [k]])
                        tmp = mat[vdofs222, vv] + mmm[:, [k]]
                        mat[vdofs222, vv] = tmp.flatten()
                    except:
                        import traceback
                        print("error", myid)
                        print(vdofs1, vdofs22, vdofs222, mmm.shape, k)
                        traceback.print_exc()

    for source_id in range(nprc):
        data = comm.bcast(shared_data, root=source_id)
        myoffset = fes2.GetMyTDofOffset()
        for v2g, elmat, vdofs1 in data:
            if v2g >= myoffset and v2g < myoffset + mat.shape[0]:
                i = v2g - myoffset
                #print("procesising this", myid, i, v2g, elmat, vdofs1)                
                mat[i, vdofs1] = mat[i, vdofs1] + elmat

    return mat
