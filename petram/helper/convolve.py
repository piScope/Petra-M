import numpy as np
from scipy.sparse import lil_matrix
import itertools
from collections import defaultdict, OrderedDict
from mfem.common.mpi_debug import nicePrint
from mfem.common.parcsr_extra import ToScipyCoo

from petram.helper.dof_map import get_empty_map

import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('convolve')

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

def delta(x, _x0, w=None):
    '''
    delta function like coefficient.
    has to return 1/w instead of 1.
    (this is for debugging)
    '''
    #return 1.0
    if np.sqrt(np.sum(x**2)) < 1e-5:
        return 1/w
    return 0.0

def zero(x):
    '''
    sample support funciton. 0.0 means that the contribution is non-zero
    only when x1 and x2 are the same location
    '''
    return 0.0

def get_rule(fe1, fe2, trans, orderinc, verbose):
    order = fe1.GetOrder() + fe2.GetOrder() + trans.OrderW() + orderinc
    if (fe1.Space() == mfem.FunctionSpace.rQk):
        assert False, "not supported"
    ir = mfem.IntRules.Get(fe1.GetGeomType(), order)
    
    if verbose:
        dprint1("Order, N Points", order, ir.GetNPoints())
    return ir

def profile_start():
    '''
    profiling start
    usage:
       pr = profile_start()
       ... do something
       profile_stop(pr)
    '''
    import cProfile
    print('starting profiler')
    pr = cProfile.Profile()
    pr.enable()
    return pr


def profile_stop(pr, sortby='cumulative'):
    '''
    profile_stop(pr, sortby='cumulative'):

    end profile
    sortby = 'cumulative', 'calls', 'cumtime', 
             'file', 'filename', 'module',
             'ncalls', pcalls', 'line', 'name',
             'nfl', stdname', 'time', 'tottime'
    '''
    from six import StringIO
    import pstats
    pr.disable()
    # print 'stopped profiler'
    lsortby = ['cumulative', 'calls', 'cumtime',
               'file', 'filename', 'module',
               'ncalls', 'pcalls', 'line', 'name',
               'nfl', 'stdname', 'time', 'tottime']
    if not sortby in lsortby:
        print('invalid sortby')
        print(lsortby)
        return

    s = StringIO()
    sortby = sortby
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print((s.getvalue()))

def convolve1d(fes1, fes2, kernel=delta, support=None,
               orderinc=5, is_complex=False,
               trial_domain='all',
               test_domain='all',
               verbose=False, coeff=None):
    '''
    fill linear operator for convolution
    \int phi_test(x) func(x-x') phi_trial(x') dx
    '''
    mat, rstart = get_empty_map(fes2, fes1, is_complex=is_complex)

    eltrans1 = fes1.GetElementTransformation(0)
    ir = get_rule(fes1.GetFE(0), fes2.GetFE(0), eltrans1, orderinc, verbose)

    shape1 = mfem.Vector()
    shape2 = mfem.Vector()

    #nicePrint("shape", mat.shape, fes2.GetNE(), fes1.GetNE())

    # communication strategy
    #   (1) x2 (ir points on test space) is collected in each nodes
    #   (2) x2 is send to other nodes
    #   (3) each nodes compute \int f(x2-x1) phi(x1)
    #   (4) non-zero results of (3) and global index should be send back

    # Step (1, 2)
    if verbose:
        dprint1("Step 1,2")    
    x2_arr = []
    i2_arr = []

    ptx = mfem.DenseMatrix(ir.GetNPoints(), 1)
    
    attrs1 = fes2.GetMesh().GetAttributeArray()
    attrs2 = fes2.GetMesh().GetAttributeArray()
    
    for i in range(fes2.GetNE()): # scan test space
        if test_domain != 'all':
            if not attrs1[i] in test_domain: continue
        eltrans = fes2.GetElementTransformation(i)
        eltrans.Transform(ir, ptx)
        x2_arr.append(ptx.GetDataArray().copy())
        i2_arr.append(i)
    if len(i2_arr) > 0:
       ptx_x2 = np.vstack(x2_arr)
       i2_arr = np.hstack(i2_arr)
    else:
       ptx_x2 = np.array([[]])
       i2_arr = np.array([])

    #nicePrint("x2 shape", ptx_x2.shape)
    if USE_PARALLEL:
        ## note: we could implement more advanced alg. to reduce
        ## the amount of data exchange..
        x2_all = comm.allgather(ptx_x2)
        i2_all = comm.allgather(i2_arr)
    else:
        x2_all = [ptx_x2]
        i2_all = [i2_arr]
    #nicePrint("x2_all shape", x2_all.shape)

    if USE_PARALLEL:
        #this is global TrueDoF (offset is not subtracted)        
        P = fes1.Dof_TrueDof_Matrix()
        P = ToScipyCoo(P).tocsr()
        VDoFtoGTDoF1 = P.indices  
        P = fes2.Dof_TrueDof_Matrix()
        P = ToScipyCoo(P).tocsr()
        VDoFtoGTDoF2 = P.indices
        
    # Step 3
    if verbose:
        dprint1("Step 3")
    vdofs1_senddata = []
    elmats_senddata = []
        
    for knode1 in range(len(x2_all)):
        x2_onenode = x2_all[knode1]
        i2_onenode = i2_all[knode1]
        elmats_all = []
        vdofs1_all = []

        # collect vdofs
        for j in range(fes1.GetNE()):
            local_vdofs = fes1.GetElementVDofs(j)
            if USE_PARALLEL:
                subvdofs2 = [VDoFtoGTDoF1[i] for i in local_vdofs]
                vdofs1_all.append(subvdofs2)
            else:
                vdofs1_all.append(local_vdofs)

        for i, x2s in zip(i2_onenode, x2_onenode): # loop over fes2
            nd2 = len(x2s)
            #nicePrint(x2s)
            elmats = []
            for j in range(fes1.GetNE()):
                if trial_domain != 'all':
                    if not attrs1[j] in trial_domain: continue
                
                # collect integration
                fe1 = fes1.GetFE(j)
                nd1 = fe1.GetDof()
                shape1.SetSize(nd1)
                eltrans = fes1.GetElementTransformation(j)

                tmp_int = np.zeros(shape1.Size(), dtype=mat.dtype)
                elmat = np.zeros((nd2, nd1), dtype=mat.dtype)

                #if myid == 0: print("fes1 idx", j)

                dataset = []
                for jj in range(ir.GetNPoints()):
                    ip1 = ir.IntPoint(jj)
                    eltrans.SetIntPoint(ip1)
                    x1 = eltrans.Transform(ip1)[0]
                    fe1.CalcShape(ip1, shape1)
                    w = eltrans.Weight() * ip1.weight
                    dataset.append((x1, w, shape1.GetDataArray().copy()))
                    
                has_contribution = False
                for kkk, x2 in enumerate(x2s):
                    tmp_int *= 0.0
                    
                    for x1, w, shape_arr in dataset:
                        if support is not None:
                            s = support((x1 + x2)/2.0)
                            if np.abs(x1-x2) > s:
                                continue

                        has_contribution = True
                        #if myid == 0: print("check here", x1, x2)
                        val = kernel(x2-x1, (x2+x1)/2.0, w=w)
                        if coeff is not None:
                            val = val* coeff((x2+x1)/2.0)

                        #shape_arr *= w*val
                        tmp_int += shape_arr*w*val
                    elmat[kkk, :] = tmp_int

                if has_contribution:
                    elmats.append((j, elmat))
                #print(elmats)
            if len(elmats) > 0:
                elmats_all.append((i, elmats))
                
        vdofs1_senddata.append(vdofs1_all)
        elmats_senddata.append(elmats_all)
        
        # send this information to knodes;
        '''
        if USE_PARALLEL:
            #nicePrint(vdofs1_all)
            #nicePrint("elmats", [len(x) for x in elmats_all])
            if myid == knode1:
                vdofs1_data = comm.gather(vdofs1_all, root=knode1)
                elmats_data = comm.gather(elmats_all, root=knode1)
            else:
                _ = comm.gather(vdofs1_all, root=knode1)
                _ = comm.gather(elmats_all, root=knode1)
        else:
            vdofs1_data = [vdofs1_all,]
            elmats_data = [elmats_all,]
        '''
    if USE_PARALLEL:
        knode1 = 0
        for vdofs1_all, elmats_all in zip(vdofs1_senddata, elmats_senddata):
            if myid == knode1:
                vdofs1_data = comm.gather(vdofs1_all, root=knode1)
                elmats_data = comm.gather(elmats_all, root=knode1)
            else:
                _ = comm.gather(vdofs1_all, root=knode1)
                _ = comm.gather(elmats_all, root=knode1)
            knode1 = knode1 + 1
    else:
        vdofs1_data = vdofs1_senddata
        elmats_data = elmats_senddata
        
    # Step 4
    if verbose:
        dprint1("Step 4")        
    shared_data = []
    mpi_rank = 0
    for vdofs1, elmats_all in zip(vdofs1_data, elmats_data): # loop over MPI nodes
        #nicePrint("len elmats", len(elmats_all))
        #for i, elmats in enumerate(elmats_all):  # corresponds to loop over fes2
        
        if verbose:
            coupling = [len(elmats) for i, elmats in elmats_all]
            nicePrint("Element coupling for rank", mpi_rank)
            nicePrint("   Average :", (0 if len(coupling)==0 else np.mean(coupling)))
            nicePrint("   Max/Min :", (0 if len(coupling)==0 else np.max(coupling)),
                                      (0 if len(coupling)==0 else np.min(coupling)))
            mpi_rank += 1
                
        for i, elmats in elmats_all:  # corresponds to loop over fes2
            vdofs2 = fes2.GetElementVDofs(i)
            fe2 = fes2.GetFE(i)
            nd2 = fe2.GetDof()
            shape2.SetSize(nd2)

            eltrans = fes2.GetElementTransformation(i)

            #for j, elmat in enumerate(elmats):                
            for j, elmat in elmats:
                #print(vdofs1[j], elmat.shape)
                #if elmat is None:
                #    continue
                
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
                    vdofs22 = [fes2.GetLocalTDofNumber(ii) for ii in vdofs2]
                    vdofs22g = [VDoFtoGTDoF2[ii] for ii in vdofs2]
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
                        else:
                            vdofs222 = vdofs2
                            mmm = mm
                        #if myid == 1:
                        #    print("check here", vdofs2, vdofs22, vdofs222)
                        #print(mmm[:, [k]])
                        tmp = mat[vdofs222, vv] + mmm[:, [k]]
                        mat[vdofs222, vv] = tmp.flatten()
                    except:
                        import traceback
                        print("error", myid)
                        #print(vdofs1, vdofs22, vdofs222, mmm.shape, k)
                        traceback.print_exc()

    if USE_PARALLEL:
        for source_id in range(nprc):
            data = comm.bcast(shared_data, root=source_id)
            myoffset = fes2.GetMyTDofOffset()
            for v2g, elmat, vdofs1 in data:
                if v2g >= myoffset and v2g < myoffset + mat.shape[0]:
                    i = v2g - myoffset
                    #print("procesising this", myid, i, v2g, elmat, vdofs1)                
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
        #print("mat", M)
    else:
        from petram.helper.block_matrix import convert_to_ScipyCoo

        M = convert_to_ScipyCoo(coo_matrix(mat, dtype=mat.dtype))
                
    return M


def convolve2d(fes1, fes2, kernel=delta, support=None,
               orderinc=5, is_complex=False,
               trial_domain='all',
               test_domain='all',
               verbose=False, coeff=None):
    '''
    fill linear operator for convolution
    \int phi_test(x) func(x-x') phi_trial(x') dx
    
    Genralized version to multi-dim
    test/trial
        ScalarFE, ScalarFE   : func is scalar
        VectorFE, ScalarFE   : func is vector (vertical)
        ScalarFE, VectorFE   : func is vector (horizontal)
        VectorFE, VectorFE   : func matrix
    '''
    mat, rstart = get_empty_map(fes2, fes1, is_complex=is_complex)

    if fes1.GetNE() == 0:
        assert False, "FESpace does not have element"
    eltrans1 = fes1.GetElementTransformation(0)
    ir = get_rule(fes1.GetFE(0), fes2.GetFE(0), eltrans1, orderinc, verbose)

    name_fes1 = fes1.FEColl().Name()[:2]
    name_fes2 = fes2.FEColl().Name()[:2]


    sdim = fes1.GetMesh().SpaceDimension()
    if name_fes1 in ['RT', 'ND']:
        shape1 = mfem.DenseMatrix()
        vdim1 = fes1.GetMesh().SpaceDimension()
    else:
        shape1 = mfem.Vector()
        vdim1 = 1
    if name_fes2 in ['RT', 'ND']:
        shape2 = mfem.DenseMatrix()
        vdim2 = fes1.GetMesh().SpaceDimension()        
    else:
        shape2 = mfem.Vector()
        vdim1 = 1        
    
    #nicePrint("shape", mat.shape, fes2.GetNE(), fes1.GetNE())

    # communication strategy
    #   (1) x2 (ir points on test space) is collected in each nodes
    #   (2) x2 is send to other nodes
    #   (3) each nodes compute \int f(x2-x1) phi(x1)
    #   (4) non-zero results of (3) and global index should be send back

    # Step (1, 2)
    if verbose:
        dprint1("Step 1,2")    
    x2_arr = []
    i2_arr = []

    ptx = mfem.DenseMatrix(ir.GetNPoints(), sdim)
    
    attrs1 = fes2.GetMesh().GetAttributeArray()
    attrs2 = fes2.GetMesh().GetAttributeArray()

    for i in range(fes2.GetNE()): # scan test space
        if test_domain != 'all':
            if not attrs1[i] in test_domain: continue
        eltrans = fes2.GetElementTransformation(i)
        eltrans.Transform(ir, ptx)
        x2_arr.append(ptx.GetDataArray().copy().transpose())
        i2_arr.append(i)
        
    if support is not None:
        supports = np.array([support(np.mean(xxx, 0))  for xxx in  x2_arr])
    else:
        supports = -np.ones(len(x2_arr))

    if len(i2_arr) > 0:
       ptx_x2 = np.stack(x2_arr)
       i2_arr = np.hstack(i2_arr)
    else:
       ptx_x2 = np.array([[[]]])
       i2_arr = np.array([])

    #nicePrint("x2 shape", ptx_x2.shape)
    if USE_PARALLEL:
        ## note: we could implement more advanced alg. to reduce
        ## the amount of data exchange..
        x2_all = comm.allgather(ptx_x2)
        i2_all = comm.allgather(i2_arr)
        s_all = comm.allgather(supports)
    else:
        x2_all = [ptx_x2]
        i2_all = [i2_arr]
        s_all = [supports]
    #nicePrint("x2_all shape", supports.shape, len(x2_all), [tmp.shape for tmp in x2_all])

    if USE_PARALLEL:
        #this is global TrueDoF (offset is not subtracted)        
        P = fes1.Dof_TrueDof_Matrix()
        P = ToScipyCoo(P).tocsr()
        VDoFtoGTDoF1 = P.indices  
        P = fes2.Dof_TrueDof_Matrix()
        P = ToScipyCoo(P).tocsr()
        VDoFtoGTDoF2 = P.indices
        
    # Step 3
    if verbose:
        dprint1("Step 3")
    vdofs1_senddata = []
    elmats_senddata = []
        
    for knode1 in range(len(x2_all)):
        #dprint1("new knode1", myid, knode1)
            
        x2_onenode = x2_all[knode1]
        i2_onenode = i2_all[knode1]
        s_onenode = s_all[knode1]
        
        elmats_all = []
        vdofs1_all = []

        # collect vdofs
        for j in range(fes1.GetNE()):
            local_vdofs = fes1.GetElementVDofs(j)
            local_vdofs = [vv if vv >=0 else  -1 -vv for vv in local_vdofs]
            if USE_PARALLEL:
                subvdofs2 = [VDoFtoGTDoF1[i] for i in local_vdofs]
                vdofs1_all.append(subvdofs2)
            else:
                vdofs1_all.append(local_vdofs)

        #if myid == 0:
        #    pr = profile_start()

        for i, x2s, su in zip(i2_onenode, x2_onenode, s_onenode): # loop over fes2
            nd2 = len(x2s)
            #nicePrint("x2s", i, x2s.shape, x2s)
            elmats = []
            for j in range(fes1.GetNE()):
                
                if trial_domain != 'all':
                    if not attrs1[j] in trial_domain: continue

                # collect integration
                fe1 = fes1.GetFE(j)
                nd1 = fe1.GetDof()
                eltrans = fes1.GetElementTransformation(j)
                dof_sign1 = np.array([1 if vv >=0 else -1
                             for vv in fes1.GetElementVDofs(j)])

                if name_fes1 in ['RT', 'ND']:
                    shape1.SetSize(nd1, vdim1)
                else:
                    shape1.SetSize(nd1)
                elmat = np.zeros((nd2, vdim2, nd1), dtype=mat.dtype)
                tmp_int = np.zeros((vdim2, nd1), dtype=mat.dtype).squeeze()

                #if myid == 0: print("fes1 idx", j)

                dataset = []
                shapes = []                
                for jj in range(ir.GetNPoints()):
                    ip1 = ir.IntPoint(jj)
                    eltrans.SetIntPoint(ip1)
                    x1 = eltrans.Transform(ip1)
                    if name_fes1 in ['RT', 'ND']:
                        fe1.CalcVShape(eltrans, shape1)
                    else:
                        fe1.CalcShape(ip1, shape1)
                    w = eltrans.Weight() * ip1.weight
                    ss = shape1.GetDataArray().copy()
                    
                    if len(ss.shape) > 1:
                        #dof_sign1 = dof_sign1.reshape(-1, 1)
                        ss = np.transpose(ss)
                    ss = ss * dof_sign1
                    dataset.append((x1, w, ss))
                    
                has_contribution = False
                for kkk, x2 in enumerate(x2s):
                    tmp_int *= 0.0
                    has_contribution2 = False                    
                    for x1, w, shape_arr in dataset:
                        s = np.sqrt(np.sum((x1-x2)**2))
                        if su >= 0 and s > su:
                            continue
                                          
                        val = kernel(x2-x1, (x2+x1)/2.0, w=w)
                        if val is None:
                            continue
                        if coeff is not  None:
                            val = val* coeff((x2+x1)/2.0)

                        tmp_int += np.dot(val, shape_arr)*w
                        has_contribution2 = True
                        
                    if has_contribution2:
                        elmat[kkk, ...] = tmp_int
                        has_contribution = True
                if has_contribution:
                    elmats.append((j, elmat))
                    
            #if myid == 0:
            #    pr.dump_stats("/home/shiraiwa/test.prf")                
            #    profile_stop(pr)
            #    assert False, "hoge"
            #    pr = profile_start()
            if len(elmats) > 0:
                elmats_all.append((i, elmats))
                
        vdofs1_senddata.append(vdofs1_all)
        elmats_senddata.append(elmats_all)
        
        # send this information to knodes;
        '''
        if USE_PARALLEL:
            #nicePrint(vdofs1_all)
            #nicePrint("elmats", [len(x) for x in elmats_all])
            if myid == knode1:
                vdofs1_data = comm.gather(vdofs1_all, root=knode1)
                elmats_data = comm.gather(elmats_all, root=knode1)
            else:
                _ = comm.gather(vdofs1_all, root=knode1)
                _ = comm.gather(elmats_all, root=knode1)
        else:
            vdofs1_data = [vdofs1_all,]
            elmats_data = [elmats_all,]
        '''
    if USE_PARALLEL:
        knode1 = 0
        for vdofs1_all, elmats_all in zip(vdofs1_senddata, elmats_senddata):
            if myid == knode1:
                vdofs1_data = comm.gather(vdofs1_all, root=knode1)
                elmats_data = comm.gather(elmats_all, root=knode1)
            else:
                _ = comm.gather(vdofs1_all, root=knode1)
                _ = comm.gather(elmats_all, root=knode1)
            knode1 = knode1 + 1
    else:
        vdofs1_data = vdofs1_senddata
        elmats_data = elmats_senddata
        
    # Step 4
    if verbose:
        dprint1("Step 4")        
    shared_data = []
    mpi_rank = 0
    for vdofs1, elmats_all in zip(vdofs1_data, elmats_data): # loop over MPI nodes
        #nicePrint("len elmats", len(elmats_all))
        #for i, elmats in enumerate(elmats_all):  # corresponds to loop over fes2
        
        if verbose:
            coupling = [len(elmats) for i, elmats in elmats_all]
            nicePrint("Element coupling for rank/count", mpi_rank, len(coupling))
            nicePrint("   Average :", (0 if len(coupling)==0 else np.mean(coupling)))
            nicePrint("   Max/Min :", (0 if len(coupling)==0 else np.max(coupling)),
                                      (0 if len(coupling)==0 else np.min(coupling)))
            mpi_rank += 1

        for i, elmats in elmats_all:  # corresponds to loop over fes2
            vdofs2 = fes2.GetElementVDofs(i)
            dof_sign2 = np.array([[1 if vv >= 0 else -1
                                   for vv in vdofs2],]).transpose()
            vdofs2 = [-1-x if x < 0 else x for x in vdofs2]            
            
            fe2 = fes2.GetFE(i)
            nd2 = fe2.GetDof()
            
            if name_fes2 in ['RT', 'ND']:
                shape2.SetSize(nd2, vdim2)
            else:
                shape2.SetSize(nd2)

            eltrans = fes2.GetElementTransformation(i)

            #for j, elmat in enumerate(elmats):                
            for j, elmat in elmats:
                #print(vdofs1[j], elmat.shape)
                #if elmat is None:
                #    continue
                
                mm = np.zeros((len(vdofs2), len(vdofs1[j])), dtype=float)

                for ii in range(ir.GetNPoints()):
                    ip2 = ir.IntPoint(ii)
                    eltrans.SetIntPoint(ip2)
                    ww = eltrans.Weight() * ip2.weight
                    
                    if name_fes2 in ['RT', 'ND']:
                        fe2.CalcVShape(eltrans, shape2)
                    else:
                        fe2.CalcShape(ip2, shape2)

                    shape2 *= ww
                    ss = shape2.GetDataArray().reshape(-1, vdim2)
                    ss = ss * dof_sign2

                    tmp_int = elmat[ii, ...].reshape(vdim1, -1)
                    tmp = np.dot(ss, tmp_int)
                    mm = mm + tmp

                # preapre shared data
                if USE_PARALLEL:
                    vdofs22 = [fes2.GetLocalTDofNumber(ii) for ii in vdofs2]
                    vdofs22g = [VDoFtoGTDoF2[ii] for ii in vdofs2]
                    kkk = 0
                    #for v2, v2g in zip(vdofs22, vdofs22g):
                    for v2, v2g in zip(vdofs22, vdofs22g):
                        if v2 < 0:
                            shared_data.append([v2g, mm[kkk, :], vdofs1[j]])
                        kkk = kkk + 1
                        
                # merge contribution to final mat
                for k, vv in enumerate(vdofs1[j]):
                    try:
                        if USE_PARALLEL:
                            mmm = mm[np.where(np.array(vdofs22) >= 0)[0], :]                            
                            vdofs222 = [x for x in vdofs22 if x >= 0]
                        else:
                            vdofs222 = vdofs2
                            mmm = mm
                        #if myid == 1:
                        #    print("check here", vdofs2, vdofs22, vdofs222)
                        #print(mmm[:, [k]])
                        tmp = mat[vdofs222, vv] + mmm[:, [k]]
                        mat[vdofs222, vv] = tmp.flatten()

                    except:
                        import traceback
                        print("error", myid)
                        #print(vdofs1, vdofs22, vdofs222, mmm.shape, k)
                        traceback.print_exc()

    if USE_PARALLEL:
        for source_id in range(nprc):
            data = comm.bcast(shared_data, root=source_id)
            myoffset = fes2.GetMyTDofOffset()
            for v2g, elmat, vdofs1 in data:
                if v2g >= myoffset and v2g < myoffset + mat.shape[0]:
                    i = v2g - myoffset
                    #print("procesising this", myid, i, v2g, elmat, vdofs1)                
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

