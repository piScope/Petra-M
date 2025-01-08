from __future__ import print_function
'''
    partial_mesh


    generate a new mesh which contains only spedified domain/boundaries.

    we want to assign boundary attribute number consistently.

    MFEM currently does not have a mechanism to assign numbers
    to ndim-2 and below elements.
    We take this informatnion from extended_connectivity data
    gathered when loading mesh. This way, edge(vertex) numbers
    in 3D (2D) mesh is properied carrid over to surface. 
'''
import os
import numpy as np

from petram.mfem_config import use_parallel
import petram.debug
dprint1, dprint2, dprint3 = petram.debug.init_dprints('partial_mesh')

partition_method = "default"   # default or 0 (0 handles disconnected mesh)

if use_parallel:
    import mfem.par as mfem

    from mpi4py import MPI
    myid = MPI.COMM_WORLD.rank
    nprc = MPI.COMM_WORLD.size
    comm = MPI.COMM_WORLD

    smyid = '{:0>6d}'.format(myid)
    from mfem.common.mpi_debug import nicePrint, niceCall
    from petram.helper.mpi_recipes import allgather, allgather_vector, gather_vector
    from petram.mesh.mesh_utils import distribute_shared_entity
else:
    import mfem.ser as mfem
    myid = 0

    def nicePrint(x):
        print(x)


def isParMesh(mesh):
    return hasattr(mesh, 'GetNGroups')


def _collect_data(index, mesh, mode, skip_vtx=False):
    '''
    collect  index : attribute

    return  idx, attrs, ivert, nverts, base
      idx, attrs: element index, attr number for elemnt
      ivert : flattened vertex
      nverts : num of vertices for each element
      base : element geometry base
    '''
    LEN = len
    if mode == 'bdr':
        GetXElementVertices = mesh.GetBdrElementVertices
        GetXBaseGeometry = mesh.GetBdrElementBaseGeometry
        attrs = mesh.GetBdrAttributeArray()
        idx = np.arange(len(attrs))[np.in1d(attrs, index)]
        attrs = attrs[idx]

    elif mode == 'dom':
        GetXElementVertices = mesh.GetElementVertices
        GetXBaseGeometry = mesh.GetElementBaseGeometry
        attrs = mesh.GetAttributeArray()

        idx = np.arange(len(attrs))[np.in1d(attrs, index)]
        attrs = attrs[idx]

    elif mode == 'edge':
        GetXElementVertices = mesh.GetEdgeVertices
        def GetXBaseGeometry(x): return 1

        s2l = mesh.extended_connectivity['surf2line']
        l2e = mesh.extended_connectivity['line2edge']
        idx = np.array(sum([l2e[ea] for ea in index], []), dtype=int)
        if len(idx) == 0:
            attrs = np.atleast_1d([]).astype(int)
        else:
            attrs = np.hstack([[ea]*len(l2e[ea]) for ea in index]).astype(int)

    elif mode == 'vertex':
        v2v = mesh.extended_connectivity['vert2vert']
        def GetXElementVertices(x): return v2v[x] if x in v2v else []
        def GetXBaseGeometry(x): return 0
        def LEN(x): return 1
        idx = np.array([x for x in index if x in v2v], dtype=int)
        if len(idx) == 0:
            attrs = np.atleast_1d([]).astype(int)
        else:
            attrs = np.hstack([va for va in index if va in v2v]).astype(int)

    else:
        assert False, "Unknown mode (_collect_data) "+mode

    if len(idx) > 0:
        ivert = [GetXElementVertices(i) for i in idx]
        nverts = np.array([LEN(x) for x in ivert], dtype=int)
        ivert = np.hstack(ivert).astype(int, copy=False)
        base = np.hstack([GetXBaseGeometry(i)
                          for i in idx]).astype(int, copy=False)
    else:
        ivert = np.array([], dtype=int)
        nverts = np.array([], dtype=int)
        base = np.array([], dtype=int)

    return idx, attrs, ivert, nverts, base


def _add_face_data(m, idx, nverts, base):
    '''
    3D mesh (face), 2D mesh (edge)
    '''
    new_v = [m.GetFaceVertices(i) for i in idx]
    #ivert  = np.hstack((ivert,  np.hstack(new_v)))
    nverts = np.hstack((nverts,  [len(kk) for kk in new_v]))
    base = np.hstack((base,  [m.GetFaceBaseGeometry(i) for i in idx]))
    # return ivert, nverts, base
    return nverts, base


def _gather_shared_vertex(mesh, u, shared_info,  *iverts):
    # u_own, iv1, iv2... = gather_shared_vertex(mesh, u, ld, md, iv1, iv2...)

    # u_own  : unique vertex id ownd by a process
    # shared_info : shared data infomation
    # iv1, iv2, ...: array of vertex is after overwriting shadow vertex
    #                to a real one, which is owned by other process

    # process shared vertex
    #    1) a vertex in sub-volume may be shadow
    #    2) the real one may not a part of sub-volume on the master node
    #    3) we always use the real vertex
    #        1) shadow vertex index is over-written to a real vertex
    #        2) make sure that the real one is added to sub-volume mesh obj.

    offset = np.hstack([0, np.cumsum(allgather(mesh.GetNV()))])
    iverts = [iv + offset[myid] for iv in iverts]
    u = u + offset[myid]  # -> global numbering

    ld, md = shared_info
    mv_list = [[] for i in range(nprc)]
    for key in ld.keys():
        mid, g_in_master = key
        if mid != myid:
            for lv, mv in zip(ld[key][0], md[key][0]):
                ic = 0
                for iv in iverts:
                    iii = np.where(iv == lv)[0]
                    ic = ic + len(iii)
                    if len(iii) > 0:
                        iv[iii] = mv
                if ic > 0:
                    mv_list[mid].append(mv)
            u = u[np.in1d(u, ld[key][0], invert=True)]
    for i in range(nprc):
        mvv = gather_vector(np.atleast_1d(mv_list[i]).astype(int), root=i)
        if i == myid:
            missing = np.unique(mvv[np.in1d(mvv, u, invert=True)])
            if len(missing) != 0:
                dprint1("adding (vertex)", missing)
                u = np.hstack((u, missing))

    u_own = np.sort(u - offset[myid])
    return [u_own]+list(iverts)  # u_own, iv1, iv2 =


def _gather_shared_element(mesh, mode, shared_info, ielem, kelem, attrs,
                           nverts, base, ivert, skip_adding=False):

    ld, md = shared_info

    if mode == 'face':
        imode = 2
    elif mode == 'edge':
        imode = 1
    else:
        imode = 0
    #
    me_list = [[] for i in range(nprc)]
    mea_list = [[] for i in range(nprc)]
    for key in ld.keys():
        mid, g_in_master = key
        if mid != myid:
            for le, me in zip(ld[key][imode], md[key][imode]):
                iii = np.where(ielem == le)[0]
                if len(iii) != 0:
                    if not skip_adding:
                        kelem[iii] = False
                    me_list[mid].append(me)
                    mea_list[mid].extend(list(attrs[iii]))
                assert len(
                    iii) < 2, "same iface (pls report this error to developer) ???"

    new_ivert = np.array([], dtype=ivert.dtype)
    for i in range(nprc):
        mev = gather_vector(np.atleast_1d(me_list[i]).astype(int), root=i)
        mea = gather_vector(np.atleast_1d(mea_list[i]).astype(int), root=i)
        if i == myid:
            check = np.in1d(mev, ielem, invert=True)
            missing, mii = np.unique(mev[check], return_index=True)
            missinga = mea[check][mii]
            if len(missing) != 0:
                dprint1("adding (face)", myid, missing)
                dprint1(len(missing), len(missinga), missinga)
                if not skip_adding:
                    nverts, base = _add_face_data(mesh, missing, nverts, base)
                    attrs = np.hstack((attrs, missinga)).astype(attrs.dtype)
                    kelem = np.hstack((kelem, [True]*len(missing)))

    if not skip_adding:
        new_ivert = new_ivert.astype(ivert.dtype)
        new_ivert = allgather_vector(new_ivert)
        #ivert  = np.hstack((ivert,  new_ivert))
    attrs = allgather_vector(attrs)
    base = allgather_vector(base)
    nverts = allgather_vector(nverts)
    kelem = allgather_vector(kelem)
    return kelem, attrs, nverts, base, ivert


def _fill_mesh_elements(omesh, vtx, indices, nverts,  attrs, base):

    cnverts = np.hstack([0, np.cumsum(nverts)])

    for i, a in enumerate(attrs):
        iv = indices[cnverts[i]:cnverts[i+1]]
        if base[i] == 1:  # segment
            el = mfem.Segment()
            el.SetAttribute(a)
            el.SetVertices(list(iv))
            el.thisown = False
            omesh.AddElement(el)
        elif base[i] == 2:  # triangle
            omesh.AddTri(list(iv), a)
        elif base[i] == 3:  # quad
            omesh.AddQuad(list(iv), a)
        elif base[i] == 4:  # tet
            omesh.AddTet(list(iv), a)
        elif base[i] == 5:  # hex
            omesh.AddHex(list(iv), a)
        else:
            assert False, "unsupported base geometry: " + str(base[i])


def _fill_mesh_bdr_elements(omesh, vtx, bindices, nbverts,
                            battrs, bbase, kbelem):

    cnbverts = np.hstack([0, np.cumsum(nbverts)])
    for i, ba in enumerate(battrs):
        if not kbelem[i]:
            dprint1("skipping")
            continue
        iv = bindices[cnbverts[i]:cnbverts[i+1]]
        if bbase[i] == 0:
            el = mfem.Point(iv[0])
            el.SetAttribute(ba)
            el.thisown = False
            omesh.AddBdrElement(el)
        elif bbase[i] == 1:
            omesh.AddBdrSegment(list(iv), ba)
        elif bbase[i] == 2:
            #if myid == 1: print(list(iv), ba)
            omesh.AddBdrTriangle(list(iv), ba)
        elif bbase[i] == 3:
            omesh.AddBdrQuad(list(iv), ba)
        else:
            assert False, "unsupported base geometry: " + str(bbase[i])

    for v in vtx:
        omesh.AddVertex(list(v))


def edge(mesh, in_attr, filename='', precision=8):
    '''
    make a new mesh which contains only spedified edges.

    in_attr : eihter
    filename : an option to save the file 
    return new surface mesh
    '''
    sdim = mesh.SpaceDimension()
    dim = mesh.Dimension()
    Nodal = mesh.GetNodalFESpace()
    hasNodal = (Nodal is not None)

    if sdim == 3 and dim == 3:
        mode = 'edge', 'vertex'
    elif sdim == 3 and dim == 2:
        mode = 'bdr', 'vertex'
    elif sdim == 2 and dim == 2:
        mode = 'bdr', 'vertex'
    elif sdim == 2 and dim == 1:
        mode = 'dom', 'vertex'
    else:
        assert False, "unsupported mdoe"

    idx, attrs, ivert, nverts, base = _collect_data(in_attr, mesh, mode[0])

    l2v = mesh.extended_connectivity['line2vert']
    in_eattr = np.unique(np.hstack([l2v[k]
                         for k in in_attr])).astype(int, copy=False)
    if isParMesh(mesh):
        in_eattr = np.unique(allgather_vector(in_eattr))
    eidx, eattrs, eivert, neverts, ebase = _collect_data(in_eattr, mesh,
                                                         mode[1])

    u, indices = np.unique(np.hstack((ivert, eivert)),
                           return_inverse=True)
    keelem = np.array([True]*len(eidx), dtype=bool)
    u_own = u

    if isParMesh(mesh):
        shared_info = distribute_shared_entity(mesh)
        u_own, ivert, eivert = _gather_shared_vertex(mesh, u, shared_info,
                                                     ivert, eivert)
    Nvert = len(u)
    if len(u_own) > 0:
        vtx = np.vstack([mesh.GetVertexArray(i) for i in u_own])
    else:
        vtx = np.array([]).reshape((-1, sdim))

    if isParMesh(mesh):
        #
        # distribute vertex/element data
        #
        base = allgather_vector(base)
        nverts = allgather_vector(nverts)
        attrs = allgather_vector(attrs)

        ivert = allgather_vector(ivert)
        eivert = allgather_vector(eivert)

        vtx = allgather_vector(vtx.flatten()).reshape(-1, sdim)

        u, indices = np.unique(np.hstack([ivert, eivert]),
                               return_inverse=True)

        #
        # take care of shared boundary (edge)
        #
        keelem, eattrs, neverts, ebase, eivert = (
            _gather_shared_element(mesh, 'vertex', shared_info, eidx,
                                   keelem, eattrs,
                                   neverts, ebase, eivert))

    indices = np.array([np.where(u == biv)[0][0] for biv in ivert])
    eindices = np.array([np.where(u == biv)[0][0] for biv in eivert])

    Nvert = len(vtx)
    Nelem = len(attrs)
    Nbelem = len(eattrs)

    dprint1("NV, NBE, NE: " +
            ",".join([str(x) for x in (Nvert, Nbelem, Nelem)]))

    omesh = mfem.Mesh(1, Nvert, Nelem, Nbelem, sdim)

    _fill_mesh_elements(omesh, vtx, indices, nverts, attrs, base)
    _fill_mesh_bdr_elements(omesh, vtx, eindices, neverts, eattrs,
                            ebase, keelem)

    omesh.FinalizeTopology()

    if hasNodal:
        odim = omesh.Dimension()

        dprint1("odim, dim, sdim", odim, " ", dim, " ", sdim)
        fec = Nodal.FEColl()
        dNodal = mfem.FiniteElementSpace(omesh, fec, sdim)
        omesh.SetNodalFESpace(dNodal)
        omesh._nodal = dNodal

        GetXDofs = Nodal.GetElementDofs
        if dim == 3:
            GetXDofs = Nodal.GetEdgeDofs
        elif dim == 2:
            GetXDofs = Nodal.GetBdrElementDofs
        elif dim == 1:
            GetXDofs = Nodal.GetElementDofs

        dGetXDofs = dNodal.GetElementDofs

        DofToVDof = Nodal.DofToVDof
        dDofToVDof = dNodal.DofToVDof

        #nicePrint(dGetNX(),',', GetNX())
        nodes = mesh.GetNodes()
        node_ptx1 = nodes.GetDataArray()

        onodes = omesh.GetNodes()
        node_ptx2 = onodes.GetDataArray()
        #nicePrint(len(idx), idx)

        if len(idx) > 0:
            dof1_idx = np.hstack([[DofToVDof(i, d) for d in range(sdim)]
                                  for j in idx
                                  for i in GetXDofs(j)])
            data = node_ptx1[dof1_idx]
        else:
            dof1_idx = np.array([])
            data = np.array([])
        if isParMesh(mesh):
            data = allgather_vector(data)
        if isParMesh(mesh):
            idx = allgather_vector(idx)
        #nicePrint(len(data), ',', len(idx))

        dof2_idx = np.hstack([[dDofToVDof(i, d) for d in range(sdim)]
                              for j in range(len(idx))
                              for i in dGetXDofs(j)])
        node_ptx2[dof2_idx] = data
        # nicePrint(len(dof2_idx))

    # this should be after setting HO nodals...
    omesh.Finalize(refine=True, fix_orientation=True)

    if isParMesh(mesh):
        if omesh.GetNE() < nprc*3:
            parts = omesh.GeneratePartitioning(1, 1)
        elif partition_method != 'default':
            nparts = min([nprc, 8])
            parts = omesh.GeneratePartitioning(nparts, 0)
        else:
            parts = None
        omesh = mfem.ParMesh(comm, omesh, parts)

    if filename != '':
        if isParMesh(mesh):
            smyid = '{:0>6d}'.format(myid)
            filename = filename + '.'+smyid
        omesh.PrintToFile(filename, precision)

    return omesh


def surface(mesh, in_attr, filename='', precision=8):
    '''
    mesh must be 
    if sdim == 3:
       a domain of   2D mesh
       a boundary of 3D mesh
    if sdim == 2:
       a domain  in 2D mesh

    in_attr : eihter
    filename : an option to save the file 
    return new surface mesh

    '''
    sdim = mesh.SpaceDimension()
    dim = mesh.Dimension()
    Nodal = mesh.GetNodalFESpace()
    hasNodal = (Nodal is not None)

    if sdim == 3 and dim == 3:
        mode = 'bdr', 'edge'
    elif sdim == 3 and dim == 2:
        mode = 'dom', 'bdr'
    elif sdim == 2 and dim == 2:
        mode = 'dom', 'bdr'
    else:
        assert False, "unsupported mdoe"

    idx, attrs, ivert, nverts, base = _collect_data(in_attr, mesh, mode[0])

    s2l = mesh.extended_connectivity['surf2line']
    in_eattr = np.unique(np.hstack([s2l[k]
                         for k in in_attr])).astype(int, copy=False)
    if isParMesh(mesh):
        in_eattr = np.unique(allgather_vector(in_eattr))

    eidx, eattrs, eivert, neverts, ebase = _collect_data(in_eattr, mesh,
                                                         mode[1])

    u, indices = np.unique(np.hstack((ivert, eivert)),
                           return_inverse=True)
    keelem = np.array([True]*len(eidx), dtype=bool)
    u_own = u

    if isParMesh(mesh):
        shared_info = distribute_shared_entity(mesh)
        u_own, ivert, eivert = _gather_shared_vertex(mesh, u, shared_info,
                                                     ivert, eivert)
    Nvert = len(u)
    if len(u_own) > 0:
        vtx = np.vstack([mesh.GetVertexArray(i) for i in u_own])
    else:
        vtx = np.array([]).reshape((-1, sdim))

    if isParMesh(mesh):
        #
        # distribute vertex/element data
        #
        base = allgather_vector(base)
        nverts = allgather_vector(nverts)
        attrs = allgather_vector(attrs)

        ivert = allgather_vector(ivert)
        eivert = allgather_vector(eivert)

        vtx = allgather_vector(vtx.flatten()).reshape(-1, sdim)

        u, indices = np.unique(np.hstack([ivert, eivert]),
                               return_inverse=True)

        #
        # take care of shared boundary (edge)
        #
        keelem, eattrs, neverts, ebase, eivert = (
            _gather_shared_element(mesh, 'edge', shared_info, eidx,
                                   keelem, eattrs,
                                   neverts, ebase, eivert,
                                   skip_adding=True))

    #indices  = np.array([np.where(u == biv)[0][0] for biv in ivert])
    #eindices = np.array([np.where(u == biv)[0][0] for biv in eivert])

    iv, ivi = np.unique(ivert, return_inverse=True)
    tmp = np.where(np.in1d(u,  ivert,  assume_unique=True))[0]
    indices = tmp[ivi]
    iv, ivi = np.unique(eivert, return_inverse=True)
    tmp = np.where(np.in1d(u,  eivert, assume_unique=True))[0]
    eindices = tmp[ivi]

    Nvert = len(vtx)
    Nelem = len(attrs)
    Nbelem = len(eattrs)

    dprint1("NV, NBE, NE: " +
            ",".join([str(x) for x in (Nvert, Nbelem, Nelem)]))

    omesh = mfem.Mesh(2, Nvert, Nelem, Nbelem, sdim)

    _fill_mesh_elements(omesh, vtx, indices, nverts, attrs, base)
    _fill_mesh_bdr_elements(omesh, vtx, eindices, neverts, eattrs,
                            ebase, keelem)

    omesh.FinalizeTopology()

    if hasNodal:
        odim = omesh.Dimension()
        dprint1("odim, dim, sdim", odim, " ", dim, " ", sdim)
        fec = Nodal.FEColl()
        dNodal = mfem.FiniteElementSpace(omesh, fec, sdim)
        omesh.SetNodalFESpace(dNodal)
        omesh._nodal = dNodal

        if sdim == 3:
            if dim == 3:
                GetXDofs = Nodal.GetBdrElementDofs
                GetNX = Nodal.GetNBE
            elif dim == 2:
                GetXDofs = Nodal.GetElementDofs
                GetNX = Nodal.GetNE
            else:
                assert False, "not supported ndim 1"
        elif sdim == 2:
            GetNX = Nodal.GetNE
            GetXDofs = Nodal.GetElementDofs

        dGetNX = dNodal.GetNE
        dGetXDofs = dNodal.GetElementDofs

        DofToVDof = Nodal.DofToVDof
        dDofToVDof = dNodal.DofToVDof

        #nicePrint(dGetNX(),',', GetNX())
        nodes = mesh.GetNodes()
        node_ptx1 = nodes.GetDataArray()

        onodes = omesh.GetNodes()
        node_ptx2 = onodes.GetDataArray()
        #nicePrint(len(idx), idx)

        if len(idx) > 0:
            dof1_idx = np.hstack([[DofToVDof(i, d) for d in range(sdim)]
                                  for j in idx
                                  for i in GetXDofs(j)])
            data = node_ptx1[dof1_idx]
        else:
            dof1_idx = np.array([])
            data = np.array([])
        if isParMesh(mesh):
            data = allgather_vector(data)
        if isParMesh(mesh):
            idx = allgather_vector(idx)
        #nicePrint(len(data), ',', len(idx))

        dof2_idx = np.hstack([[dDofToVDof(i, d) for d in range(sdim)]
                              for j in range(len(idx))
                              for i in dGetXDofs(j)])
        node_ptx2[dof2_idx] = data

    # this should be after setting HO nodals...
    omesh.Finalize(refine=True, fix_orientation=True)

    if isParMesh(mesh):
        if omesh.GetNE() < nprc*3:
            parts = omesh.GeneratePartitioning(1, 1)
        elif partition_method != 'default':
            nparts = min([nprc, 8])
            parts = omesh.GeneratePartitioning(nparts, 0)
        else:
            parts = None
        omesh = mfem.ParMesh(comm, omesh, parts)
        #omesh = mfem.ParMesh(comm, omesh)

    if filename != '':
        if isParMesh(mesh):
            smyid = '{:0>6d}'.format(myid)
            filename = filename + '.'+smyid
        omesh.PrintToFile(filename, precision)

    return omesh


def volume(mesh, in_attr, filename='', precision=8):
    '''
    make a new mesh which contains only spedified attributes.

    note: 
       1) boundary elements are also copied and bdr_attributes
          are maintained
       2) in parallel, new mesh must be geometrically continuous.
          this routine does not check it

    mesh must have sdim == 3:
    in_attr : domain attribute
    filename : an option to save the file 

    return new volume mesh
    '''
    in_attr = np.atleast_1d(in_attr)
    sdim = mesh.SpaceDimension()
    dim = mesh.Dimension()
    Nodal = mesh.GetNodalFESpace()
    hasNodal = (Nodal is not None)

    if sdim != 3:
        assert False, "sdim must be three for volume mesh"
    if dim != 3:
        assert False, "sdim must be three for volume mesh"

    idx, attrs, ivert, nverts, base = _collect_data(in_attr, mesh, 'dom')

    v2s = mesh.extended_connectivity['vol2surf']
    in_battr = np.unique(np.hstack([v2s[k]
                         for k in in_attr])).astype(int, copy=False)
    if isParMesh(mesh):
        in_battr = np.unique(allgather_vector(in_battr))

    bidx, battrs, bivert, nbverts, bbase = _collect_data(in_battr, mesh, 'bdr')
    iface = np.array([mesh.GetBdrElementEdgeIndex(i) for i in bidx],
                     dtype=int)

    # note u is sorted unique
    u, indices = np.unique(np.hstack((ivert, bivert)),
                           return_inverse=True)

    kbelem = np.array([True]*len(bidx), dtype=bool)
    u_own = u

    if isParMesh(mesh):
        shared_info = distribute_shared_entity(mesh)
        u_own, ivert, bivert = _gather_shared_vertex(mesh, u, shared_info,
                                                     ivert, bivert)

    if len(u_own) > 0:
        vtx = np.vstack([mesh.GetVertexArray(i) for i in u_own])
    else:
        vtx = np.array([]).reshape((-1, sdim))

    if isParMesh(mesh):
        #
        # distribute vertex/element data
        #
        base = allgather_vector(base)
        nverts = allgather_vector(nverts)
        attrs = allgather_vector(attrs)

        ivert = allgather_vector(ivert)
        bivert = allgather_vector(bivert)

        vtx = allgather_vector(vtx.flatten()).reshape(-1, sdim)

        u, indices = np.unique(np.hstack([ivert, bivert]), return_inverse=True)

        #
        # take care of shared boundary (face)
        #
        #  2018.11.28
        #   skip_adding is on. This basically skip shared_element
        #   processing. Check em3d_TEwg7 if you need to remov this.
        #
        kbelem, battrs, nbverts, bbase, bivert = (
            _gather_shared_element(mesh, 'face', shared_info, iface,
                                   kbelem, battrs,
                                   nbverts, bbase, bivert,
                                   skip_adding=True))

    #indices0  = np.array([np.where(u == biv)[0][0] for biv in ivert])
    #bindices0 = np.array([np.where(u == biv)[0][0] for biv in bivert])

    iv, ivi = np.unique(ivert, return_inverse=True)
    tmp = np.where(np.in1d(u,  ivert,  assume_unique=True))[0]
    indices = tmp[ivi]
    iv, ivi = np.unique(bivert, return_inverse=True)
    tmp = np.where(np.in1d(u,  bivert, assume_unique=True))[0]
    bindices = tmp[ivi]

    #print('check', np.sum(np.abs(indices - indices0)))

    Nvert = len(vtx)
    Nelem = len(attrs)
    Nbelem = np.sum(kbelem)  # len(battrs)

    dprint1("NV, NBE, NE: " +
            ",".join([str(x) for x in (Nvert, Nbelem, Nelem)]))

    omesh = mfem.Mesh(3, Nvert, Nelem, Nbelem, sdim)
    #omesh = mfem.Mesh(3, Nvert, Nelem, 0, sdim)

    _fill_mesh_elements(omesh, vtx, indices, nverts, attrs, base)
    _fill_mesh_bdr_elements(omesh, vtx, bindices, nbverts, battrs,
                            bbase, kbelem)

    omesh.FinalizeTopology()

    if hasNodal:
        odim = omesh.Dimension()
        dprint1("odim, dim, sdim", odim, " ", dim, " ", sdim)

        fec = Nodal.FEColl()
        dNodal = mfem.FiniteElementSpace(omesh, fec, sdim)
        omesh.SetNodalFESpace(dNodal)
        omesh._nodal = dNodal

        GetXDofs = Nodal.GetElementDofs
        GetNX = Nodal.GetNE
        dGetXDofs = dNodal.GetElementDofs
        dGetNX = dNodal.GetNE

        DofToVDof = Nodal.DofToVDof
        dDofToVDof = dNodal.DofToVDof

        #nicePrint(dGetNX(),',', GetNX())
        nodes = mesh.GetNodes()
        node_ptx1 = nodes.GetDataArray()

        onodes = omesh.GetNodes()
        node_ptx2 = onodes.GetDataArray()
        #nicePrint(len(idx), idx)

        if len(idx) > 0:
            dof1_idx = np.hstack([[DofToVDof(i, d) for d in range(sdim)]
                                  for j in idx
                                  for i in GetXDofs(j)])
            data = node_ptx1[dof1_idx]
        else:
            dof1_idx = np.array([])
            data = np.array([])
        if isParMesh(mesh):
            data = allgather_vector(data)
        if isParMesh(mesh):
            idx = allgather_vector(idx)
        #nicePrint(len(data), ',', len(idx))

        dof2_idx = np.hstack([[dDofToVDof(i, d) for d in range(sdim)]
                              for j in range(len(idx))
                              for i in dGetXDofs(j)])
        node_ptx2[dof2_idx] = data

    # this should be after setting HO nodals...
    omesh.Finalize(refine=True, fix_orientation=True)

    if isParMesh(mesh):
        if omesh.GetNE() < nprc*3:
            parts = omesh.GeneratePartitioning(1, 1)
        elif partition_method != 'default':
            nparts = min([nprc, 8])
            parts = omesh.GeneratePartitioning(nparts, 0)
        else:
            parts = None
        omesh = mfem.ParMesh(comm, omesh, parts)
        #omesh = mfem.ParMesh(comm, omesh)

    if filename != '':
        if isParMesh(mesh):
            smyid = '{:0>6d}'.format(myid)
            filename = filename + '.'+smyid
        omesh.PrintToFile(filename, precision)

    return omesh
