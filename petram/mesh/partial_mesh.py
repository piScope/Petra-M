import os
import numpy as np

from petram.mfem_config import use_parallel

if use_parallel:
   import mfem.par as mfem
   
   from mpi4py import MPI
   myid = MPI.COMM_WORLD.rank
   nprc = MPI.COMM_WORLD.size   
   comm  = MPI.COMM_WORLD
   
   smyid = '{:0>6d}'.format(myid)
   from mfem.common.mpi_debug import nicePrint, niceCall
   from petram.helper.mpi_recipes import allgather, allgather_vector, gather_vector
   
else:
   import mfem.ser as mfem
   myid = 0

def collect_data(index, mesh, mode, skip_vtx= False):
   
    if mode == 'bdr':
        GetXAttributeArray  = mesh.GetBdrAttributeArray
        GetXElementVertices = mesh.GetBdrElementVertices
        GetXBaseGeometry    = mesh.GetBdrElementBaseGeometry
        attrs = mesh.GetBdrAttributeArray()
    elif mode == 'dom':
        GetXAttributeArray  = mesh.GetAttributeArray
        GetXElementVertices = mesh.GetElementVertices
        GetXBaseGeometry    = mesh.GetElementBaseGeometry
        attrs = mesh.GetAttributeArray()
        
    idx = np.arange(len(attrs))[np.in1d(attrs, index)]
    attrs = attrs[idx]
                 
    if len(idx) > 0:
        ivert = [GetXElementVertices(i) for i in idx]
        nverts = np.array([len(x) for x in ivert], dtype=int)                 
        ivert = np.hstack(ivert).astype(int, copy=False)
        base = np.hstack([GetXBaseGeometry(i)
                          for i in idx]).astype(int, copy=False)
    else:
        ivert = np.array([], dtype=int)
        nverts= np.array([], dtype=int)
        base = np.array([], dtype=int)
        
    return idx, attrs, ivert, nverts, base

def add_face_data(m, idx, ivert, nverts, base):
    '''
    3D mesh (face), 2D mesh (edge)
    '''
    new_v = [m.GetFaceVertices(i) for i in idx]
    ivert  = np.hstack((ivert,  np.hstack(new_v)))
    nverts = np.hstack((nverts,  [len(kk) for kk in new_v]))
    base   = np.hstack((base,  [m.GetFaceBaseGeometry(i) for i in idx]))
    return ivert, nverts, base

 
def distribute_shared_vertex(pmesh):
    master_entry = []
    local_data = {}
    master_data = {}
        
    offset_v = np.hstack([0, np.cumsum(allgather(pmesh.GetNV()))])    
    offset_e = np.hstack([0, np.cumsum(allgather(pmesh.GetNEdges()))])
    offset_f = np.hstack([0, np.cumsum(allgather(pmesh.GetNFaces()))])

    ng = pmesh.GetNGroups()

    from mfem.par import intp
    def GroupEdge(j, iv):
        edge = intp(); o = intp()
        pmesh.GroupEdge(j, iv, edge, o)
        return edge.value()
    def GroupFace(j, iv):
        face = intp(); o = intp()
        pmesh.GroupFace(j, iv, face, o)
        return face.value()
     
    for j in range(ng):
        if j == 0: continue
        nv = pmesh.GroupNVertices(j)
        sv = np.array([pmesh.GroupVertex(j, iv) for iv in range(nv)])
        ne = pmesh.GroupNEdges(j)
        se = np.array([GroupEdge(j, iv) for iv in range(ne)])
        nf = pmesh.GroupNFaces(j)
        sf = np.array([GroupFace(j, iv) for iv in range(nf)])

        data = (sv + offset_v[myid],
                se + offset_e[myid],
                sf + offset_f[myid])
        local_data[(pmesh.gtopo.GetGroupMasterRank(j),
                    pmesh.gtopo.GetGroupMasterGroup(j))] = data

        if pmesh.gtopo.IAmMaster(j):
            master_entry.append((myid, j,))
            mv = sv + offset_v[myid]
            me = se + offset_e[myid]
            mf = sf + offset_f[myid]
            data = (mv, me, mf)
        else:
            data = None
        master_data[(pmesh.gtopo.GetGroupMasterRank(j),
                     pmesh.gtopo.GetGroupMasterGroup(j))] = data
        
    master_entry = comm.gather(master_entry)
    if myid == 0: master_entry = sum(master_entry, [])
    master_entry = comm.bcast(master_entry)
    for entry in master_entry:
        master_id = entry[0]
        if master_id == myid:
            data = master_data[entry]
        else:
            data = None
        data = comm.bcast(data, root=master_id)
        if entry in master_data:
            master_data[entry] = data
            
    return local_data, master_data

def surface(mesh, in_attr, filename = '', precision=8):
    '''
    make a new mesh which contains only spedified boundaries.

    we want to assign boundary attribute number consistently.

    MFEM currently does not have a mechanism to assign numbers
    to ndim-2 and below elements.
    We take this informatnion from extended_connectivity data
    gathered when loading mesh. This way, edge(vertex) numbers
    in 3D (2D) mesh is properied carrid over to surface. 

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

    '''
    if sdim == 3:
        if dim == 3:
            GetXAttributeArray  = mesh.GetBdrAttributeArray
            GetXElementVertices = mesh.GetBdrElementVertices
            GetXBaseGeometry    = mesh.GetBdrElementBaseGeometry
        elif dim == 2:
            GetXAttributeArray  = mesh.GetAttributeArray
            GetXElementVertices = mesh.GetElementVertices
            GetXBaseGeometry    = mesh.GetElementBaseGeometry
        else:
            assert False, "not supprint sdim==3, dim==1"
    elif sdim == 2:
        GetXAttributeArray  = mesh.GetAttributeArray
        GetXElementVertices = mesh.GetElementVertices
        GetXBaseGeometry    = mesh.GetElementBaseGeometry
    else:
        assert False, "not supprint sdim==1"
    attrs = GetXAttributeArray()
    idx = np.arange(len(attrs))[np.in1d(attrs, in_attr)]
    attrs = attrs[idx]
    
    if len(idx) > 0:
        ivert = np.hstack([GetXElementVertices(i) for i in idx]).astype(int, copy=False)
        u, indices = np.unique(ivert, return_inverse = True)
    
        nverts= np.hstack([len(GetXElementVertices(i)) for i in idx]).astype(int, copy=False)
        base = np.hstack([GetXBaseGeometry(i) for i in idx]).astype(int, copy=False)
        vtx = np.vstack([mesh.GetVertexArray(i) for i in u])
    else:
        ivert = np.array([], dtype=int)
        u, indices = np.unique(ivert, return_inverse = True)
    
        nverts= np.array([], dtype=int)
        base = np.array([], dtype=int)
        vtx = np.array([]).reshape((-1, sdim))
    '''
    if sdim == 3 and dim == 3: mode = 'bdr', 'edge'
    elif sdim == 3 and dim == 2: mode = 'dom', 'bdr'
    elif sdim == 2 and dim == 2: mode = 'dom', 'bdr'

    idx, attrs, ivert, nverts, base = collect_data(in_attr, mesh, mode[0])
    
    s2e = mesh.extended_connectivity['surf2edge']
    in_eattr = np.unique(np.hstack([s2e[k] for k in in_attr]))
    eidx, eattrs, eivert, neverts, ebase = collect_data(in_eattr, mesh, mode[1])
    
    Nvert = len(u)
    Nelem = len(idx)
    
    if use_parallel:
        #accumulate all info...
        Nelem = np.sum(allgather(Nelem))
        base = allgather_vector(base)
        nverts = allgather_vector(nverts)
        attrs = allgather_vector(attrs)

        offset = np.hstack([0, np.cumsum(allgather(mesh.GetNV()))])
        ivert = ivert + offset[myid] # -> global numbering
        u = u +  offset[myid]        # -> global numbering
        
        ld, md = distribute_shared_vertex(mesh)
        ## eliminat shared vertices from data collection
        
        vtx_mask = np.array([True]*len(u))

        for key in ld.keys():
            mid, g_in_master = key
            if mid == myid: continue
            for lv, mv in zip(ld[key][0], md[key][0]):
                if mv == -1: continue               
                idx0 =  np.where(ivert == lv)[0]
                if len(idx0) > 0:
                    ivert[idx0] = mv
            idx0 = np.in1d(u, ld[key][0])
            Nvert = Nvert - np.sum(idx0)
            vtx_mask[idx0] = False
        #nicePrint("Nvert", Nvert)
        Nvert = allgather(Nvert)
        cNvert = np.hstack([0, np.cumsum(Nvert)])
        Nvert = np.sum(Nvert)
        #nicePrint(cNvert)
        ivert = allgather_vector(ivert)
        u, indices = np.unique(ivert, return_inverse = True)
        size = vtx.shape
        if len(vtx) > 0: vtx = vtx[vtx_mask, :]
        vtx = allgather_vector(vtx.flatten()).reshape(-1, sdim)


    cnverts = np.hstack([0, np.cumsum(nverts)])
    #nicePrint(list(indices), list(cnverts), Nelem, vtx.shape)

    omesh = mfem.Mesh(2, Nvert, Nelem, 0, sdim)

    for i in range(Nelem):
        iv = indices[cnverts[i]:cnverts[i+1]]

        if base[i] == 2:  # triangle
            omesh.AddTri(list(iv), attrs[i])
        elif base[i] == 3: # quad
            omesh.AddQuad(list(iv), attrs[i])
        else:
            assert False, "unsupported base geometry: " + str(base[i])

    for i in range(Nvert):
         omesh.AddVertex(list(vtx[i]))

    omesh.FinalizeTopology()
    omesh.Finalize(refine=False, fix_orientation=True)

    if hasNodal:
        odim = omesh.Dimension()
        print("odim", odim)
        fec = Nodal.FEColl()
        dNodal = mfem.FiniteElementSpace(omesh, fec, sdim)
        omesh.SetNodalFESpace(dNodal)
        omesh._nodal= dNodal

        if sdim == 3:
           if dim == 3:
               GetXDofs        =  Nodal.GetBdrElementDofs
               GetNX           =  Nodal.GetNBE
           elif dim == 2:
               GetXDofs        =  Nodal.GetElementDofs
               GetNX           =  Nodal.GetNE               
           else:
               assert False, "not supported ndim 1" 
           if odim == 3:
               dGetXDofs       = dNodal.GetBdrElementDofs
               dGetNX          = dNodal.GetNBE                              
           elif odim == 2:
               dGetXDofs       = dNodal.GetElementDofs
               dGetNX          = dNodal.GetNE               
           else:
               assert False, "not supported ndim (3->1)" 
        elif sdim == 2:
           GetXDofs         =  Nodal.GetElementDofs
           dGetXDofs        = dNodal.GetElementDofs
           
        DofToVDof        =  Nodal.DofToVDof
        dDofToVDof       = dNodal.DofToVDof

        nicePrint(dGetNX(),',', GetNX())
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
        if use_parallel: data  = allgather_vector(data)
        if use_parallel: idx  = allgather_vector(idx)
        #nicePrint(len(data), ',', len(idx))

        dof2_idx = np.hstack([[dDofToVDof(i, d) for d in range(sdim)]
                              for j in range(len(idx))
                              for i in dGetXDofs(j)])
        node_ptx2[dof2_idx] = data 
        #nicePrint(len(dof2_idx))

    #mesh.FinalizeTriMesh(1,1, True)
    if filename != '':
        if use_parallel:
            smyid = '{:0>6d}'.format(myid)
            filename = filename +'.'+smyid
        omesh.PrintToFile(filename, precision)

    return omesh

def volume(mesh, in_attr, filename = '', precision=8):
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

    if sdim != 3: assert False, "sdim must be three for volume mesh"
    if dim != 3: assert False, "sdim must be three for volume mesh"

    idx, attrs, ivert, nverts, base = collect_data(in_attr, mesh, 'dom')
    
    v2s = mesh.extended_connectivity['vol2surf']
    in_battr = np.unique(np.hstack([v2s[k] for k in in_attr]))
    bidx, battrs, bivert, nbverts, bbase = collect_data(in_battr, mesh, 'bdr')
    iface = np.array([mesh.GetBdrElementEdgeIndex(i) for i in bidx], dtype=int)

    # note u is sorted unique
    u, indices = np.unique(np.hstack((ivert, bivert)),
                           return_inverse = True)


    Nelem = len(idx)
    Nbelem = len(bidx)
    kbelem = np.array([True]*Nbelem, dtype=bool)
    u_own = u
    
    if use_parallel:
    #if False:
        # process shared vertex
        #  note
        #    1) a vertex in sub-volume may be shadow
        #    2) the real one may not a part of sub-volume on the master node
        #    3) we always use the real vertex
        #        1) shadow vertex index is over-written to a real vertex
        #        2) make sure that the real one is added to sub-volume mesh obj.
       
        offset = np.hstack([0, np.cumsum(allgather(mesh.GetNV()))])
        ivert = ivert + offset[myid] # -> global numbering
        bivert = bivert + offset[myid] # -> global numbering
        u = u +  offset[myid] # -> global numbering

        ld, md = distribute_shared_vertex(mesh)

        mv_list = [[] for i in range(nprc)]
        for key in ld.keys():
            mid, g_in_master = key
            if mid != myid:
               for lv, mv in zip(ld[key][0], md[key][0]):
                   iii =  np.where(ivert == lv)[0]
                   jjj =  np.where(bivert == lv)[0]

                   if len(iii) > 0 or len(jjj) > 0:
                       if len(iii) > 0:
                           ivert[iii] = mv
                       if len(jjj) > 0:
                           bivert[jjj] = mv
                       mv_list[mid].append(mv)
               u = u[np.in1d(u, ld[key][0], invert=True)]
        for i in range(nprc):
            mvv = gather_vector(np.atleast_1d(mv_list[i]).astype(int), root=i)
            if i == myid:
               missing = np.unique(mvv[np.in1d(mvv, u, invert=True)])
               if len(missing) != 0:
                   print "adding (vertex)", missing
                   u = np.hstack((u, missing))               
                  
        u_own = np.sort(u - offset[myid])               
    Nvert = len(u_own)
    if len(u_own) > 0:
        vtx = np.vstack([mesh.GetVertexArray(i) for i in u_own])    
    else:
        vtx = np.array([]).reshape((-1, sdim))

    if use_parallel:
        #accumulate all info...
        base = allgather_vector(base)
        nverts = allgather_vector(nverts)
        attrs = allgather_vector(attrs)
        
        Nvert = allgather(Nvert)
        cNvert = np.hstack([0, np.cumsum(Nvert)])
        Nvert = np.sum(Nvert)

        Nelem = np.sum(allgather(Nelem))        
        ivert = allgather_vector(ivert)
        bivert = allgather_vector(bivert)
        
        vtx = allgather_vector(vtx.flatten()).reshape(-1, sdim)

        u, indices = np.unique(np.hstack([ivert,bivert]), return_inverse = True)

        #
        # take care of shared boundary (face)
        #
        mf_list = [[] for i in range(nprc)]
        mfa_list = [[] for i in range(nprc)]
        for key in ld.keys():
            mid, g_in_master = key
            if mid != myid:        
               for lf, mf in zip(ld[key][2], md[key][2]):
                   iii =  np.where(iface == lf)[0]
                   if len(iii) != 0:
                       kbelem[iii] = False
                       mf_list[mid].append(mf)
                       mfa_list[mid].extend(list(battrs[iii]))
                       Nbelem -= 1
                   assert len(iii)<2, "same iface (pls report this error to developer) ???"
        for i in range(nprc):        
            mfv = gather_vector(np.atleast_1d(mf_list[i]).astype(int), root=i)
            mfa = gather_vector(np.atleast_1d(mfa_list[i]).astype(int), root=i)
            if i == myid:            
               check = np.in1d(mfv, iface, invert=True)
               missing, mii = np.unique(mfv[check], return_index=True)
               missinga = mfa[check][mii]
               if len(missing) != 0:                       
                   print "adding (face)", missing
                   bivert, nbverts, bbase  = add_face_data(mesh, missing, bivert,
                                                           nbverts, bbase)
                   print len(missing), len(missinga), missinga
                   battrs = np.hstack((battrs, missinga))
                   Nbelem  += len(missing)
                   kbelem = np.hstack((kbelem, [True]*len(missing)))

        Nbelem = np.sum(allgather(Nbelem))       
        battrs = allgather_vector(battrs)
        bbase = allgather_vector(bbase)
        nbverts = allgather_vector(nbverts)
        kbelem  = allgather_vector(kbelem)        
        
    indices  = np.array([np.where(u == biv)[0][0] for biv in ivert])
    bindices = np.array([np.where(u == biv)[0][0] for biv in bivert])

    if myid ==0: print("NV, NBE, NE: " +
                       ",".join([str(x) for x in (Nvert, Nbelem, Nelem)]))
    
    cnverts = np.hstack([0, np.cumsum(nverts)])
    cnbverts = np.hstack([0, np.cumsum(nbverts)])

    omesh = mfem.Mesh(3, Nvert, Nelem, Nbelem, sdim)
    #omesh = mfem.Mesh(3, Nvert, Nelem, 0, sdim)
         
    for i in range(Nelem):
        iv = indices[cnverts[i]:cnverts[i+1]]
        if base[i] == 4: # tet
            omesh.AddTet(list(iv), attrs[i])
        elif base[i] == 5: # hex
            omesh.AddHex(list(iv), attrs[i])
        else:
            assert False, "unsupported base geometry: " + str(base[i])

    for i in range(Nbelem):
        if not kbelem[i]:
           print "skipping"
           continue
        iv = bindices[cnbverts[i]:cnbverts[i+1]]
        if bbase[i] == 2:  # triangle
            omesh.AddBdrTriangle(list(iv), battrs[i])
        elif bbase[i] == 3: # quad
            omesh.AddBdrQuad(list(iv), battrs[i])
        else:
            assert False, "unsupported base geometry: " + str(base[i])

    for i in range(Nvert):
        omesh.AddVertex(list(vtx[i]))
        
    omesh.FinalizeTopology()
    omesh.Finalize(refine=False, fix_orientation=True)

    if hasNodal:
        odim = omesh.Dimension()
        fec = Nodal.FEColl()
        dNodal = mfem.FiniteElementSpace(omesh, fec, sdim)
        omesh.SetNodalFESpace(dNodal)
        omesh._nodal= dNodal

        GetXDofs        =  Nodal.GetElementDofs
        GetNX           =  Nodal.GetNE               
        dGetXDofs       = dNodal.GetElementDofs
        dGetNX          = dNodal.GetNE               
           
        DofToVDof        =  Nodal.DofToVDof
        dDofToVDof       = dNodal.DofToVDof

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
        if use_parallel: data  = allgather_vector(data)
        if use_parallel: idx  = allgather_vector(idx)
        #nicePrint(len(data), ',', len(idx))

        dof2_idx = np.hstack([[dDofToVDof(i, d) for d in range(sdim)]
                              for j in range(len(idx))
                              for i in dGetXDofs(j)])
        node_ptx2[dof2_idx] = data 
        #nicePrint(len(dof2_idx))

    #mesh.FinalizeTriMesh(1,1, True)
    if filename != '':
        if use_parallel:
            smyid = '{:0>6d}'.format(myid)
            filename = filename +'.'+smyid
        omesh.PrintToFile(filename, precision)

    return omesh
 

