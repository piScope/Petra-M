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

if use_parallel:
   import mfem.par as mfem
   
   from mpi4py import MPI
   myid = MPI.COMM_WORLD.rank
   nprc = MPI.COMM_WORLD.size   
   comm  = MPI.COMM_WORLD
   
   smyid = '{:0>6d}'.format(myid)
   from mfem.common.mpi_debug import nicePrint, niceCall
   from petram.helper.mpi_recipes import allgather, allgather_vector, gather_vector
   from petram.mesh.mesh_utils import distribute_shared_entity
else:
   import mfem.ser as mfem
   myid = 0

def isParMesh(mesh):
    return hasattr(mesh, 'GetNGroups')
    
def _collect_data(index, mesh, mode, skip_vtx= False):
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
        GetXBaseGeometry    = mesh.GetBdrElementBaseGeometry
        attrs = mesh.GetBdrAttributeArray()
        
        idx = np.arange(len(attrs))[np.in1d(attrs, index)]
        attrs = attrs[idx]
        
    elif mode == 'dom':
        GetXElementVertices = mesh.GetElementVertices
        GetXBaseGeometry    = mesh.GetElementBaseGeometry
        attrs = mesh.GetAttributeArray()
        
        idx = np.arange(len(attrs))[np.in1d(attrs, index)]
        attrs = attrs[idx]
        
    elif mode == 'edge':        
        GetXElementVertices = mesh.GetEdgeVertices
        GetXBaseGeometry    = lambda x: 1
        
        s2l = mesh.extended_connectivity['surf2line']
        l2e = mesh.extended_connectivity['line2edge']
        idx = sum([l2e[ea] for ea in index], [])
        if len(idx) == 0:
            attrs = np.atleast_1d([]).astype(int)
        else:
            attrs = np.hstack([[ea]*len(l2e[ea]) for ea in index]).astype(int)
        
    elif mode == 'vertex':
        v2v = mesh.extended_connectivity['vert2vert']       
        GetXElementVertices = lambda x: v2v[x]
        GetXBaseGeometry    = lambda x: 0
        LEN    = lambda x: 1
        idx = list(index)
        if len(idx) == 0:
            attrs =  np.atleast_1d([]).astype(int)
        else:
            attrs =  np.hstack([va for va in index]).astype(int)
           
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
        nverts= np.array([], dtype=int)
        base = np.array([], dtype=int)
        
    return idx, attrs, ivert, nverts, base

def _add_face_data(m, idx, nverts, base):
    '''
    3D mesh (face), 2D mesh (edge)
    '''
    new_v = [m.GetFaceVertices(i) for i in idx]
    #ivert  = np.hstack((ivert,  np.hstack(new_v)))
    nverts = np.hstack((nverts,  [len(kk) for kk in new_v]))
    base   = np.hstack((base,  [m.GetFaceBaseGeometry(i) for i in idx]))
    return ivert, nverts, base

 
def _gather_shared_vetex(mesh, u, shared_info,  *iverts):
    # u_own, iv1, iv2... = gather_shared_vetex(mesh, u, ld, md, iv1, iv2...)

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
    u = u +  offset[myid] # -> global numbering
    
    ld, md = shared_info
    mv_list = [[] for i in range(nprc)]
    for key in ld.keys():
        mid, g_in_master = key
        if mid != myid:
            for lv, mv in zip(ld[key][0], md[key][0]):
                ic = 0
                for iv in iverts:
                    iii =  np.where(iv == lv)[0]
                    ic = ic + len(iii)
                    if len(iii)>0:
                        iv[iii] = mv
                if ic > 0: mv_list[mid].append(mv)                        
            u = u[np.in1d(u, ld[key][0], invert=True)]
    for i in range(nprc):
        mvv = gather_vector(np.atleast_1d(mv_list[i]).astype(int), root=i)
        if i == myid:
            missing = np.unique(mvv[np.in1d(mvv, u, invert=True)])
            if len(missing) != 0:
                print "adding (vertex)", missing
                u = np.hstack((u, missing))               

    u_own = np.sort(u - offset[myid])
    return [u_own]+list(iverts) ## u_own, iv1, iv2 =

def _gather_shared_element(mesh, mode, shared_info, ielem, kelem, attrs,
                          nverts, base):
   
    ld, md = shared_info
    imode = 1 if mode == 'edge' else 2
    #
    me_list = [[] for i in range(nprc)]
    mea_list = [[] for i in range(nprc)]
    nicePrint("ielem", "kelem", ielem, kelem)
    nicePrint("local", [np.in1d(ielem, ld[key][imode]) for key in ld])
    for key in ld.keys():
        mid, g_in_master = key
        if mid != myid:
           for le, me in zip(ld[key][imode], md[key][imode]): 
               iii =  np.where(ielem == le)[0]
               if len(iii) != 0:
                   kelem[iii] = False
                   me_list[mid].append(me)
                   mea_list[mid].extend(list(attrs[iii]))
               assert len(iii)<2, "same iface (pls report this error to developer) ???"
    nicePrint("me_list", me_list)           
    for i in range(nprc):        
        mev = gather_vector(np.atleast_1d(me_list[i]).astype(int), root=i)
        mea = gather_vector(np.atleast_1d(mea_list[i]).astype(int), root=i)
        if i == myid:            
           check = np.in1d(mev, ielem, invert=True)
           missing, mii = np.unique(mev[check], return_index=True)
           missinga = mea[check][mii]
           if len(missing) != 0:                       
               print "adding (face)", missing
               nverts, base  = add_face_data(mesh, missing, nverts, base)
               print len(missing), len(missinga), missinga
               attrs = np.hstack((attrs, missinga))
               kelem = np.hstack((kelem, [True]*len(missing)))

    attrs = allgather_vector(attrs)
    base  = allgather_vector(base)
    nverts = allgather_vector(nverts)
    kelem  = allgather_vector(kelem)
    return kelem, attrs, nverts, base    
        
def _fill_mesh_elements(omesh, vtx, indices, nverts,  attrs, base):

    cnverts = np.hstack([0, np.cumsum(nverts)])

    
    for i, a in enumerate(attrs):
        iv = indices[cnverts[i]:cnverts[i+1]]
        if base[i] == 1:  # segment
            el = mfem.Segment()
            el.SetAttribute(a)
            el.SetVertices(list(iv))
            el.thisown=False
            omesh.AddElement(el)
        elif base[i] == 2:  # triangle
            omesh.AddTri(list(iv), a)
        elif base[i] == 3: # quad
            omesh.AddQuad(list(iv), a)
        elif base[i] == 4: # tet
            omesh.AddTet(list(iv), a)
        elif base[i] == 5: # hex
            omesh.AddHex(list(iv), a)
        else:
            assert False, "unsupported base geometry: " + str(base[i])
            
def _fill_mesh_bdr_elements(omesh, vtx, bindices, nbverts,
                            battrs, bbase, kbelem):

    cnbverts = np.hstack([0, np.cumsum(nbverts)])   
    for i, ba in enumerate(battrs):
        if not kbelem[i]:
           print "skipping"
           continue
        iv = bindices[cnbverts[i]:cnbverts[i+1]]
        if bbase[i] == 0:
             el = mfem.Point(iv[0])
             el.SetAttribute(ba)
             el.thisown=False
             omesh.AddBdrElement(el)
        elif bbase[i] == 1:  
             omesh.AddBdrSegment(list(iv), ba)
        elif bbase[i] == 2:  
             omesh.AddBdrTriangle(list(iv), ba)
        elif bbase[i] == 3:
            omesh.AddBdrQuad(list(iv), ba)
        else:
            assert False, "unsupported base geometry: " + str(bbase[i])

    for v in vtx: omesh.AddVertex(list(v))
    
def edge(mesh, in_attr, filename = '', precision=8):
    '''
    make a new mesh which contains only spedified boundaries.

    mesh must be 
    if sdim == 3:
       not supported
    if dim == 2:
       a boundary in 2D mesh
    elif dim == 1:
       a domain in 1D mesh


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
    in_eattr = np.unique(np.hstack([l2v[k] for k in in_attr]))
    eidx, eattrs, eivert, neverts, ebase = _collect_data(in_eattr, mesh,
                                                          mode[1])
        
    u, indices = np.unique(np.hstack((ivert, eivert)),
                           return_inverse = True)
    keelem = np.array([True]*len(eidx), dtype=bool)    
    u_own = u
    
    if isParMesh(mesh):
        shared_info = distribute_shared_entity(mesh)       
        u_own, ivert, eivert = _gather_shared_vetex(mesh, u, shared_info,
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

        u, indices = np.unique(np.hstack([ivert,eivert]),
                               return_inverse = True)

        #
        # take care of shared boundary (edge)
        #
        keelem, eattrs, neverts, ebase = (
            _gather_shared_element(mesh, 'edge', shared_info, eidx,
                                   keelem, eattrs,
                                   neverts, ebase))
        
        
    indices  = np.array([np.where(u == biv)[0][0] for biv in ivert])
    eindices = np.array([np.where(u == biv)[0][0] for biv in eivert])

    Nvert = len(vtx)
    Nelem = len(attrs)    
    Nbelem = len(eattrs)

    if myid ==0: print("NV, NBE, NE: " +
                       ",".join([str(x) for x in (Nvert, Nbelem, Nelem)]))
    

    omesh = mfem.Mesh(1, Nvert, Nelem, Nbelem, sdim)

    _fill_mesh_elements(omesh, vtx, indices, nverts, attrs, base)
    _fill_mesh_bdr_elements(omesh, vtx, eindices, neverts, eattrs,
                            ebase, keelem)

    omesh.FinalizeTopology()
    omesh.Finalize(refine=True, fix_orientation=True)

    if hasNodal:
        assert False, "high order edge mesh is not supported"
        '''
        odim = omesh.Dimension()
        print("odim, dim, sdim", odim, " ", dim, " ", sdim)
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
           GetNX           =  Nodal.GetNE                          
           dGetNX          = dNodal.GetNE                          
           GetXDofs         =  Nodal.GetElementDofs
           dGetXDofs        = dNodal.GetElementDofs
           
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
        '''
    if isParMesh(mesh):              
        omesh = mfem.ParMesh(comm, omesh)

    if filename != '':
        if isParMesh(mesh):                         
            smyid = '{:0>6d}'.format(myid)
            filename = filename +'.'+smyid
        omesh.PrintToFile(filename, precision)
        
    return omesh
    
def surface(mesh, in_attr, filename = '', precision=8):
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
    in_eattr = np.unique(np.hstack([s2l[k] for k in in_attr]))
    eidx, eattrs, eivert, neverts, ebase = _collect_data(in_eattr, mesh,
                                                          mode[1])
    nicePrint("eidx", eidx, eattrs, len(eattrs))
    #nicePrint(len(np.hstack((eivert, ivert))))
    u, indices = np.unique(np.hstack((ivert, eivert)),
                           return_inverse = True)
    keelem = np.array([True]*len(eidx), dtype=bool)    
    u_own = u
    
    if isParMesh(mesh):                                
        shared_info = distribute_shared_entity(mesh)       
        u_own, ivert, eivert = _gather_shared_vetex(mesh, u, shared_info,
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

        u, indices = np.unique(np.hstack([ivert,eivert]),
                               return_inverse = True)

        #
        # take care of shared boundary (edge)
        #
        keelem, eattrs, neverts, ebase = (
            _gather_shared_element(mesh, 'edge', shared_info, eidx,
                                   keelem, eattrs,
                                   neverts, ebase))
        
        
    indices  = np.array([np.where(u == biv)[0][0] for biv in ivert])
    eindices = np.array([np.where(u == biv)[0][0] for biv in eivert])

    Nvert = len(vtx)
    Nelem = len(attrs)    
    Nbelem = len(eattrs)

    if myid ==0: print("NV, NBE, NE: " +
                       ",".join([str(x) for x in (Nvert, Nbelem, Nelem)]))
    

    omesh = mfem.Mesh(2, Nvert, Nelem, Nbelem, sdim)

    _fill_mesh_elements(omesh, vtx, indices, nverts, attrs, base)
    _fill_mesh_bdr_elements(omesh, vtx, eindices, neverts, eattrs,
                            ebase, keelem)

    omesh.FinalizeTopology()
    omesh.Finalize(refine=True, fix_orientation=True)

    if hasNodal:
        odim = omesh.Dimension()
        print("odim, dim, sdim", odim, " ", dim, " ", sdim)
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
           GetNX           =  Nodal.GetNE                          
           dGetNX          = dNodal.GetNE                          
           GetXDofs         =  Nodal.GetElementDofs
           dGetXDofs        = dNodal.GetElementDofs
           
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
        if isParMesh(mesh): data  = allgather_vector(data)
        if isParMesh(mesh): idx  = allgather_vector(idx)
        #nicePrint(len(data), ',', len(idx))

        dof2_idx = np.hstack([[dDofToVDof(i, d) for d in range(sdim)]
                              for j in range(len(idx))
                              for i in dGetXDofs(j)])
        node_ptx2[dof2_idx] = data 
        #nicePrint(len(dof2_idx))

    if isParMesh(mesh):
        omesh = mfem.ParMesh(comm, omesh)

    if filename != '':
        if isParMesh(mesh):       
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

    idx, attrs, ivert, nverts, base = _collect_data(in_attr, mesh, 'dom')
    
    v2s = mesh.extended_connectivity['vol2surf']
    in_battr = np.unique(np.hstack([v2s[k] for k in in_attr]))
    bidx, battrs, bivert, nbverts, bbase = _collect_data(in_battr, mesh, 'bdr')
    iface = np.array([mesh.GetBdrElementEdgeIndex(i) for i in bidx],
                     dtype=int)

    # note u is sorted unique
    u, indices = np.unique(np.hstack((ivert, bivert)),
                           return_inverse = True)

    kbelem = np.array([True]*len(bidx), dtype=bool)
    u_own = u
    
    if isParMesh(mesh):       
        shared_info = distribute_shared_entity(mesh)       
        u_own, ivert, bivert = _gather_shared_vetex(mesh, u, shared_info,
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

        u, indices = np.unique(np.hstack([ivert,bivert]), return_inverse = True)

        #
        # take care of shared boundary (face)
        #
        kbelem, battrs, nbverts, bbase = (
            _gather_shared_element(mesh, 'face', shared_info, iface,
                                   kbelem, battrs,
                                   nbverts, bbase))
        
        
    indices  = np.array([np.where(u == biv)[0][0] for biv in ivert])
    bindices = np.array([np.where(u == biv)[0][0] for biv in bivert])

    
    Nvert = len(vtx)
    Nelem = len(attrs)    
    Nbelem = len(battrs)

    if myid ==0: print("NV, NBE, NE: " +
                       ",".join([str(x) for x in (Nvert, Nbelem, Nelem)]))

    
    omesh = mfem.Mesh(3, Nvert, Nelem, Nbelem, sdim)
    #omesh = mfem.Mesh(3, Nvert, Nelem, 0, sdim)
    
    _fill_mesh_elements(omesh, vtx, indices, nverts, attrs, base)
    _fill_mesh_bdr_elements(omesh, vtx, bindices, nbverts, battrs,
                            bbase, kbelem)
   
    omesh.FinalizeTopology()
    omesh.Finalize(refine=True, fix_orientation=True)

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
        if isParMesh(mesh): data  = allgather_vector(data)
        if isParMesh(mesh): idx  = allgather_vector(idx)
        #nicePrint(len(data), ',', len(idx))

        dof2_idx = np.hstack([[dDofToVDof(i, d) for d in range(sdim)]
                              for j in range(len(idx))
                              for i in dGetXDofs(j)])
        node_ptx2[dof2_idx] = data 
        #nicePrint(len(dof2_idx))

    if isParMesh(mesh):
        omesh = mfem.ParMesh(comm, omesh)
        
    if filename != '':
        if isParMesh(mesh):
            smyid = '{:0>6d}'.format(myid)
            filename = filename +'.'+smyid
        omesh.PrintToFile(filename, precision)

    return omesh
 

