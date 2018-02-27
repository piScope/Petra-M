import os
import numpy as np

from petram.mfem_config import use_parallel

if use_parallel:
   import mfem.par as mfem
   
   from mpi4py import MPI
   myid = MPI.COMM_WORLD.rank
   comm  = MPI.COMM_WORLD
   
   smyid = '{:0>6d}'.format(myid)
   from mfem.common.mpi_debug import nicePrint, niceCall
   from petram.helper.mpi_recipes import allgather, allgather_vector
   
else:
   import mfem.ser as mfem
   myid = 0

def distribute_shared_vertex(pmesh):
    master_entry = []
    local_data = {}
    master_data = {}

    offset = np.hstack([0, np.cumsum(allgather(pmesh.GetNV()))])
    ng = pmesh.GetNGroups()    
    for j in range(ng):
        if j == 0: continue
        nv = pmesh.GroupNVertices(j)
        sv = np.array([pmesh.GroupVertex(j, iv) for iv in range(nv)])
        
        local_data[(pmesh.gtopo.GetGroupMasterRank(j),
                    pmesh.gtopo.GetGroupMasterGroup(j))] = sv + offset[myid]

        if pmesh.gtopo.IAmMaster(j):
            master_entry.append((myid, j, pmesh.GroupNVertices(j),))
            mv = sv + offset[myid]
        else:
            mv = None
        master_data[(pmesh.gtopo.GetGroupMasterRank(j),
                    pmesh.gtopo.GetGroupMasterGroup(j))] = mv
        
    master_entry = comm.gather(master_entry)
    if myid == 0: master_entry = sum(master_entry, [])
    master_entry = comm.bcast(master_entry)
    for master_id, g_in_master, size in master_entry:
        entry = (master_id, g_in_master)
        if master_id == myid:
            data = master_data[entry]
        else:
            data = None
        data = comm.bcast(data, root=master_id)
        if entry in master_data:
            master_data[entry] = data
    return local_data, master_data

def surface(mesh, index, filename = '', precision=8):
    '''
    mesh must be 
    if sdim == 3:
       a domain of   2D mesh
       a boundary of 3D mesh
    if sdim == 2:
       a domain  in 2D mesh

    index : eihter
    filename : an option to save the file 
    return new surface mesh

    # MFEM Geometry Types (see mesh/geom.hpp):
    #
    # POINT       = 0
    # SEGMENT     = 1
    # TRIANGLE    = 2
    # SQUARE      = 3
    # TETRAHEDRON = 4
    # CUBE        = 5
    '''
    sdim = mesh.SpaceDimension()
    dim = mesh.Dimension()
    Nodal = mesh.GetNodalFESpace()
    hasNodal = (Nodal is not None)    

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
    idx = np.arange(len(attrs))[np.in1d(attrs, index)]
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
    
    Nvert = len(u)
    Nelem = len(idx)
    
    #cnverts = np.hstack([0, np.cumsum(nverts)])    
    #indices  = np.hstack([indices[cnverts[i]:cnverts[i+1]] for i in range(len(cnverts)-1)])

    if use_parallel:
        #accumulate all info...
        Nelem = np.sum(allgather(Nelem))
        base = allgather_vector(base)
        nverts = allgather_vector(nverts)
        attrs = allgather_vector(attrs)

        
        ld, md = distribute_shared_vertex(mesh)

        #nicePrint("ld", ld)
        #nicePrint("md", md)        
        offset = np.hstack([0, np.cumsum(allgather(mesh.GetNV()))])

        ivert = ivert + offset[myid] # -> global numbering
        u = u +  offset[myid] # -> global numbering
        #nicePrint(u)        
        ## eliminat shared vertices from data collection
        
        vtx_mask = np.array([True]*len(u))
        #nicePrint(ivert)
        #nicePrint("Nvert", Nvert)        
        for key in ld.keys():
            mid, g_in_master = key
            if mid == myid: continue
            for lv, mv in zip(ld[key], md[key]):
                idx0 =  np.where(ivert == lv)[0]
                if len(idx0) > 0:
                    ivert[idx0] = mv
            idx0 = np.in1d(u, ld[key])
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

def volume(mesh, index, filename = '', precision=8):
    '''
    mesh must be 
    if sdim == 3:
       a domain of   2D mesh
       a boundary of 3D mesh
    if sdim == 2:
       a domain  in 2D mesh

    index : eihter
    filename : an option to save the file 
    return new surface mesh

    # MFEM Geometry Types (see mesh/geom.hpp):
    #
    # POINT       = 0
    # SEGMENT     = 1
    # TRIANGLE    = 2
    # SQUARE      = 3
    # TETRAHEDRON = 4
    # CUBE        = 5
    '''
    sdim = mesh.SpaceDimension()
    dim = mesh.Dimension()
    Nodal = mesh.GetNodalFESpace()
    hasNodal = (Nodal is not None)    

    if sdim != 3: assert False, "sdim must be three for volume mesh"
    if dim != 3: assert False, "sdim must be three for volume mesh"    

    GetXAttributeArray  = mesh.GetAttributeArray
    GetXElementVertices = mesh.GetElementVertices
    GetXBaseGeometry    = mesh.GetElementBaseGeometry
    
    attrs = GetXAttributeArray()
    idx = np.arange(len(attrs))[np.in1d(attrs, index)]
    attrs = attrs[idx]
    
    if len(idx) > 0:
        ivert = np.hstack([GetXElementVertices(i)
                           for i in idx]).astype(int, copy=False)
        u, indices = np.unique(ivert, return_inverse = True)
    
        nverts= np.hstack([len(GetXElementVertices(i))
                           for i in idx]).astype(int, copy=False)
        base = np.hstack([GetXBaseGeometry(i)
                          for i in idx]).astype(int, copy=False)
        vtx = np.vstack([mesh.GetVertexArray(i) for i in u])
    else:
        ivert = np.array([], dtype=int)
        u, indices = np.unique(ivert, return_inverse = True)
    
        nverts= np.array([], dtype=int)
        base = np.array([], dtype=int)
        vtx = np.array([]).reshape((-1, sdim))
    
    Nvert = len(u)
    Nelem = len(idx)
    
    #cnverts = np.hstack([0, np.cumsum(nverts)])    
    #indices  = np.hstack([indices[cnverts[i]:cnverts[i+1]] for i in range(len(cnverts)-1)])

    if use_parallel:
        #accumulate all info...
        Nelem = np.sum(allgather(Nelem))
        base = allgather_vector(base)
        nverts = allgather_vector(nverts)
        attrs = allgather_vector(attrs)

        
        ld, md = distribute_shared_vertex(mesh)

        #nicePrint("ld", ld)
        #nicePrint("md", md)        
        offset = np.hstack([0, np.cumsum(allgather(mesh.GetNV()))])

        ivert = ivert + offset[myid] # -> global numbering
        u = u +  offset[myid] # -> global numbering
        #nicePrint(u)        
        ## eliminat shared vertices from data collection
        
        vtx_mask = np.array([True]*len(u))
        #nicePrint(ivert)
        #nicePrint("Nvert", Nvert)        
        for key in ld.keys():
            mid, g_in_master = key
            if mid == myid: continue
            for lv, mv in zip(ld[key], md[key]):
                idx0 =  np.where(ivert == lv)[0]
                if len(idx0) > 0:
                    ivert[idx0] = mv
            idx0 = np.in1d(u, ld[key])
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

    omesh = mfem.Mesh(3, Nvert, Nelem, 0, sdim)

    for i in range(Nelem):
        iv = indices[cnverts[i]:cnverts[i+1]]

        if base[i] == 2:  # triangle
            omesh.AddTri(list(iv), attrs[i])
        elif base[i] == 3: # quad
            omesh.AddQuad(list(iv), attrs[i])
        elif base[i] == 4: # tet
            omesh.AddTet(list(iv), attrs[i])
        elif base[i] == 5: # hex
            omesh.AddHex(list(iv), attrs[i])
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

        GetXDofs        =  Nodal.GetElementDofs
        GetNX           =  Nodal.GetNE               
        dGetXDofs       = dNodal.GetElementDofs
        dGetNX          = dNodal.GetNE               
           
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
 

