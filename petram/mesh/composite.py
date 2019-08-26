from collections import defaultdict
from itertools import combinations
from sets import Set
import numpy as np


def merge_domain_3d(mesh, domain_list, reorder_dom=True):

    if not hasattr(mesh, 'extended_connectivity'): 
        from petram.mesh.mesh_utils import  get_extended_connectivity       
        get_extended_connectivity(mesh)               
    ec = mesh.extended_connectivity

    v2s = ec['vol2surf']
    s2l = ec['surf2line']

    from petram.mesh.mesh_utils import vol2line, line2surf,line2vol
    
    v2l = vol2line(v2s, s2l)
    l2v = line2vol(v2l)
    l2s = line2surf(s2l)
    
    # find surfaces to remove
    # find edge to remove
    remove_surf=[]
    remove_edge = Set()
    merge_vol = {}
    for d in domain_list:
       merge_vol[d] = 0
    touched_dom = []    
    for i, j in combinations(domain_list, 2):
        iscts = np.intersect1d(v2s[i], v2s[j])
        if len(iscts) != 0: touched_dom.append((i,j))
        remove_surf.append(iscts)
    remove_surf = list(np.hstack(remove_surf).astype(int))
    
    for s in remove_surf:
        for l in s2l[s]:
            for v in v2l:
                if l in v2l[v] and not v in domain_list:
                    break
            else:
                remove_edge.add(l)
        for l in s2l[s]:
            connected_surf = [s for s in l2s[l] if not s in remove_surf]
            if len(connected_surf) != 2: continue
            v1 = [v for v in v2s if not v in domain_list and connected_surf[0] in v2s[v]]
            v2 = [v for v in v2s if not v in domain_list and connected_surf[1] in v2s[v]]
            if len(Set(v1+v2)) == 1:
               remove_edge.add(l)                        
               
    remove_edge = list(remove_edge)                


    # check merging to one domain
    if len(touched_dom)==0:
        assert False, "domains are not connected"
    c = list(touched_dom[0])
    for i in range(len(touched_dom)):
         for j in range(i, len(touched_dom)):
             if touched_dom[j][0] in c and not touched_dom[j][1] in c:
                 c.append(touched_dom[j][1])
             if touched_dom[j][1] in c and not touched_dom[j][0] in c:
                 c.append(touched_dom[j][0])
    print("touched_domain", c)
    if len(c) != len(domain_list): 
        assert False, "domains are not connected"       
    
    # collect faces to merge
    merge_face0 = []
    for k, l in enumerate(remove_edge):
        for f in merge_face0:
            if np.intersect1d(list(f), l2s[l]).size != 0:
                 for s in l2s[l]:
                     if not s in remove_surf: f.add(s)
                 break
        else:
            sss = [s for s in l2s[l] if not s in remove_surf]
            merge_face0.append(Set(sss))
    print(merge_face0)
    merge_face = {}
    for k, ss in enumerate(merge_face0):
        for s in ss:
            merge_face[s] = k
                 
    # mapping of bdr (face) attributes
    idx = 1
    bdrattr_map = {}
    for s in s2l:
        if s in remove_surf: continue
        if not s in merge_face:
            bdrattr_map[s] = idx
            idx = idx + 1
    for s, k in merge_face.items():
        if s in remove_surf: continue        
        bdrattr_map[s] = k + idx        

    # mapping of domaon attributes
    if reorder_dom:
        idx = 1
        domattr_map = {}
        for v in v2s:
            if not v in merge_vol:
                domattr_map[v] = idx
                idx = idx + 1
        for v, k in merge_vol.items():
            domattr_map[v] = k + idx
    else:
        domattr_map = {}
        for v in v2s:
            if not v in merge_vol:
                domattr_map[v] = v
        idx = max(v2s.keys())+1
        for v, k in merge_vol.tems():
            domattr_map[v] = k + idx
        
    print("merge_vol", merge_vol)
    print("merge_surf", merge_face)    
    print("remove_edge", remove_edge)
    print("remove_surf", remove_surf)

    print("domattr_map", domattr_map)
    print("bdrattr_map", bdrattr_map)    

    # check if edge can be removed...
        
    
    NV = mesh.GetNV()
    NE = mesh.GetNE()
    NBE = mesh.GetNBE()    

    bdrattr = mesh.GetBdrAttributeArray()
    domattr = mesh.GetAttributeArray()
    
    Nbelem = np.sum([x in bdrattr_map for x in bdrattr])

    print("NV, NE, NBE", NV, NE, Nbelem)

    import sys
    if 'mfem.par' in sys.modules:
       import mfem.par as mfem
    else:
       import mfem.par as mfem        
    omesh = mfem.Mesh(3, NV, NE, Nbelem, 3)

    for k in range(NV):
        v = mesh.GetVertexArray(k)
        omesh.AddVertex(list(v))

    for k in range(NE):
        iv = mesh.GetElementVertices(k)
        a = domattr_map[domattr[k]]
        if len(iv) == 4:
            omesh.AddTet(list(iv), a)
        if len(iv) == 8:
            omesh.AddHex(list(iv), a)
            
    for k in range(NBE):
        iv = mesh.GetBdrElementVertices(k)
        if not bdrattr[k] in bdrattr_map: continue
        a = bdrattr_map[bdrattr[k]]
        
        if len(iv) == 3:
            omesh.AddBdrTriangle(list(iv), a)
        if len(iv) == 4:
            omesh.AddBdrQuad(list(iv), a)

    omesh.FinalizeTopology()
    omesh.Finalize(refine=True, fix_orientation=True)
            
    return omesh    


