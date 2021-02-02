import numpy as np
import ifigure.utils.geom
from ifigure.interactive import figure, solid
import matplotlib.cm as cm

from .mesh_viewer import open_meshviewer
from petram.helper.geom import connect_pairs

def dim2name_bdry(dim):
    if dim == 3:
        return  'face'
    elif dim == 2:
        return 'edge'
    elif dim == 1:
        return 'point'
    
def dim2name_domain(dim):
    if dim == 3:
        return  'volume'
    elif dim == 2:
        return 'face'
    elif dim == 1:
        return 'edge'


cmap_name = 'Paired'

def get_cmap(name):
    '''
    adjust cmap behavior difference between 1.5 and 2.0
    '''
    m = cm.get_cmap(name)
    data = cm.datad[name]
    m.N = len(data[data.keys()[0]])
    try:
         m(3)
    except IndexError:
         pass
    return m
    
def plot_bdrymesh(meshname = '', mesh = None, idx = 'all',
                  viewer = None, cmap=None, name = '',
                  return_data = False, linewidths = 1.0):
    if mesh is None:  mesh = globals()[meshname]
    if viewer is None:  viewer = open_meshviewer()
    if cmap is None: cmap = cmap_name

    key = dim2name_bdry(mesh.Dimension())
    nbdr = mesh.bdr_attributes.Size()
    kbdr = mesh.bdr_attributes.ToList()
    nbe = mesh.GetNBE()

    
    idxarr = kbdr if str(idx) == 'all' else idx    
    attr = mesh.GetBdrAttributeArray()
    ivert = np.vstack([mesh.GetBdrElement(i).GetVerticesArray() for i in range(nbe)])

    cmap = get_cmap(cmap)
    dom_check_vert = [None]*mesh.bdr_attributes.Size()
    dom_check_idx = [None]*mesh.bdr_attributes.Size()    
    viewer.update(False)

    for kk, ii in enumerate(idxarr):
#        idx = np.where(attr == ii)[0]
#        data = np.dstack([np.vstack([mesh.GetVertexArray(int(k)) for k in ivert[x]]) 
#                          for x in idx])
        idx = mesh.GetBdrArray(ii)
        data = np.dstack([np.vstack([mesh.GetVertexArray(k)
                          for k in mesh.GetBdrElement(x).GetVerticesArray()])
                          for x in idx])

        dom_check_vert[kk] = mesh.GetBdrElementVertices(idx[0])
        dom_check_idx[kk] = idx[0]
        #ivert[attr == ii, :]])
        data = np.rollaxis(data, 2, 0)  ### [i_triangle, i_vert, xyz]
        #data[:,:,2] =         data[:,:,2]*100

        obj= viewer.solid(data, facecolor = cmap(kk % cmap.N),#loat(ii)/nbdr),
#                          edgecolor=(0,0,0,1), alpha = 1.0)
                          alpha = 1.0, linewidths=linewidths)
#                          view_offset = (0.0, 0.0, -0.01, 0.0))
        if obj is not None: obj.rename(name + key + '_'+str(ii))

    viewer.update(True)        
    if return_data: return data
#    return dom_check_vert
    return dom_check_idx


def plot_domainmesh(meshname = '', mesh = None, idx = 'all',
                  viewer = None, cmap=None, name = '',
                    return_data = False, linewidths = 1.0):
    if mesh is None:  mesh = globals()[meshname]
    if viewer is None:  viewer = open_meshviewer()
    if cmap is None: cmap = cmap_name
    
    key = dim2name_domain(mesh.Dimension())    
    ndom = mesh.attributes.Size()
    kdom = mesh.attributes.ToList()
    nele = mesh.GetNE()
    
    idxarr = kdom if str(idx) == 'all' else idx
    
    attr = mesh.GetAttributeArray()
    ivert = np.vstack([mesh.GetElement(i).GetVerticesArray() for i in range(nele)])

    cmap = get_cmap(cmap)
    dom_check_vert = [None]*mesh.attributes.Size()
    dom_check_idx = [None]*mesh.attributes.Size()    
    viewer.update(False)

    for kk, ii in enumerate(idxarr):
        idx = np.where(attr == ii)[0]
        data = np.dstack([np.vstack([mesh.GetVertexArray(k)
                          for k in mesh.GetElement(x).GetVerticesArray()])
                          for x in idx])

        data = np.rollaxis(data, 2, 0)  ### [i_triangle, i_vert, xyz]
        obj= viewer.solid(data, facecolor = cmap(kk % cmap.N), #float(ii)/nele),
#                          edgecolor=(0,0,0,1), alpha = 1.0)
                          alpha = 1.0, linewidths=linewidths)
        if obj is not None: obj.rename(name + key + '_'+str(ii))

    viewer.update(True)        
    if return_data: return data

def find_domain_bdr(mesh, dom_check_list):
    ndom = mesh.attributes.Max()
    ne   = mesh.GetNE()
    kbdr = mesh.bdr_attributes.ToList()

    domainbdr = []
    for i in range(ndom):
        domainbdr.append([])

    '''
    This old method scan all elements...
    for i in range(ndom): domainbdr[i] = list()
    for i in range(ne):
        iattr =  mesh.GetElement(i).GetAttribute()
        x = mesh.GetElementVertices(i)
        idx = np.where([all(y in x for y in t) for t in dom_check_list])[0]
        domainbdr[iattr-1].extend([kbdr[j] for j in idx])
    '''
    for ibdr in dom_check_list:
        idx = mesh.GetBdrElementEdgeIndex(ibdr)
        elem1, elem2 = mesh.GetFaceElements(idx)
        if elem1 != -1:
            i = mesh.GetAttribute(elem1)-1
            domainbdr[i].append(mesh.GetBdrAttribute(ibdr))
        if elem2 != -1:
            i = mesh.GetAttribute(elem2)-1
            domainbdr[i].append(mesh.GetBdrAttribute(ibdr))
            
    return domainbdr
            
def plot_bdr(meshname = '', mesh = None, idx = 'all', viewer = None,
             name = '', return_data = False,  **kwargs):
    if mesh is None:  mesh = globals()[meshname]
    if viewer is None:  viewer = open_meshviewer()

    nbdr = mesh.bdr_attributes.Size()
    kbdr = mesh.bdr_attributes.ToList()
    nbe = mesh.GetNBE()

    idxarr = kbdr if str(idx) == 'all' else idx
    attr = np.array([mesh.GetBdrElement(i).GetAttribute() for i in range(nbe)])
#    ivert = np.vstack([mesh.GetBdrElement(i).GetVerticesArray() for i in range(nbe)])
#    print [x for x in ivert[attr == 1, :]]


    for ii in idxarr:
        idx = np.where(attr == ii)[0]
        edges = np.array([mesh.GetBdrElementEdges(i)[0] for i in idx]).flatten()
        dom_check_vert[ii] = mesh.GetBdrElementVertices(idx[0])
        #ange(nbe)
#                          if mesh.GetBdrElement(i).GetAttribute()==ii]).flatten()
        d = {}
        for x in edges:
            d[x] = x in d
        edges = [x for x in d.keys() if not d[x]]
        ivert = [mesh.GetEdgeVertices(x) for x in edges]
        ivert = connect_pairs(ivert)
        vv = np.vstack([mesh.GetVertexArray(i) for i in ivert])
        obj = viewer.plot(vv[:,0], vv[:,1], vv[:,2], **kwargs)
        if obj is not None: obj.rename(name + 'bdry_'+str(ii))        

    if return_data: return vv
    return dom_check_vert


def plot_domain(meshname = '', mesh = None, idx = 'all',
                viewer = None, cmap=None, return_data = False):
    if mesh is None:  mesh = globals()[meshname]
    if viewer is None:  viewer = open_meshviewer()
    if cmap is None: cmap = cmap_name

    cmap = get_cmap(cmap)    

    nbdr = mesh.bdr_attributes.Size()
    ndom = mesh.attributes.Size()    
    kdom = mesh.attributes.ToList()
    nbe = mesh.GetNBE()
    nel = mesh.GetNE()    

    idxarr = kdom if str(idx) == 'all' else idx    
    dom_attr = np.array([mesh.GetElement(i).GetAttribute() for i in range(nel)])
    bdr_attr = np.array([mesh.GetBdrElement(i).GetAttribute() for i in range(nbe)])
    dfaces = np.array([mesh.GetElementFaces(i)[0] for i in range(nel)])
    bfaces = [mesh.GetBdrElementFace(i)[0] for i in range(nbe)]
    bfaces0 = np.unique(bfaces)
    viewer.update(False)
    for kk, ii in enumerate(idxarr):
        domain_faces = np.unique(np.array([dfaces[dom_attr==ii]]).flatten())
        faces = np.intersect1d(domain_faces, bfaces0)
        bindex = [bfaces.index(f) for f in faces]
        bindex = np.unique(bdr_attr[bindex])
        
        data = np.dstack([np.vstack([mesh.GetVertexArray(x)
                                     for x in mesh.GetFaceVertices(kk)]) for kk in faces])
        data = np.rollaxis(data, 2, 0)  ### [i_triangle, i_vert, xyz]    
        #data[:,:,2] =         data[:,:,2]*100
        col =cmap(kk % cmap.N)#float(ii)/ndom)
        obj= viewer.solid(data, facecolor = col, 
                          linewidth = 0.0)
        if obj is not None: obj.rename('domain_'+str(ii))                
        plot_bdr(mesh = mesh, idx = bindex, viewer = viewer,  color = [0,0,0,1],
                 name = 'domain_'+str(ii)+'_')
    viewer.update(True)        
    if return_data: return data

    
