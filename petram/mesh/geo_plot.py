import numpy as np

def expand_vertex_data(X, vertex_idx, element_id):
    '''
    expand index data using element_id, so that
   
       surface edges will have duplicate vertex,
       this way,,
         1) the normal vector on the edge become discutinous
         2) the elment index on the surface becomes constant
    '''
    k = 0
    verts = []
    iele = []
    iarr = []

    nel = vertex_idx.shape[-1]

    def iter_unique_idx(x):
        '''
        return unique element and all indices for each eleement

        note: this is to avoid using where in the loop like this
           # for kk in np.unique(element_id):
           #    idx = np.where(element_id == kk)[0]
           # (using where inside the loop makes it very slow as 
           # number of unique elements increases
        '''
        uv, uc = np.unique(x, return_counts=True)
        idx = np.argsort(x)
        uc = np.hstack((0, np.cumsum(uc)))
        for i, v in enumerate(uv):
            yield (v, idx[uc[i]:uc[i+1]])

    for kk, idx in iter_unique_idx(element_id):
        iverts = vertex_idx[idx].flatten()
        iv, idx = np.unique(iverts, return_inverse = True)
        
        verts.append(X[iv])
        iele.append(idx.reshape(-1, nel)+k)
        k = k + len(iv)
        iarr.append(np.zeros(len(iv))+kk)

    array_idx = np.hstack(iarr).astype(int)
    elem_idx = np.vstack(iele).astype(int)
    verts = np.vstack(verts)

    return verts, elem_idx, array_idx

# we visit nodes in counter clock wise.
gmsh_ho_el_idx = {'triangle6': ([0, 3, 5], [3, 1, 4], [5, 3, 4], [5, 4, 2]),
                  'triangle10':([0, 3, 8], [8, 3, 9], [3, 4, 9], [9, 4, 5], [1, 5, 4],
                                [7, 8, 9], [7, 9, 6], [6, 9, 5], [2, 7, 6]),
                  'triangle15':([0, 3, 11], [11, 3, 12], [12, 3, 4], [4, 13, 12], [13, 4, 5],
                                [13, 5, 6], [6, 5, 1], [10, 11, 12], [10, 12, 14],
                                [14, 12, 13], [14, 13, 7], [7, 13, 6], [9, 10, 14],
                                [9, 14, 8], [8, 14, 7], [2, 9, 8]),
                  'triangle21':([0, 3, 14], [3, 15, 14], [3, 4, 15], [4, 18, 15],
                                [4, 5, 18], [5, 16, 18], [5, 6, 16], [6, 7, 16],
                                [6, 1, 7], [13, 14, 15], [13, 15, 20],
                                [15, 18, 20], [18, 19, 20], [18, 16, 19], [16, 8, 19],
                                [16, 7, 8], [12, 13, 20], [20, 17, 12], [20, 19, 17],
                                [19, 9, 17], [19, 8, 9], [11, 12, 17], [11, 17, 10],
                                [10, 17, 9], [2, 11, 10]),}

gmsh_ho_bel_idx = {'triangle6':([0, 3], [3, 1], [1, 4], [4, 2], [2, 5], [5, 0]),
                   'triangle10':([0, 3], [3, 4], [4, 1],
                                 [1, 5], [5, 6], [6, 2],
                                 [2, 7], [7, 8], [8, 0]),
                   'triangle15':([0, 3], [3, 4], [4, 5], [5, 1],
                                 [1, 6], [6, 7], [7, 8], [8, 2],
                                 [2, 9], [9, 10], [10, 11], [11, 0]),
                   'triangle21':([0, 3], [3, 4], [4, 5], [5, 6], [6, 1],
                                 [1, 7], [7, 8], [8, 9], [9, 10], [10, 2],
                                 [2, 11], [11, 12], [12, 13], [13, 14], [14, 0]),}
             
def expand_vertex_data_gmsh_ho(X, vertex_idx, element_id, el_type):
    '''
    expand index data using element_id, so that
   
       surface edges will have duplicate vertex,
       this way,,
         1) the normal vector on the edge become discutinous
         2) the elment index on the surface becomes constant
    '''
    k = 0
    verts = []
    iele = []
    iarr = []
    iedge = []

    nel = vertex_idx.shape[-1]

    def iter_unique_idx(x):
        '''
        return unique element and all indices for each eleement

        note: this is to avoid using where in the loop like this
           # for kk in np.unique(element_id):
           #    idx = np.where(element_id == kk)[0]
           # (using where inside the loop makes it very slow as 
           # number of unique elements increases
        '''
        uv, uc = np.unique(x, return_counts=True)
        idx = np.argsort(x)
        uc = np.hstack((0, np.cumsum(uc)))
        for i, v in enumerate(uv):
            yield (v, idx[uc[i]:uc[i+1]])

    for kk, idx in iter_unique_idx(element_id):
        iverts = vertex_idx[idx].flatten()
        iv, idx = np.unique(iverts, return_inverse=True)

        verts.append(X[iv])
        tmp = idx.reshape(-1, nel)+k

        el = [tmp[:, x] for x in gmsh_ho_el_idx[el_type]]
        bel = [tmp[:, x] for x in gmsh_ho_bel_idx[el_type]]
        elem_idx = np.vstack(el)
        ed_idx = np.vstack(bel)

        iele.append(elem_idx)
        iedge.append(ed_idx)
        k = k + len(iv)
        iarr.append(np.zeros(len(iv))+kk)

    array_idx = np.hstack(iarr).astype(int)
    elem_idx = np.vstack(iele).astype(int)
    edge_idx = np.vstack(iedge).astype(int)
    verts = np.vstack(verts)

    return verts, elem_idx, edge_idx, array_idx


def call_solid1(viewer, name, verts, elem_idx, array_idx, lw,
                edge_idx=None):
    # template for faces    
    obj = viewer.solid(verts, elem_idx,
                       array_idx=array_idx,
                       facecolor=(0.7, 0.7, 0.7, 1.0),
                       linewidth=lw,
                       edge_idx=edge_idx)  

    obj.rename(name)
    obj.set_gl_hl_use_array_idx(True)
    return obj

def call_solid2(viewer, name, verts, elem_idx, array_idx=None, lw=1.5):
    # template for lines
    obj = viewer.solid(verts, elem_idx,
                       array_idx=array_idx,
                       linewidth=lw,
                       facecolor=(0, 0, 0, 1.0),
                       edgecolor=(0, 0, 0, 1.0),
#                           view_offset = (0, 0, -0.001, 0),
                       draw_last=True)
    obj.rename(name)
    obj.set_gl_hl_use_array_idx(True)
    return obj

def plot_geometry(viewer,  ret,  geo_phys = 'geometrical', lw = 0):
    viewer.cls()
    viewer.set_hl_color((1, 0, 0))

    print('plot_geometry')
    X, cells, pt_data, cell_data, field_data = ret

    if 'triangle_x' in cells:
        verts = cell_data['X_refined_face']
        elem_idx = cells['triangle_x']
        array_idx = cell_data['triangle_x'][geo_phys]
        eelem_idx = cells['triangle_xe']

        call_solid1(viewer, 'face_t', verts, elem_idx, array_idx, lw,
                    edge_idx=eelem_idx)

    elif 'triangle' in cells:
        verts, elem_idx, array_idx = expand_vertex_data(X, cells['triangle'],
                                                        cell_data['triangle'][geo_phys])

        call_solid1(viewer, 'face_t', verts, elem_idx, array_idx, lw)

    if 'quad_x' in cells:
        verts = cell_data['X_refined_face']
        elem_idx = cells['quad_x']
        array_idx = cell_data['quad_x'][geo_phys]
        eelem_idx = cells['quad_xe']
        call_solid1(viewer, 'face_r', verts, elem_idx, array_idx, lw,
                    edge_idx=eelem_idx)

    elif 'quad' in cells:        
        verts, elem_idx, array_idx = expand_vertex_data(X, cells['quad'],
                                       cell_data['quad'][geo_phys])

        call_solid1(viewer, 'face_r', verts, elem_idx, array_idx, lw)
        
    if 'line_x' in cells:
        verts = cell_data['X_refined_edge']
        elem_idx = cells['line_x']
        array_idx = cell_data['line_x'][geo_phys]
        call_solid2(viewer, 'edge', verts, elem_idx, array_idx)        

    elif 'line' in cells and len(cells['line']) > 0:
        verts, elem_idx, array_idx = expand_vertex_data(X, cells['line'],
                                                           cell_data['line'][geo_phys])
        call_solid2(viewer, 'edge', verts, elem_idx, array_idx)                

    if 'vertex' in cells:
        if 'vertex_mask' in cells:
            vidx = cells['vertex'][cells['vertex_mask']]
            aidx = cell_data['vertex'][geo_phys][cells['vertex_mask']]            
        else:
            vidx = cells['vertex']
            aidx = cell_data['vertex'][geo_phys]
        vert = np.atleast_2d(np.squeeze(X[vidx]))
        if len(vert) > 0:
            obj= viewer.plot(vert[:,0],
                             vert[:,1],
                             vert[:,2], 'ok',
                             array_idx = aidx,
                             linewidth = 0)
            obj.rename('point')
            obj.set_gl_hl_use_array_idx(True)
    viewer.set_sel_mode(viewer.get_sel_mode())

def oplot_meshed(viewer,  ret):

    ax =viewer.get_axes()
    for name, obj in ax.get_children():
        if name.endswith('_meshed'):
            viewer.cls(obj = obj)

    try:
        X, cells, pt_data, cell_data, field_data = ret
    except ValueError:
        return
    
    meshed_face = []
    if 'triangle' in cells:
        verts, elem_idx, array_idx = expand_vertex_data(X, cells['triangle'],
                                       cell_data['triangle']['geometrical'])


        print(verts.shape, elem_idx.shape, array_idx.shape)
        obj = viewer.solid(verts, elem_idx,
                           array_idx = array_idx,
                           facecolor = (0.7, 0.7, 0.7, 1.0),
                           edgecolor = (0, 0, 0, 1),
                           linewidth = 1,
                           view_offset = (0, 0, -0.0005, 0))

        obj.rename('face_t_meshed')
        obj.set_gl_hl_use_array_idx(True)

        meshed_face.extend(list(np.unique(cell_data['triangle']['geometrical'])))
        

    def handle_gmsh_ho_elements(el):
        elem_idx = cells[el]
        array_idx = cell_data[el]['geometrical']
        verts, elem_idx, edge_idx, array_idx = expand_vertex_data_gmsh_ho(X, elem_idx, array_idx, el)

        obj = viewer.solid(verts, elem_idx,
                           array_idx = array_idx,
                           facecolor = (0.7, 0.7, 0.7, 1.0),
                           edgecolor = (0, 0, 0, 1),
                           linewidth = 1,
                           view_offset = (0, 0, -0.0005, 0),
                           edge_idx=edge_idx)
        obj.rename('face_t_meshed')
        obj.set_gl_hl_use_array_idx(True)

        meshed_face.extend(list(array_idx))
    if 'triangle6' in cells:
        handle_gmsh_ho_elements('triangle6')
    if 'triangle10' in cells:
        handle_gmsh_ho_elements('triangle10')
    if 'triangle15' in cells:
        handle_gmsh_ho_elements('triangle15')
    if 'triangle21' in cells:
        handle_gmsh_ho_elements('triangle21')
        
    if 'quad' in cells:
        verts, elem_idx, array_idx = expand_vertex_data(X, cells['quad'],
                                       cell_data['quad']['geometrical'])

        #print verts.shape, elem_idx.shape, array_idx.shape
        obj = viewer.solid(verts, elem_idx,
                           array_idx = array_idx,
                           facecolor = (0.7, 0.7, 0.7, 1.0),
                           edgecolor = (0, 0, 0, 1),
#                           view_offset = (0, 0, -0.0005, 0),
                           linewidth = 1,)

                      
        obj.rename('face_r_meshed')
        obj.set_gl_hl_use_array_idx(True)
        meshed_face.extend(list(np.unique(cell_data['quad']['geometrical'])))

    hide_face_meshmode(viewer, meshed_face)
    '''
    s, v = viewer._s_v_loop['mesh']
    facesa = []
    if len(v)>0:  # in 3D starts with faces from shown volumes
        all_surfaces = np.array(list(s), dtype=int)        
        for key in v:
            if not key in viewer._mhidden_volume:
                facesa.extend(v[key])
        facesa = np.unique(facesa)
        mask  = np.logical_not(np.in1d(all_surfaces, facesa))
        facesa = list(all_surfaces[mask])
        
    facesa.extend(viewer._mhidden_face)

    for name, obj in ax.get_children():
        if name.startswith('face') and not name.endswith('meshed'):
            h = list(np.unique(facesa + meshed_face))
            obj.hide_component(h)
        if name.startswith('face') and  name.endswith('meshed'):
            h = list(np.unique(facesa))
            obj.hide_component(h)
    '''
    if 'line' in cells:
        vert = np.squeeze(X[cells['line']][:,0,:])
        if vert.size > 3:
            obj= viewer.plot(vert[:,0],
                    vert[:,1],
                    vert[:,2], 'ob',
                    array_idx = cell_data['line']['geometrical'],
                    linewidth = 0)
#                    view_offset = (0, 0, -0.005, 0))

            verts, elem_idx, array_idx = expand_vertex_data(X, cells['line'],
                                       cell_data['line']['geometrical'])

            obj.rename('edge_meshed')
            meshed_edge = list(np.unique(cell_data['line']['geometrical']))
    else:
        meshed_edge = []

    hide_edge_meshmode(viewer, meshed_edge)
    '''
    s, v = viewer._s_v_loop['mesh']
    edgesa = viewer._mhidden_edge
        
    for name, obj in ax.get_children():
        if name.startswith('edge') and not name.endswith('meshed'):
            h = list(np.unique(edgesa + meshed_edge))
            if obj.hasvar('idxset'): obj.hide_component(h)        
    '''
    viewer.set_sel_mode(viewer.get_sel_mode())


def hide_face_meshmode(viewer, meshed_face):
    ax =viewer.get_axes()    
    s, v = viewer._s_v_loop['mesh']
    facesa = []
    if v is not None and len(v)>0:  # in 3D starts with faces from shown volumes
        all_surfaces = np.array(list(s), dtype=int)
        for key in v:
            if not key in viewer._mhidden_volume:
                facesa.extend(v[key])
        facesa = np.unique(facesa)
        mask  = np.logical_not(np.in1d(all_surfaces, facesa))
        facesa = list(all_surfaces[mask])
        
    facesa.extend(viewer._mhidden_face)
        
    for name, obj in ax.get_children():
        if name.startswith('face') and not name.endswith('meshed'):
            h = list(np.unique(facesa + meshed_face))
            obj.hide_component(h)
        if name.startswith('face') and  name.endswith('meshed'):
            h = list(np.unique(facesa))
            obj.hide_component(h)

def hide_edge_meshmode(viewer, meshed_edge):
    ax =viewer.get_axes()    
    s, v = viewer._s_v_loop['mesh']
    edgesa = viewer._mhidden_edge
        
    for name, obj in ax.get_children():
        if name.startswith('edge') and not name.endswith('meshed'):
            h = list(np.unique(edgesa + meshed_edge))
            if obj.hasvar('idxset'): obj.hide_component(h)        
            
