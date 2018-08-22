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
    for kk in np.unique(element_id):
        idx = np.where(element_id == kk)[0]

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

def call_solid1(viewer, name, verts, elem_idx, array_idx, lw):
    # template for faces    
    obj = viewer.solid(verts, elem_idx,
                      array_idx = array_idx,
                      facecolor = (0.7, 0.7, 0.7, 1.0),
                      linewidth = lw)

    obj.rename(name)
    obj._artists[0].set_gl_hl_use_array_idx(True)
    return obj

def call_solid2(viewer, name, verts, elem_idx, array_idx=None, lw=1.5):
    # template for lines
    obj = viewer.solid(verts, elem_idx,
                      array_idx = array_idx,
                      linewidth = lw,
                      facecolor = (0, 0, 0, 1.0),
                      edgecolor = (0, 0, 0, 1.0),
#                           view_offset = (0, 0, -0.001, 0),
                      draw_last = True)
    obj.rename(name)
    obj._artists[0].set_gl_hl_use_array_idx(True)
    return obj

def plot_geometry(viewer,  ret,  geo_phys = 'geometrical', lw = 0):

    viewer.cls()
    viewer.set_hl_color((1,0,0))
    
    X, cells, pt_data, cell_data, field_data = ret

    if 'triangle_x' in cells:
        verts = cell_data['X_refined_face']
        elem_idx = cells['triangle_x']
        array_idx = cell_data['triangle_x'][geo_phys]
        call_solid1(viewer, 'face_t', verts, elem_idx, array_idx, lw)
        eelem_idx = cells['triangle_xe']        
        call_solid2(viewer, 'face_te', verts, eelem_idx, lw=0)
        
    elif 'triangle' in cells:
        verts, elem_idx, array_idx = expand_vertex_data(X, cells['triangle'],
                                       cell_data['triangle'][geo_phys])

        call_solid1(viewer, 'face_t', verts, elem_idx, array_idx, lw)

    if 'quad_x' in cells:
        verts = cell_data['X_refined_face']
        elem_idx = cells['quad_x']
        array_idx = cell_data['quad_x'][geo_phys]
        call_solid1(viewer, 'face_r', verts, elem_idx, array_idx, lw)
        eelem_idx = cells['quad_xe']        
        call_solid2(viewer, 'face_re', verts, eelem_idx, lw = 0)
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
        vert = np.atleast_2d(np.squeeze(X[cells['vertex']]))
        if len(vert) > 0:
            obj= viewer.plot(vert[:,0],
                             vert[:,1],
                             vert[:,2], 'ok',
                             array_idx = cell_data['vertex'][geo_phys],
                             linewidth = 0)
            obj.rename('point')
            obj._artists[0].set_gl_hl_use_array_idx(True)
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


        #print verts.shape, elem_idx.shape, array_idx.shape
        obj = viewer.solid(verts, elem_idx,
                           array_idx = array_idx,
                           facecolor = (0.7, 0.7, 0.7, 1.0),
                           edgecolor = (0, 0, 0, 1),
                           linewidth = 1,
                           view_offset = (0, 0, -0.0005, 0))

        obj.rename('face_t_meshed')
        obj._artists[0].set_gl_hl_use_array_idx(True)

        meshed_face.extend(list(np.unique(cell_data['triangle']['geometrical'])))
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
        obj._artists[0].set_gl_hl_use_array_idx(True)
        meshed_face.extend(list(np.unique(cell_data['quad']['geometrical'])))

    for name, obj in ax.get_children():
        if name.startswith('face') and not name.endswith('meshed'):
            h = list(np.unique(obj.hidden_component + meshed_face))
            obj.hide_component(h)        
    
    if 'line' in cells:

        vert = np.squeeze(X[cells['line']][:,0,:])
        obj= viewer.plot(vert[:,0],
                    vert[:,1],
                    vert[:,2], 'ob',
                    array_idx = cell_data['line']['geometrical'],
                    linewidth = 0)
#                    view_offset = (0, 0, -0.005, 0))

        verts, elem_idx, array_idx = expand_vertex_data(X, cells['line'],
                                       cell_data['line']['geometrical'])

        obj.rename('edge_meshed')
        '''
        obj._artists[0].set_gl_hl_use_array_idx(True)        
        #print verts.shape, elem_idx.shape, array_idx.shape
        obj = viewer.solid(verts, elem_idx,
                           array_idx = array_idx,
                           facecolor = (0.7, 0.7, 0.7, 1.0),
                           edgecolor = (0, 0, 0, 1),
                           linewidth = 1,
                           view_offset = (0, 0, -0.0005, 0))

        obj.rename('edge_meshed')
        obj._artists[0].set_gl_hl_use_array_idx(True)
        '''
        meshed_edge = list(np.unique(cell_data['line']['geometrical']))
    else:
        meshed_edge = []
        
    for name, obj in ax.get_children():
        if name.startswith('edge') and not name.endswith('meshed'):
            h = list(np.unique(obj.hidden_component + meshed_edge))
            if obj.hasvar('idxset'): obj.hide_component(h)        

    viewer.set_sel_mode(viewer.get_sel_mode())

    

