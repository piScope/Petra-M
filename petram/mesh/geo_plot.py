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
            yield (v, idx[uc[i]:uc[i + 1]])

    for kk, idx in iter_unique_idx(element_id):
        iverts = vertex_idx[idx].flatten()
        iv, idx = np.unique(iverts, return_inverse=True)

        verts.append(X[iv])
        iele.append(idx.reshape(-1, nel) + k)
        k = k + len(iv)
        iarr.append(np.zeros(len(iv)) + kk)

    array_idx = np.hstack(iarr).astype(int)
    elem_idx = np.vstack(iele).astype(int)
    verts = np.vstack(verts)

    return verts, elem_idx, array_idx


# we visit nodes in counter clock wise.
gmsh_ho_el_idx = {'triangle6': ([0, 3, 5], [3, 1, 4], [5, 3, 4], [5, 4, 2]),
                  'triangle10': ([0, 3, 8], [8, 3, 9], [3, 4, 9], [9, 4, 5], [1, 5, 4],
                                 [7, 8, 9], [7, 9, 6], [6, 9, 5], [2, 7, 6]),
                  'triangle15': ([0, 3, 11], [11, 3, 12], [12, 3, 4], [4, 13, 12], [13, 4, 5],
                                 [13, 5, 6], [6, 5, 1], [
                                     10, 11, 12], [10, 12, 14],
                                 [14, 12, 13], [14, 13, 7], [
                                     7, 13, 6], [9, 10, 14],
                                 [9, 14, 8], [8, 14, 7], [2, 9, 8]),
                  'triangle21': ([0, 3, 14], [3, 15, 14], [3, 4, 15], [4, 18, 15],
                                 [4, 5, 18], [5, 16, 18], [
                                     5, 6, 16], [6, 7, 16],
                                 [6, 1, 7], [13, 14, 15], [13, 15, 20],
                                 [15, 18, 20], [18, 19, 20], [
                                     18, 16, 19], [16, 8, 19],
                                 [16, 7, 8], [12, 13, 20], [
                                     20, 17, 12], [20, 19, 17],
                                 [19, 9, 17], [19, 8, 9], [
                                     11, 12, 17], [11, 17, 10],
                                 [10, 17, 9], [2, 11, 10]),

                  'quad9': ([0, 4, 7], [7, 4, 8], [8, 4, 1], [8, 1, 5], [3, 7, 8],
                            [3, 8, 6], [6, 8, 5], [6, 5, 2],),
                  'quad16': ([0, 4, 11], [11, 4, 12], [4, 5, 12], [12, 5, 13], [5, 1, 13],
                             [13, 1, 6], [11, 12, 10], [10, 12, 15], [
                                 12, 13, 15], [15, 13, 14],
                             [13, 6, 14], [14, 6, 7], [10, 15, 3], [
                                 3, 15, 9], [15, 14, 9],
                             [9, 14, 8], [14, 7, 8], [8, 7, 2]),
                  'quad25': ([0, 4, 15], [15, 4, 16], [4, 5, 16], [16, 5, 20], [5, 6, 20],
                             [20, 6, 17], [6, 1, 17], [17, 1, 7], [
                                 15, 16, 14], [14, 16, 23],
                             [16, 20, 23], [23, 20, 24], [
                                 20, 17, 24], [24, 17, 21], [17, 7, 21],
                             [21, 7, 8], [14, 23, 13], [13, 23, 19], [
                                 23, 24, 19], [19, 24, 22],
                             [24, 21, 22], [22, 21, 18], [
                                 21, 8, 18], [18, 8, 9], [13, 19, 3],
                             [3, 19, 12], [19, 22, 12], [
                                 12, 22, 11], [22, 18, 11],
                             [11, 18, 10], [18, 9, 10], [10, 9, 2]),
                  'quad36': ([0, 4, 19], [19, 4, 20], [4, 5, 20], [20, 5, 24], [5, 6, 24],
                             [24, 6, 25], [6, 7, 25], [
                                 25, 7, 21], [7, 1, 21], [21, 1, 8],
                             [19, 20, 18], [18, 20, 31], [20, 24, 31], [
                                 31, 24, 32], [24, 25, 32],
                             [32, 25, 33], [25, 21, 33], [
                                 33, 21, 26], [21, 8, 26], [26, 8, 9],
                             [18, 31, 17], [17, 31, 30], [31, 32, 30], [
                                 30, 32, 35], [32, 33, 35],
                             [35, 33, 34], [33, 26, 34], [
                                 34, 26, 27], [26, 9, 27], [27, 9, 10],
                             [17, 30, 16], [16, 30, 23], [30, 35, 23], [
                                 23, 35, 29], [35, 34, 29],
                             [29, 34, 28], [34, 27, 28], [28, 27, 22], [
                                 27, 10, 22], [22, 10, 11],
                             [16, 23, 3], [3, 23, 15], [23, 29, 15], [
                                 15, 29, 14], [29, 28, 14],
                             [14, 28, 13], [28, 22, 13], [13, 22, 12], [22, 11, 12], [12, 11, 2]),
                  }

gmsh_ho_bel_idx = {'triangle6': ([0, 3], [3, 1], [1, 4], [4, 2], [2, 5], [5, 0]),
                   'triangle10': ([0, 3], [3, 4], [4, 1],
                                  [1, 5], [5, 6], [6, 2],
                                  [2, 7], [7, 8], [8, 0]),
                   'triangle15': ([0, 3], [3, 4], [4, 5], [5, 1],
                                  [1, 6], [6, 7], [7, 8], [8, 2],
                                  [2, 9], [9, 10], [10, 11], [11, 0]),
                   'triangle21': ([0, 3], [3, 4], [4, 5], [5, 6], [6, 1],
                                  [1, 7], [7, 8], [8, 9], [9, 10], [10, 2],
                                  [2, 11], [11, 12], [12, 13], [13, 14], [14, 0]),
                   'quad9': ([0, 4], [4, 1], [1, 5], [5, 2], [2, 6], [6, 3],
                             [3, 7], [7, 0]),
                   'quad16': ([0, 4], [4, 5], [5, 1], [1, 6], [6, 7], [7, 2],
                              [2, 8], [8, 9], [9, 3], [3, 10], [10, 11], [11, 0]),
                   'quad25': ([0, 4], [4, 5], [5, 6], [6, 1],
                              [1, 7], [7, 8], [8, 9], [9, 2],
                              [2, 10], [10, 11], [11, 12], [12, 3],
                              [3, 13], [13, 14], [14, 15], [15, 0]),
                   'quad36': ([0, 4], [4, 5], [5, 6], [6, 7], [7, 1],
                              [1, 8], [8, 9], [8, 10], [10, 11], [11, 2],
                              [2, 12], [12, 13], [13, 14], [14, 15], [15, 3],
                              [3, 16], [16, 17], [17, 18], [18, 19], [19, 0],)}


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
            yield (v, idx[uc[i]:uc[i + 1]])

    for kk, idx in iter_unique_idx(element_id):
        iverts = vertex_idx[idx].flatten()
        iv, idx = np.unique(iverts, return_inverse=True)

        verts.append(X[iv])
        tmp = idx.reshape(-1, nel) + k

        el = [tmp[:, x] for x in gmsh_ho_el_idx[el_type]]
        bel = [tmp[:, x] for x in gmsh_ho_bel_idx[el_type]]
        elem_idx = np.vstack(el)
        ed_idx = np.vstack(bel)

        iele.append(elem_idx)
        iedge.append(ed_idx)
        k = k + len(iv)
        iarr.append(np.zeros(len(iv)) + kk)

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


def plot_geometry(viewer, ret, geo_phys='geometrical', lw=0):
    '''
    plot_geometry
    '''
    viewer.cls()
    viewer.set_hl_color((1, 0, 0))

    X, cells, pt_data, cell_data, field_data = ret

    if 'triangle_x' in cells or 'quad_x' in cells:
        # merge two cases as triangle data
        verts = cell_data['X_refined_face']

        elem_idx0 = []
        eelem_idx0 = []
        array_idx = np.zeros(len(verts), dtype=int)

        for n in ['triangle_x', 'quad_x']:
            if not n in cells:
                continue
            xxx = cells[n]
            if n == 'quad_x':
                elem_idx0.append(xxx[:, :3])
                elem_idx0.append(xxx[:, [0, 2, 3]])
            else:
                elem_idx0.append(xxx)
            eelem_idx0.append(cells[n + 'e'])
            idx = cell_data[n][geo_phys]
            array_idx[np.unique(cells[n])] = idx

        elem_idx = np.vstack(elem_idx0)
        eelem_idx = np.vstack(eelem_idx0)

        call_solid1(viewer, 'face_t', verts, elem_idx, array_idx, lw,
                    edge_idx=eelem_idx)

    elif 'triangle' in cells or 'quad' in cells:
        verts0 = []
        elem_idx0 = []
        eelem_idx0 = []
        array_idx0 = []

        l_verts = 0
        for n in ['triangle', 'quad']:
            if not n in cells:
                continue
            verts, elem_idx, array_idx = expand_vertex_data(X, cells[n],
                                                            cell_data[n][geo_phys])
            verts0.append(verts)
            xxx = elem_idx if l_verts == 0 else elem_idx + l_verts

            if n == 'quad':
                elem_idx0.append(xxx[:, :3])
                elem_idx0.append(xxx[:, [0, 2, 3]])
                eelem_idx0.extend([xxx[:, :2],
                                   xxx[:, 1:3],
                                   xxx[:, -2:],
                                   xxx[:, [0, -1]]])
            else:
                elem_idx0.append(xxx)
                eelem_idx0.extend([xxx[:, :2],
                                   xxx[:, -2:],
                                   xxx[:, [0, 2]]])
            l_verts = l_verts + len(verts)
            array_idx0.append(array_idx)

        verts = np.vstack(verts0)
        elem_idx = np.vstack(elem_idx0)
        array_idx = np.hstack(array_idx0)
        eelem_idx = np.vstack(eelem_idx0)

        call_solid1(viewer, 'face_t', verts, elem_idx, array_idx, lw,
                    edge_idx=eelem_idx)

    '''
    if 'triangle_x' in cells:
        elem_idx = cells['triangle_x']
        array_idx = cell_data['triangle_x'][geo_phys]
        eelem_idx = cells['triangle_xe']

        array_idx = np.zeros(len(verts))


        array_idx[np.unique(elem_idx.flatten()] =

        ### to handle mixed mesh, where verts contains vertices for tri and quad
        ar, idx = np.unique(np.hstack([elem_idx.flatten(),
                                       eelem_idx.flatten()]), return_inverse=True)
        elem_idx = idx[:np.prod(elem_idx.shape)].reshape(elem_idx.shape)
        eelem_idx = idx[np.prod(elem_idx.shape):].reshape(eelem_idx.shape)
        verts = verts[ar]


        call_solid1(viewer, 'face_t', verts, elem_idx, array_idx, lw,
                    edge_idx=eelem_idx)
    if 'quad_x' in cells:
        verts = cell_data['X_refined_face']
        elem_idx = cells['quad_x']
        array_idx = cell_data['quad_x'][geo_phys]
        eelem_idx = cells['quad_xe']

        ### to handle mixed mesh, where verts contains vertices for tri and quad
        ar, idx = np.unique(np.hstack([elem_idx.flatten(),
                                       eelem_idx.flatten()]), return_inverse=True)
        elem_idx = idx[:np.prod(elem_idx.shape)].reshape(elem_idx.shape)
        eelem_idx = idx[np.prod(elem_idx.shape):].reshape(eelem_idx.shape)
        verts = verts[ar]

        call_solid1(viewer, 'face_r', verts, elem_idx, array_idx, lw,
                    edge_idx=eelem_idx)
    '''

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
        if len(vidx) > 0:
            vert = np.atleast_2d(np.squeeze(X[vidx]))
        #if len(vert) > 0:
            x = vert[:, 0]
            if vert.shape[-1] > 1:
                y = vert[:, 1]
            else:
                y = x*0
            if vert.shape[-1] > 2:
                z = vert[:, 2]
            else:
                z = y*0
            obj = viewer.plot(x, y, z, 'ok',
                              array_idx=aidx,
                              linewidth=0)
            obj.rename('point')
            obj.set_gl_hl_use_array_idx(True)
    viewer.set_sel_mode(viewer.get_sel_mode())


def oplot_meshed(viewer, ret):
    print("oplot_meshed")
    ax = viewer.get_axes()
    for name, obj in ax.get_children():
        if name.endswith('_meshed'):
            viewer.cls(obj=obj)

    try:
        X, cells, pt_data, cell_data, field_data = ret
    except ValueError:
        return

    meshed_face = []
    if 'triangle' in cells:
        verts, elem_idx, array_idx = expand_vertex_data(X, cells['triangle'],
                                                        cell_data['triangle']['geometrical'])

        obj = viewer.solid(verts, elem_idx,
                           array_idx=array_idx,
                           facecolor=(0.7, 0.7, 0.7, 1.0),
                           edgecolor=(0, 0, 0, 1),
                           linewidth=1,
                           view_offset=(0, 0, -0.0005, 0))

        obj.rename('face_t_meshed')
        obj.set_gl_hl_use_array_idx(True)

        meshed_face.extend(
            list(
                np.unique(
                    cell_data['triangle']['geometrical'])))

    def handle_gmsh_ho_elements(el):
        elem_idx = cells[el]
        array_idx = cell_data[el]['geometrical']
        verts, elem_idx, edge_idx, array_idx = expand_vertex_data_gmsh_ho(
            X, elem_idx, array_idx, el)

        obj = viewer.solid(verts, elem_idx,
                           array_idx=array_idx,
                           facecolor=(0.7, 0.7, 0.7, 1.0),
                           edgecolor=(0, 0, 0, 1),
                           linewidth=1,
                           view_offset=(0, 0, -0.0005, 0),
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

        # print verts.shape, elem_idx.shape, array_idx.shape
        obj = viewer.solid(verts, elem_idx,
                           array_idx=array_idx,
                           facecolor=(0.7, 0.7, 0.7, 1.0),
                           edgecolor=(0, 0, 0, 1),
                           #                           view_offset = (0, 0, -0.0005, 0),
                           linewidth=1,)

        obj.rename('face_r_meshed')
        obj.set_gl_hl_use_array_idx(True)
        meshed_face.extend(list(np.unique(cell_data['quad']['geometrical'])))

    if 'quad9' in cells:
        handle_gmsh_ho_elements('quad9')
    if 'quad16' in cells:
        handle_gmsh_ho_elements('quad16')
    if 'quad25' in cells:
        handle_gmsh_ho_elements('quad25')
    if 'quad36' in cells:
        handle_gmsh_ho_elements('quad36')

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
        vert = np.squeeze(X[cells['line']][:, 0, :])
        if vert.size > 3:
            obj = viewer.plot(vert[:, 0],
                              vert[:, 1],
                              vert[:, 2], 'ob',
                              array_idx=cell_data['line']['geometrical'],
                              linewidth=0)
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
    ax = viewer.get_axes()
    s, v = viewer._s_v_loop['mesh']
    facesa = []
    if v is not None and len(
            v) > 0:  # in 3D starts with faces from shown volumes
        all_surfaces = np.array(list(s), dtype=int)
        for key in v:
            if not key in viewer._mhidden_volume:
                facesa.extend(v[key])
        facesa = np.unique(facesa)
        mask = np.logical_not(np.in1d(all_surfaces, facesa))
        facesa = list(all_surfaces[mask])

    facesa.extend(viewer._mhidden_face)

    for name, obj in ax.get_children():
        if name.startswith('face') and not name.endswith('meshed'):
            h = list(np.unique(facesa + meshed_face))
            obj.hide_component(h)
        if name.startswith('face') and name.endswith('meshed'):
            h = list(np.unique(facesa))
            obj.hide_component(h)


def hide_edge_meshmode(viewer, meshed_edge):
    ax = viewer.get_axes()
    s, v = viewer._s_v_loop['mesh']
    edgesa = viewer._mhidden_edge

    for name, obj in ax.get_children():
        if name.startswith('edge') and not name.endswith('meshed'):
            h = list(np.unique(edgesa + meshed_edge))
            if obj.hasvar('idxset'):
                obj.hide_component(h)
