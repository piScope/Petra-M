import wx
import numpy as np
from collections import defaultdict

import petram
from petram.utils import get_pkg_datafile
#import petram.geom

fdotbk = get_pkg_datafile(petram.pi, 'icon',  'dot_bk.png')
fedgebk = get_pkg_datafile(petram.pi, 'icon', 'line_bk.png')
ffacebk = get_pkg_datafile(petram.pi, 'icon', 'face_bk.png')
fdot = get_pkg_datafile(petram.pi, 'icon',  'dot.png')
fedge = get_pkg_datafile(petram.pi, 'icon', 'line.png')
fface = get_pkg_datafile(petram.pi, 'icon', 'face.png')
fdom = get_pkg_datafile(petram.pi, 'icon', 'domain.png')
fshow = get_pkg_datafile(petram.pi, 'icon', 'show.png')
fshowall = get_pkg_datafile(petram.pi, 'icon', 'showall.png')
fhide = get_pkg_datafile(petram.pi, 'icon', 'hide.png')
fsolid = get_pkg_datafile(petram.pi, 'icon', 'solid.png')
ftrans = get_pkg_datafile(petram.pi, 'icon', 'transparent.png')


def _select_x(evt, mode, mask):
    viewer = evt.GetEventObject().GetTopLevelParent()
    viewer._sel_mode = mode
    viewer.canvas.unselect_all()
    viewer.set_picker_mask(mask)
    for name, child in viewer.get_axes().get_children():
        if hasattr(child, 'setSelectedIndesx'):
            child.setSelectedIndex([])
    viewer.canvas.unselect_all()
    viewer.draw()


def select_dot(evt):
    _select_x(evt, 'point', 'point')


def select_edge(evt):
    _select_x(evt, 'edge', 'edge')


def select_face(evt):
    _select_x(evt, 'face', 'face')


def select_volume(evt):
    _select_x(evt, 'volume', 'face')


def refresh(navibar, btnls):
    viewer = navibar.GetTopLevelParent()
    mode = viewer._sel_mode
    for btnl in btnls:
        if isinstance(btnl, str):
            continue
        if btnl.tg == 2:
            if btnl.btask == mode:
                btnl.SetToggled(True)
                btnl.SetBitmap(btnl.bitmap2)
            else:
                btnl.SetToggled(False)
                btnl.SetBitmap(btnl.bitmap1)


def show_all(evt):
    viewer = evt.GetEventObject().GetTopLevelParent()
    mode = viewer._sel_mode
    ax = viewer.get_axes()

    namestart = mode if mode != 'volume' else 'face'
    for name, child in ax.get_children():
        if name.startswith('point'):
            continue
        if name.startswith('face'):
            child.hide_component([])
        elif name.startswith('edge'):
            child.hide_component([])

    viewer._hidden_volume = []
    viewer._hidden_face = []
    viewer._hidden_edge = []
    viewer._dom_bdr_sel = ([], [], [], [])
    viewer.draw_all()


def hide_elem(evt, inverse=False):
    viewer = evt.GetEventObject().GetTopLevelParent()
    mode = viewer._sel_mode

    ax = viewer.get_axes()
    namestart = mode if mode != 'volume' else 'face'
    objs = [child for name, child in ax.get_children()
            if name.startswith(namestart)]
    Eobjs = [child for name, child in ax.get_children()
             if name.startswith('edge')]
    Fobjs = [child for name, child in ax.get_children()
             if name.startswith('face')]
    s2l, v2s = viewer._s_v_loop['phys']

    if mode == 'volume':
        facesa = []
        facesb = []
        selected_volume = viewer._dom_bdr_sel[0]

        target_volumes = selected_volume
        if not inverse:
            target_volumes.extend(viewer._hidden_volume)
        for key in v2s.keys():
            if key in target_volumes:
                facesa.extend(v2s[key])
            else:
                facesb.extend(v2s[key])
        if inverse:
            for o in objs:
                o.hide_component(facesa, inverse=True)
            hidden_volume = [x for x in v2s.keys() if not x in selected_volume]
            viewer._hidden_volume = hidden_volume
        else:
            facesa = np.unique(np.array(facesa))
            facesb = np.unique(np.array(facesb))
            new_hide = list(np.setdiff1d(facesa, facesb, True))
            for o in objs:
                idx = o.hidden_component
                idx = list(set(idx+new_hide))
                o.hide_component(idx)

            viewer._hidden_volume.extend(selected_volume)
            viewer._hidden_volume = list(set(viewer._hidden_volume))

    elif mode == 'face' or mode == 'edge':
        for o in objs:
            idx = o.getSelectedIndex()
            if inverse:
                idx = list(set(idx))
            else:
                idx = list(set(o.hidden_component+idx))
            o.hide_component(idx, inverse=inverse)
    elif mode == 'point':
        pass
    else:
        pass

    if mode == 'volume' or mode == 'face':
        from petram.mesh.mesh_utils import line2surf
        hidden_face = sum([o.hidden_component for o in Fobjs], [])
        l2s = line2surf(s2l)

        dd = defaultdict(int)
        for f in hidden_face:
            for x in s2l[f]:
                dd[x] = dd[x]+1
        dd = dict(dd)
        hide_this_edge = [x for x in dd if dd[x] == len(l2s[x])]

        # hide_this_edge = [l for l, ss in l2s.items()
        #              if np.intersect1d(ss, hidden_face).size==len(ss)]

        for o in Eobjs:
            #idx = o.getSelectedIndex()
            idx = list(set(o.hidden_component+hide_this_edge))
            o.hide_component(idx)

    viewer.canvas.unselect_all()
    viewer._dom_bdr_sel = ([], [], [], [])
    viewer.draw_all()


def show_only(evt):
    hide_elem(evt, inverse=True)


def toggle_dot(evt):
    viewer = evt.GetEventObject().GetTopLevelParent()
    mode = viewer._sel_mode

    ax = viewer.get_axes()
    if not ax.has_child('point'):
        return

    if ax.point.isSuppressed:
        ax.point.onUnSuppress()
    else:
        ax.point.onSuppress()
    wx.CallAfter(ax.point.set_gl_hl_use_array_idx, True)
    viewer.canvas.unselect_all()
    # viewer.draw_all()


def toggle_edge(evt):
    viewer = evt.GetEventObject().GetTopLevelParent()
    mode = viewer._sel_mode

    ax = viewer.get_axes()
    if not ax.has_child('edge'):
        return

    if ax.edge.isSuppressed:
        ax.edge.onUnSuppress()
    else:
        ax.edge.onSuppress()

    wx.CallAfter(ax.edge.set_gl_hl_use_array_idx, True)
    viewer.canvas.unselect_all()
    # viewer.draw_all()


def toggle_face(evt):
    viewer = evt.GetEventObject().GetTopLevelParent()
    mode = viewer._sel_mode

    ax = viewer.get_axes()
    children = [(name, child)
                for name, child in ax.get_children() if name.startswith('face')]
    isSuppressed = any([c.isSuppressed for n, c in children])

    if isSuppressed:
        for n, c in children:
            c.onUnSuppress()
            wx.CallAfter(c.set_gl_hl_use_array_idx, True)
    else:
        for n, c in children:
            c.onSuppress()
            wx.CallAfter(c.set_gl_hl_use_array_idx, True)

    viewer.canvas.unselect_all()
    wx.CallAfter(viewer.canvas.refresh_hl)


def make_solid(evt):
    viewer = evt.GetEventObject().GetTopLevelParent()
    mode = viewer._sel_mode

    ax = viewer.get_axes()
    for name, child in ax.get_children():
        if len(child._artists) > 0:
            child.set_alpha(1.0, child._artists[0])
    viewer.canvas.unselect_all()
    viewer.draw_all()


def make_transp(evt):
    viewer = evt.GetEventObject().GetTopLevelParent()
    mode = viewer._sel_mode

    ax = viewer.get_axes()
    for name, child in ax.get_children():
        if len(child._artists) > 0:
            child.set_alpha(0.75, child._artists[0])
    viewer.canvas.unselect_all()
    viewer.draw_all()


btask = [
    ('dot',    fdot,  2, 'select vertex', select_dot),
    ('edge',   fedge, 2, 'select edge', select_edge),
    ('face',   fface, 2, 'select face', select_face),
    ('domain', fdom,  2, 'select domain', select_volume),
    ('---', None, None, None),
    ('toggledot',    fdotbk,  0, 'toggle vertex', toggle_dot),
    ('toggleedge',   fedgebk, 0, 'toggle edge', toggle_edge),
    ('toggleface',   ffacebk, 0, 'toggle face', toggle_face),
    ('---', None, None, None),
    ('mshow',  fshowall,  0, 'show all', show_all),
    ('mhide',  fhide,  0, 'hide selection', hide_elem),
    ('mshowonly',  fshow,  0, 'show only', show_only),
    ('---', None, None, None),
    ('solid',  fsolid,  0, 'solid', make_solid),
    ('transpaent',  ftrans,  0, 'transparent', make_transp), ]
