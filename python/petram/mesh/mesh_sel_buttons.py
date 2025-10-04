import wx
import numpy as np
from collections import defaultdict

import petram
from petram.utils import get_pkg_datafile

fdotbk = get_pkg_datafile(petram.pi, 'icon',  'dot_bk.png')
fedgebk = get_pkg_datafile(petram.pi, 'icon', 'line_bk.png')
ffacebk = get_pkg_datafile(petram.pi, 'icon', 'face_bk.png')
fdot = get_pkg_datafile(petram.pi, 'icon',  'dot.png')
fedge = get_pkg_datafile(petram.pi, 'icon', 'line.png')
fface = get_pkg_datafile(petram.pi, 'icon', 'face.png')
fdom = get_pkg_datafile(petram.pi, 'icon', 'domain.png')
showall = get_pkg_datafile(petram.pi, 'icon', 'showall.png')
fshow = get_pkg_datafile(petram.pi, 'icon', 'show.png')
hide = get_pkg_datafile(petram.pi, 'icon', 'hide.png')
fsolid = get_pkg_datafile(petram.pi, 'icon', 'solid.png')
ftrans = get_pkg_datafile(petram.pi, 'icon', 'transparent.png')


from petram.pi.sel_buttons import _select_x

def select_dot(evt):
    _select_x(evt, 'point', 'point')
    
def select_edge(evt):
    _select_x(evt, 'edge', 'edge')
    
def select_face(evt):
    _select_x(evt, 'face', 'face')
    
def select_volume(evt):
    _select_x(evt, 'volume', 'face')
    
def show_all(evt):
    viewer = evt.GetEventObject().GetTopLevelParent()
    mode = viewer._sel_mode

    ax = viewer.get_axes()
    
    Fobjs = [child for name, child in ax.get_children() if name.startswith("face")
            and not name.endswith("meshed")]
    mFobjs = [child for name, child in ax.get_children() if name.startswith("face")
            and name.endswith("meshed")]
    Eobjs = [child for name, child in ax.get_children() if name.startswith("edge")
            and not name.endswith("meshed")]
    mEobjs = [child for name, child in ax.get_children() if name.startswith("edge")
            and name.endswith("meshed")]


    def show_all_obj(objs, meshed_objs):
        idx = []        
        for o in meshed_objs:
            if o.hasvar('idxset'):
                o.hide_component([])
            idx = idx + list(np.unique(o.getvar('array_idx')))
        for o in objs:
            if o.hasvar('idxset'):            
                o.hide_component(idx)        
    
    if mode == 'volume':
         show_all_obj(Fobjs, mFobjs)
         show_all_obj(Eobjs, mEobjs)         
    elif mode == 'face':
         show_all_obj(Fobjs, mFobjs)
         show_all_obj(Eobjs, mEobjs)         
    elif mode == 'edge':
         show_all_obj(Fobjs, mFobjs)
         show_all_obj(Eobjs, mEobjs)         
    elif mode == 'point':
        if ax.point.hasvar('idxset'):            
             ax.point.hide_component([])                        
    else:
        pass
    
    viewer._mhidden_volume = []
    viewer._mhidden_face = []
    viewer._mhidden_edge = []
    viewer._dom_bdr_sel  = ([], [], [], [])            
    
    viewer.draw_all()    

def hide_elem(evt, inverse=False):
    viewer = evt.GetEventObject().GetTopLevelParent()
    mode = viewer._sel_mode
    ax = viewer.get_axes()
    
    namestart = mode if mode != 'volume' else 'face'
    namestart2 =  namestart + '_meshed'    
    objs = [child for name, child in ax.get_children() if name.startswith(namestart)
            and not name.startswith(namestart2)]
    meshed_objs = [child for name, child in ax.get_children() if name.startswith(namestart2)]
    
    Eobjs = [child for name, child in ax.get_children() if name.startswith('edge')]
    Fobjs = [child for name, child in ax.get_children() if name.startswith('face')]

    sel = viewer.canvas.selection
    #if len(sel) != 1:  return
    sel_objs = [s().figobj for s in sel]

    s2l, v2s = viewer._s_v_loop['mesh']    
    if mode == 'volume':
        facesa = []
        facesb = []        
        
        selected_volume = viewer._dom_bdr_sel[0]
        target_volumes = selected_volume

        if not inverse:
            target_volumes.extend(viewer._mhidden_volume)
        for key in v2s.keys():
            if key in target_volumes:
                facesa.extend(v2s[key])
            else:
                facesb.extend(v2s[key])
        if inverse:
            for o in objs:                         
                o.hide_component(facesa, inverse=True)
            for o in meshed_objs:                         
                o.hide_component(facesa, inverse=True)
            hidden_volume = [x for x in v2s.keys() if not x
                             in selected_volume]            
            viewer._mhidden_volume = hidden_volume
        else:
            facesa = np.unique(np.array(facesa))
            facesb = np.unique(np.array(facesb))
            new_hide = list(np.setdiff1d(facesa, facesb, True))
            for o in objs:                                     
                idx = o.hidden_component
                idx = list(set(idx+new_hide))
                o.hide_component(idx)
            for o in meshed_objs:                                     
                idx = o.hidden_component
                idx = list(set(idx+new_hide))
                o.hide_component(idx)
            tmp = viewer._mhidden_volume
            tmp.extend(selected_volume)
            viewer._mhidden_volume = list(set(tmp))
    elif mode == 'face' or mode == 'edge':
        idx1 = []
        for o in objs + meshed_objs:
            idx1.extend(o.getSelectedIndex())
        for o in objs + meshed_objs:
            idx = list(set(o.hidden_component+idx1))        
            o.hide_component(idx, inverse=inverse)
        if mode == "face":
            if inverse:
                hidden_face = [x for x in s2l.keys() if not x
                                 in idx]
            else:
                hidden_face = idx
            viewer._mhidden_face = hidden_face
        if mode == "edge":
            if inverse:
                all_edges = list(np.unique(np.hstack(([s2l[x] for x in s2l.keys()]))))
                hidden_edge = [x for x in all_edges if not x in idx]
            else:
                hidden_edge = idx
            viewer._mhidden_edge = hidden_edge
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
        
        #hide_this_edge = [l for l, ss in l2s.items()
        #              if np.intersect1d(ss, hidden_face).size==len(ss)]

        for o in Eobjs:
            #idx = o.getSelectedIndex()
            idx = list(set(o.hidden_component+hide_this_edge))
            if o.hasvar('idxset'):            
                o.hide_component(idx)
    
    viewer.canvas.unselect_all()    
    viewer.draw_all()
    
def show_only(evt):    
    hide_elem(evt, inverse=True)

def _toggle_any(evt, txt):
    viewer = evt.GetEventObject().GetTopLevelParent()
    mode = viewer._sel_mode

    ax = viewer.get_axes()
    objs = [child for name, child in ax.get_children() if name.startswith(txt)]

    c_status = any([o.isSuppressed for o in objs])
    for o in objs:
        if c_status:
            o.onUnSuppress()
        else:
            o.onSuppress()
        wx.CallAfter(o.set_gl_hl_use_array_idx, True)
    viewer.canvas.unselect_all()
    evt.Skip()
    
def toggle_dot(evt):
    _toggle_any(evt, 'point')
def toggle_edge(evt):
    _toggle_any(evt, 'edge')
    
from petram.pi.sel_buttons import toggle_face    

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
    
btask = [('mdot',    fdot,  2, 'select vertex', select_dot),
         ('medge',   fedge, 2, 'select edge', select_edge),
         ('mface',   fface, 2, 'select face', select_face),
         ('mdomain', fdom,  2, 'select domain', select_volume),
         ('---', None, None, None),
         ('mtoggledot',    fdotbk,  0, 'toggle vertex', toggle_dot),
         ('mtoggleedge',   fedgebk, 0, 'toggle edge', toggle_edge),
         ('mtoggleface',   ffacebk, 0, 'toggle face', toggle_face),             
         ('---', None, None, None),
         ('mshow',   showall,  0, 'show all', show_all),
         ('mhide',   hide,  0, 'hide selection', hide_elem),        
         ('mshowonly',  fshow,  0, 'show only', show_only),    
         ('---', None, None, None),
         ('msolid',  fsolid,  0, 'solid', make_solid),
         ('mtranspaent',  ftrans,  0, 'transparent', make_transp),]
            
