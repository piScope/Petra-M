import wx
import numpy as np
import petram
from petram.utils import get_pkg_datafile

fdotbk = get_pkg_datafile(petram.pi, 'icon',  'dot_bk.png')
fedgebk = get_pkg_datafile(petram.pi, 'icon', 'line_bk.png')
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
    
    namestart = mode if mode != 'volume' else 'face'
    namestart2 =  namestart + '_meshed'    
    objs = [child for name, child in ax.get_children() if name.startswith(namestart)
            and not name.startswith(namestart2)]
    meshed_objs = [child for name, child in ax.get_children() if name.startswith(namestart2)]
    
    if mode == 'volume' or mode == 'face' or mode == 'edge':
        idx = []
        for o in meshed_objs:
            o.hide_component([])
            idx = idx + list(np.unique(o.getvar('array_idx')))
        for o in objs:            
            o.hide_component(idx)        
    elif mode == 'point':
        ax.point.hide_component([])                        
    else:
        pass
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

    sel = viewer.canvas.selection
    if len(sel) != 1:  return
    sel_objs = [s().figobj for s in sel]
        
    if mode == 'volume':
        facesa = []
        facesb = []        
        s, v = viewer._s_v_loop['mesh']
        selected_volume = viewer._dom_bdr_sel        
        for key in v.keys():
            if key in selected_volume:
                facesa.extend(v[key])
            else:
                facesb.extend(v[key])
        if inverse:
            for o in objs:                         
                o.hide_component(facesa, inverse=True)
            hidden_volume = [x for x in v.keys() if not x
                             in selected_volume]            
            viewer._hidden_volume = hidden_volume
        else:
            facesa = np.unique(np.array(facesa))
            facesb = np.unique(np.array(facesb))
            new_hide = list(np.setdiff1d(facesa, facesb, True))
            for o in objs:                                     
                idx = o.hidden_component
                idx = list(set(idx+new_hide))
                o.hide_component(idx)
                
    elif mode == 'face' or mode == 'edge':
        for o in objs + meshed_objs:
            if not o in sel_objs: continue
            idx = o.getSelectedIndex()
            idx = list(set(o.hidden_component+idx))        
            o.hide_component(idx, inverse=inverse)        
    elif mode == 'point':
        pass
    else:
        pass
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
         ('toggledot',    fdotbk,  0, 'toggle vertex', toggle_dot),
         ('toggleedge',   fedgebk, 0, 'toggle edge', toggle_edge),
         ('---', None, None, None),
         ('mshow',   showall,  0, 'show all', show_all),
         ('mhide',   hide,  0, 'hide selection', hide_elem),        
         ('mshowonly',  fshow,  0, 'show only', show_only),    
         ('---', None, None, None),
         ('solid',  fsolid,  0, 'solid', make_solid),
         ('transpaent',  ftrans,  0, 'transparent', make_transp),]
            
