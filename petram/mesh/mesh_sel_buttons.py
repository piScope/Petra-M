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
    if mode == 'volume':
        if ax.has_child('face_meshed'):
            ax.face_meshed.hide_component([])
            idx = ax.face_meshed.getvar('array_idx')
            idx = list(np.unique(idx))
        else:
            idx = []
        ax.face.hide_component(idx)        
    elif mode == 'face':
        if ax.has_child('face_meshed'):        
            ax.face_meshed.hide_component([])
            idx = ax.face_meshed.getvar('array_idx')
            idx = list(np.unique(idx))
        else:
            idx = []
        ax.face.hide_component(idx)        
    elif mode == 'edge':
        if ax.has_child('edge_meshed'):                
            #ax.edge_meshed.hide_component([])
            idx = ax.face_meshed.getvar('array_idx')
            idx = list(np.unique(idx))
        else:
            idx = []
        ax.edge.hide_component(idx)        
    elif mode == 'point':
        ax.point.hide_component([])                        
    else:
        pass
    viewer.draw_all()    

def hide_elem(evt, inverse=False):
    viewer = evt.GetEventObject().GetTopLevelParent()
    mode = viewer._sel_mode

    def hide(ax, name, inverse=False):
        if not ax.has_child(name): return
        obj = ax.get_child(name = name)
        idx = obj.getSelectedIndex()
        idx = list(set(obj.hidden_component+idx))        
        obj.hide_component(idx, inverse=inverse)        
        

    ax = viewer.get_axes()
    sel = viewer.canvas.selection
    if len(sel) != 1:  return
    names = [s().figobj.name for s in sel]
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
            ax.face.hide_component(facesa, inverse=True)
            hidden_volume = [x for x in v.keys() if not x
                             in selected_volume]            
            viewer._hidden_volume = hidden_volume
        else:
            facesa = np.unique(np.array(facesa))
            facesb = np.unique(np.array(facesb))
            new_hide = list(np.setdiff1d(facesa, facesb, True))
            idx = ax.face.hidden_component
            idx = list(set(idx+new_hide))
            ax.face.hide_component(idx)        
    elif mode == 'face':
        if 'face' in names: hide(ax, 'face', invserse=inverse)
        if 'face_meshed' in names: hide(ax, 'face_meshed',
                                        inverse=inverse)                
    elif mode == 'edge':
        if 'edge' in names: hide(ax, 'edge', inverse=inverse)
        if 'edge_meshed' in names: hide(ax, 'edge_meshed',
                                        inverse=inverse)                
    elif mode == 'point':
        pass
    else:
        pass
    viewer.canvas.unselect_all()    
    viewer.draw_all()
    
def show_only(evt):    
    hide_elem(evt, inverse=True)

def toggle_dot(evt):
    viewer = evt.GetEventObject().GetTopLevelParent()
    mode = viewer._sel_mode

    ax = viewer.get_axes()
    if ax.has_child('point'):
       if ax.point.isSuppressed:
           ax.point.onUnSuppress()
       else:
           ax.point.onSuppress()        
    viewer.canvas.unselect_all()
    viewer.draw_all()
    
def toggle_edge(evt):
    viewer = evt.GetEventObject().GetTopLevelParent()
    mode = viewer._sel_mode

    ax = viewer.get_axes()
    if ax.has_child('edge'): 
        if ax.edge.isSuppressed:
            ax.edge.onUnSuppress()
        else:
            ax.edge.onSuppress()
    if ax.has_child('edge_meshed'): 
        if ax.edge_meshed.isSuppressed:
            ax.edge_meshed.onUnSuppress()
        else:
            ax.edge_meshed.onSuppress()        
    viewer.canvas.unselect_all()
    viewer.draw_all()
     
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
            
