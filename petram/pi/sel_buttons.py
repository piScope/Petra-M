import numpy as np

from petram.utils import get_pkg_datafile
import petram.geom

fdot = get_pkg_datafile(petram.pi, 'icon',  'dot.png')
fedge = get_pkg_datafile(petram.pi, 'icon', 'line.png')
fface = get_pkg_datafile(petram.pi, 'icon', 'face.png')
fdom = get_pkg_datafile(petram.pi, 'icon', 'domain.png')
fshow = get_pkg_datafile(petram.pi, 'icon', 'show.png')
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
    viewer.draw()

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
        if isinstance(btnl, str): continue
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
    if mode == 'volume':
        ax.face.hide_component([])
    elif mode == 'face':
        ax.face.hide_component([])        
    elif mode == 'edge':
        ax.edge.hide_component([])                
    else:
        pass
    viewer.draw_all()    

def hide_elem(evt):
    viewer = evt.GetEventObject().GetTopLevelParent()
    mode = viewer._sel_mode

    ax = viewer.get_axes()
    if mode == 'volume':
        facesa = []
        facesb = []        
        s, v = viewer._s_v_loop['phys']
        selected_volume = viewer._dom_bdr_sel
        for key in v.keys():
            if key in selected_volume:
                facesa.extend(v[key])
            else:
                facesb.extend(v[key])
        facesa = np.unique(np.array(facesa))
        facesb = np.unique(np.array(facesb))
        new_hide = list(np.setdiff1d(facesa, facesb, True))
        idx = ax.face.hidden_component
        idx = list(set(idx+new_hide))
        ax.face.hide_component(idx)        
    elif mode == 'face':
        idx = ax.face.getSelectedIndex()
        idx = list(set(ax.face.hidden_component+idx))        
        ax.face.hide_component(idx)        
    elif mode == 'edge':
        idx = ax.edge.getSelectedIndex()
        idx = list(set(ax.edge.hidden_component+idx))        
        ax.edge.hide_component(idx)                
    elif mode == 'point':
        pass
    else:
        pass
    viewer.canvas.unselect_all()
    viewer.draw_all()
               
def make_solid(evt):
    viewer = evt.GetEventObject().GetTopLevelParent()
    mode = viewer._sel_mode

    ax = viewer.get_axes()
    for name, child in ax.get_children():    
        child.set_alpha(1.0, child._artists[0])
    viewer.canvas.unselect_all()
    viewer.draw_all()
    
def make_transp(evt):
    viewer = evt.GetEventObject().GetTopLevelParent()
    mode = viewer._sel_mode

    ax = viewer.get_axes()
    for name, child in ax.get_children():    
        child.set_alpha(0.75, child._artists[0])
    viewer.canvas.unselect_all()
    viewer.draw_all()
        
btask = [
         ('edge',   fedge, 2, 'select edge', select_edge),
         ('face',   fface, 2, 'select face', select_face),
         ('domain', fdom,  2, 'select domain', select_volume),
         ('---', None, None, None),
         ('mshow',  fshow,  0, 'show all', show_all),
         ('mhide',  fhide,  0, 'hide selection', hide_elem),
         ('---', None, None, None),
         ('solid',  fsolid,  0, 'solid', make_solid),
         ('transpaent',  ftrans,  0, 'transparent', make_transp),]

