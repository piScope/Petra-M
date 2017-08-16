from petram.utils import get_pkg_datafile
import petram.geom

fdot = get_pkg_datafile(petram.pi, 'icon',  'dot.png')
fedge = get_pkg_datafile(petram.pi, 'icon', 'line.png')
fface = get_pkg_datafile(petram.pi, 'icon', 'face.png')
fdom = get_pkg_datafile(petram.pi, 'icon', 'domain.png')

def _select_x(evt, mode, mask):
    viewer = evt.GetEventObject().GetTopLevelParent()
    viewer._sel_mode = mode
    viewer.canvas.unselect_all()    
    viewer.set_picker_mask(mask)
    for name, child in viewer.get_axes().get_children():
        if hasattr(child, 'setSelectedIndesx'):
            child.setSelectedIndex([])
    viewer.draw()

def select_dot(evt):
    print(evt.GetEventObject())
    
def select_edge(evt):
    print(evt.GetEventObject())
    
def select_face(evt):
    print(evt.GetEventObject())
    
def select_dom(evt):
    print(evt.GetEventObject())    
    
btask = [('face',   fface, 2, 'select face', select_face),
         ('domain', fdom,  2, 'select domain', select_dom),]
