from __future__ import print_function

import os
import numpy as np
import wx
from collections import OrderedDict
import traceback
from ifigure.utils.cbook import BuildPopUpMenu
from ifigure.utils.edit_list import EditListPanel, ScrolledEditListPanel
from ifigure.utils.edit_list import EDITLIST_CHANGED,  EDITLIST_CHANGING
from ifigure.utils.edit_list import EDITLIST_SETFOCUS
from ifigure.widgets.miniframe_with_windowlist import MiniFrameWithWindowList
from ifigure.widgets.miniframe_with_windowlist import DialogWithWindowList

import petram
import petram.helper.pickle_wrapper as pickle

try:
    import treemixin 
except ImportError:
    from wx.lib.mixins import treemixin

from petram.mfem_model import MFEM_ModelRoot

from ifigure.ifigure_config import rcdir
petram_model_scratch=os.path.join(rcdir, 'petram_model_scratch')

from petram.utils import get_pkg_datafile
#import petram.geom

fdot = get_pkg_datafile(petram.pi, 'icon',  'dot.png')
fedge = get_pkg_datafile(petram.pi, 'icon', 'line.png')
fface = get_pkg_datafile(petram.pi, 'icon', 'face.png')
fdom = get_pkg_datafile(petram.pi, 'icon', 'domain.png')

im1 = wx.Image(fdom, wx.BITMAP_TYPE_PNG)
image1 = im1.ConvertToBitmap()
im2 = wx.Image(fface, wx.BITMAP_TYPE_PNG)
image2 = im2.ConvertToBitmap()
im3 = wx.Image(fedge, wx.BITMAP_TYPE_PNG)
image3 = im3.ConvertToBitmap()
im4 = wx.Image(fdot, wx.BITMAP_TYPE_PNG)
image4 = im4.ConvertToBitmap()

w,h = image4.GetSize()
il = wx.ImageList(w, h)
il.Add(image1)
il.Add(image2)
il.Add(image3)
il.Add(image4)

from petram.pi.simple_frame_plus import SimpleFramePlus

class EntityTree(treemixin.VirtualTree, wx.TreeCtrl):
    def __init__(self, *args, **kwargs):
        self.topwindow = kwargs.pop('topwindow')
        self.entitydim = kwargs.pop('entitydim')
        self.d= {}
        self.keys = []
        super(EntityTree, self).__init__(*args, **kwargs)
        
    def OnGetItemText(self, indices):
        if len(indices) == 1:
            txt = str(self.keys[indices[0]])
        elif len(indices) == 2:
            l = sorted(self.d[self.keys[indices[0]]])
            txt =  str(l[indices[1]])
        else:
            txt = ''
        return txt
    
    def OnGetItemTextColour(self, indices):
        return wx.BLACK
        
    def OnGetItemFont(self, indices):
        return wx.NORMAL_FONT        
        
    def OnGetChildrenCount(self, indices):
        if len(indices) == 0:
            return len(self.keys)
        elif len(indices) == 1:
            return len(self.d[self.keys[indices[0]]])
        else:
            return 0
        return 0

    def GetSelection(self):
        ## this returns only one selection
        ## called when only one element is assumed to be selected
        ret = self.GetSelections()
        if len(ret) == 0: return None
        return ret[0]
    
    def isMultipleSelection(self):
        return len(self.GetSelections()) > 1

    def set_dict(self, d):
        if d is None:
            d = {}
        self.d = d
        self.keys = sorted(list(d))
        
class SelectionPalette(SimpleFramePlus):
    def __init__(self, parent, id, title, model = None):
                       
        '''
        (use this style if miniframe is used)
        style=(wx.CAPTION|
                       wx.CLOSE_BOX|
                       wx.MINIMIZE_BOX| 
                       wx.RESIZE_BORDER|
                       wx.FRAME_FLOAT_ON_PARENT)
        '''
        style =  wx.CAPTION|wx.RESIZE_BORDER|wx.SYSTEM_MENU
        style = (wx.CAPTION|
                 wx.CLOSE_BOX|
                 wx.MINIMIZE_BOX| 
                 wx.RESIZE_BORDER|
                 wx.FRAME_FLOAT_ON_PARENT)
        #        wx.FRAME_TOOL_WINDOW : this styles may not work on Windows/Mac

        #style = wx.RESIZE_BORDER
        super(SelectionPalette, self).__init__(parent, id, title, style=style)
        
        #self.tree.SetSizeHints(150, -1, maxW=150)
        self.nb = wx.Notebook(self)
        self.nb.SetImageList(il)
        
        self.SetSizer(wx.BoxSizer(wx.VERTICAL))
        self.GetSizer().Add(self.nb,  1, wx.EXPAND|wx.ALL, 1)
        self.cb_update = wx.CheckBox(self, wx.ID_ANY, 'Update Figure')
        self.GetSizer().Add(self.cb_update,  0, wx.EXPAND|wx.ALL, 5)        
        
        self.p1 = wx.Panel(self.nb)
        self.p2 = wx.Panel(self.nb)
        self.p3 = wx.Panel(self.nb)
        self.p4 = wx.Panel(self.nb)
        
        self.p1.SetSizer(wx.BoxSizer(wx.HORIZONTAL))
        self.p2.SetSizer(wx.BoxSizer(wx.HORIZONTAL))
        self.p3.SetSizer(wx.BoxSizer(wx.HORIZONTAL))
        self.p4.SetSizer(wx.BoxSizer(wx.HORIZONTAL))        
        
        self.p1sizer = wx.BoxSizer(wx.VERTICAL)
        self.p2sizer = wx.BoxSizer(wx.VERTICAL)
        self.p3sizer = wx.BoxSizer(wx.VERTICAL)
        self.p4sizer = wx.BoxSizer(wx.VERTICAL)        
 
        self.p1.GetSizer().Add(self.p1sizer, 1, wx.EXPAND)
        self.p2.GetSizer().Add(self.p2sizer, 1, wx.EXPAND)
        self.p3.GetSizer().Add(self.p3sizer, 1, wx.EXPAND)
        self.p4.GetSizer().Add(self.p4sizer, 1, wx.EXPAND)

        self.nb.AddPage(self.p1, "", imageId=0)
        self.nb.AddPage(self.p2, "", imageId=1)
        self.nb.AddPage(self.p3, "", imageId=2)
        self.nb.AddPage(self.p4, "", imageId=3)
        
        self.tree1 = EntityTree(self.p1, topwindow = self, entitydim=3,
                            style = wx.TR_DEFAULT_STYLE|wx.TR_MULTIPLE)
        self.tree2 = EntityTree(self.p2, topwindow = self, entitydim=2,
                            style = wx.TR_DEFAULT_STYLE|wx.TR_MULTIPLE)
        self.tree3 = EntityTree(self.p3, topwindow = self, entitydim=1,
                            style = wx.TR_DEFAULT_STYLE|wx.TR_MULTIPLE)
        self.tree4 = EntityTree(self.p4, topwindow = self, entitydim=0,
                            style = wx.TR_DEFAULT_STYLE|wx.TR_MULTIPLE)

        self.p1sizer.Add(self.tree1, 1, wx.EXPAND)
        self.p2sizer.Add(self.tree2, 1, wx.EXPAND)
        self.p3sizer.Add(self.tree3, 1, wx.EXPAND)
        self.p4sizer.Add(self.tree4, 1, wx.EXPAND)

        self.tree1.Bind(wx.EVT_CONTEXT_MENU,
                  self.OnItemRightClick0)
        self.tree2.Bind(wx.EVT_CONTEXT_MENU,
                  self.OnItemRightClick0)
        self.tree3.Bind(wx.EVT_CONTEXT_MENU,
                  self.OnItemRightClick0)
        self.tree4.Bind(wx.EVT_CONTEXT_MENU,
                  self.OnItemRightClick0)
        
        self.Bind(wx.EVT_TREE_ITEM_RIGHT_CLICK, 
                  self.OnItemRightClick)
        self.Bind(wx.EVT_TREE_SEL_CHANGED, 
                  self.OnItemSelChanged)
        self.Bind(wx.EVT_TREE_SEL_CHANGING, 
                  self.OnItemSelChanging)
        self.Bind(wx.EVT_TREE_KEY_DOWN,
                  self.OnKeyDown)
        self.Bind(wx.EVT_KEY_UP,self.OnKeyUp)
        self.tree1.Bind(wx.EVT_KEY_UP,self.OnKeyUp)
        self.tree2.Bind(wx.EVT_KEY_UP,self.OnKeyUp)
        self.tree3.Bind(wx.EVT_KEY_UP,self.OnKeyUp)
        self.tree4.Bind(wx.EVT_KEY_UP,self.OnKeyUp)

        self.bind_sel_events()
        
        #s.Add(self.tree, 0, wx.EXPAND|wx.ALL, 1)
        #s2.Add(self.nb, 1, wx.EXPAND|wx.ALL, 1)        
        wx.GetApp().add_palette(self)
        self.Layout()
        wx.CallAfter(self.OnRefreshTree)
     
        self.panels = {}
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.Bind(wx.EVT_CHILD_FOCUS, self.OnChildFocus)        
        self._focus_idx = None
        self._focus_obj = None
        self._copied_item = None
        self.SetSize((200,400))
        self.trees = [self.tree1, self.tree2, self.tree3, self.tree4]
        self.loops = (None, None, None, None)  #backup
        self._old_item = None
        self._key_down = (-1, False)
        self._do_veto_changing = True
        self._changed_proccessed = True        
        self._current_selection = []
        self._selected_items = []

    def get_item_from_indices(self, ktree, indices):
        tree = self.trees[ktree]
        parent = tree.GetRootItem()
        for c in indices:
           item, hoge = tree.GetFirstChild(parent)
           if not item.IsOk(): return None
           for i in range(c):
               item = tree.GetNextSibling(item)
               if not item.IsOk(): return None               
           parent = item

        return parent

    def select_by_indices(self, ktree, indices):
        tree = self.trees[ktree]
        tree.UnselectAll()
        items = tree.GetSelections()
        cindices = [tree.GetIndexOfItem(ii) for ii in items]
        for i in cindices:
            if not i in indices:            
               item = self.get_item_from_indices(ktree, i)
               if item is not None:
                   tree.SelectItem(item, select = False)
        for i in indices:
            if not i in cindices:            
                item = self.get_item_from_indices(ktree, i)
                if item is not None:
                    tree.SelectItem(item)
                

    def CallClose(self, evt):
        self.Close()

    def OnChildFocus(self, evt):
        self.GetParent()._palette_focus = 'selection'
        self.OnRefreshTree()
        evt.Skip()

    def OnItemRightClick0(self, e):
        menus = [('Refresh', self.OnRefreshTree, None)]           
        m  = wx.Menu()
        BuildPopUpMenu(m, menus, eventobj=self)
        self.PopupMenu(m, 
                       e.GetPoint())
        m.Destroy()
        
    def OnItemRightClick(self, e):
        tree = self.trees[self.nb.GetSelection()]
        menus = []        
        if tree.isMultipleSelection():
            items = tree.GetSelections()
            indices = [tree.GetIndexOfItem(ii) for ii in items]
        else:    
            indices = tree.GetIndexOfItem(tree.GetSelection())
        menus = menus + [('Refresh', self.OnRefreshTree, None)]           
        m  = wx.Menu()
        BuildPopUpMenu(m, menus, eventobj=self)
        self.PopupMenu(m, 
                       e.GetPoint())
        m.Destroy()

    def OnExportToShell(self, evt):
        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
 
        import wx
        import ifigure.widgets.dialog as dialog
        
        app = wx.GetApp().TopWindow
        app.shell.lvar[mm.name()] = mm
        app.shell.SendShellEnterEvent()
        ret=dialog.message(app, mm.name() + ' is exported', 'Export', 0)
        
        
    def OnRefreshTree(self, evt=None):
        v = self.GetParent()

        if not v._view_mode in v._s_v_loop:
            return 
        loops = v._s_v_loop[v._view_mode]
        #loops = v._figure_data[v._view_mode]        
        if (self.loops[0] is loops[0] and
            self.loops[1] is loops[1]):
            return

        
        self.tree1.set_dict(loops[1])
        self.tree2.set_dict(loops[0])
        if self.GetParent().book.page1.axes1.has_child('edge'):
            edges = np.unique(self.GetParent().book.page1.axes1.edge.getvar('array_idx'))
        else:
            edges = []
        d = {x:[] for x in edges}

        if self.GetParent().book.page1.axes1.has_child('point'):        
            points = self.GetParent().book.page1.axes1.point.getvar('array_idx')
        else:
            points = []
        d2 = {x:[] for x in points}
        
        self.loops = (loops[0], loops[1], d, d2)
        
        self.tree3.set_dict(d)
        self.tree4.set_dict(d2)        

        self.tree1.RefreshItems()
        self.tree2.RefreshItems()
        self.tree3.RefreshItems() 
        self.tree4.RefreshItems()       
        if evt is not None: evt.Skip()

    def bind_sel_events(self):
        self.Bind(wx.EVT_TREE_SEL_CHANGED, 
                  self.OnItemSelChanged)
        self.Bind(wx.EVT_TREE_SEL_CHANGING, 
                  self.OnItemSelChanging)
    def unbind_sel_events(self):
        self.Unbind(wx.EVT_TREE_SEL_CHANGED)
        self.Unbind(wx.EVT_TREE_SEL_CHANGING)
        
    def OnKeyUp(self, evt):
        self._key_down = (-1, False)
        
    def OnKeyDown(self, evt):
        # check if alt key is pushed
        sel = self.nb.GetSelection()
        if evt.GetKeyEvent().GetKeyCode() == wx.WXK_ALT:
            self._key_down = (sel, True)
        else:
            self._key_down = (-1, False)

    def OnItemSelChanging(self, evt):
        #print("event changing")
        sel = self.nb.GetSelection()
        tree = self.trees[sel]
        
        if not evt.GetOldItem().IsOk():
            evt.Skip()
            return
        if not evt.GetItem().IsOk():
            evt.Skip()
            return 
        
        if (tree.GetIndexOfItem(evt.GetOldItem()) ==
            tree.GetIndexOfItem(evt.GetItem())):
           #print("calling Veto")
           evt.Veto()

    
    def OnItemSelChanged(self, evt = None):
        #print("event changed")
        ktree = self.nb.GetSelection()
        tree = self.trees[ktree]
        self._do_veto_changing = True
        self._changed_proccessed = True

        if tree.GetSelection() is None: return
        
        if (self._key_down[0] == ktree) and self._key_down[1]:
            #print("check1 ", self._selected)
            item = tree.GetIndexOfItem(evt.GetItem())
            #print("check2 ", item)
            if not (item in self._selected):
                new_selected = self._selected + [item]
                #print("calling new selection:", new_selected)
                wx.CallAfter(self.select_by_indices, ktree, new_selected)
                self._selected = new_selected
                evt.Skip()
                return
        else:
            items = tree.GetSelections()
            indices = [tree.GetIndexOfItem(ii) for ii in items]
            self._selected = indices
            
        #print("items in changed", self._selected)
        
        '''
        if ((self._key_down[0] == sel) and self._key_down[1] and
            not self._do_vete_changing):
            #self.unbind_sel_events()
            if evt.GetItem() is evt.GetOldItem():
                print("removing ",evt.GetOldItem())
                tree.SelectItem(evt.GetOldItem(), select = False)
                #evt.Veto()                
            else:
                print("adding ", tree.GetIndexOfItem(self._old_item))
                tree.SelectItem(self._old_item)
                #evt.Veto()
            #wx.CallAfter(self.bind_sel_events)
            self._do_vete_changing = True
            print("1")
        else:
            print("2")            
            self._do_vete_changing = False

        '''

        if self.cb_update.GetValue():
            v= [];s=[];l=[];p=[]
            if ktree == 0:
                d = self.loops[1]
                keys = sorted(list(d))
                for i in self._selected:
                    if len(i) == 1:
                        v.append(keys[i[0]])
                    else:
                        s.append(sorted(d[keys[i[0]]])[i[1]])
            elif ktree == 1:
                d = self.loops[0]
                keys = sorted(list(d))
                for i in self._selected:
                    if len(i) == 1:
                        s.append(keys[i[0]])
                    else:
                        l.append(sorted(d[keys[i[0]]])[i[1]])
            elif ktree == 2:
                d = self.loops[2]
                keys = sorted(list(d))
                for i in self._selected:
                    if len(i) == 1:
                        l.append(keys[i[0]])
                    else:
                        p.append(sorted(d[keys[i[0]]])[i[1]])
            elif ktree == 3:
                d = self.loops[3]
                keys = sorted(list(d))
                for i in self._selected:
                    if len(i) == 1:
                        p.append(keys[i[0]])
                    else:
                        pass
            else:
                pass

            vv = self.GetParent()
            if len(v) > 0:
                vv.change_panel_button('domain')
                vv.highlight_domain(v)
                vv._dom_bdr_sel = (v, [], [], [])
                
            elif len(s) > 0:
                vv.change_panel_button('face')                
                vv.highlight_face(s)
                vv._dom_bdr_sel = ([], s, [], [])
                
            elif len(l) > 0:
                vv.change_panel_button('edge')
                vv.highlight_edge(l)
                vv._dom_bdr_sel = ([], [], l, [])
                
            elif len(p) > 0:
                vv.change_panel_button('dot')                
                vv.highlight_point(p)
                vv._dom_bdr_sel = ([], [], [], p)
                
            else:
                vv.highlight_none()
                vv._dom_bdr_sel = ([], [], [], [])
                
        if evt is not None:
            evt.Skip()
    
    def OnClose(self, evt):
        wx.GetApp().rm_palette(self)
        self.GetParent().selection_palette = None
        evt.Skip()
        

        

        
        

