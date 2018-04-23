import wx
from collections import OrderedDict
import traceback
from ifigure.utils.cbook import BuildPopUpMenu
from ifigure.utils.edit_list import EditListPanel, ScrolledEditListPanel
from ifigure.utils.edit_list import EDITLIST_CHANGED,  EDITLIST_CHANGING
from ifigure.utils.edit_list import EDITLIST_SETFOCUS
from ifigure.widgets.miniframe_with_windowlist import MiniFrameWithWindowList
from ifigure.widgets.miniframe_with_windowlist import DialogWithWindowList

try:
    import treemixin 
except ImportError:
    from wx.lib.mixins import treemixin

from petram.mfem_model import MFEM_ModelRoot

class ModelTree(treemixin.VirtualTree, wx.TreeCtrl):
    def __init__(self, *args, **kwargs):
        self.topwindow = kwargs.pop('topwindow')
        super(ModelTree, self).__init__(*args, **kwargs)
        
    def OnGetItemText(self, indices):
        item = self.topwindow.model.GetItem(indices)
        txt = self.topwindow.model.GetItemText(indices)

        if item.has_ns():
            info =item.get_info_str()
            if info != "":
                txt = txt + "(" + info + ")"
        if hasattr(item, 'isGeom') and hasattr(item, '_newobjs'):
                txt = txt + '('+','.join(item._newobjs) + ')'            
        return txt
    def OnGetItemTextColour(self, indices):
        item = self.topwindow.model.GetItem(indices)
        if item.enabled:
            return wx.BLACK
        else:
            return wx.Colour(128,128,128)
        
    def OnGetItemFont(self, indices):
        item = self.topwindow.model.GetItem(indices)
        if item.enabled:
            return wx.NORMAL_FONT
        else:
            return wx.ITALIC_FONT       
        
    def OnGetChildrenCount(self, indices):
        return self.topwindow.model.GetChildrenCount(indices)

#class DlgEditModel(MiniFrameWithWindowList):
class DlgEditModel(DialogWithWindowList):
    def __init__(self, parent, id, title, model = None):
                       
        self.model = model if not model is None else MFEM_ModelRoot()
        '''
        (use this style if miniframe is used)
        style=wx.CAPTION|
                       wx.CLOSE_BOX|
                       wx.MINIMIZE_BOX| 
                       wx.RESIZE_BORDER|
                       wx.FRAME_FLOAT_ON_PARENT,
        '''
        style = wx.DEFAULT_DIALOG_STYLE|wx.RESIZE_BORDER
        super(DlgEditModel, self).__init__(parent, id, title, style=style)
        
        self.splitter = wx.SplitterWindow(self, wx.ID_ANY,
                                 style=wx.SP_NOBORDER|wx.SP_LIVE_UPDATE|wx.SP_3DSASH)

        self.tree = ModelTree(self.splitter, topwindow = self)
        #self.tree.SetSizeHints(150, -1, maxW=150)
        self.nb = wx.Notebook(self.splitter)
        self.splitter.SplitVertically(self.tree, self.nb)
        self.splitter.SetMinimumPaneSize(150)
        
        self.p1 = wx.Panel(self.nb)
        self.p2 = wx.Panel(self.nb)
        self.p3 = wx.Panel(self.nb)
        self.p4 = wx.Panel(self.nb)        
        self.nb.AddPage(self.p1, "Config.")
#        self.nb.AddPage(self.p2, "Selection")
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
        
        self.SetSizer(wx.BoxSizer(wx.VERTICAL))
        s = self.GetSizer()
        #s2 = wx.BoxSizer(wx.HORIZONTAL)
        s.Add(self.splitter,  1, wx.EXPAND|wx.ALL, 1)
        self.Bind(wx.EVT_TREE_ITEM_RIGHT_CLICK, 
                  self.OnItemRightClick)
        self.Bind(wx.EVT_TREE_SEL_CHANGED, 
                  self.OnItemSelChanged)
        #s.Add(self.tree, 0, wx.EXPAND|wx.ALL, 1)
        #s2.Add(self.nb, 1, wx.EXPAND|wx.ALL, 1)        
        wx.GetApp().add_palette(self)
        self.Layout()
        wx.CallAfter(self.tree.RefreshItems)
        self.panels = {}
        self.Bind(wx.EVT_CLOSE, self.OnClose)
        self.Bind(EDITLIST_CHANGED, self.OnEL_Changed)        
        self.Bind(EDITLIST_CHANGING, self.OnEL_Changing)
        self.Bind(EDITLIST_SETFOCUS, self.OnEL_SetFocus)
        self.Bind(wx.EVT_CHILD_FOCUS, self.OnChildFocus)        
        self._focus_idx = None
        self._focus_obj = None
        self.SetSize((600,400))
        
    def OnChildFocus(self, evt):
        self.GetParent()._palette_focus = 'edit'                
        evt.Skip()
        
    def OnItemRightClick(self, e):
        tree = self.tree
        indices = tree.GetIndexOfItem(tree.GetSelection())
        mm = self.model.GetItem(indices)
        menus = []
        for cls in mm.get_possible_child():
           def add_func(evt, cls = cls, indices = indices, tree = tree,
                        model = self.model):
               txt = cls.__name__.split('_')[-1]               
               name = model.GetItem(indices).add_item(txt, cls)
               viewer = self.GetParent()               
               viewer.model.scripts.helpers.rebuild_ns()
               engine = viewer.engine
               model.GetItem(indices)[name].postprocess_after_add(engine)
               tree.RefreshItems()
           txt = cls.__name__.split('_')[-1]
           menus=menus+[('Add '+txt, add_func, None),]
        for t, m in mm.get_special_menu():
           menus=menus+[(t, m, None),]            
        menus = menus + [('---', None, None)]
        if mm.has_ns() and not mm.hide_ns_menu:
            if mm.ns_name is not None:
                menus.append(("Delete NS.",  self.OnDelNS, None))
                if hasattr(mm, '_global_ns'):
                    menus.append(("Initialize Dataset", self.OnInitDataset, None))      
            else:
                menus.append(("Add NS...",  self.OnAddNS, None))

        menus.extend(mm.get_editor_menus())
        if mm.can_delete:
            if menus[-1][0] != '---':
               menus = menus + [('---', None, None)]
            if mm.enabled:
               menus = menus + [('Disable', self.OnDisableItem, None)]
            else:
               menus = menus + [('Enable', self.OnEnableItem, None)]
            menus = menus + [('Duplicate', self.OnDuplicateItemFromModel, 
                              None)]
            if not mm.mustbe_firstchild:
               menus = menus + [('+Move...', None, None),
                                ('Up', self.OnMoveItemUp, None),
                                ('Down', self.OnMoveItemDown, None),
                                ('To...', self.OnMoveItemTo, None),
                                ('!', None, None),]
            menus = menus + [('Delete', self.OnDeleteItemFromModel, None)]
        if menus[-1][0] != '---':
             menus = menus + [('---', None, None)]
                                    
        menus = menus + [('Refresh', self.OnRefreshTree, None)]
        menus = menus + [('Export to shell', self.OnExportToShell, None)]        
        m  = wx.Menu()
        BuildPopUpMenu(m, menus, eventobj=self)
        self.PopupMenu(m, 
                       e.GetPoint())
        m.Destroy()

    def OnExportToShell(self, evt):
        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
 
        import wx
        app = wx.GetApp().TopWindow
        app.shell.lvar[mm.name()] = mm

    def OnDuplicateItemFromModel(self, evt):
        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
        name = mm.name()
        base, num = mm.split_digits()
        parent = mm.parent

#        mm._parent = None
        import cPickle as pickle
        newmm = pickle.loads(pickle.dumps(mm))
#        mm._parent = parent
        index = parent.keys().index(name)
        nums = []
        for key in parent.keys():
           print key, base
           base0 = ''.join([k for k in key if not k.isdigit()])
           if base0 != base: continue
           print nums
           nums.append(int(''.join([k for k in key if k.isdigit()])))
        
        parent.insert_item(index+1, base+str(long(max(nums))+1), newmm)
        self.tree.RefreshItems()        
        
    def OnDeleteItemFromModel(self, evt):
        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
        text = self.model.GetItemText(indices)
        del mm.parent[text]   
        self.tree.RefreshItems()

    def OnRefreshTree(self, evt=None):
        self.tree.RefreshItems()
        if evt is not None: evt.Skip()

    def OnItemSelChanged(self, evt = None):
        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
#        if not mm.__class__ in self.panels.keys():

        for k in self.panels.keys():
            p1panel, p2panel, p3panel, p4panel = self.panels[k]
            self.p1sizer.Detach(p1panel)
            self.p2sizer.Detach(p2panel)
            self.p3sizer.Detach(p3panel)
            self.p4sizer.Detach(p4panel)            
            p1panel.Hide()
            p2panel.Hide()
            p3panel.Hide()
            p4panel.Hide()            
        self.generate_panel(mm)

        self._cpanels = self.panels[mm.fullname()]
        p1panel, p2panel, p3panel, p4panel = self.panels[mm.fullname()]               
        
        if mm.has_2nd_panel:
            if self.nb.GetPageCount() == 1:
               self.nb.AddPage(self.p2, "Selection")
            self.p1sizer.Add(p1panel, 1, wx.EXPAND|wx.ALL, 1)
            self.p2sizer.Add(p2panel, 1, wx.EXPAND|wx.ALL, 1)
            p1panel.SetValue(mm.get_panel1_value())
            p2panel.SetValue(mm.get_panel2_value())
            p1panel.Show()
            p2panel.Show()
            self.p1.Layout()
            self.p2.Layout()
            if mm.has_3rd_panel:
                if self.nb.GetPageCount() == 2:
                    self.nb.AddPage(self.p3, "Init/NL.")

                self.p3sizer.Add(p3panel, 1, wx.EXPAND|wx.ALL, 1)
                p3panel.SetValue(mm.get_panel3_value())
                p3panel.Show()
                self.p3.Layout()
            else:
                if self.nb.GetPageCount() > 3:  self.nb.RemovePage(3)            
                if self.nb.GetPageCount() > 2:  self.nb.RemovePage(2)
                p3panel.Hide()                                
                
            if mm.has_4th_panel:
                if self.nb.GetPageCount() == 3:
                    self.nb.AddPage(self.p4, "Time Dep.")
                self.p4sizer.Add(p4panel, 1, wx.EXPAND|wx.ALL, 1)
                p4panel.SetValue(mm.get_panel4_value())
                p4panel.Show()                
                #for c in p4panel.GetChildren(): c.Show()
                self.p4.Layout()
            else:
                if self.nb.GetPageCount() > 3:  self.nb.RemovePage(3)            
                p4panel.Hide()                
        else:
            if self.nb.GetPageCount() > 3:  self.nb.RemovePage(3)            
            if self.nb.GetPageCount() > 2:  self.nb.RemovePage(2)
            if self.nb.GetPageCount() > 1:  self.nb.RemovePage(1)
            self.p1sizer.Add(p1panel, 1, wx.EXPAND|wx.ALL, 1)
            p1panel.SetValue(mm.get_panel1_value())
            p1panel.Show()
            self.p1.Layout()
            
        self._focus_idx = None
        from petram.model import Bdry, Domain, Pair
        from petram.phys.phys_model import PhysModule
        viewer = self.GetParent()
        engine = viewer.engine
        if hasattr(mm, '_sel_index'):
            self._focus_idx = 0
            if not mm.enabled:
                viewer.highlight_none()
                viewer._dom_bdr_sel = ([], [], [], [])
                
            elif isinstance(mm, Bdry):
                if not hasattr(mm, '_sel_index') or mm.sel_index == 'remaining':
                    phys = mm.get_root_phys()
                    engine.assign_sel_index(phys)

                if mm.dim == 3:
                    viewer.canvas.toolbar.ClickP1Button('face')                    
                    viewer.highlight_face(mm._sel_index)
                    viewer._dom_bdr_sel = ([], mm._sel_index, [], [])
                elif mm.dim == 2:                    
                    viewer.canvas.toolbar.ClickP1Button('edge')                    
                    viewer.highlight_edge(mm._sel_index)
                    viewer._dom_bdr_sel = ([], [], mm._sel_index, [],)
                elif mm.dim == 1:                                        
                    viewer.canvas.toolbar.ClickP1Button('dot')                    
                    viewer.highlight_point(mm._sel_index)
                    viewer._dom_bdr_sel = ([], [], [], mm._sel_index, )
                else:
                    pass
                
            elif isinstance(mm, Domain):
                if not hasattr(mm, '_sel_index') or mm.sel_index == 'remaining':
                    phys = mm.get_root_phys()
                    engine.assign_sel_index(phys)

                if mm.dim == 3:
                    viewer.canvas.toolbar.ClickP1Button('domain')                    
                    viewer.highlight_domain(mm._sel_index)
                    viewer._dom_bdr_sel = (mm._sel_index, [], [], [])                    
                elif mm.dim == 2:
                    viewer.canvas.toolbar.ClickP1Button('face')                    
                    viewer.highlight_face(mm._sel_index)
                    viewer._dom_bdr_sel = ([], mm._sel_index, [], [])
                elif mm.dim == 1:
                    viewer.canvas.toolbar.ClickP1Button('edge')                    
                    viewer.highlight_edge(mm._sel_index)
                    viewer._dom_bdr_sel = ([], [], mm._sel_index, [],)
                else:
                    pass
        elif isinstance(mm, PhysModule):
            if not mm.enabled:
                viewer.highlight_none()
                viewer._dom_bdr_sel = ([], [], [], [])
            else:
                if not hasattr(mm, '_phys_sel_index') or mm.sel_index == 'all':
                    engine.assign_sel_index(mm)
                if hasattr(mm, '_phys_sel_index'):
                    # need this if in case mesh is not loaded....
                    if mm.dim == 3:
                        viewer.canvas.toolbar.ClickP1Button('domain')                    
                        viewer.highlight_domain(mm._phys_sel_index)
                        viewer._dom_bdr_sel = (mm._phys_sel_index, [], [], [])                    
                    elif mm.dim == 2:
                        viewer.canvas.toolbar.ClickP1Button('face')                    
                        viewer.highlight_face(mm._phys_sel_index)
                        viewer._dom_bdr_sel = ([], mm._phys_sel_index, [], [])
                    elif mm.dim == 1:
                        viewer.canvas.toolbar.ClickP1Button('edge')                    
                        viewer.highlight_edge(mm._phys_sel_index)
                        viewer._dom_bdr_sel = ([], [], mm._phys_sel_index, [],)
                    else:
                        pass
        else:
            pass
        if evt is not None:
            mm.onItemSelChanged(evt)      
            evt.Skip()
        
    def OnClose(self, evt):
        wx.GetApp().rm_palette(self)
        self.GetParent().editdlg = None
        evt.Skip()

    def generate_panel(self, mm):
        if mm.fullname() in self.panels and not mm.always_new_panel:
            self.update_panel_label(mm)
        else:
            self.panels[mm.fullname()] = (ScrolledEditListPanel(self.p1,
                                                           list =  mm.panel1_param(), 
                                                           tip=mm.panel1_tip()),
                                     EditListPanel(self.p2, list =  mm.panel2_param(),
                                                   tip=mm.panel2_tip()),
                                     EditListPanel(self.p3, list =  mm.panel3_param(),
                                                   tip=mm.panel3_tip()),
                                     EditListPanel(self.p4, list =  mm.panel4_param(),
                                                   tip=mm.panel4_tip()),)
        
    def update_panel_label(self, mm):
        self.panels[mm.fullname()][0].update_label(mm.panel1_param())
        self.panels[mm.fullname()][1].update_label(mm.panel2_param())
        self.panels[mm.fullname()][2].update_label(mm.panel3_param())
        self.panels[mm.fullname()][3].update_label(mm.panel4_param())                
                             
    def OnEL_Changed(self, evt):
        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
  
        p1children = self.p1sizer.GetChildren()
        phys = None
        viewer_update = False
        if len(p1children) > 0:
            elp1 = p1children[0].GetWindow()
            v1 = elp1.GetValue()
            viewer_update = mm.import_panel1_value(v1)
            try:
                phys = mm.get_root_phys()
            except:
                pass
            elp1.SetValue(mm.get_panel1_value())
            
        if mm.has_2nd_panel:
            p2children = self.p2sizer.GetChildren()
            if len(p2children) > 0:
                elp2 = p2children[0].GetWindow()
                v2 = elp2.GetValue()
                viewer_update = mm.import_panel2_value(v2)
                elp2.SetValue(mm.get_panel2_value())
            
        if mm.has_3rd_panel:                
            p3children = self.p3sizer.GetChildren()
            if len(p3children) > 0:
                elp3 = p3children[0].GetWindow()
                v3 = elp3.GetValue()
                viewer_update = mm.import_panel3_value(v3)
                elp3.SetValue(mm.get_panel3_value())

        if mm.has_4th_panel:                
            p4children = self.p4sizer.GetChildren()
            if len(p4children) > 0:
                elp4 = p4children[0].GetWindow()
                v4 = elp4.GetValue()
                viewer_update = mm.import_panel4_value(v4)
                elp4.SetValue(mm.get_panel4_value())
                
        if phys is not None:
           viewer = self.GetParent()
           try:
               engine = viewer.engine.assign_sel_index(phys)
           except:
               traceback.print_exc()
               
        if viewer_update:
            mm.update_after_ELChanged(self)
        self.tree.RefreshItems()            
        evt.Skip()

    def OnEL_Changing(self, evt):
        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)

    def set_model(self, model):
        self.model = model
        self.tree.RefreshItems()
        
    def OnEL_SetFocus(self, evt):
        try:
           indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        except:
           return 
        mm = self.model.GetItem(indices)
        from petram.model import Bdry, Point, Pair, Domain, Edge

        if isinstance(mm, Bdry):
            try:
                id_list = evt.GetEventObject().GetValue()
            except ValueError:
                return
        self._focus_idx = evt.widget_idx
        #print  self._focus_obj

    def OnAddNS(self, evt):
        import   ifigure.widgets.dialog as dialog
        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
        txt = self.model.GetItemText(indices)        
        ret, txt = dialog.textentry(self, 
                                     "Enter namespace name", "New NS...", txt.lower())
        if not ret: return
        mm.new_ns(txt)
        self.tree.RefreshItems()
        evt.Skip()
         
    def OnDelNS(self, evt):
        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
        mm.delete_ns()
        self.tree.RefreshItems()
        evt.Skip()
        
    def OnInitDataset(self, evt):
        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
        if not hasattr(mm, '_global_ns'): return
        if not 'dataset_names' in mm._global_ns: return
        names = mm._global_ns['dataset_names']
        viewer = self.GetParent()               
        viewer.model.scripts.helpers.init_dataset(mm.ns_name, names)                       
        self.tree.RefreshItems()
        evt.Skip()

    def OnEvalNS(self, evt):
        viewer = self.GetParent()
        engine = viewer.engine
        model = viewer.book.get_pymodel()
        model.scripts.helpers.rebuild_ns()
        evt.Skip()

    def OnDisableItem(self, evt):
        import   ifigure.widgets.dialog as dialog
        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
        mm.enabled = False
        self.tree.RefreshItems()

        # if it is phys, handle possible change of what remaining means                  
        if not hasattr(mm, 'get_root_phys'): return        
        phys = mm.get_root_phys()   
        if phys is not None:
           viewer = self.GetParent()
           try:
               viewer.engine.assign_sel_index(phys)
           except:
               traceback.print_exc()


    def OnEnableItem(self, evt):
        import   ifigure.widgets.dialog as dialog
        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
        mm.enabled = True
        self.tree.RefreshItems()

        # if it is phys, handle possible change of what remaining means                          
        if not hasattr(mm, 'get_root_phys'): return
        phys = mm.get_root_phys() 
        if phys is not None:
           viewer = self.GetParent()
           try:
               viewer.engine.assign_sel_index(phys)
           except:
               traceback.print_exc()

    def get_selected_mm(self):
        import   ifigure.widgets.dialog as dialog
        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
        return mm
        

    def select_next_enabled(self):
        item = self.tree.GetSelection()
        while True:
           item = self.tree.GetNextSibling(item)
           if not item.IsOk(): return 
           indices = self.tree.GetIndexOfItem(item)
           mm = self.model.GetItem(indices)
           if mm.enabled:
               self.tree.SelectItem(item)
               return
        wx.CallAfter(self.Refresh, False)

    @staticmethod                         
    def MoveItemInList(l, i1,i2):
        if i1 > i2:
           return   l[0:i2] + [l[i1]] + l[i2:i1] + l[i1+1:len(l)]
        elif  i1 < i2: 
           return   l[0:i1] + l[i1+1:i2+1] + [l[i1]]+ l[i2+1:len(l)]
                             
    def OnMoveItemUp(self, evt):
        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
        p = mm.parent
        names = list(p._contents)

        idx = names.index(mm.name())
        if idx == 0: return

        new_names = self.MoveItemInList(names, idx, idx-1)

        p._contents = OrderedDict((k, p._contents[k]) for k in new_names)        
        self.tree.RefreshItems()
                             
    def OnMoveItemDown(self, evt):
        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
        p = mm.parent
        names = list(p._contents)

        idx = names.index(mm.name())
        if idx == len(names)-1: return

        new_names = self.MoveItemInList(names, idx, idx+1)

        p._contents = OrderedDict((k, p._contents[k]) for k in new_names)
        self.tree.RefreshItems()

    def OnMoveItemTo(self, evt):
        from   ifigure.utils.edit_list import DialogEditList                
        import   ifigure.widgets.dialog as dialog
        
        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
        p = mm.parent
        names = list(p._contents)
        idx = names.index(mm.name())
        

        list6 = [
               ["New parent", p.name(), 0],
               ["Index ", str(idx), 0],]
        value = DialogEditList(list6, modal = True, 
                               style = wx.DEFAULT_DIALOG_STYLE|wx.RESIZE_BORDER,
                               tip = None, 
                               parent=self,
                               title = 'Move item to...')
        if not value[0]: return

        if value[1][0] != p.name():
            try:
                assert False, "Moving under diffent parent is not supported"
            except AssertionError:
                dialog.showtraceback(parent = self,
                           txt='Moving under diffent parent is not supported',
                           title='Error',
                           traceback=traceback.format_exc())
                return
            
        new_idx = int(value[1][1])
        names = list(p._contents)
        new_idx = max([0, new_idx])
        new_idx = min([len(names)-1, new_idx])

        idx = names.index(mm.name())        
        new_names = self.MoveItemInList(names, idx, new_idx)
        p._contents = OrderedDict((k, p._contents[k]) for k in new_names)
        self.tree.RefreshItems()
        
    def isSelectionPanelOpen(self):
        from petram.model import Bdry, Point, Pair, Domain, Edge        
        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)

        false_value = False, '', [], []        
        try:
           phys = mm.get_root_phys()
        except:
           return false_value

        if self.nb.GetPageCount() == 0: return false_value
        if (self.nb.GetPageCount() > 0 and 
            self.nb.GetSelection() != 1):return false_value
       
        is_wc = mm.is_wildcard_in_sel()

        idx = []
        labels = []
        for a, b, c in zip(mm.is_wildcard_in_sel(),
                           mm.panel2_sel_labels(),
                           mm.panel2_all_sel_index()):
            if not a:
                labels.append(b)                
                idx.append(c)
            else:
                labels.append('')
                idx.append([])

        tnames = ['domain', 'bdry', 'edge', 'point']
        if isinstance(mm, Domain):
            tt = 0
        elif isinstance(mm, Bdry):
            tt = 1
        elif isinstance(mm, Edge):
            tt = 2
        elif isinstance(mm, Pair):
            tt = 1
        else:
           return false_value
        if mm.dim == 2: tt += 1
        if mm.dim == 1: tt += 2
        
        return True, tnames[tt], idx, labels

    def add_remove_AreaSelection(self,  idx, rm= False, flag=0):
        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
        try:
           phys = mm.get_root_phys()
        except:
           return 

        sidx = [' ' + str(x) for x in idx]
        if flag == 0:
            tgt = [int(x) for x in mm.sel_index]
        elif flag == 1:
            tgt = [int(x) for x in mm.src_index]
        else:
            pass

        if rm:
            for x in idx: 
                if int(x) in tgt: tgt.remove(x)
        else:
            for x in idx: 
                if not int(x) in tgt: tgt.append(x)

        sidx = [' ' + str(x) for x in tgt]
        if len(tgt) > 0:  sidx[0] = str(tgt[0])
        if flag == 0:
            mm.sel_index = sidx
        elif flag == 1:
            mm.src_index = sidx
        else:
            pass    
        if phys is not None:
           viewer = self.GetParent()
           try:
               engine = viewer.engine.assign_sel_index(phys)
           except:
               traceback.print_exc()
        self.OnItemSelChanged(None)               
        #evt.Skip()
        
        

