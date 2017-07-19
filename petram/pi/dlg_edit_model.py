import wx
from collections import OrderedDict
import traceback
from ifigure.utils.cbook import BuildPopUpMenu
from ifigure.utils.edit_list import EditListPanel, EDITLIST_CHANGED,  EDITLIST_CHANGING
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
            if item.get_ns_name() is not None:
                txt = txt + '(NS:'+item.ns_name + ')'
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
        self.nb.AddPage(self.p1, "Config.")
#        self.nb.AddPage(self.p2, "Selection")
        self.p1.SetSizer(wx.BoxSizer(wx.HORIZONTAL))
        self.p2.SetSizer(wx.BoxSizer(wx.HORIZONTAL))
        self.p3.SetSizer(wx.BoxSizer(wx.HORIZONTAL))
        
        self.p1sizer = wx.BoxSizer(wx.VERTICAL)
        self.p2sizer = wx.BoxSizer(wx.VERTICAL)
        self.p3sizer = wx.BoxSizer(wx.VERTICAL)
        
        self.p1.GetSizer().Add(self.p1sizer, 1, wx.EXPAND)
        self.p2.GetSizer().Add(self.p2sizer, 1, wx.EXPAND)
        self.p3.GetSizer().Add(self.p3sizer, 1, wx.EXPAND)
        
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

        self._focus_idx = None
        self._focus_obj = None        
        
    def OnItemRightClick(self, e):
        tree = self.tree
        indices = tree.GetIndexOfItem(tree.GetSelection())
        mm = self.model.GetItem(indices)
        menus = []
        for cls in mm.get_possible_child():
           def add_func(evt, cls = cls, indices = indices, tree = tree,
                        model = self.model):
               txt = cls.__name__.split('_')[-1]               
               model.GetItem(indices).add_item(txt, cls)
               viewer = self.GetParent()               
               viewer.model.scripts.helpers.rebuild_ns()                        
               tree.RefreshItems()
           txt = cls.__name__.split('_')[-1]
           menus=menus+[('Add '+txt, add_func, None),]

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
        parent.insert_item(index+1, base+str(long(num)+1), newmm)
        self.tree.RefreshItems()        
        
    def OnDeleteItemFromModel(self, evt):
        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
        text = self.model.GetItemText(indices)
        del mm.parent[text]   
        self.tree.RefreshItems()

    def OnRefreshTree(self, evt):
        self.tree.RefreshItems()

    def OnItemSelChanged(self, evt = None):
        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
        if not mm.__class__ in self.panels.keys():
            self.generate_panel(mm)
        for k in self.panels.keys():
            p1panel, p2panel, p3panel = self.panels[k]
            self.p1sizer.Detach(p1panel)
            self.p2sizer.Detach(p2panel)
            self.p3sizer.Detach(p3panel)
            p1panel.Hide()
            p2panel.Hide()
            p3panel.Hide()
        if mm.has_2nd_panel:
            if self.nb.GetPageCount() == 1:
               self.nb.AddPage(self.p2, "Selection")
            p1panel = self.panels[mm.__class__][0]
            p2panel = self.panels[mm.__class__][1]
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
                    self.nb.AddPage(self.p3, "init/NL.")
                p3panel = self.panels[mm.__class__][2]
                self.p3sizer.Add(p3panel, 1, wx.EXPAND|wx.ALL, 1)
                p3panel.SetValue(mm.get_panel3_value())
                p3panel.Show()
                self.p3.Layout()
        else:
            if self.nb.GetPageCount() > 2:  self.nb.RemovePage(2)
            if self.nb.GetPageCount() > 1:  self.nb.RemovePage(1)
            p1panel = self.panels[mm.__class__][0]
            self.p1sizer.Add(p1panel, 1, wx.EXPAND|wx.ALL, 1)
            p1panel.SetValue(mm.get_panel1_value())
            p1panel.Show()
            self.p1.Layout()
            
        self._focus_idx = None
        from petram.model import Bdry, Domain, Pair
        viewer = self.GetParent()
        engine = viewer.engine
        if hasattr(mm, '_sel_index'):
            self._focus_idx = 0
            if isinstance(mm, Bdry):
                if not hasattr(mm, '_sel_index') or mm.sel_index == 'remaining':
                    phys = mm.get_root_phys()
                    engine.assign_sel_index(phys)
                viewer.highlight_bdry(mm._sel_index)
            elif isinstance(mm, Domain):
                if not hasattr(mm, '_sel_index') or mm.sel_index == 'remaining':
                    phys = mm.get_root_phys()
                    engine.assign_sel_index(phys)
                viewer.highlight_domain(mm._sel_index)
        if evt is not None: evt.Skip()
        
    def OnClose(self, evt):
        wx.GetApp().rm_palette(self)
        self.GetParent().editdlg = None
        evt.Skip()

    def generate_panel(self, mm):
        self.panels[mm.__class__] = (EditListPanel(self.p1, list =  mm.panel1_param(), 
                                                   tip=mm.panel1_tip()),
                                     EditListPanel(self.p2, list =  mm.panel2_param(),
                                                   tip=mm.panel2_tip()),
                                     EditListPanel(self.p3, list =  mm.panel3_param(),
                                                   tip=mm.panel3_tip()),)
                                     
    def OnEL_Changed(self, evt):
        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
  
        p1children = self.p1sizer.GetChildren()
        phys = None
        if len(p1children) > 0:
            elp1 = p1children[0].GetWindow()
            v1 = elp1.GetValue()
            mm.import_panel1_value(v1)
            try:
                phys = mm.get_root_phys()
            except:
                pass
        if mm.has_2nd_panel:
            p2children = self.p2sizer.GetChildren()
            if len(p2children) > 0:
                elp2 = p2children[0].GetWindow()
                v2 = elp2.GetValue()
                mm.import_panel2_value(v2)
        if mm.has_3rd_panel:                
            p3children = self.p3sizer.GetChildren()
            if len(p3children) > 0:
                elp3 = p3children[0].GetWindow()
                v3 = elp3.GetValue()
                mm.import_panel3_value(v3)
        if phys is not None:
           viewer = self.GetParent()
           try:
               engine = viewer.engine.assign_sel_index(phys)
           except:
               traceback.print_exc()
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

    def OnEnableItem(self, evt):
        import   ifigure.widgets.dialog as dialog
        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
        mm.enabled = True
        self.tree.RefreshItems()

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
        from collections import OrderedDict
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
        from collections import OrderedDict
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
                
        if isinstance(mm, Domain):
            tt = 'domain'
        elif isinstance(mm, Bdry):
            tt = 'bdry'            
        elif isinstance(mm, Edge):
            tt = 'edge'
        elif isinstance(mm, Pair):
            tt = 'pair'
        else:
           return false_value

        return True, tt, idx, labels

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
        
        

