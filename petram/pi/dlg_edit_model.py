from __future__ import print_function
from petram.pi.simple_frame_plus import SimpleFramePlus

import os
import wx
from collections import OrderedDict
import traceback
from ifigure.utils.cbook import BuildPopUpMenu
from ifigure.utils.edit_list import EditListPanel, ScrolledEditListPanel
from ifigure.utils.edit_list import EDITLIST_CHANGED,  EDITLIST_CHANGING
from ifigure.utils.edit_list import EDITLIST_SETFOCUS
from ifigure.widgets.miniframe_with_windowlist import MiniFrameWithWindowList
from ifigure.widgets.miniframe_with_windowlist import DialogWithWindowList
import ifigure.widgets.dialog as dialog

import petram.helper.pickle_wrapper as pickle

try:
    import treemixin
except ImportError:
    from wx.lib.mixins import treemixin

from petram.mfem_model import MFEM_ModelRoot

from ifigure.ifigure_config import rcdir
petram_model_scratch = os.path.join(rcdir, 'petram_model_scratch')


class ModelTree(treemixin.VirtualTree, wx.TreeCtrl):
    def __init__(self, *args, **kwargs):
        self.topwindow = kwargs.pop('topwindow')
        super(ModelTree, self).__init__(*args, **kwargs)

    def OnGetItemText(self, indices):
        item = self.topwindow.model.GetItem(indices)
        txt = self.topwindow.model.GetItemText(indices)

        # if item.has_ns() or item.has_nsref():
        info = item.get_info_str()
        if info != "":
            txt = txt + "(" + info + ")"
        if hasattr(item, 'isGeom') and hasattr(item, '_newobjs'):
            if len(item._newobjs) < 10:
                txt = txt + '('+','.join(item._newobjs) + ')'
            else:
                txt = txt + '(total '+str(len(item._newobjs)) + ' items)'
        return txt

    def OnGetItemTextColour(self, indices):
        item = self.topwindow.model.GetItem(indices)
        if item.enabled:
            return wx.BLACK
        else:
            return wx.Colour(128, 128, 128)

    def OnGetItemFont(self, indices):
        item = self.topwindow.model.GetItem(indices)
        if item.enabled:
            return wx.NORMAL_FONT
        else:
            return wx.ITALIC_FONT

    def OnGetChildrenCount(self, indices):
        return self.topwindow.model.GetChildrenCount(indices)

    def GetSelection(self):
        # this returns only one selection
        # called when only one element is assumed to be selected
        ret = self.GetSelections()
        if len(ret) == 0:
            return None
        return ret[0]

    def isMultipleSelection(self):
        return len(self.GetSelections()) > 1

# class DlgEditModel(MiniFrameWithWindowList):


# class DlgEditModel(DialogWithWindowList):


class DlgEditModel(SimpleFramePlus):
    def __init__(self, parent, id, title, model=None):

        self.model = model if not model is None else MFEM_ModelRoot()
        '''
        (use this style if miniframe is used)
        style=(wx.CAPTION|
                       wx.CLOSE_BOX|
                       wx.MINIMIZE_BOX| 
                       wx.RESIZE_BORDER|
                       wx.FRAME_FLOAT_ON_PARENT)
        '''
        style = wx.CAPTION | wx.RESIZE_BORDER | wx.SYSTEM_MENU
        style = (wx.CAPTION |
                 wx.CLOSE_BOX |
                 wx.MINIMIZE_BOX |
                 wx.RESIZE_BORDER |
                 wx.FRAME_FLOAT_ON_PARENT)
        #        wx.FRAME_TOOL_WINDOW  this style may not work on Mac/Windows

        #style = wx.RESIZE_BORDER
        super(DlgEditModel, self).__init__(parent, id, title, style=style)

        self.splitter = wx.SplitterWindow(self, wx.ID_ANY,
                                          style=wx.SP_NOBORDER | wx.SP_LIVE_UPDATE | wx.SP_3DSASH)

        p0 = wx.Panel(self.splitter)
        p0.SetSizer(wx.BoxSizer(wx.HORIZONTAL))
        p0sizer = wx.BoxSizer(wx.VERTICAL)
        p0.GetSizer().Add(p0sizer, 1, wx.EXPAND)

        self.tree = ModelTree(p0, topwindow=self,
                              style=wx.TR_DEFAULT_STYLE | wx.TR_MULTIPLE)

        from ifigure.utils.wx3to4 import wxNamedColour
        p0.SetBackgroundColour(wxNamedColour('White'))

        p0sizer.Add(self.tree, 1, wx.EXPAND | wx.ALL, 1)
        #self.tree.SetSizeHints(150, -1, maxW=150)
        self.nb = wx.Notebook(self.splitter)
        self.splitter.SplitVertically(p0, self.nb)
        self.splitter.SetMinimumPaneSize(150)
        wx.CallAfter(self.splitter.SetSashPosition, 150, True)

        self.p1 = wx.Panel(self.nb)
        self.p2 = wx.Panel(self.nb)
        self.p3 = wx.Panel(self.nb)
        self.p4 = wx.Panel(self.nb)
        self.p1.SetBackgroundColour(wx.Colour(235, 235, 235, 255))
        self.p2.SetBackgroundColour(wx.Colour(235, 235, 235, 255))
        self.p3.SetBackgroundColour(wx.Colour(235, 235, 235, 255))
        self.p4.SetBackgroundColour(wx.Colour(235, 235, 235, 255))
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
        s.Add(self.splitter,  1, wx.EXPAND | wx.ALL, 1)
        self.Bind(wx.EVT_TREE_ITEM_RIGHT_CLICK,
                  self.OnItemRightClick)
        self.Bind(wx.EVT_TREE_SEL_CHANGED,
                  self.OnItemSelChanged)
        self.Bind(wx.EVT_TREE_SEL_CHANGING,
                  self.OnItemSelChanging)
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
        self._copied_item = None
        self._opened_dlg = None
        self._enable = True
        self.SetSize((600, 400))

        wx.CallAfter(self.CentreOnParent)
        #hbox = wx.BoxSizer(wx.HORIZONTAL)
        #self.GetSizer().Add(hbox, 0, wx.EXPAND|wx.ALL,5)
        #button=wx.Button(self, wx.ID_ANY, "Close")
        #button.Bind(wx.EVT_BUTTON, self.CallClose)
        # hbox.AddStretchSpacer()
        #hbox.Add(button, 0, wx.ALL,1)

    def CallClose(self, evt):
        self.Close()

    def OnChildFocus(self, evt):
        self.GetParent()._palette_focus = 'edit'
        evt.Skip()

    def OnItemRightClick(self, e):
        tree = self.tree
        menus = []
        if tree.isMultipleSelection():
            items = tree.GetSelections()
            indices = [tree.GetIndexOfItem(ii) for ii in items]
            mm = [self.model.GetItem(ii) for ii in indices]
            if all([mm[0].parent == m.parent for m in mm]):
                if mm[0].enabled:
                    menus = menus + [('Disable', self.OnDisableItemMult, None)]
                else:
                    menus = menus + [('Enable', self.OnEnableItemMult, None)]
                #menus.append(('Duplicate', self.OnDuplicateItemFromModelMult, None))
                menus.append(('Copy', self.OnCopyItemFromModelMult, None))
            if all([m.can_delete for m in mm]):
                menus.append(('Delete', self.OnDeleteItemFromModelMult, None))

        else:
            indices = tree.GetIndexOfItem(tree.GetSelection())
            mm = self.model.GetItem(indices)
            for xxxx in mm.get_possible_child_menu():

                submenu, cls = xxxx
                if cls is not None:
                    txt = cls.fancy_menu_name()
                    txt2 = cls.fancy_tree_name()

                    def add_func(evt, cls=cls, indices=indices, tree=tree,
                                 namebase=txt2, model=self.model):
                        parent = model.GetItem(indices)

                        # build stop is a flag for precedual construction of geom/mesh
                        if hasattr(parent, '_build_stop'):
                            before, after = parent.build_stop
                        elif hasattr(parent.parent, 'build_stop'):
                            before, after = parent.parent.build_stop
                        else:
                            before, after = None, None

                        try:
                            name = parent.add_item(namebase, cls,
                                                   before=before, after=after)
                        except:
                            dialog.showtraceback(parent=self,
                                                 txt="Failed to add child",
                                                 title='Error',
                                                 traceback=traceback.format_exc())
                            return

                        child = parent[name]
                        try:
                            child.on_created_in_tree()
                        except:
                            del parent[name]
                            dialog.showtraceback(parent=self,
                                                 txt="Failed to add child",
                                                 title='Error',
                                                 traceback=traceback.format_exc())
                            return

                        viewer = self.GetParent()
                        viewer.model.scripts.helpers.rebuild_ns()
                        engine = viewer.engine
                        model.GetItem(indices)[
                            name].postprocess_after_add(engine)
                        tree.RefreshItems()

                        viewer.engine.run_mesh_extension_prep(reset=True)

                        old_item = tree.GetItemByIndex(parent.GetIndices())
                        tree.Expand(old_item)
                        tree.SelectItem(old_item, select=False)

                        new_item = tree.GetItemByIndex(child.GetIndices())
                        tree.SelectItem(new_item)
                        evt.Skip()
                else:
                    add_func = None
                if len(submenu) != 0:
                    if submenu == "!":
                        if add_func is not None:
                            menus = menus+[('Add '+txt, add_func, None), ]
                        menus = menus+[('!', None, None), ]
                    else:
                        menus = menus+[('+'+submenu, None, None), ]
                        if add_func is not None:
                            menus = menus+[('Add '+txt, add_func, None), ]
                else:
                    menus = menus+[('Add '+txt, add_func, None), ]
            for t, m, m2 in mm.get_special_menu(e):
                menus = menus+[(t, m, m2), ]
            menus = menus + [('---', None, None)]
            if (mm.has_ns() or mm.has_nsref()) and not mm.hide_ns_menu:
                if mm.ns_name is not None:
                    menus.append(("Delete NS.",  self.OnDelNS, None))
                    if hasattr(mm, '_global_ns'):
                        menus.append(
                            ("Initialize Dataset", self.OnInitDataset, None))
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
                                  None),
                                 ('Copy', self.OnCopyItemFromModel,
                                  None), ]

            if os.path.exists(petram_model_scratch):
                menus = menus + [('Paste Item', self.OnPasteItemToModel,
                                  None)]
            if mm.can_delete:
                if not mm.mustbe_firstchild:
                    menus = menus + [('+Move...', None, None),
                                     ('Up', self.OnMoveItemUp, None),
                                     ('Down', self.OnMoveItemDown, None),
                                     ('To...', self.OnMoveItemTo, None),
                                     ('!', None, None), ]
                menus = menus + [('Delete', self.OnDeleteItemFromModel, None)]
            if mm.can_rename:
                menus = menus + [('Rename...', self.OnRenameItem, None)]
            if menus[-1][0] != '---':
                menus = menus + [('---', None, None)]

            menus = menus + [('Export to shell', self.OnExportToShell, None)]

        menus = menus + [('Refresh', self.OnRefreshTree, None)]
        m = wx.Menu()
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
        app.shell.SendShellEnterEvent()
        ret = dialog.message(app, mm.name() + ' is exported', 'Export', 0)

    def OnCopyItemFromModelMult(self, evt):
        tree = self.tree
        items = tree.GetSelections()
        indices = [tree.GetIndexOfItem(ii) for ii in items]
        mm = [self.model.GetItem(ii) for ii in indices]
        base = [m.split_digits()[0] for m in mm]

        _copied_item = (base, mm)

        fid = open(petram_model_scratch, 'wb')
        pickle.dump(_copied_item, fid)
        fid.close()

    def OnCopyItemFromModel(self, evt):
        return self.OnCopyItemFromModelMult(evt)

    def OnPasteItemToModel(self, evt):
        if self.tree.isMultipleSelection():
            return

        try:
            fid = open(petram_model_scratch, 'rb')
            _copied_item = pickle.load(fid)
            fid.close()
        except:
            import traceback
            traceback.print_exc()
            return

        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
        for cbase, cmm in zip(*_copied_item):
            if not cmm.__class__ in mm.get_possible_child():
                print("Cannot paste "+cmm.__class__.__name__)
                continue
            mm.add_itemobj(cbase, cmm)
        self.tree.RefreshItems()
        self.OnEvalNS(evt)

    def OnDuplicateItemFromModelMult(self, evt):
        pass

    def OnDuplicateItemFromModel(self, evt):
        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        self.tree.SelectItem(self.tree.GetSelection(), select=False)

        mm = self.model.GetItem(indices)
        name = mm.name()
        base, num = mm.split_digits()
        parent = mm.parent

        newmm = pickle.loads(pickle.dumps(mm))

        index = list(parent).index(name)
        nums = []
        for key in parent.keys():
            #base0 = ''.join([k for k in key if not k.isdigit()])
            base0, num = parent[key].split_digits()
            if base0 != base:
                continue

            #nums.append(int(''.join([k for k in key if k.isdigit()])))
            nums.append(int(num))

        parent.insert_item(index+1, base+str(int(max(nums))+1), newmm)
        self.tree.RefreshItems()

        new_item = self.tree.GetItemByIndex(newmm.GetIndices())
        self.tree.SelectItem(new_item)
        self.OnEvalNS(evt)

    def OnDeleteItemFromModelMult(self, evt):
        tree = self.tree
        items = tree.GetSelections()
        indices = [tree.GetIndexOfItem(ii) for ii in items]
        mmm = [self.model.GetItem(ii) for ii in indices]
        texts = [self.model.GetItemText(ii) for ii in indices]
        for mm, text in zip(mmm, texts):
            del mm.parent[text]
        self.tree.RefreshItems()
        self.OnItemSelChanged()

    def OnDeleteItemFromModel(self, evt):
        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
        text = self.model.GetItemText(indices)
        del mm.parent[text]
        self.tree.RefreshItems()
        self.OnItemSelChanged()

    def OnRenameItem(self, evt):
        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)

        def callback(value):
            mm.rename(value[1])
            self.tree.RefreshItems()
            self.Enable(True)

        self.open_textentry(parent=self, message='Enter new name', title='Rename',
                            def_string=mm.name()+'_renamed',
                            ok_callback=callback)

    def OnRefreshTree(self, evt=None):
        self.tree.RefreshItems()
        if evt is not None:
            evt.Skip()

    def OnItemSelChanging(self, evt):
        if self.tree.GetSelection() is None:
            return

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

    def show_panel(self, mm):
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
                self.nb.AddPage(self.p2, mm.panel2_tabname())
            else:
                self.nb.SetPageText(1, mm.panel2_tabname())

            self.p1sizer.Add(p1panel, 1, wx.EXPAND | wx.ALL, 1)
            self.p2sizer.Add(p2panel, 1, wx.EXPAND | wx.ALL, 1)
            p1panel.SetValue(mm.get_panel1_value())
            p2panel.SetValue(mm.get_panel2_value())
            p1panel.Show()
            p2panel.Show()
            self.p1.Layout()
            self.p2.Layout()
            if mm.has_3rd_panel:
                if self.nb.GetPageCount() == 2:
                    self.nb.AddPage(self.p3, mm.panel3_tabname())
                else:
                    self.nb.SetPageText(2, mm.panel3_tabname())

                self.p3sizer.Add(p3panel, 1, wx.EXPAND | wx.ALL, 1)
                p3panel.SetValue(mm.get_panel3_value())
                p3panel.Show()
                self.p3.Layout()
            else:
                if self.nb.GetPageCount() > 3:
                    self.nb.RemovePage(3)
                if self.nb.GetPageCount() > 2:
                    self.nb.RemovePage(2)
                p3panel.Hide()

            if mm.has_4th_panel:
                if self.nb.GetPageCount() == 3:
                    self.nb.AddPage(self.p4, mm.panel4_tabname())
                else:
                    self.nb.SetPageText(3, mm.panel4_tabname())

                self.p4sizer.Add(p4panel, 1, wx.EXPAND | wx.ALL, 1)
                p4panel.SetValue(mm.get_panel4_value())
                p4panel.Show()
                #for c in p4panel.GetChildren(): c.Show()
                self.p4.Layout()
            else:
                if self.nb.GetPageCount() > 3:
                    self.nb.RemovePage(3)
                p4panel.Hide()
        else:
            if self.nb.GetPageCount() > 3:
                self.nb.RemovePage(3)
            if self.nb.GetPageCount() > 2:
                self.nb.RemovePage(2)
            if self.nb.GetPageCount() > 1:
                self.nb.RemovePage(1)
            self.p1sizer.Add(p1panel, 1, wx.EXPAND | wx.ALL, 1)
            p1panel.SetValue(mm.get_panel1_value())
            p1panel.Show()
            p2panel.Hide()
            p3panel.Hide()
            p4panel.Hide()
            self.p1.Layout()

        if not self._enable:
            self.Enable(False)

    def OnItemSelChanged(self, evt=None):

        if self.tree.GetSelection() is None:
            return

        from petram.phys.aux_operator import AUX_Operator
        from petram.phys.aux_variable import AUX_Variable

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
                self.nb.AddPage(self.p2, mm.panel2_tabname())
            else:
                self.nb.SetPageText(1, mm.panel2_tabname())

            self.p1sizer.Add(p1panel, 1, wx.EXPAND | wx.ALL, 1)
            self.p2sizer.Add(p2panel, 1, wx.EXPAND | wx.ALL, 1)
            p1panel.SetValue(mm.get_panel1_value())
            p2panel.SetValue(mm.get_panel2_value())
            p1panel.Show()
            p2panel.Show()
            self.p1.Layout()
            self.p2.Layout()
            if mm.has_3rd_panel:
                if self.nb.GetPageCount() == 2:
                    self.nb.AddPage(self.p3, mm.panel3_tabname())
                else:
                    self.nb.SetPageText(2, mm.panel3_tabname())

                self.p3sizer.Add(p3panel, 1, wx.EXPAND | wx.ALL, 1)
                p3panel.SetValue(mm.get_panel3_value())
                p3panel.Show()
                self.p3.Layout()
            else:
                if self.nb.GetPageCount() > 3:
                    self.nb.RemovePage(3)
                if self.nb.GetPageCount() > 2:
                    self.nb.RemovePage(2)
                p3panel.Hide()

            if mm.has_4th_panel:
                if self.nb.GetPageCount() == 3:
                    self.nb.AddPage(self.p4, mm.panel4_tabname())
                else:
                    self.nb.SetPageText(3, mm.panel4_tabname())

                self.p4sizer.Add(p4panel, 1, wx.EXPAND | wx.ALL, 1)
                p4panel.SetValue(mm.get_panel4_value())
                p4panel.Show()
                #for c in p4panel.GetChildren(): c.Show()
                self.p4.Layout()
            else:
                if self.nb.GetPageCount() > 3:
                    self.nb.RemovePage(3)
                p4panel.Hide()
        else:
            if self.nb.GetPageCount() > 3:
                self.nb.RemovePage(3)
            if self.nb.GetPageCount() > 2:
                self.nb.RemovePage(2)
            if self.nb.GetPageCount() > 1:
                self.nb.RemovePage(1)
            self.p1sizer.Add(p1panel, 1, wx.EXPAND | wx.ALL, 1)
            p1panel.SetValue(mm.get_panel1_value())
            p1panel.Show()
            p2panel.Hide()
            p3panel.Hide()
            p4panel.Hide()
            self.p1.Layout()

        self._focus_idx = None

        from petram.model import Bdry, Domain, Point, Pair
        from petram.phys.phys_model import PhysModule

        viewer = self.GetParent()
        engine = viewer.engine

        if isinstance(mm, PhysModule):
            if not mm.enabled:
                viewer.highlight_none()
                viewer._dom_bdr_sel = ([], [], [], [])
            else:
                if not hasattr(mm, '_phys_sel_index') or mm.sel_index == 'all':
                    engine.assign_sel_index(mm)
                if hasattr(mm, '_phys_sel_index'):
                    # need this if in case mesh is not loaded....
                    if mm.dim == 3:
                        viewer.change_panel_button('domain')
                        viewer.highlight_domain(mm._phys_sel_index)
                        viewer._dom_bdr_sel = (mm._phys_sel_index, [], [], [])
                    elif mm.dim == 2:
                        viewer.change_panel_button('face')
                        viewer.highlight_face(mm._phys_sel_index)
                        viewer._dom_bdr_sel = ([], mm._phys_sel_index, [], [])
                    elif mm.dim == 1:
                        viewer.change_panel_button('edge')
                        viewer.highlight_edge(mm._phys_sel_index)
                        viewer._dom_bdr_sel = ([], [], mm._phys_sel_index, [],)
                    else:
                        pass

        elif hasattr(mm, '_sel_index'):
            self._focus_idx = 0
            if not mm.enabled:
                viewer.highlight_none()
                viewer._dom_bdr_sel = ([], [], [], [])

            elif isinstance(mm, Bdry):
                if not hasattr(mm, '_sel_index') or mm.sel_index == 'remaining':
                    phys = mm.get_root_phys()
                    engine.assign_phys_pp_sel_index()
                    engine.assign_sel_index(phys)

                if mm.dim == 3:
                    viewer.change_panel_button('face')
                    viewer.highlight_face(mm._sel_index)
                    viewer._dom_bdr_sel = ([], mm._sel_index, [], [])
                elif mm.dim == 2:
                    viewer.change_panel_button('edge')
                    viewer.highlight_edge(mm._sel_index)
                    viewer._dom_bdr_sel = ([], [], mm._sel_index, [],)
                elif mm.dim == 1:
                    viewer.change_panel_button('dot')
                    viewer.highlight_point(mm._sel_index)
                    viewer._dom_bdr_sel = ([], [], [], mm._sel_index, )
                else:
                    pass

            elif isinstance(mm, Domain):
                if not hasattr(mm, '_sel_index') or mm.sel_index == 'remaining':
                    phys = mm.get_root_phys()
                    engine.assign_phys_pp_sel_index()
                    engine.assign_sel_index(phys)

                if mm.dim == 3:
                    viewer.change_panel_button('domain')
                    viewer.highlight_domain(mm._sel_index)
                    viewer._dom_bdr_sel = (mm._sel_index, [], [], [])
                elif mm.dim == 2:
                    viewer.change_panel_button('face')
                    viewer.highlight_face(mm._sel_index)
                    viewer._dom_bdr_sel = ([], mm._sel_index, [], [])
                elif mm.dim == 1:
                    viewer.change_panel_button('edge')
                    viewer.highlight_edge(mm._sel_index)
                    viewer._dom_bdr_sel = ([], [], mm._sel_index, [],)
                else:
                    pass

            elif isinstance(mm, Point):
                if not hasattr(mm, '_sel_index') or mm.sel_index == 'remaining':
                    phys = mm.get_root_phys()
                    engine.assign_phys_pp_sel_index()
                    engine.assign_sel_index(phys)

                viewer.change_panel_button('vertex')
                viewer.highlight_point(mm._sel_index)
                viewer._dom_bdr_sel = ([], [], [], mm._sel_index)

        elif isinstance(mm, AUX_Operator) or isinstance(mm, AUX_Variable):
            if not mm.enabled:
                viewer.highlight_none()
                viewer._dom_bdr_sel = ([], [], [], [])
            else:
                mm2 = mm.parent
                if not hasattr(mm2, '_phys_sel_index') or mm2.sel_index == 'all':
                    engine.assign_sel_index(mm2)
                if hasattr(mm2, '_phys_sel_index'):
                    # need this if in case mesh is not loaded....
                    if mm2.dim == 3:
                        viewer.change_panel_button('domain')
                        viewer.highlight_domain(mm2._phys_sel_index)
                        viewer._dom_bdr_sel = (mm2._phys_sel_index, [], [], [])
                    elif mm2.dim == 2:
                        viewer.change_panel_button('face')
                        viewer.highlight_face(mm2._phys_sel_index)
                        viewer._dom_bdr_sel = ([], mm2._phys_sel_index, [], [])
                    elif mm2.dim == 1:
                        viewer.change_panel_button('edge')
                        viewer.highlight_edge(mm2._phys_sel_index)
                        viewer._dom_bdr_sel = (
                            [], [], mm2._phys_sel_index, [],)
                    else:
                        pass

        else:
            pass

        if evt is not None:
            mm.onItemSelChanged(evt)
            evt.Skip()

        if not self._enable:
            self.Enable(False)

    def OnClose(self, evt):
        wx.GetApp().rm_palette(self)
        self.GetParent().editdlg = None
        evt.Skip()

    def generate_panel(self, mm):
        if mm.fullname() in self.panels and not mm.always_new_panel:
            self.update_panel_label(mm)
        else:
            self.panels[mm.fullname()] = (ScrolledEditListPanel(self.p1,
                                                                list=mm.panel1_param(),
                                                                tip=mm.panel1_tip()),
                                          EditListPanel(self.p2, list=mm.panel2_param(),
                                                        tip=mm.panel2_tip()),
                                          EditListPanel(self.p3, list=mm.panel3_param(),
                                                        tip=mm.panel3_tip()),
                                          EditListPanel(self.p4, list=mm.panel4_param(),
                                                        tip=mm.panel4_tip()),)

    def update_panel_label(self, mm):
        self.panels[mm.fullname()][0].update_label(mm.panel1_param())
        self.panels[mm.fullname()][1].update_label(mm.panel2_param())
        self.panels[mm.fullname()][2].update_label(mm.panel3_param())
        self.panels[mm.fullname()][3].update_label(mm.panel4_param())

    def import_selected_panel_value(self, evt=None):
        if self.tree.GetSelection() is None:
            return
        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)

        p1children = self.p1sizer.GetChildren()
        phys = None
        viewer_update = False
        if len(p1children) > 0:
            elp1 = p1children[0].GetWindow()
            v1 = elp1.GetValue()

            viewer_update = mm.import_panel1_value(v1) or viewer_update
            try:
                phys = mm.get_root_phys()
            except:
                pass
            elp1.SetValue(mm.get_panel1_value())
            elp1.update_label(mm.panel1_param())

        if mm.has_2nd_panel:
            p2children = self.p2sizer.GetChildren()
            if len(p2children) > 0:
                elp2 = p2children[0].GetWindow()
                v2 = elp2.GetValue()
                viewer_update = mm.import_panel2_value(v2) or viewer_update
                elp2.SetValue(mm.get_panel2_value())

        if mm.has_3rd_panel:
            p3children = self.p3sizer.GetChildren()
            if len(p3children) > 0:
                elp3 = p3children[0].GetWindow()
                v3 = elp3.GetValue()
                viewer_update = mm.import_panel3_value(v3) or viewer_update
                elp3.SetValue(mm.get_panel3_value())

        if mm.has_4th_panel:
            p4children = self.p4sizer.GetChildren()
            if len(p4children) > 0:
                elp4 = p4children[0].GetWindow()
                v4 = elp4.GetValue()
                viewer_update = mm.import_panel4_value(v4) or viewer_update
                elp4.SetValue(mm.get_panel4_value())

        if phys is not None:
            viewer = self.GetParent()
            try:
                viewer.engine.run_mesh_extension_prep(reset=True)
                engine = viewer.engine.assign_sel_index(phys)
            except:
                traceback.print_exc()

        if viewer_update:
            flag1 = mm.update_after_ELChanged(self)
            if evt is not None:
                flag2 = mm.update_after_ELChanged2(evt)
            else:
                flag2 = False
            if flag1 or flag2:
                wx.CallAfter(self.show_panel, mm)

        self.tree.RefreshItems()
        return viewer_update

    def OnEL_Changed(self, evt):
        viewer_update = self.import_selected_panel_value(evt)
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
        # print  self._focus_obj

    def OnAddNS(self, evt):
        import ifigure.widgets.dialog as dialog
        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
        txt = self.model.GetItemText(indices)

        def callback(value):
            mm.new_ns(value[1])
            self.tree.RefreshItems()
            self.Enable(True)

        self.open_textentry(self,
                            "Enter namespace name",
                            "New NS...",
                            txt.lower(),
                            ok_callback=callback)
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
        if not hasattr(mm, '_global_ns'):
            return
        if not 'dataset_names' in mm._global_ns:
            return
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

    def _enabile_mm(self, value):
        tree = self.tree
        items = tree.GetSelections()
        indices = [tree.GetIndexOfItem(ii) for ii in items]
        mm = [self.model.GetItem(ii) for ii in indices]

        for m in mm:
            m.enabled = value
        self.tree.RefreshItems()

        mm = mm[0]
        if not hasattr(mm, 'get_root_phys'):
            return

        phys = mm.get_root_phys()
        if phys is not None:
            viewer = self.GetParent()
            try:
                viewer.engine.assign_sel_index(phys)
            except:
                traceback.print_exc()

    def OnDisableItemMult(self, evt):
        self._enabile_mm(False)

    def OnEnableItemMult(self, evt):
        self._enabile_mm(True)

    def OnDisableItem(self, evt):
        self._enabile_mm(False)

    def OnEnableItem(self, evt):
        self._enabile_mm(True)

    def get_selected_mm(self):
        import ifigure.widgets.dialog as dialog
        if self.tree.GetSelection() is None:
            return

        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
        return mm

    def select_next_enabled(self):
        if self.tree.GetSelection() is None:
            return

        item = self.tree.GetSelection()
        item0 = item
        self.tree.UnselectAll()
        while True:
            item = self.tree.GetNextSibling(item)
            if not item.IsOk():
                self.tree.SelectItem(item0)
                return
            indices = self.tree.GetIndexOfItem(item)
            mm = self.model.GetItem(indices)
            if mm.enabled:
                self.tree.SelectItem(item)
                return
        self.tree.SelectItem(item0)
        wx.CallAfter(self.Refresh, False)

    @staticmethod
    def MoveItemInList(l, i1, i2):
        if i1 > i2:
            return l[0:i2] + [l[i1]] + l[i2:i1] + l[i1+1:len(l)]
        elif i1 < i2:
            return l[0:i1] + l[i1+1:i2+1] + [l[i1]] + l[i2+1:len(l)]

    def OnMoveItemUp(self, evt):
        if self.tree.GetSelection() is None:
            return

        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
        p = mm.parent
        names = list(p._contents)

        idx = names.index(mm.name())
        if idx == 0:
            return

        new_names = self.MoveItemInList(names, idx, idx-1)

        p._contents = OrderedDict((k, p._contents[k]) for k in new_names)
        self.tree.RefreshItems()

        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
        self.show_panel(mm)

    def OnMoveItemDown(self, evt):
        if self.tree.GetSelection() is None:
            return

        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
        p = mm.parent
        names = list(p._contents)

        idx = names.index(mm.name())
        if idx == len(names)-1:
            return

        new_names = self.MoveItemInList(names, idx, idx+1)

        p._contents = OrderedDict((k, p._contents[k]) for k in new_names)
        self.tree.RefreshItems()

        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
        self.show_panel(mm)

    def OnMoveItemTo(self, evt):
        from ifigure.utils.edit_list import DialogEditList
        import ifigure.widgets.dialog as dialog

        if self.tree.GetSelection() is None:
            return

        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
        p = mm.parent
        names = list(p._contents)
        idx = names.index(mm.name())

        list6 = [
            ["New parent", p.name(), 0],
            ["Index (0-base)", str(idx), 0], ]
        value = DialogEditList(list6, modal=True,
                               style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
                               tip=None,
                               parent=self,
                               title='Move item to...')
        if not value[0]:
            return

        if value[1][0] != p.name():
            try:
                assert False, "Moving under diffent parent is not supported"
            except AssertionError:
                dialog.showtraceback(parent=self,
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

        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)
        self.show_panel(mm)

    def isSelectionPanelOpen(self):
        from petram.model import Bdry, Point, Pair, Domain, Edge
        false_value = False, '', [], []

        if self.tree.GetSelection() is None:
            return false_value
        indices = self.tree.GetIndexOfItem(self.tree.GetSelection())
        mm = self.model.GetItem(indices)

        try:
            phys = mm.get_root_phys()
        except:
            return false_value

        if self.nb.GetPageCount() == 0:
            return false_value
        if (self.nb.GetPageCount() > 0 and
                self.nb.GetSelection() != 1):
            return false_value

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
        if mm.dim == 2:
            tt += 1
        if mm.dim == 1:
            tt += 2

        return True, tnames[tt], idx, labels

    def add_remove_AreaSelection(self,  idx, rm=False, flag=0):
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
                if int(x) in tgt:
                    tgt.remove(x)
        else:
            for x in idx:
                if not int(x) in tgt:
                    tgt.append(x)

        sidx = [' ' + str(x) for x in tgt]
        if len(tgt) > 0:
            sidx[0] = str(tgt[0])
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
        # evt.Skip()

    def Hide(self):
        super(DlgEditModel, self).Hide()
        if self._opened_dlg is not None:
            self._opened_dlg.Hide()

    def Show(self):
        super(DlgEditModel, self).Show()
        if self._opened_dlg is not None:
            self._opened_dlg.Show()

    def Raise(self):
        super(DlgEditModel, self).Raise()
        if self._opened_dlg is not None:
            self._opened_dlg.Raise()

    def Enable(self, value):
        super(DlgEditModel, self).Enable(value)
        self._enable = value
        self.enable_panels(value)

    def enable_panels(self, value):
        self.p1.Enable(value)
        self.p2.Enable(value)
        self.p3.Enable(value)
        self.p4.Enable(value)

    def textentry_close(self, value):
        self._opened_dlg = None
        self.Enable(True)

    def open_textentry(self, parent=None, message='', title='', def_string='', center=False,
                       ok_callback=None):
        l = max(len(def_string), 50)
        list = [[None, message+" "*l, 2],
                ["", def_string, 0], ]

        from ifigure.utils.edit_list import EditListMiniFrame
        dlg = EditListMiniFrame(parent, wx.ID_ANY, title, list=list, nobutton=False,
                                ok_callback=ok_callback,
                                close_callback=self.textentry_close)
        w, h = dlg.GetSize()
        dlg.Show()
        dlg.Raise()
        self._opened_dlg = dlg
        parent.Enable(False)

    def show_progress_bar(self, message, title='In progress', count=5):
        dlg = dialog.progressbar(self, message, title, count)
        dlg.Show()
        wx.GetApp().Yield()

        return dlg
