import os
import wx
import sys
import subprocess as sp

from ifigure.utils.edit_list import EditListPanel, EDITLIST_CHANGED
from ifigure.utils.cbook import BuildPopUpMenu

font_h = None
font_w = None
font = None
font_label = None


def set_default_font():
    size = 12
    font = wx.Font(pointSize=size, family=wx.DEFAULT,
                   style=wx.NORMAL,  weight=wx.NORMAL,
                   faceName='Consolas')
    globals()['font_label'] = wx.Font(pointSize=size, family=wx.DEFAULT,
                                      style=wx.NORMAL,  weight=wx.BOLD,
                                      faceName='Consolas')
    dc = wx.ScreenDC()
    dc.SetFont(font)
    w, h = dc.GetTextExtent('A')
    globals()['font_h'] = h*1.5
    globals()['font_w'] = w
    globals()['font'] = font


def install_from_github(url, update=False):
    """
    Installs a Python library from a GitHub repo.

    """
    if not url.startswith("git+"):
        url = "git+" + url
    command = [sys.executable, "-m", "pip", "install"]
    if update:
        command.append("-U")
    command.append(url)
    try:
        # python -m pip install
        sp.check_call(command,
                      # stdout=sp.PIPE,
                      # stderr=sp.STDOUT,
                      )
    except sp.CalledProcessError as e:
        print(f"Installation failed. Error: {e}")
        print(f"Output: {e.output.decode()}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


class pkglist_popup(wx.Menu):
    def __init__(self, parent):
        super(pkglist_popup, self).__init__()
        self.parent = parent
        pkg = parent.packages[parent.selected_row]

        self.target_pkg = pkg
        if pkg["html_url"] == "?":
            menus = [('Recheck Repos...', self.onRecheck, None),
                     ('** can not update from here ** ', None, None),]
        elif pkg["latest"] == "?":
            menus = [('Recheck Repos...', self.onRecheck, None),
                     ('** can not update from here ** ', None, None),]
        elif pkg["installed"] == "no":
            menus = [('Install', self.onInstall, None),
                     ('Recheck Repos...', self.onRecheck, None),]
        else:
            menus = [('Update', self.onUpdate, None),
                     ('Recheck Repos...', self.onRecheck, None),]

        BuildPopUpMenu(self, menus)

    def onInstall(self, evt):
        install_from_github(self.target_pkg["html_url"])
        self.parent.update_done = True
        self.parent.do_recheck()

    def onUpdate(self, evt):
        install_from_github(self.target_pkg["html_url"])
        self.parent.update_done = True
        self.parent.do_recheck()

    def onRecheck(self, evt):
        self.parent.do_recheck()


class dlg_packages(wx.Dialog):
    def __init__(self, parent, id=wx.ID_ANY, title='packages'):
        from petram.utils import get_user_config

        config = get_user_config()
        self.repos = config["repos"]

        set_default_font()

        wx.Dialog.__init__(self, parent, wx.ID_ANY, title,
                           style=wx.STAY_ON_TOP | wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        sizer = wx.BoxSizer(wx.VERTICAL)

        # grid
        self.SetSizer(sizer)

        self.grid = wx.grid.Grid(self)
        self.grid.CreateGrid(10, 5)
        self.grid.HideRowLabels()
        self.grid.SetSelectionMode(wx.grid.Grid.SelectRows)
        self.grid.SetDefaultCellFont(font)
        self.grid.SetLabelFont(font_label)
        self.grid.SetColLabelSize(int(font_h))
        self.grid.SetDefaultRowSize(int(font_h), True)
        self.grid.EnableDragColSize(True)
        # self.grid.SetTable(VarViewerGridTable(None, self.grid))
        self.grid.SetColLabelValue(0, "package")
        self.grid.SetColLabelValue(1, "version")
        self.grid.SetColLabelValue(2, "description")
        self.grid.SetColLabelValue(3, "url")
        self.grid.SetColLabelValue(4, "lastest")

        sizer.Add(self.grid, 1, wx.EXPAND, 0)

        # buttons
        sizer0 = wx.BoxSizer(wx.HORIZONTAL)

        okbutton = wx.Button(self, wx.ID_OK, "OK")
        sizer0.AddStretchSpacer()
        sizer0.Add(okbutton, 0, wx.ALIGN_CENTER | wx.ALL, 1)
        okbutton.Bind(wx.EVT_BUTTON, self.onOK)

        sizer.Add(sizer0, 0, wx.EXPAND | wx.ALL, 10)

        self.fill_grid()

        # if add_palette:
        wx.GetApp().add_palette(self)

        #
        self.grid.Bind(wx.grid.EVT_GRID_SELECT_CELL, self.onCellSelected)
        self.grid.Bind(wx.grid.EVT_GRID_CELL_RIGHT_CLICK, self.onRightRelease)
        #
        self.selected_row = -1
        self.update_done = False

        self.Show()
        wx.CallAfter(self._myRefresh)

    def fill_grid(self):
        from petram.remote.get_repo_info import (get_local_packages,
                                                 get_repo_info)

        urls = [x['url'] for x in self.repos]

        self.packages = get_repo_info(urls=urls)

        nrow = self.grid.GetNumberRows()
        ldif = len(self.packages) - nrow

        if ldif > 0:
            self.grid.AppendRows(ldif)
        elif ldif < 0:
            self.grid.DeleteRows(0, -ldif)
        else:
            pass

        for k, p in enumerate(self.packages):
            self.grid.SetCellValue(k, 0, p['module'])
            self.grid.SetCellValue(k, 1, p['version'])
            self.grid.SetCellValue(k, 2, p['description'])
            self.grid.SetCellValue(k, 3, p['html_url'])
            self.grid.SetCellValue(k, 4, p['latest'])
        self.grid.AutoSizeColumns()

    def onCellSelected(self, evt):
        self.selected_row = evt.GetRow()
        evt.Skip()

    def onRightRelease(self, evt):
        if self.selected_row < 0:
            return
        m = pkglist_popup(self)
        self.PopupMenu(m,
                       evt.GetPosition())
        m.Destroy()

    def do_recheck(self):
        self.fill_grid()
        self.selected_row = -1
        wx.CallAfter(self._myRefresh)

    def onOK(self, evt):
        self.Close()

    def _myRefresh(self):
        self.Fit()
        self.Layout()


def check_packages(parent):
    dlg = dlg_packages(parent)

    def close_dlg(evt, dlg=dlg):
        if dlg.update_done:
            from ifigure.widgets.dialog import message
            wx.CallAfter(message, parent,
                         "Packages are updated. Restart piScope, in order to use updated modules.",
                         style=0,
                         title="Update recommended")

        dlg.Destroy()
    dlg.Bind(wx.EVT_CLOSE, close_dlg)
    return dlg
